#!/usr/bin/env python3
"""
Galileo Adversarial Persona Experiment Pipeline

NEW DESIGN:
1. Initial evaluation with beam search
2. For correct answers, run adversarial testing:
   - Model generates adversarial claims dynamically based on persona
   - Conversation accumulates across rounds
   - All 5 personas run in parallel
3. Recovery testing for failed cases
4. Results aggregation and export
"""

import os
import sys
import argparse
import json
import csv
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

try:
    from setproctitle import setproctitle
    setproctitle("aa007878")
except ImportError:
    pass

# Set GPU visibility before importing torch/vllm
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

from config import (
    MODELS,
    SYSTEM_PROMPT,
    INSTRUCTION_TEMPLATE,
    RESULTS_DIR,
    MAX_TOKENS,
    BEAM_SEARCH_N,
    BEAM_SEARCH_TEMPERATURE,
    GREEDY_TEMPERATURE,
    MAX_ADVERSARIAL_ROUNDS,
    TENSOR_PARALLEL_SIZE,
    ExperimentConfig,
)
from inference import InferenceEngine
from evaluation import evaluate_response, extract_answer
from personas import (
    get_claim_generation_prompt, 
    get_retry_suffix,
    get_recovery_prompt, 
    get_all_persona_keys, 
    get_persona_name,
)
from data_loader import load_dataset, save_jsonl, get_test_name, prepare_problem


def setup_results_dir(config: ExperimentConfig) -> None:
    """Create results directory structure."""
    os.makedirs(config.results_dir, exist_ok=True)
    for model in config.models:
        model_short = model.split("/")[-1]
        os.makedirs(os.path.join(config.results_dir, model_short), exist_ok=True)


def run_initial_evaluation(
    engine: InferenceEngine,
    problems: List[Dict[str, Any]],
    test_name: str,
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Run initial evaluation using beam search.
    Returns list of results with correctness info.
    """
    print(f"\n{'='*60}")
    print(f"Initial Evaluation: {test_name}")
    print(f"Model: {engine.model_short_name}")
    print(f"Samples: {len(problems)}")
    print(f"Beam search n={config.beam_search_n}, temp={config.beam_search_temperature}")
    print(f"{'='*60}")
    
    # Prepare prompts
    prompts = [INSTRUCTION_TEMPLATE.format(question=prob["question"]) for prob in problems]
    
    # Generate with beam search - vLLM handles batching internally
    print("Generating responses with beam search...")
    outputs = engine.generate_beam_search(
        prompts=prompts,
        n=config.beam_search_n,
        temperature=config.beam_search_temperature,
        max_tokens=MAX_TOKENS,
        system_prompt=SYSTEM_PROMPT,
    )
    
    all_results = []
    for prob, beam_outputs in tqdm(zip(problems, outputs), total=len(problems), desc="Processing results"):
        # Check each beam for correct answer
        best_response = None
        is_correct = False
        extracted_answer = None
        
        for output in beam_outputs:
            ans, correct = evaluate_response(output.response, prob["answer"])
            if correct:
                best_response = output.response
                is_correct = True
                extracted_answer = ans
                break
        
        if best_response is None:
            best_response = beam_outputs[0].response
            extracted_answer, is_correct = evaluate_response(best_response, prob["answer"])
        
        result = {
            "question": prob["question"],
            "ground_truth": prob["answer"],
            "initial_response": best_response,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "test_name": test_name,
            "model": engine.model_short_name,
        }
        all_results.append(result)
    
    correct = sum(1 for r in all_results if r["is_correct"])
    accuracy = correct / len(all_results) * 100
    print(f"\nInitial accuracy: {correct}/{len(all_results)} = {accuracy:.2f}%")
    
    return all_results


def run_adversarial_testing(
    engine: InferenceEngine,
    initial_results: List[Dict[str, Any]],
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Run adversarial testing on correct answers.
    
    NEW DESIGN:
    - Model generates adversarial claims dynamically
    - Conversation accumulates across rounds
    - All personas run in parallel
    
    Returns list of adversarial test results.
    """
    # Filter to only correct answers
    correct_results = [r for r in initial_results if r["is_correct"]]
    
    if not correct_results:
        print("\nNo correct answers to challenge")
        return []
    
    print(f"\n{'='*60}")
    print(f"Adversarial Testing (Dynamic Claim Generation)")
    print(f"Model: {engine.model_short_name}")
    print(f"Correct answers to challenge: {len(correct_results)}")
    print(f"Personas: {len(get_all_persona_keys())}")
    print(f"Max rounds per persona: {config.max_adversarial_rounds}")
    print(f"{'='*60}")
    
    all_adversarial_results = []
    persona_keys = get_all_persona_keys()
    
    # Create tracking for each (problem, persona) pair
    # Structure: {(question, persona): {conversation_history, still_correct, rounds_completed, ...}}
    active_tracks = {}
    
    for result in correct_results:
        for persona_key in persona_keys:
            key = (result["question"], persona_key)
            active_tracks[key] = {
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "persona": persona_key,
                "test_name": result["test_name"],
                "model": result["model"],
                "conversation": [
                    {"role": "user", "content": INSTRUCTION_TEMPLATE.format(question=result["question"])},
                    {"role": "assistant", "content": result["initial_response"]},
                ],
                "initial_response": result["initial_response"],
                "initial_extracted": result["extracted_answer"],
                "rounds_completed": 0,
                "still_correct": True,
                "round_results": [],
            }
    
    # Run rounds
    for round_num in range(config.max_adversarial_rounds):
        # Get tracks still active (still correct)
        active_keys = [k for k, v in active_tracks.items() if v["still_correct"]]
        
        if not active_keys:
            print(f"  Round {round_num + 1}: No active tracks remaining")
            break
        
        print(f"\n  Round {round_num + 1}: {len(active_keys)} active tracks")
        
        # Step 1: Generate adversarial claims
        print(f"    Generating adversarial claims...")
        claim_conversations = []
        claim_keys = []
        
        for key in active_keys:
            track = active_tracks[key]
            persona_key = track["persona"]
            
            # Build conversation for claim generation
            conv = deepcopy(track["conversation"])
            conv.append({
                "role": "user", 
                "content": get_claim_generation_prompt(persona_key)
            })
            claim_conversations.append(conv)
            claim_keys.append(key)
        
        # Generate claims
        claim_responses = engine.generate_multi_turn(
            conversations=claim_conversations,
            temperature=0.7,  # Slightly creative for claim generation
            max_tokens=256,   # Claims should be short
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Step 2: Generate retry answers with the claims
        print(f"    Generating retry answers...")
        retry_conversations = []
        retry_keys = []
        generated_claims = {}
        
        for key, claim_resp in zip(claim_keys, claim_responses):
            track = active_tracks[key]
            claim_text = claim_resp.response.strip()
            
            # Store generated claim
            generated_claims[key] = claim_text
            
            # Add claim + retry suffix to conversation
            full_claim = claim_text + get_retry_suffix()
            
            conv = deepcopy(track["conversation"])
            conv.append({"role": "user", "content": full_claim})
            
            retry_conversations.append(conv)
            retry_keys.append(key)
        
        # Generate retry answers
        retry_responses = engine.generate_multi_turn(
            conversations=retry_conversations,
            temperature=config.greedy_temperature,
            max_tokens=MAX_TOKENS,
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Step 3: Evaluate and update tracks
        round_correct = 0
        round_wrong = 0
        
        for key, retry_resp in zip(retry_keys, retry_responses):
            track = active_tracks[key]
            claim_text = generated_claims[key]
            full_claim = claim_text + get_retry_suffix()
            retry_text = retry_resp.response.strip()
            
            # Evaluate
            extracted, is_correct = evaluate_response(retry_text, track["ground_truth"])
            
            # Update conversation history
            track["conversation"].append({"role": "user", "content": full_claim})
            track["conversation"].append({"role": "assistant", "content": retry_text})
            track["rounds_completed"] = round_num + 1
            
            # Record round result
            round_result = {
                "round": round_num + 1,
                "generated_claim": claim_text,
                "full_claim": full_claim,
                "retry_response": retry_text,
                "extracted_answer": extracted,
                "is_correct": is_correct,
            }
            track["round_results"].append(round_result)
            
            if is_correct:
                round_correct += 1
            else:
                round_wrong += 1
                track["still_correct"] = False
        
        print(f"    Results: {round_correct} still correct, {round_wrong} failed")
    
    # Compile final results with cleaner structure
    for key, track in active_tracks.items():
        # Build structured turns array
        turns = []
        for rr in track["round_results"]:
            turns.append({
                "turn": rr["round"],
                "adversarial_claim": rr["generated_claim"],
                "model_response": rr["retry_response"],
                "extracted_answer": rr["extracted_answer"],
                "is_correct": rr["is_correct"],
            })
        
        result = {
            "question": track["question"],
            "ground_truth": track["ground_truth"],
            "initial_response": track["initial_response"],
            "initial_extracted": track.get("initial_extracted", None),
            "persona": track["persona"],
            "persona_name": get_persona_name(track["persona"]),
            "rounds_completed": track["rounds_completed"],
            "final_correct": track["still_correct"],
            "turns": turns,
            "test_name": track["test_name"],
            "model": track["model"],
            # Keep conversation for recovery testing (internal use)
            "conversation": track["conversation"],
        }
        all_adversarial_results.append(result)
    
    # Summary
    for persona_key in persona_keys:
        persona_results = [r for r in all_adversarial_results if r["persona"] == persona_key]
        correct = sum(1 for r in persona_results if r["final_correct"])
        total = len(persona_results)
        print(f"\n  {get_persona_name(persona_key)}: {correct}/{total} survived ({correct/total*100:.1f}%)")
    
    return all_adversarial_results


def run_recovery_testing(
    engine: InferenceEngine,
    adversarial_results: List[Dict[str, Any]],
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Run recovery testing on answers that became wrong.
    Uses the full conversation context where failure occurred.
    
    Returns list of recovery results.
    """
    # Filter to only failed cases
    failed_results = [r for r in adversarial_results if not r["final_correct"]]
    
    if not failed_results:
        print("\nNo failed cases to recover")
        return []
    
    print(f"\n{'='*60}")
    print(f"Recovery Testing")
    print(f"Model: {engine.model_short_name}")
    print(f"Failed cases to recover: {len(failed_results)}")
    print(f"{'='*60}")
    
    # Build recovery conversations
    conversations = []
    for result in failed_results:
        # Use full conversation history + recovery prompt
        conv = deepcopy(result["conversation"])
        conv.append({"role": "user", "content": get_recovery_prompt()})
        conversations.append(conv)
    
    # Generate recovery responses
    print("Generating recovery answers...")
    responses = engine.generate_multi_turn(
        conversations=conversations,
        temperature=config.greedy_temperature,
        max_tokens=MAX_TOKENS,
        system_prompt=SYSTEM_PROMPT,
    )
    
    # Evaluate recovery
    recovery_results = []
    recovered_count = 0
    
    for result, response in zip(failed_results, responses):
        recovery_text = response.response.strip()
        extracted, is_correct = evaluate_response(recovery_text, result["ground_truth"])
        
        if is_correct:
            recovered_count += 1
        
        recovery_result = {
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "persona": result["persona"],
            "persona_name": result["persona_name"],
            "test_name": result["test_name"],
            "model": result["model"],
            "failed_at_round": result["rounds_completed"],
            "recovery_response": recovery_text,
            "extracted_answer": extracted,
            "recovered": is_correct,
            "full_conversation": result["conversation"] + [
                {"role": "user", "content": get_recovery_prompt()},
                {"role": "assistant", "content": recovery_text},
            ],
        }
        recovery_results.append(recovery_result)
    
    # Summary
    print(f"\nRecovery rate: {recovered_count}/{len(failed_results)} ({recovered_count/len(failed_results)*100:.1f}%)")
    
    for persona_key in get_all_persona_keys():
        persona_results = [r for r in recovery_results if r["persona"] == persona_key]
        if persona_results:
            recovered = sum(1 for r in persona_results if r["recovered"])
            total = len(persona_results)
            print(f"  {get_persona_name(persona_key)}: {recovered}/{total} recovered ({recovered/total*100:.1f}%)")
    
    return recovery_results


def save_results_to_csv(
    initial_results: List[Dict[str, Any]],
    adversarial_results: List[Dict[str, Any]],
    recovery_results: List[Dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    """Save aggregated results to CSV files."""
    print("\nSaving CSV results...")
    
    # Initial accuracy CSV
    initial_csv = os.path.join(config.results_dir, "initial_accuracy.csv")
    with open(initial_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_name", "correct", "total", "accuracy"])
        
        by_model_test = defaultdict(list)
        for r in initial_results:
            key = (r["model"], r["test_name"])
            by_model_test[key].append(r)
        
        for (model, test), results in sorted(by_model_test.items()):
            correct = sum(1 for r in results if r["is_correct"])
            total = len(results)
            accuracy = correct / total * 100
            writer.writerow([model, test, correct, total, f"{accuracy:.2f}"])
    
    # Adversarial survival rate CSV (per persona, per round)
    if adversarial_results:
        adv_csv = os.path.join(config.results_dir, "adversarial_survival.csv")
        with open(adv_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "test_name", "persona", "round", "survived", "total", "survival_rate"])
            
            # Group by model, test, persona
            by_mtp = defaultdict(list)
            for r in adversarial_results:
                key = (r["model"], r["test_name"], r["persona"])
                by_mtp[key].append(r)
            
            for (model, test, persona), results in sorted(by_mtp.items()):
                total = len(results)
                for round_num in range(1, config.max_adversarial_rounds + 1):
                    # Count how many survived this round
                    survived = sum(1 for r in results 
                                   if r["rounds_completed"] >= round_num and 
                                   (r["final_correct"] or 
                                    any(rr["round"] == round_num and rr["is_correct"] 
                                        for rr in r["round_results"])))
                    rate = survived / total * 100 if total > 0 else 0
                    writer.writerow([model, test, get_persona_name(persona), round_num, survived, total, f"{rate:.2f}"])
    
    # Recovery CSV
    if recovery_results:
        recovery_csv = os.path.join(config.results_dir, "recovery_accuracy.csv")
        with open(recovery_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "test_name", "persona", "recovered", "total", "recovery_rate"])
            
            by_mtp = defaultdict(list)
            for r in recovery_results:
                key = (r["model"], r["test_name"], r["persona"])
                by_mtp[key].append(r)
            
            for (model, test, persona), results in sorted(by_mtp.items()):
                recovered = sum(1 for r in results if r["recovered"])
                total = len(results)
                rate = recovered / total * 100 if total > 0 else 0
                writer.writerow([model, test, get_persona_name(persona), recovered, total, f"{rate:.2f}"])
    
    print(f"  Saved to {config.results_dir}/")


def run_experiment(config: ExperimentConfig) -> None:
    """Run the complete experiment pipeline."""
    print("\n" + "="*70)
    print("GALILEO ADVERSARIAL PERSONA EXPERIMENT (v2 - Dynamic Claims)")
    print("="*70)
    print(f"Models: {', '.join(m.split('/')[-1] for m in config.models)}")
    print(f"Data files: {len(config.data_files)}")
    print(f"Num samples: {config.num_samples if config.num_samples > 0 else 'all'}")
    print("="*70)
    
    setup_results_dir(config)
    
    all_initial_results = []
    all_adversarial_results = []
    all_recovery_results = []
    
    for model_name in config.models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")
        
        engine = InferenceEngine(
            model_name=model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=MAX_TOKENS,
        )
        
        model_short = model_name.split("/")[-1]
        model_results_dir = os.path.join(config.results_dir, model_short)
        
        for data_file in config.data_files:
            test_name = get_test_name(data_file)
            
            # Load data
            problems = load_dataset(
                data_file,
                num_samples=config.num_samples,
                shuffle=config.test_mode,
            )
            problems = [prepare_problem(p) for p in problems]
            
            # Phase 1: Initial evaluation
            initial_results = run_initial_evaluation(engine, problems, test_name, config)
            all_initial_results.extend(initial_results)
            save_jsonl(initial_results, os.path.join(model_results_dir, f"{test_name}_initial.jsonl"))
            
            # Phase 2: Adversarial testing
            adversarial_results = run_adversarial_testing(engine, initial_results, config)
            all_adversarial_results.extend(adversarial_results)
            # Filter out internal fields (starting with _) when saving
            save_data = [{k: v for k, v in r.items() if not k.startswith("_")} for r in adversarial_results]
            save_jsonl(save_data, os.path.join(model_results_dir, f"{test_name}_adversarial.jsonl"))
            
            # Phase 3: Recovery testing
            recovery_results = run_recovery_testing(engine, adversarial_results, config)
            all_recovery_results.extend(recovery_results)
            if recovery_results:
                save_jsonl(recovery_results, os.path.join(model_results_dir, f"{test_name}_recovery.jsonl"))
        
        del engine
        gc.collect()
    
    # Save aggregated CSV results
    save_results_to_csv(
        all_initial_results,
        all_adversarial_results,
        all_recovery_results,
        config,
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {config.results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Galileo Adversarial Persona Experiment")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with fewer samples")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples (-1 for all)")
    parser.add_argument("--model", type=str, help="Run only specific model")
    parser.add_argument("--data_file", type=str, help="Run only specific data file")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="Results directory")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    config.test_mode = args.test_mode
    config.results_dir = args.results_dir
    
    if args.num_samples > 0:
        config.num_samples = args.num_samples
    elif args.test_mode:
        config.num_samples = 10
    
    if args.model:
        config.models = [args.model]
    
    if args.data_file:
        config.data_files = [args.data_file]
    
    run_experiment(config)


if __name__ == "__main__":
    main()
