#!/usr/bin/env python3
"""
Galileo Adversarial Persona Experiment Pipeline

This script runs the complete experiment:
1. Initial evaluation with beam search
2. Adversarial testing with 5 personas
3. Recovery testing
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
from evaluation import evaluate_response, extract_answer, compute_accuracy
from personas import get_adversarial_prompt, get_recovery_prompt, get_all_persona_keys, get_persona_name
from data_loader import load_dataset, save_jsonl, append_jsonl, get_test_name, prepare_problem


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
    prompts = []
    for prob in problems:
        prompt = INSTRUCTION_TEMPLATE.format(question=prob["question"])
        prompts.append(prompt)
    
    # Generate with beam search
    print("Generating responses with beam search...")
    batch_size = 16
    all_results = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Initial eval"):
        batch_prompts = prompts[i:i+batch_size]
        batch_problems = problems[i:i+batch_size]
        
        batch_outputs = engine.generate_beam_search(
            prompts=batch_prompts,
            n=config.beam_search_n,
            temperature=config.beam_search_temperature,
            max_tokens=MAX_TOKENS,
            system_prompt=SYSTEM_PROMPT,
        )
        
        for prob, outputs in zip(batch_problems, batch_outputs):
            # Check each beam for correct answer
            best_response = None
            is_correct = False
            extracted_answer = None
            
            for output in outputs:
                ans, correct = evaluate_response(output.response, prob["answer"])
                if correct:
                    best_response = output.response
                    is_correct = True
                    extracted_answer = ans
                    break
            
            # If no correct answer found, use first response
            if best_response is None:
                best_response = outputs[0].response
                extracted_answer, is_correct = evaluate_response(best_response, prob["answer"])
            
            result = {
                "question": prob["question"],
                "ground_truth": prob["answer"],
                "model_response": best_response,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "test_name": test_name,
                "model": engine.model_short_name,
                "phase": "initial",
            }
            all_results.append(result)
    
    # Report accuracy
    correct = sum(1 for r in all_results if r["is_correct"])
    accuracy = correct / len(all_results) * 100
    print(f"\nInitial accuracy: {correct}/{len(all_results)} = {accuracy:.2f}%")
    
    return all_results


def run_adversarial_testing(
    engine: InferenceEngine,
    initial_results: List[Dict[str, Any]],
    config: ExperimentConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run adversarial testing on correct answers.
    
    Returns dict mapping persona to list of adversarial results.
    """
    # Filter to only correct answers
    correct_results = [r for r in initial_results if r["is_correct"]]
    
    print(f"\n{'='*60}")
    print(f"Adversarial Testing")
    print(f"Model: {engine.model_short_name}")
    print(f"Correct answers to challenge: {len(correct_results)}")
    print(f"Max rounds per persona: {config.max_adversarial_rounds}")
    print(f"{'='*60}")
    
    all_adversarial_results = {persona: [] for persona in get_all_persona_keys()}
    
    for persona_key in get_all_persona_keys():
        persona_name = get_persona_name(persona_key)
        print(f"\nTesting persona: {persona_name}")
        
        # Track results per round
        round_results = []
        current_problems = correct_results.copy()
        
        for round_num in range(config.max_adversarial_rounds):
            if not current_problems:
                break
            
            print(f"  Round {round_num + 1}: {len(current_problems)} problems")
            
            # Build multi-turn conversations
            conversations = []
            problem_map = []
            
            for prob in current_problems:
                # Get the previous response (initial or last adversarial)
                if round_num == 0:
                    prev_response = prob["model_response"]
                else:
                    prev_response = prob.get("last_response", prob["model_response"])
                
                # Build conversation
                messages = [
                    {"role": "user", "content": INSTRUCTION_TEMPLATE.format(question=prob["question"])},
                    {"role": "assistant", "content": prev_response},
                    {"role": "user", "content": get_adversarial_prompt(persona_key, round_num)},
                ]
                conversations.append(messages)
                problem_map.append(prob)
            
            # Generate responses
            batch_size = 16
            responses = []
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i+batch_size]
                batch_outputs = engine.generate_multi_turn(
                    conversations=batch,
                    temperature=config.greedy_temperature,
                    max_tokens=MAX_TOKENS,
                    system_prompt=SYSTEM_PROMPT,
                )
                responses.extend(batch_outputs)
            
            # Evaluate responses
            still_correct = []
            for prob, response in zip(problem_map, responses):
                extracted, is_correct = evaluate_response(response.response, prob["ground_truth"])
                
                result = {
                    "question": prob["question"],
                    "ground_truth": prob["ground_truth"],
                    "model_response": response.response,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "persona": persona_key,
                    "round": round_num + 1,
                    "model": engine.model_short_name,
                    "test_name": prob["test_name"],
                    "phase": "adversarial",
                }
                round_results.append(result)
                
                if is_correct:
                    # Update for next round
                    prob["last_response"] = response.response
                    still_correct.append(prob)
            
            current_problems = still_correct
            
            # Report round accuracy
            round_correct = len(still_correct)
            round_total = len(problem_map)
            print(f"    Still correct: {round_correct}/{round_total} = {round_correct/round_total*100:.2f}%")
        
        all_adversarial_results[persona_key] = round_results
    
    return all_adversarial_results


def run_recovery_testing(
    engine: InferenceEngine,
    adversarial_results: Dict[str, List[Dict[str, Any]]],
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Run recovery testing on answers that became wrong.
    
    Returns list of recovery results.
    """
    # Find all problems that became wrong after adversarial testing
    problems_to_recover = []
    
    for persona_key, results in adversarial_results.items():
        # Group by question
        by_question = defaultdict(list)
        for r in results:
            by_question[r["question"]].append(r)
        
        for question, rounds in by_question.items():
            # Check if ended up wrong
            last_round = max(rounds, key=lambda x: x["round"])
            if not last_round["is_correct"]:
                problems_to_recover.append({
                    "question": question,
                    "ground_truth": last_round["ground_truth"],
                    "last_response": last_round["model_response"],
                    "persona": persona_key,
                    "final_round": last_round["round"],
                    "test_name": last_round["test_name"],
                })
    
    if not problems_to_recover:
        print("\nNo problems to recover (all remained correct)")
        return []
    
    print(f"\n{'='*60}")
    print(f"Recovery Testing")
    print(f"Model: {engine.model_short_name}")
    print(f"Problems to recover: {len(problems_to_recover)}")
    print(f"{'='*60}")
    
    # Build conversations with recovery prompt
    conversations = []
    for prob in problems_to_recover:
        messages = [
            {"role": "user", "content": INSTRUCTION_TEMPLATE.format(question=prob["question"])},
            {"role": "assistant", "content": prob["last_response"]},
            {"role": "user", "content": get_recovery_prompt()},
        ]
        conversations.append(messages)
    
    # Generate responses
    batch_size = 16
    responses = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        batch_outputs = engine.generate_multi_turn(
            conversations=batch,
            temperature=config.greedy_temperature,
            max_tokens=MAX_TOKENS,
            system_prompt=SYSTEM_PROMPT,
        )
        responses.extend(batch_outputs)
    
    # Evaluate recovery
    recovery_results = []
    for prob, response in zip(problems_to_recover, responses):
        extracted, is_correct = evaluate_response(response.response, prob["ground_truth"])
        
        result = {
            "question": prob["question"],
            "ground_truth": prob["ground_truth"],
            "model_response": response.response,
            "extracted_answer": extracted,
            "is_correct": is_correct,
            "recovered": is_correct,
            "persona": prob["persona"],
            "failed_at_round": prob["final_round"],
            "model": engine.model_short_name,
            "test_name": prob["test_name"],
            "phase": "recovery",
        }
        recovery_results.append(result)
    
    # Report recovery rate
    recovered = sum(1 for r in recovery_results if r["recovered"])
    total = len(recovery_results)
    print(f"\nRecovery rate: {recovered}/{total} = {recovered/total*100:.2f}%")
    
    return recovery_results


def save_results_to_csv(
    initial_results: List[Dict[str, Any]],
    adversarial_results: Dict[str, List[Dict[str, Any]]],
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
        
        # Group by model and test
        by_model_test = defaultdict(list)
        for r in initial_results:
            key = (r["model"], r["test_name"])
            by_model_test[key].append(r)
        
        for (model, test), results in sorted(by_model_test.items()):
            correct = sum(1 for r in results if r["is_correct"])
            total = len(results)
            accuracy = correct / total * 100
            writer.writerow([model, test, correct, total, f"{accuracy:.2f}"])
    
    # Adversarial accuracy CSV
    adv_csv = os.path.join(config.results_dir, "adversarial_accuracy.csv")
    with open(adv_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_name", "persona", "round", "correct", "total", "accuracy", "drop_from_initial"])
        
        for persona_key, results in adversarial_results.items():
            if not results:
                continue
            
            # Group by model, test, round
            by_model_test_round = defaultdict(list)
            for r in results:
                key = (r["model"], r["test_name"], r["round"])
                by_model_test_round[key].append(r)
            
            for (model, test, round_num), round_results in sorted(by_model_test_round.items()):
                correct = sum(1 for r in round_results if r["is_correct"])
                total = len(round_results)
                accuracy = correct / total * 100
                
                # Calculate drop from initial
                initial_key = (model, test)
                initial_for_test = [r for r in initial_results if (r["model"], r["test_name"]) == initial_key]
                initial_correct = sum(1 for r in initial_for_test if r["is_correct"])
                drop = initial_correct - correct
                
                writer.writerow([
                    model, test, get_persona_name(persona_key), round_num,
                    correct, total, f"{accuracy:.2f}", drop
                ])
    
    # Recovery accuracy CSV
    if recovery_results:
        recovery_csv = os.path.join(config.results_dir, "recovery_accuracy.csv")
        with open(recovery_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "test_name", "persona", "recovered", "total", "recovery_rate"])
            
            # Group by model, test, persona
            by_model_test_persona = defaultdict(list)
            for r in recovery_results:
                key = (r["model"], r["test_name"], r["persona"])
                by_model_test_persona[key].append(r)
            
            for (model, test, persona), results in sorted(by_model_test_persona.items()):
                recovered = sum(1 for r in results if r["recovered"])
                total = len(results)
                rate = recovered / total * 100
                writer.writerow([model, test, get_persona_name(persona), recovered, total, f"{rate:.2f}"])
    
    print(f"  Saved to {config.results_dir}/")


def run_experiment(config: ExperimentConfig) -> None:
    """Run the complete experiment pipeline."""
    print("\n" + "="*70)
    print("GALILEO ADVERSARIAL PERSONA EXPERIMENT")
    print("="*70)
    print(f"Models: {', '.join(m.split('/')[-1] for m in config.models)}")
    print(f"Data files: {len(config.data_files)}")
    print(f"Test mode: {config.test_mode}")
    print(f"Num samples: {config.num_samples if config.num_samples > 0 else 'all'}")
    print("="*70)
    
    setup_results_dir(config)
    
    # Collect all results
    all_initial_results = []
    all_adversarial_results = {p: [] for p in get_all_persona_keys()}
    all_recovery_results = []
    
    for model_name in config.models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")
        
        # Load model
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
            
            # Save initial results
            save_jsonl(
                initial_results,
                os.path.join(model_results_dir, f"{test_name}_initial.jsonl")
            )
            
            # Phase 2: Adversarial testing
            adversarial_results = run_adversarial_testing(engine, initial_results, config)
            for persona, results in adversarial_results.items():
                all_adversarial_results[persona].extend(results)
            
            # Save adversarial results
            all_adv = []
            for results in adversarial_results.values():
                all_adv.extend(results)
            save_jsonl(
                all_adv,
                os.path.join(model_results_dir, f"{test_name}_adversarial.jsonl")
            )
            
            # Phase 3: Recovery testing
            recovery_results = run_recovery_testing(engine, adversarial_results, config)
            all_recovery_results.extend(recovery_results)
            
            # Save recovery results
            if recovery_results:
                save_jsonl(
                    recovery_results,
                    os.path.join(model_results_dir, f"{test_name}_recovery.jsonl")
                )
        
        # Clean up model
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
    
    # Build config
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
    
    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
