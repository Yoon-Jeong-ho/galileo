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
# (default is also set in config.py; keep this aligned with shared-server policy)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")

from config import (
    MODELS,
    SYSTEM_PROMPT,
    INSTRUCTION_TEMPLATE,
    RESULTS_DIR,
    MAX_TOKENS,
    MAX_MODEL_LEN,
    BEAM_SEARCH_N,
    BEAM_SEARCH_TEMPERATURE,
    GREEDY_TEMPERATURE,
    MAX_ADVERSARIAL_ROUNDS,
    TENSOR_PARALLEL_SIZE,
    ExperimentConfig,
)
from inference import InferenceEngine
from evaluation import evaluate_response
from personas import (
    get_claim_generation_prompt,
    get_retry_suffix,
    get_recovery_prompt,
    get_all_persona_keys,
    get_persona_name,
)
from data_loader import load_dataset, save_jsonl, get_test_name, prepare_problem
from tasks import get_task, format_mcqa_options


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
    
    # Task spec (assume single task per dataset file)
    task_name = problems[0].get("task", "math") if problems else "math"
    task_spec = get_task(task_name)

    # Prepare prompts
    prompts = []
    for prob in problems:
        if task_spec.answer_style == "mcqa":
            opts = format_mcqa_options(prob.get("choices", []))
            prompts.append(task_spec.instruction_template.format(question=prob["question"], options=opts))
        else:
            prompts.append(task_spec.instruction_template.format(question=prob["question"]))
    
    # Generate with beam search - vLLM handles batching internally
    print("Generating responses with beam search...")
    outputs = engine.generate_beam_search(
        prompts=prompts,
        n=config.beam_search_n,
        temperature=config.beam_search_temperature,
        max_tokens=config.max_tokens,
        system_prompt=task_spec.system_prompt,
    )
    
    all_results = []
    for prob, beam_outputs in tqdm(zip(problems, outputs), total=len(problems), desc="Processing results"):
        # Check each beam for correct answer
        best_response = None
        is_correct = False
        extracted_answer = None
        
        for output in beam_outputs:
            ans, correct, _metrics = evaluate_response(output.response, prob["ground_truth"], answer_style=task_spec.answer_style)
            if correct:
                best_response = output.response
                is_correct = True
                extracted_answer = ans
                break
        
        if best_response is None:
            best_response = beam_outputs[0].response
            extracted_answer, is_correct, _metrics = evaluate_response(best_response, prob["ground_truth"], answer_style=task_spec.answer_style)
        
        result = {
            "question": prob["question"],
            "ground_truth": prob["ground_truth"],
            "initial_response": best_response,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "test_name": test_name,
            "model": engine.model_short_name,
            "task": prob.get("task", "math"),
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
    # Task spec (single task per dataset file)
    task_spec = get_task(correct_results[0].get("task", "math"))

    print(f"Adversarial Testing (Dynamic Claim Generation)")
    print(f"Model: {engine.model_short_name}")
    print(f"Correct answers to challenge: {len(correct_results)}")
    print(f"Personas: {len(get_all_persona_keys())}")
    print(f"Max rounds per persona: {config.max_adversarial_rounds}")
    print(f"{'='*60}")
    
    all_adversarial_results = []
    persona_keys = config.personas if getattr(config, "personas", None) else get_all_persona_keys()
    
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
                "task": result.get("task", "math"),
                "choices": result.get("choices", []),
                "conversation": [
                    {"role": "user", "content": (
                        task_spec.instruction_template.format(
                            question=result["question"],
                            options=format_mcqa_options(result.get("choices", [])),
                        )
                        if task_spec.answer_style == "mcqa"
                        else task_spec.instruction_template.format(question=result["question"])
                    )},
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
            system_prompt=task_spec.system_prompt,
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
            full_claim = claim_text + get_retry_suffix(task_spec.answer_style)
            
            conv = deepcopy(track["conversation"])
            conv.append({"role": "user", "content": full_claim})
            
            retry_conversations.append(conv)
            retry_keys.append(key)
        
        # Generate retry answers
        retry_responses = engine.generate_multi_turn(
            conversations=retry_conversations,
            temperature=config.greedy_temperature,
            max_tokens=config.max_tokens,
            system_prompt=task_spec.system_prompt,
        )
        
        # Step 3: Evaluate and update tracks
        round_correct = 0
        round_wrong = 0
        
        for key, retry_resp in zip(retry_keys, retry_responses):
            track = active_tracks[key]
            claim_text = generated_claims[key]
            full_claim = claim_text + get_retry_suffix(task_spec.answer_style)
            retry_text = retry_resp.response.strip()
            
            # Evaluate
            extracted, is_correct, _metrics = evaluate_response(retry_text, track["ground_truth"], answer_style=task_spec.answer_style)
            
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
        
        ever_incorrect = any(not rr["is_correct"] for rr in track["round_results"])
        first_incorrect_round = next((rr["round"] for rr in track["round_results"] if not rr["is_correct"]), None)

        result = {
            "question": track["question"],
            "ground_truth": track["ground_truth"],
            "initial_response": track["initial_response"],
            "initial_extracted": track.get("initial_extracted", None),
            "persona": track["persona"],
            "persona_name": get_persona_name(track["persona"]),
            "rounds_completed": track["rounds_completed"],
            # NOTE: "final_correct" is correctness at the end of Phase 2 (after R rounds).
            "final_correct": track["still_correct"],
            # "flipped" matches the paper definition for recovery@flip: whether the trace
            # became incorrect at least once during rounds 1..R (even if it later recovered).
            "flipped": ever_incorrect,
            "first_failure_round": first_incorrect_round,
            "turns": turns,
            "test_name": track["test_name"],
            "model": track["model"],
            "task": track.get("task", "math"),
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
    # Filter to traces that flipped at least once during Phase 2 (rounds 1..R).
    # IMPORTANT: this matches the paper definition of recovery@flip. We should NOT
    # restrict to "final incorrect" only, because a trace may flip and then recover
    # within Phase 2.
    failed_results = [r for r in adversarial_results if r.get("flipped", False)]
    
    if not failed_results:
        print("\nNo failed cases to recover")
        return []
    
    print(f"\n{'='*60}")
    # Task spec (single task per dataset file)
    task_spec = get_task(failed_results[0].get("task", "math"))

    print(f"Recovery Testing")
    print(f"Model: {engine.model_short_name}")
    print(f"Flip cases to recover (flipped at least once): {len(failed_results)}")
    print(f"{'='*60}")
    
    # Build recovery conversations
    conversations = []
    for result in failed_results:
        # Use full conversation history + recovery prompt
        conv = deepcopy(result["conversation"])
        conv.append({"role": "user", "content": get_recovery_prompt(task_spec.answer_style, variant=getattr(config, "recovery_variant", "baseline"))})
        conversations.append(conv)
    
    # Generate recovery responses
    print("Generating recovery answers...")
    responses = engine.generate_multi_turn(
        conversations=conversations,
        temperature=config.greedy_temperature,
        max_tokens=config.max_tokens,
        system_prompt=task_spec.system_prompt,
    )
    
    # Evaluate recovery
    recovery_results = []
    recovered_count = 0
    
    for result, response in zip(failed_results, responses):
        recovery_text = response.response.strip()
        extracted, is_correct, _metrics = evaluate_response(recovery_text, result["ground_truth"], answer_style=task_spec.answer_style)
        
        if is_correct:
            recovered_count += 1
        
        recovery_result = {
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "persona": result["persona"],
            "persona_name": result["persona_name"],
            "test_name": result["test_name"],
            "model": result["model"],
            "task": result.get("task", "math"),
            "failed_at_round": result.get("first_failure_round", None),
            "recovery_response": recovery_text,
            "extracted_answer": extracted,
            "recovered": is_correct,
            "full_conversation": result["conversation"] + [
                {"role": "user", "content": get_recovery_prompt(task_spec.answer_style, variant=getattr(config, "recovery_variant", "baseline"))},
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
                                    any(rr["turn"] == round_num and rr["is_correct"] 
                                        for rr in r["turns"])))
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

    # Best-effort reproducibility (dataset sampling, any Python-side randomness)
    try:
        import random
        random.seed(config.seed)
    except Exception:
        pass

    
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
            max_model_len=config.max_model_len,
        )
        
        model_short = model_name.split("/")[-1]
        model_results_dir = os.path.join(config.results_dir, model_short)
        
        for data_file in config.data_files:
            test_name = get_test_name(data_file)
            
            # Load data
            problems = load_dataset(
                data_file,
                num_samples=config.num_samples,
                shuffle=(config.test_mode or (config.num_samples > 0)),
                seed=config.seed,
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

    # run control
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with fewer samples")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples (-1 for all)")

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling/shuffling")

    # scope
    parser.add_argument("--model", type=str, default=None, help="Run only a specific model (HuggingFace name)")
    parser.add_argument("--data_file", type=str, default=None, help="Run only a specific JSONL data file")

    # paths
    parser.add_argument("--data_dir", type=str, default=None, help="Override DATA_DIR (expects JSONL files)")
    parser.add_argument("--results_dir", type=str, default=None, help="Override RESULTS_DIR")

    # vLLM / generation
    parser.add_argument("--max_tokens", type=int, default=None, help="Max new tokens per generation (overrides config)")
    parser.add_argument("--max_model_len", type=int, default=None, help="Max context length for vLLM (overrides config)")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Tensor parallel size (overrides config)")
    parser.add_argument("--greedy_temperature", type=float, default=None, help="Decoding temperature for adversarial/recovery turns (overrides config)")
    parser.add_argument("--recovery_variant", type=str, default=None, choices=["baseline","reinforce_correct","verify_then_answer"], help="Recovery prompt variant ablation")

    # persona selection
    parser.add_argument(
        "--personas",
        type=str,
        default=None,
        help=(
            "Comma-separated persona keys to run in Phase 2 (e.g., 'soft_pressure,authority_claim'). "
            "Special: 'all' (default) or 'no_control' (exclude control_reask)."
        ),
    )

    args = parser.parse_args()

    config = ExperimentConfig()

    config.seed = int(args.seed)

    # CLI overrides
    config.test_mode = args.test_mode

    if args.num_samples > 0:
        config.num_samples = args.num_samples
    elif args.test_mode:
        config.num_samples = 10

    if args.model:
        config.models = [args.model]

    # Data selection precedence:
    # - If --data_dir is provided, we treat it as the root directory.
    # - If --data_file is ALSO provided, restrict to that single file within data_dir.
    # - Otherwise, run all *.jsonl under data_dir.
    if args.data_dir is not None:
        import os as _os
        if args.data_file:
            config.data_files = [_os.path.join(args.data_dir, str(args.data_file))]
        else:
            config.data_files = [
                _os.path.join(args.data_dir, f)
                for f in _os.listdir(args.data_dir)
                if f.endswith(".jsonl")
            ]
    elif args.data_file:
        config.data_files = [args.data_file]

    if args.results_dir is not None:
        config.results_dir = args.results_dir

    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens

    if args.max_model_len is not None:
        config.max_model_len = args.max_model_len

    if args.tensor_parallel_size is not None:
        config.tensor_parallel_size = args.tensor_parallel_size

    if args.greedy_temperature is not None:
        config.greedy_temperature = float(args.greedy_temperature)

    if args.recovery_variant is not None:
        config.recovery_variant = str(args.recovery_variant)

    if args.personas is not None:
        raw = str(args.personas).strip()
        if raw.lower() == "all" or raw == "":
            config.personas = []
        elif raw.lower() == "no_control":
            config.personas = [p for p in get_all_persona_keys(include_control=False)]
        else:
            config.personas = [p.strip() for p in raw.split(",") if p.strip()]

    run_experiment(config)


if __name__ == "__main__":
    main()
