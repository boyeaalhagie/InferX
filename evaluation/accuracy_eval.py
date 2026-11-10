#!/usr/bin/env python3
"""
InferX — Accuracy Evaluation for Precision Study (Experiment E)

Compares output quality of BF16 vs FP16 vs INT8 on the same fixed prompts.
BF16 is the reference (highest fidelity on H100). We measure:
  - ROUGE-L: longest common subsequence overlap
  - Token exact match rate
  - Output length consistency

Usage (via SLURM):
    python3 evaluation/accuracy_eval.py \
        --model ~/InferX/models/qwen2.5-7b \
        --output-file evaluation/accuracy_results.json

No extra dependencies beyond torch + transformers.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed evaluation prompts — same for every dtype run
EVAL_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in machine learning.",
    "What is the capital of France and what is it known for historically?",
    "Describe how a transformer neural network processes a sequence of tokens.",
    "What are the main causes of climate change according to scientific consensus?",
    "Explain the concept of recursion in programming with a simple example.",
    "What is photosynthesis and why is it important for life on Earth?",
    "Describe the difference between RAM and ROM in computer systems.",
    "What is the Pythagorean theorem and when is it used?",
    "Explain how HTTPS protects data transmitted over the internet.",
    "What is the difference between a compiler and an interpreter?",
    "Describe the water cycle and its importance to the global climate.",
    "What are prime numbers and why are they important in cryptography?",
    "Explain what a neural network is and how it learns from data.",
    "What is the significance of the speed of light in physics?",
    "Describe how a database index improves query performance.",
    "What is the difference between TCP and UDP network protocols?",
    "Explain the concept of entropy in thermodynamics.",
    "What is a binary search tree and what are its advantages?",
    "Describe how the human immune system fights viral infections.",
    "What is quantum entanglement and why is it considered strange?",
]

MAX_NEW_TOKENS = 128


# ---------------------------------------------------------------------------
# Simple ROUGE-L implementation (no extra deps)
# ---------------------------------------------------------------------------

def lcs_length(x: list, y: list) -> int:
    """Length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score between two strings (word-level)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_overlap(reference: str, hypothesis: str) -> float:
    """Fraction of reference tokens also present in hypothesis (recall-style)."""
    ref_set = set(reference.lower().split())
    hyp_set = set(hypothesis.lower().split())
    if not ref_set:
        return 0.0
    return len(ref_set & hyp_set) / len(ref_set)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_outputs(
    model_path: str,
    dtype_str: str,
    prompts: list,
) -> list[str]:
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    print(f"\n[accuracy_eval] Loading model in {dtype_str}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = dtype_map[dtype_str]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    print(f"[accuracy_eval] Model loaded ({dtype_str}).")

    outputs = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(next(model.parameters()).device)

            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # greedy — deterministic
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the new tokens (not the prompt)
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(prompts)} done")

    del model
    torch.cuda.empty_cache()
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-file",
                   default=str(REPO_ROOT / "evaluation" / "accuracy_results.json"))
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"],
                   help="Dtypes to evaluate. BF16 is always the reference.")
    args = p.parse_args()

    # Ensure bf16 is first (reference)
    dtypes = args.dtypes
    if "bf16" not in dtypes:
        dtypes = ["bf16"] + dtypes
    elif dtypes[0] != "bf16":
        dtypes = ["bf16"] + [d for d in dtypes if d != "bf16"]

    print("=" * 60)
    print("InferX — Accuracy Evaluation")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Dtypes:   {dtypes}")
    print(f"Prompts:  {len(EVAL_PROMPTS)}")
    print(f"Output:   {MAX_NEW_TOKENS} tokens per prompt")

    all_outputs: dict[str, list[str]] = {}
    for dtype in dtypes:
        t0 = time.time()
        all_outputs[dtype] = generate_outputs(args.model, dtype, EVAL_PROMPTS)
        elapsed = time.time() - t0
        print(f"[{dtype}] generation complete in {elapsed:.1f}s")

    # Score each non-reference dtype against bf16
    reference_outputs = all_outputs["bf16"]
    scores: dict[str, dict] = {}

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON (vs BF16 reference)")
    print("=" * 60)

    for dtype in dtypes:
        if dtype == "bf16":
            continue
        rouge_scores = []
        overlap_scores = []
        length_ratios = []

        for ref, hyp in zip(reference_outputs, all_outputs[dtype]):
            rouge_scores.append(rouge_l(ref, hyp))
            overlap_scores.append(token_overlap(ref, hyp))
            ref_len = len(ref.split())
            hyp_len = len(hyp.split())
            length_ratios.append(hyp_len / max(1, ref_len))

        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        avg_len_ratio = sum(length_ratios) / len(length_ratios)
        min_rouge = min(rouge_scores)

        scores[dtype] = {
            "rouge_l_avg": avg_rouge,
            "rouge_l_min": min_rouge,
            "token_overlap_avg": avg_overlap,
            "length_ratio_avg": avg_len_ratio,
        }

        print(f"\n  {dtype} vs bf16:")
        print(f"    ROUGE-L (avg):      {avg_rouge:.3f}  (1.0 = identical)")
        print(f"    ROUGE-L (worst):    {min_rouge:.3f}")
        print(f"    Token overlap:      {avg_overlap:.3f}")
        print(f"    Length ratio:       {avg_len_ratio:.3f}  (1.0 = same length)")

        # Qualitative flag
        if avg_rouge >= 0.90:
            verdict = "NEGLIGIBLE quality difference"
        elif avg_rouge >= 0.75:
            verdict = "MINOR quality difference"
        elif avg_rouge >= 0.50:
            verdict = "MODERATE quality difference"
        else:
            verdict = "SIGNIFICANT quality difference"
        print(f"    Verdict:            {verdict}")

    # Sample comparison
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (Prompt 0)")
    print("=" * 60)
    print(f"Prompt: {EVAL_PROMPTS[0][:80]}...")
    print(f"\nBF16:\n  {reference_outputs[0][:300]}")
    for dtype in dtypes:
        if dtype != "bf16":
            print(f"\n{dtype}:\n  {all_outputs[dtype][0][:300]}")

    # Write results
    result = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(EVAL_PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "reference_dtype": "bf16",
        "scores": scores,
        "outputs": all_outputs,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
