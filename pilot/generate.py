"""Sample thinking-mode rollouts. Generation stops at </think>: only the reasoning is judged.

python -m pilot.generate --model qwen3-4b [--limit 2]
"""

import argparse
import json
import time

import torch

from pilot.config import MAX_THINKING_TOKENS, MODELS, ROLLOUTS_PER_PROMPT, SAMPLING, SEED, model_dir
from pilot.data import load_prompts
from pilot.io import read_jsonl
from pilot.models import load, render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, help="only the first N prompts (smoke test)")
    args = parser.parse_args()
    spec = MODELS[args.model]
    out_path = model_dir(spec.name) / "rollouts.jsonl"

    prompts = [p for p in load_prompts() if p["set"] in ROLLOUTS_PER_PROMPT]
    if args.limit:
        prompts = prompts[: args.limit]
    done = {(r["id"], r["sample"]) for r in read_jsonl(out_path)}
    jobs = [(p, s) for p in prompts for s in range(ROLLOUTS_PER_PROMPT[p["set"]]) if (p["id"], s) not in done]
    print(f"{len(jobs)} rollouts to generate ({len(done)} already done)")

    model, tokenizer = load(spec)
    stop_ids = [tokenizer.convert_tokens_to_ids("</think>"), tokenizer.eos_token_id]
    rendered = {p["id"]: render(tokenizer, p["text"]) for p in prompts}
    jobs.sort(key=lambda job: len(rendered[job[0]["id"]]))  # less padding per batch
    torch.manual_seed(SEED + len(done))

    start = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(jobs), spec.batch_size):
            batch = jobs[i : i + spec.batch_size]
            inputs = tokenizer([rendered[p["id"]] for p, _ in batch], return_tensors="pt", padding=True).to("cuda")
            # A static KV cache is preallocated once per batch. The default dynamic cache grows by
            # concatenation, which fragments GPU memory until Windows spills it into system RAM.
            with torch.no_grad():
                output = model.generate(**inputs, **SAMPLING, max_new_tokens=MAX_THINKING_TOKENS,
                                        eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
                                        cache_implementation="static", disable_compile=True)
            for (prompt, sample), tokens in zip(batch, output[:, inputs.input_ids.shape[1] :]):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                record = {
                    "id": prompt["id"], "set": prompt["set"], "sample": sample,
                    "cot": text.replace("<think>", "").split("</think>")[0].strip(),
                    "finished": "</think>" in text,
                    "n_tokens": int((tokens != tokenizer.pad_token_id).sum()),
                }
                f.write(json.dumps(record) + "\n")
            f.flush()
            n = i + len(batch)
            print(f"{n}/{len(jobs)} rollouts, {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
