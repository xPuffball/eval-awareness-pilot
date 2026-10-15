"""Representation measures, read at the last prompt token (before the model has reasoned at all).

probe: projection onto the eval-minus-deploy mean difference of the 16 contrastive pairs (Hua et al.).
lens:  Jacobian-lens log-probability mass on eval words ("test", "evaluation", "benchmark", ...),
       using Neuronpedia's pre-fitted lens. This asks whether the model is disposed to *say* an
       eval word, without it saying anything.

Both are computed at every layer and saved to results/<model>/representation.npz.

python -m pilot.represent --model qwen3-4b
"""

import argparse

import jlens
import numpy as np
import torch

from pilot.config import LENS_REPO, MODELS, model_dir
from pilot.data import CONTRASTIVE_PAIRS, EVAL_WORDS, load_prompts
from pilot.models import load, render


@torch.no_grad()
def last_token_residuals(lens_model, tokenizer, messages: list[str]) -> torch.Tensor:
    """Residual stream after every block at the final prompt token: [n_messages, n_layers, d_model]."""
    residuals = []
    for message in messages:
        input_ids = tokenizer(render(tokenizer, message), return_tensors="pt").input_ids.to("cuda")
        with jlens.ActivationRecorder(lens_model.layers, at=range(lens_model.n_layers)) as recorder:
            lens_model.forward(input_ids)
        residuals.append(torch.stack([recorder.activations[l][0, -1] for l in range(lens_model.n_layers)]))
    return torch.stack(residuals).float().cpu()


def eval_token_ids(tokenizer) -> list[int]:
    """Variants of EVAL_WORDS (with/without leading space, capitalised) that are a single token."""
    ids = set()
    for word in EVAL_WORDS:
        for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1:
                ids.add(tokens[0])
    return sorted(ids)


@torch.no_grad()
def lens_eval_mass(lens_model, lens, residuals: torch.Tensor, token_ids: list[int]) -> np.ndarray:
    """log P_lens(any eval word) at each fitted layer: [n_messages, n_lens_layers]."""
    scores = []
    for layer in lens.source_layers:
        logits = lens_model.unembed(lens.transport(residuals[:, layer].cuda(), layer)).float()
        scores.append(logits.log_softmax(-1)[:, token_ids].logsumexp(-1).cpu())
    return torch.stack(scores, dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    spec = MODELS[args.model]

    model, tokenizer = load(spec)
    lens_model = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=spec.lens_file)
    prompts = load_prompts()

    deploy = last_token_residuals(lens_model, tokenizer, [d for d, _ in CONTRASTIVE_PAIRS])
    evaluation = last_token_residuals(lens_model, tokenizer, [e for _, e in CONTRASTIVE_PAIRS])
    direction = evaluation.mean(0) - deploy.mean(0)
    direction = direction / direction.norm(dim=-1, keepdim=True)  # [n_layers, d_model]

    residuals = last_token_residuals(lens_model, tokenizer, [p["text"] for p in prompts])
    token_ids = eval_token_ids(tokenizer)
    print("eval tokens:", [tokenizer.decode([t]) for t in token_ids])

    np.savez(
        model_dir(spec.name) / "representation.npz",
        ids=np.array([p["id"] for p in prompts]),
        sets=np.array([p["set"] for p in prompts]),
        probe=torch.einsum("nld,ld->nl", residuals, direction).numpy(),
        lens=lens_eval_mass(lens_model, lens, residuals, token_ids),
        lens_layers=np.array(lens.source_layers),
    )
    print("saved", model_dir(spec.name) / "representation.npz")


if __name__ == "__main__":
    main()
