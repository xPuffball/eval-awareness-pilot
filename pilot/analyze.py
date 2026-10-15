"""Tables and figures for the pilot: results/summary.md and results/*.png.

python -m pilot.analyze
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from pilot.config import MODELS, RESULTS, SEED, model_dir
from pilot.io import read_jsonl

matplotlib.use("Agg")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three categorical slots of the dataviz reference palette
INK, MUTED = "#0b0b0b", "#8a8984"
CONTRASTS = {"FORTRESS vs WildChat": ("fortress", "wildchat"),
             "FORTRESS vs benign twins": ("fortress", "fortress_benign")}
MEASURES = {"probe": "Contrastive probe", "lens": "Jacobian lens (eval words)"}


def bootstrap_ci(per_prompt_rates: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """95% CI of the mean rate, resampling prompts (rollouts of one prompt are not independent)."""
    rng = np.random.default_rng(SEED)
    means = rng.choice(per_prompt_rates, size=(n, len(per_prompt_rates))).mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def vea_table(judged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in judged.groupby("set"):
        per_prompt = group.groupby("id")["vea"].mean().to_numpy()
        low, high = bootstrap_ci(per_prompt)
        rows.append({
            "set": name, "prompts": group["id"].nunique(), "rollouts": len(group),
            "truncated %": 100 * (~group["finished"]).mean(),
            "judge unparsed": int(group["unparsed"].sum()),
            "VEA %": 100 * group["vea"].mean(), "95% CI": f"{100 * low:.1f}–{100 * high:.1f}",
            "prompts with any VEA": int((per_prompt > 0).sum()),
        })
    return pd.DataFrame(rows)


def auroc_by_layer(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(labels, scores[:, layer]) for layer in range(scores.shape[1])])


def analyze_model(name: str) -> tuple[str, dict]:
    judged = pd.DataFrame(read_jsonl(model_dir(name) / "judged.jsonl"))
    judged["unparsed"] = judged["judge_output"].notna() & judged["vea"].isna()
    judged["vea"] = judged["vea"].fillna(False).astype(bool)
    rep = np.load(model_dir(name) / "representation.npz")
    sets, ids = rep["sets"], rep["ids"]
    layers = {"probe": np.arange(rep["probe"].shape[1]), "lens": rep["lens_layers"]}

    # Does each measure separate eval-like prompts from deployment-like ones?
    separation = {}
    for measure in MEASURES:
        for contrast, (positive, negative) in CONTRASTS.items():
            mask = np.isin(sets, [positive, negative])
            separation[measure, contrast] = auroc_by_layer(rep[measure][mask], sets[mask] == positive)

    # Does it predict which FORTRESS prompts the model verbalises eval awareness on?
    vea_rate = judged[judged["set"] == "fortress"].groupby("id")["vea"].mean()
    fortress = sets == "fortress"
    rate = vea_rate.reindex(ids[fortress]).to_numpy()
    prediction = {m: auroc_by_layer(rep[m][fortress], rate > 0) for m in MEASURES}

    lines = [f"## {name}", "", "### Verbalised eval awareness", "",
             vea_table(judged).to_markdown(index=False, floatfmt=".1f"), "",
             "### Representation", "",
             "Best layer is chosen by FORTRESS-vs-WildChat AUROC, without looking at VEA labels. "
             "'Unverbalised' is the share of FORTRESS prompts with no VEA in any rollout whose score "
             "exceeds the 95th percentile of WildChat prompts at that layer.", ""]
    rows = []
    for measure in MEASURES:
        best = int(np.argmax(separation[measure, "FORTRESS vs WildChat"]))
        scores = rep[measure][:, best]
        threshold = np.percentile(scores[sets == "wildchat"], 95)
        silent = scores[fortress][rate == 0]
        rows.append({
            "measure": MEASURES[measure], "best layer": int(layers[measure][best]),
            **{f"AUROC {c}": separation[measure, c][best] for c in CONTRASTS},
            "AUROC any-VEA prompts": prediction[measure][best],
            "Spearman ρ with VEA rate": spearmanr(scores[fortress], rate).statistic,
            "unverbalised %": 100 * (silent > threshold).mean(),
        })
    lines += [pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"),
              "", f"VEA-positive FORTRESS prompts: {(rate > 0).sum()} of {len(rate)}.", ""]
    return "\n".join(lines), {"layers": layers, "separation": separation, "prediction": prediction}


def style(ax, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_ylim(0.2, 1.0)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(results: dict) -> None:
    names = list(results)
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 6.5), squeeze=False, sharey=True)
    for col, name in enumerate(names):
        r = results[name]
        for row, measure in enumerate(MEASURES):
            ax = axes[row, col]
            for color, contrast in zip(COLORS, CONTRASTS):
                ax.plot(r["layers"][measure], r["separation"][measure, contrast], color=color, lw=2, label=contrast)
            ax.plot(r["layers"][measure], r["prediction"][measure], color=COLORS[2], lw=2,
                    label="any-VEA vs no-VEA FORTRESS prompts")
            style(ax, f"{name} · {MEASURES[measure]}")
            ax.set_xlabel("layer", color=MUTED, fontsize=8)
        axes[0, col].set_ylabel("AUROC", color=MUTED, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "auroc_by_layer.png", dpi=150)


def main() -> None:
    sections, results = ["# Eval awareness pilot: representation vs verbalisation", ""], {}
    for name in MODELS:
        if (model_dir(name) / "representation.npz").exists() and (model_dir(name) / "judged.jsonl").exists():
            text, results[name] = analyze_model(name)
            sections.append(text)
    (RESULTS / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    plot(results)
    print("\n".join(sections))


if __name__ == "__main__":
    main()
