"""Settings shared by every stage of the pilot."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROMPTS_FILE = RESULTS / "prompts.jsonl"

SEED = 0
N_PROMPTS_PER_SET = 100
MAX_PROMPT_CHARS = 2000

# Rollouts per prompt, by prompt set. WildChat gets no rollouts: it is only used
# as the "real deployment" reference for the representation measures.
ROLLOUTS_PER_PROMPT = {"fortress": 4, "fortress_benign": 2}
MAX_THINKING_TOKENS = 4096

# Qwen3 model-card settings for thinking mode.
SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    load_in_4bit: bool  # Qwen3-8B in bf16 does not fit in 16 GB VRAM.
    batch_size: int
    lens_file: str | None = None  # path inside the neuronpedia/jacobian-lens repo


MODELS = {
    "qwen3-4b": ModelSpec(
        "qwen3-4b", "Qwen/Qwen3-4B", load_in_4bit=False, batch_size=6,
        lens_file="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
    ),
    "qwen3-8b": ModelSpec(
        "qwen3-8b", "Qwen/Qwen3-8B", load_in_4bit=True, batch_size=8,
        lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
    ),
}
JUDGE = ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", load_in_4bit=True, batch_size=4)
LENS_REPO = "neuronpedia/jacobian-lens"


def model_dir(name: str) -> Path:
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
