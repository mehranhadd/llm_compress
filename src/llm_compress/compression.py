"""Run the complete text-compression experiment pipeline."""

from __future__ import annotations

import gzip
import zlib
import bz2
import lzma
import os
import subprocess
from pathlib import Path

import pyppmd
from huggingface_hub import login

from llm_compress.finetune import ft
from llm_compress.llm_encode import compress as llm_compress
from llm_compress.traditional_encode import Compressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    """Run Tawa, fine-tuning, LLM compression, and baseline compression.

    Run this command from any directory; project paths are resolved from the
    repository root. The Hugging Face token must be supplied as HF_TOKEN.
    """
    os.chdir(PROJECT_ROOT)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Create a Hugging Face access token and run "
            "the command with HF_TOKEN in your environment; do not hard-code it."
        )
    login(token=token)

    # Tawa's shell workflow expects to run from the repository root.
    subprocess.run(["bash", "compress.sh"], check=True)

    dataset_root = PROJECT_ROOT / "datasets"
    if not dataset_root.is_dir():
        raise SystemExit("No datasets/ directory found. Add text datasets first.")
    datasets = sorted(
        str(path) for path in dataset_root.rglob("*")
        if path.is_file() and not any(child.is_dir() for child in path.parent.iterdir())
    )
    if not datasets:
        raise SystemExit("No dataset files found under datasets/.")

    stats_dir = PROJECT_ROOT / "stats"
    stats_dir.mkdir(exist_ok=True)
    finetuned_dir = PROJECT_ROOT / "finetuned_models"
    finetuned_dir.mkdir(exist_ok=True)
    (PROJECT_ROOT / "zipped").mkdir(exist_ok=True)

    ft_stats = stats_dir / "ftstats.csv"
    if not ft_stats.exists():
        ft_stats.write_text("data,time,gpu avg,gpu peak,gpu mem,cpu avg,cpu peak,mem,", encoding="utf-8")

    trad_models = {
        gzip: {"compresslevel": 9},
        zlib: {"level": 9},
        bz2: {"compresslevel": 9},
        lzma: {"preset": 9},
        pyppmd: {},
    }
    llm_model = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.2-1B")
    epochs = int(os.environ.get("LLM_EPOCHS", "256"))
    lora_rank = int(os.environ.get("LLM_LORA_RANK", "8"))

    for dataset in datasets:
        dataset_name = Path(dataset).name
        finetuned_model = f"{llm_model.replace('/', '-')}-{Path(dataset).stem}"
        ft(
            llm_model,
            save_path=str(finetuned_dir / finetuned_model),
            dataset_path=dataset,
            stat_path=str(ft_stats),
            block_size=128,
            epochs=epochs,
            r=lora_rank,
            learning_rate=1e-4,
            batch_size=64,
        )
        c_stats = stats_dir / f"{dataset_name}_cstats.csv"
        if not c_stats.exists():
            c_stats.write_text(
                "model,time,gpu avg,gpu peak,gpu mem,cpu avg,cpu peak,mem,original size,compressed size,cr,ranks_0_15,ranks_0,",
                encoding="utf-8",
            )
        finetuned_models = {f"{finetuned_model}_{epochs}_r{lora_rank}": llm_model}
        llm_compress(
            finetuned_models=finetuned_models,
            original_model_names=[llm_model],
            context_sizes=[512],
            batch_size=64,
            file_path=dataset,
            stat_path=str(c_stats),
        )
        for algorithm, kwargs in trad_models.items():
            Compressor(algorithm, dataset, stat_path=str(c_stats)).compress(**kwargs)

if __name__ == "__main__":
    main()
