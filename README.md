# Comparing Text Compression Capabilities of LLMs with Traditional Algorithms

### The official repository of **Compraing Text Compression Capabilities of Large Language Models with Traditional Compression Algorithms**. [1]

This research project compares text compression using language models with traditional compressors. It records compression ratio, elapsed time, CPU/GPU usage, memory use, and LLM token-ranking metrics. The LLM fine-tuning and encoding workflows are adapted from [FineZip](https://github.com/fazalmittu/FineZip).

## Methods

Traditional baselines include Python's `gzip`, `zlib`, `bz2`, and `lzma`, plus [`pyppmd`](https://pyppmd.readthedocs.io/). The project also includes the Tawa toolkit [2] in `Tawa-0.7/`. The default language model is [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B); change it with the `LLM_MODEL` environment variable.

## Requirements

- Python 3.11 (the original project was tested with Python 3.11.2)
- A compatible CUDA/PyTorch setup and sufficient GPU memory for the selected model
- A Hugging Face account/token with access to any gated model you use
- The native Tawa executables included in `Tawa-0.7/` must be usable on your operating system

Install Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run an experiment

Place plain-text datasets under `datasets/` (subdirectories are supported).

Set your token in the environment rather than writing it into source code:

```bash
export HF_TOKEN="your_hugging_face_token"
python compress.py
```

Optional settings let you choose a model and reduce the experiment size while testing:

```bash
export LLM_MODEL="meta-llama/Llama-3.2-1B"
export LLM_EPOCHS=2
export LLM_LORA_RANK=8
python compress.py
```

`LLM_EPOCHS` is passed to the training loop as `max_steps` in the current implementation. Small values are useful for smoke tests, but they do not produce research-quality results. The full workflow runs Tawa, fine-tunes the LLM, compresses each dataset, and evaluates traditional baselines. It can take hours or days and requires significant compute resources.

## Output directories

- `stats/`: CSV summaries of training and compression metrics
- `finetuned_models/`: fine-tuned model adapters/checkpoints
- `zipped/`: compressed outputs and token-rank logs
- `models/`, `tawa_logs/`: Tawa models and resource logs
- `output/`: Trainer working files

Run lightweight project checks with:

```bash
python -m unittest discover -s tests -v
```

This check does not launch fine-tuning or require a GPU. Full end-to-end validation should be performed separately on a small dataset in an environment with the required model access and hardware.

## Reference

[1] Mehran Haddadi, William J. Teahan (2026). [Comparing Text Compression Capabilities of Large Language Models with Traditional Compression Algorithms](https://aclanthology.org/2026.eacl-srw.16/), *In Proceedings of the EACL 2026 SRW*.

[2]: William J. Teahan (2018). [A Compression-Based Toolkit for Modelling and Processing Natural Language Text](https://doi.org/10.3390/info9120294).
