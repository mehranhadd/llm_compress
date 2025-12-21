# Comparing Text Compression Capabilities of LLMs with Traditional Compression Algorithms.

> [!NOTE]
> `llm_encode.py` and `finetune.py` are adapted from  [FineZip](https://github.com/fazalmittu/FineZip)

Compressing text data using traditional compression approaches and LLMs and recording the compression time and resources usage.

## Models

Used traditional methods are standard Python libraries of:
- `zlib`
- `bz2`
- `lzma`

as well as [`pyppmd`](https://pyppmd.readthedocs.io/en/latest/) Python library and Tawa toolkit (implemented in C) [1].

The used LLM for our experiments is [LLaMA 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B). Any LLM from [HuggingFace](https://huggingface.co/) can be utilised by editing the LLM name in `compress.py`.

## Metrics

The following metrics are logged and stored in CSV files:
- Compression ratio $=$ $\frac{compressed\_size}{original\_size}$
- Compression duration
- Peak and average GPU usage
- Peak GPU memory usage
- Peak and average CPU usage
- Peak memory usage
- Percentage of the true tokens ranked 0 among the predicted tokens by the LLM
- Percentage of the true tokens ranked between 0 and 15 among the predicted tokens by the LLM

> [!NOTE]
> GPU and GPU memory usage are not recorded for Tawa executions.
>
> Ranking metrics are recorded only for the LLM.

## How to Use

> [!IMPORTANT]
> The code is tested on Python version 3.11.2

To use, first clone this repository:

```
gh repo clone https://github.com/mehranhaddadi13/llm_compress
```

Then, change the current directory to the directory of the repo:

```
cd llm_compress
```

Next, make a directory named `datasets` and move your datasets to it:

```
mkdir datasets
```

Then, install the requirements:

```
python -m pip insatll -r requirements.txt
```

Finally, edit `compress.py` and add your HuggingFace login token , before running it:

```
python compress.py
```

> [!WARNING]
> Each fine-tuning and compression processes via LLMs take hours to days to finish, depending on the available resources and the size of the LLM and dataset.

[1]: William J. Teahan 2018. [A Compression-Based Toolkit for Modelling and Processing Natural Language Text.](https://doi.org/10.3390/info9120294)