# Comparing Text Compression Capabilities of LLMs with Traditional Compression Algorithms.

Compressing text data using traditional compression approaches and LLMs and recording the compression time and resources usage.

Used traditional methods are standard Python libraries of:
- `zlib`
- `bz2`
- `lzma`

as well as [`pyppmd`](https://pyppmd.readthedocs.io/en/latest/) Python library and Tawa toolkit (implemented in C) [1].

The used LLM for our experiments is [LLaMA 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B). Any LLM from [HuggingFace](https://huggingface.co/) can be utilised by editing the LLM name in `compress.py`.

> [!NOTE]
> `llm_encode.py` and `finetune.py` are adapted from  [FineZip](https://github.com/fazalmittu/FineZip)

[1]: William J. Teahan 2018. [A Compression-Based Toolkit for Modelling and Processing Natural Language Text.](https://doi.org/10.3390/info9120294)
