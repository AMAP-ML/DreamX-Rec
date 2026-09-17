# IntBMoE Demo

This folder contains one shared BlockMoE implementation and task-specific examples for language modeling, image classification, and POI recommendation. See the [repository root README](../README.md) for the method, results, and citation.

Enter the demo directory from the repository root first; all commands below run from this directory:

```bash
cd intbmoe_demo
```

The launchers use eight GPU processes by default; set `NPROC` to another value when needed. Timestamped logs and checkpoints are written under `../output/<experiment>/`.

## Environment

Create the shared Python 3.10 environment:

```bash
conda env create -f environment.yml
conda activate intbmoe
```

To reuse an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

## 1. Language Modeling on MiniPile

The language model uses an 18-layer LLaMA-style causal Transformer with hidden
size 768. Every FFN is replaced by IntBMoE with 8 blocks, 8 basis experts, and
top-2 block routing.

**Step 1: Install FlashAttention 2.** It is installed separately because it
compiles CUDA extensions against the PyTorch version in the active environment.

```bash
pip install flash-attn==2.5.6 --no-build-isolation
```

**Step 2: Download and prepare MiniPile.** The preparation script downloads
[MiniPile from Hugging Face](https://huggingface.co/datasets/JeanKaddour/minipile),
reuses the included 32K BPE tokenizer in `nlp/data/tokenizer`, tokenizes each
split, and packs the tokens into fixed-length sequences of 1,024 tokens.

```bash
python3 nlp/prepare_data.py
```

The raw and packed datasets are stored under `nlp/data`. Completed downloads
and preprocessing results are reused when the command is run again.

**Step 3: Start training.** The launcher reads the packed dataset produced in
Step 2 and starts an eight-GPU job by default.

```bash
bash run_nlp.sh
```

Set `NPROC` to change the number of GPUs. If the packed dataset already exists
elsewhere, set `DATA_PATH` to the directory that contains `tokenized/` and
`meta.json`:

```bash
DATA_PATH=/path/to/minipile NPROC=1 bash run_nlp.sh
```

The training log is written to `../output/nlp_<timestamp>/train.log`.

## 2. Image Classification on ImageNet-1K

The vision model is an eight-layer DeiT-Tiny-style Transformer. IntBMoE
replaces the FFN in alternating layers 0, 2, 4, and 6; patch tokens are routed
through IntBMoE while the class token retains a dense FFN.

**Step 1: Obtain dataset access.** Accept the access conditions for
[ImageNet-1K on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
then authenticate on the training machine:

```bash
huggingface-cli login
```

**Step 2: Download ImageNet-1K.** The download script stores all splits in the
Hugging Face Datasets cache at `/var/tmp/hf_datasets_cache/` by default and
prints the split sizes and one sample after loading completes.

```bash
bash cv/download_data.sh
```

To use another cache directory, pass it through `DATA_PATH`:

```bash
DATA_PATH=/path/to/hf_cache bash cv/download_data.sh
```

The result remains in Hugging Face Datasets cache format; no ImageFolder
conversion is required.

**Step 3: Start training.** The launcher reads the train and validation splits
from `/var/tmp/hf_datasets_cache/` by default and starts an eight-GPU job.

```bash
bash run_cv.sh
```

Set `NPROC` to change the number of GPUs. When Step 2 used another cache
directory, pass the same `DATA_PATH` to the training command:

```bash
DATA_PATH=/path/to/hf_cache NPROC=1 bash run_cv.sh
```

The training log is written to `../output/cv_<timestamp>/train.log`.

## 3. POI Recommendation on IntTravel

This example adapts the public IntTravel FIS representation to an IntBMoE POI
recommendation model. It supports up to 40 newest-first `[F, I, S]` sessions
plus 6 user-profile tokens, for a maximum sequence length of 126.

**Step 1: Obtain the raw tables.** Clone the
[IntTravel repository](https://github.com/DreamX-Rec/IntTravel), which includes
a runnable example under `data_process/raw_data`:

```bash
git clone https://github.com/DreamX-Rec/IntTravel.git /path/to/IntTravel
```

The full public data is available from the
[IntTravel dataset on Hugging Face](https://huggingface.co/datasets/GD-ML/IntTravel_dataset).

The raw-data directory must contain these four CSV files:

```text
user_action.csv
user_profile.csv
poi_info.csv
poi_info_groupby_geographic.csv
```

**Step 2: Run IntTravel data processing once.** Use the processing scripts from
the cloned IntTravel repository. The first command constructs the FIS sequences
and labels. The second converts them into the feature arrays consumed by the
recommendation model.

```bash
python3 /path/to/IntTravel/data_process/data_processor.py \
  --input-dir /path/to/IntTravel/data_process/raw_data \
  --output-dir /path/to/IntTravel/data_process/output \
  --skip-column-check

python3 /path/to/IntTravel/data_process/post_process_features.py \
  --input-file /path/to/IntTravel/data_process/output/multi_task_input_seq_all.csv \
  --output-file /path/to/IntTravel/data_process/output/processed_features.csv
```

Processing only needs to be repeated when the raw data changes.

**Step 3: Start training.** The launcher reads
`../data_process/output/processed_features.csv` by default and does not process
the four raw tables again. Training uses eight GPUs by default.

```bash
bash run_recommendation.sh
```

To use a processed feature file in another location or change the GPU count:

```bash
DATA_PATH=/path/to/processed_features.csv NPROC=1 bash run_recommendation.sh
```

The default PyTorch backend uses dense embedding gradients for multi-GPU DDP.
Single-process training can opt into sparse embedding gradients:

```bash
NPROC=1 bash run_recommendation.sh --sparse-embedding
```

The training log is written to
`../output/recommendation_<timestamp>/train.log`.

**Step 4 (optional): Use RecIS.** The default pure-PyTorch backend materializes
only IDs present in the supplied data. The optional RecIS backend provides
dynamic hash tables and requires a separate build of
[`RecIS v1.2.0`](https://github.com/alibaba/RecIS/releases/tag/v1.2.0):

```bash
DATA_PATH=/path/to/IntTravel/data_process/output/processed_features.csv \
  bash run_recommendation.sh --backend recis
```
