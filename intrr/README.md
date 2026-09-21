# IntRR

[Paper](https://arxiv.org/abs/2602.20704) · [Citation](../README.md#papers-and-citation)

This folder contains the public semantic indexing and next-item recommendation implementation of IntRR. The Recursive-Assignment Network (RAN) uses item unique IDs (UIDs) to refine semantic ID (SID) representations and performs recursive SID decoding with one backbone token per item. See the [repository root README](../README.md#intrr-align-semantic-items-and-reduce-sequence-length) for the method and its role in DreamX-Rec.

Enter this directory from the repository root first; all commands below run from this directory:

```bash
cd intrr
```

## Environment

Use a dedicated Python 3.10 environment on Linux with CUDA-compatible GPUs and Bash 4 or later. The dependency file pins PyTorch 2.6.0 with CUDA 12.4 and the Lightning, Transformers, and TensorFlow dependencies used by this pipeline.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## 1. Prepare the Data

Prepare the Amazon Beauty, Sports, or Toys data using the [GRID data preparation workflow](https://github.com/snap-research/GRID). The experiment configurations read TFRecord files from the following directories, shown here for Sports:

```text
data/amazon_data/sports/
├── items/          # Item IDs and text for semantic embedding generation
├── training/       # Training interaction sequences
├── evaluation/     # Validation interaction sequences
└── testing/        # Test interaction sequences
```

Item records provide `id` and `text`; sequence records provide `user_id` and `sequence_data`. Keep item IDs consistent across item records, interaction sequences, embeddings, and SID mappings. Replace `sports` with `beauty` or `toys` when using another supported dataset.

The datasets, pretrained model weights, and generated embedding/SID artifacts are not bundled. This pipeline uses its own Amazon data format and does not read the shared IntTravel processed CSV.

## 2. Generate Semantic IDs

The generation script extracts item embeddings, trains a semantic indexing model, and exports item-to-SID mappings. The default embedding model is `google/flan-t5-xl`, with 2,048-dimensional embeddings, three indexing hierarchies, and a codebook width of 128.

```bash
CUDA_VISIBLE_DEVICES=0 bash gen_sid.sh --datasets sports --sid-methods rkmeans
```

Use `--datasets` and `--sid-methods` with comma-separated values to run other combinations. SID generation accepts `rkmeans`, `rqvae`, and `rvq`; the recommendation training scripts use `vqvae` for the last option.

The script prints the generated embedding and SID paths. Both artifacts use the filename `pickle/merged_predictions_tensor.pt` under their respective run directories in `logs/`.

Before training, update [configs/dataset_config.sh](configs/dataset_config.sh):

- Set `DATASET_EMBEDDING_PATHS` to the generated item embedding file.
- Update `get_sid_path` to resolve to the generated SID file for each dataset/method pair.
- Check `DATASET_ITEM_COUNTS` against the prepared item vocabulary.

The checked-in paths refer to earlier runs; `gen_sid.sh` does not update them automatically. Keep the generated SID dimensions and model settings aligned. The default indexing export appends a collision-disambiguation code to the three learned levels; the IntRR training configuration expects four SID levels.

## 3. Train and Evaluate IntRR

After configuring the artifact paths, start one experiment:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_intrr.sh \
  --datasets sports --seeds 42 --sid-type rkmeans --max-parallel 1
```

The launcher uses [configs/experiment/intrr_train_flat.yaml](configs/experiment/intrr_train_flat.yaml). This configuration uses a history length of 20 items, beam-search generation with prefix checking, and separate training, validation, and test splits. It selects the best checkpoint by `val/recall@10` and runs test evaluation after training.

Use `--datasets`, `--seeds`, and `--sid-type` with comma-separated values for multiple experiments. `--max-parallel` controls the number of concurrent experiments, not the number of GPUs. Its default is four; each experiment's trainer uses all visible GPUs. Set `CUDA_VISIBLE_DEVICES` to select the devices and use `--max-parallel 1` when experiments share them. Model, batch-size, and trainer settings are defined in the experiment YAML.

Launcher logs are written to `logs/intrr_<timestamp>/`. Hydra stores each training run under `logs/train/runs/intrr/<run-id>/`, including checkpoints, CSV metrics, and TensorBoard logs. The checkpoint callback retains the best checkpoint; it does not configure a separate last-epoch checkpoint.

## 4. Run the TIGER Baseline

The TIGER launcher uses the same prepared data and SID paths:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_tiger.sh \
  --datasets sports --seeds 42 --sid-type rkmeans --max-parallel 1
```

Its experiment settings are in [configs/experiment/tiger_train_flat.yaml](configs/experiment/tiger_train_flat.yaml). Launcher logs are written to `logs/tiger_<timestamp>/`. Use matching datasets, SID artifacts, and evaluation settings when comparing it with IntRR.

## Acknowledgments

This implementation builds on [GRID](https://github.com/snap-research/GRID) by Snap Research. We thank the GRID team for open-sourcing the semantic indexing and generative recommendation framework. See [LICENSE](LICENSE) and [third-party notices](notices.txt) for the applicable terms and attributions.
