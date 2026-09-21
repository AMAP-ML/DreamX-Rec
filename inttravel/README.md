# IntTravel

This folder contains the public S/I/F implementation of IntTravel. It loads `data_process/output/processed_features.csv` directly. Each sample is encoded as valid action tokens, six profile tokens, then trailing padding. The model uses a 3-layer task-guided HyperConnection encoder, task-specific selective gating, and DSFNet.

Run the following commands from the repository root:

```bash
cd inttravel
pip install -r requirements.txt
bash run.sh
```

`--batch-size` is the per-device batch size. Use `--data-path` to load another compatible processed CSV.

Evaluation is last1-only. For each of `where`, `how`, `when`, and `via`, the newest valid position is excluded from training and used as that task's evaluation position. Datasets with at most 100 million users use all users for this sequence-level protocol. Larger datasets reserve users with `rand_1 < 0.005` as a disjoint 0.5% test set because full-dataset evaluation is too expensive. The reported metrics are HR@1/HR@5/CIR for POI tasks, accuracy/top-3 for `how`, and accuracy/absolute class error for `when`.

The Torch backend supports one process on CPU or one GPU. Checkpoints are atomically written to `<output-dir>/checkpoint_last.pt` after every epoch. `--epochs` is the target total epoch count.

```bash
bash run.sh --embedding-backend torch --epochs 2 --output-dir output/inttravel
bash run.sh --embedding-backend torch --epochs 4 --output-dir output/inttravel --resume
```

The RecIS backend requires CUDA. It supports one GPU by default and multi-GPU DDP through `NPROC`. Every rank participates in training, evaluation, and checkpoint collectives; only rank 0 prints metrics. Its checkpoint root is `<output-dir>/recis_checkpoint`, and resume is exact at epoch boundaries with the same world size.

```bash
bash run.sh --embedding-backend recis --output-dir output/inttravel-recis
NPROC=2 bash run.sh --embedding-backend recis --output-dir output/inttravel-recis-2gpu
NPROC=2 bash run.sh --embedding-backend recis --epochs 2 --output-dir output/inttravel-recis-2gpu --resume
```
