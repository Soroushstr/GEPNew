"""
run_experiment.py
All-in-one gene essentiality prediction: data loading, training, and cross-species testing.
Designed for SLURM submission — all output is printed to stdout (redirect in .slurm file)
and all artefacts (model, metrics, predictions, plots) are saved to results_dir/experiment_name/.
"""

# ============================================================
# Imports
# ============================================================
import os
import sys
import json
import time
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")       # must be set before pyplot import (no display on cluster)
import matplotlib.pyplot as plt

import torch
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Project modules (same directory)
from seq_encoder import build_dataset
from gnn_models   import CustomGAT
from pipeline_v   import train, test

# ============================================================
# CONFIG — edit this block before submitting the SLURM job
# ============================================================
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)

CONFIG = {
    # --- Data ---
    "data_dir":        os.path.join(_BASE_DIR, "data"),
    "train_species":   "elegans",
    "test_species":    ["melanogaster", "sapiens", "arabidopsis", "saccharomyces"],
    "k":               3,

    # --- Model architecture ---
    "emb_dim":         128,
    "hidden_dim":      128,
    "num_layers":      2,
    "dropout":         0.2,
    "pool":            "mean",

    # --- Training (changes 1.1 / 1.2 / 1.3 active in pipeline_v.py) ---
    "batch_size":             64,
    "epoch_n":                100,
    "learning_rate":          1e-3,
    "use_weighted_loss":      True,    # 1.3: class-weighted CE (no sampler)
    "label_smoothing":        0.05,    # 1.3: mild smoothing
    "use_scheduler":          True,
    "scheduler_patience":     3,
    "scheduler_factor":       0.5,
    "use_gradient_clipping":  True,
    "clip_value":             1.0,
    "val_split":              0.2,
    "random_seed":            111,
    "early_stopping_patience":  10,
    "early_stopping_min_delta": 0.001,
    "cm_log_interval":          5,

    # --- Output ---
    "results_dir":      os.path.join(_BASE_DIR, "results"),
    "experiment_name":  "elegans_mcc_v1",      # change per run
}


# ============================================================
# Helpers
# ============================================================

def print_separator(title="", char="=", width=60):
    if title:
        print(f"\n{char*width}")
        print(f"  {title}")
        print(f"{char*width}")
    else:
        print(char * width)


def save_json(obj, path):
    def _serialize(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, list):
            return [_serialize(x) for x in v]
        return str(v)
    with open(path, "w") as f:
        json.dump({k: _serialize(v) for k, v in obj.items()}, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    total_start = time.time()

    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # === Output directory ===
    out_dir = os.path.join(CONFIG["results_dir"], CONFIG["experiment_name"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results directory: {out_dir}")

    # Save the full run config immediately so it is always present
    save_json(CONFIG, os.path.join(out_dir, "run_config.json"))

    # ------------------------------------------------------------------ load training data
    print_separator("Loading training data")
    train_genes  = os.path.join(CONFIG["data_dir"], f"{CONFIG['train_species']}_genes.fasta")
    train_labels = os.path.join(CONFIG["data_dir"], f"{CONFIG['train_species']}_labels.txt")

    graphs, vocab = build_dataset(train_genes, train_labels, k=CONFIG["k"])
    label_counts  = Counter(int(g.y) for g in graphs)
    print(f"Species : {CONFIG['train_species']}")
    print(f"Graphs  : {len(graphs)}")
    print(f"Vocab   : {len(vocab)} k-mers  (k={CONFIG['k']})")
    print(f"Labels  : {dict(label_counts)}  "
          f"(essential rate: {label_counts[1]/len(graphs)*100:.1f}%)")

    model_path  = os.path.join(out_dir, "model.pt")
    vocab_path  = os.path.join(out_dir, "vocab.pth")
    config_path = os.path.join(out_dir, "config.pth")

    torch.save(vocab, vocab_path)

    model_config = {
        "vocab_size": len(vocab),
        "emb_dim":    CONFIG["emb_dim"],
        "hidden_dim": CONFIG["hidden_dim"],
        "num_layers": CONFIG["num_layers"],
        "dropout":    CONFIG["dropout"],
        "pool":       CONFIG["pool"],
    }
    torch.save(model_config, config_path)

    # ------------------------------------------------------------------ build model
    model = CustomGAT(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : CustomGAT  |  params: {n_params:,}")

    # ------------------------------------------------------------------ train
    print_separator("Training")
    train_results = train(
        graphs=graphs,
        model=model,
        batch_size=CONFIG["batch_size"],
        epoch_n=CONFIG["epoch_n"],
        learning_rate=CONFIG["learning_rate"],
        use_weighted_loss=CONFIG["use_weighted_loss"],
        label_smoothing=CONFIG["label_smoothing"],
        use_scheduler=CONFIG["use_scheduler"],
        scheduler_patience=CONFIG["scheduler_patience"],
        scheduler_factor=CONFIG["scheduler_factor"],
        use_gradient_clipping=CONFIG["use_gradient_clipping"],
        clip_value=CONFIG["clip_value"],
        model_path=model_path,
        device=device,
        val_split=CONFIG["val_split"],
        random_seed=CONFIG["random_seed"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        early_stopping_min_delta=CONFIG["early_stopping_min_delta"],
        cm_log_interval=CONFIG["cm_log_interval"],
    )

    save_json(train_results, os.path.join(out_dir, "training_history.json"))
    print(f"Training history saved.")

    # ------------------------------------------------------------------ test on each species
    print_separator("Cross-species evaluation")
    all_metrics = []

    for species in CONFIG["test_species"]:
        print_separator(f"Testing: {species}", char="-")

        test_genes  = os.path.join(CONFIG["data_dir"], f"{species}_genes.fasta")
        test_labels = os.path.join(CONFIG["data_dir"], f"{species}_labels.txt")

        vocab_loaded = torch.load(vocab_path)
        test_graphs, _ = build_dataset(
            test_genes, test_labels, k=CONFIG["k"], vocab=vocab_loaded
        )
        test_counts = Counter(int(g.y) for g in test_graphs)
        print(f"  {len(test_graphs)} graphs | "
              f"essential rate: {test_counts[1]/len(test_graphs)*100:.1f}%")

        # Reload model architecture fresh for each species
        model_cfg  = torch.load(config_path)
        test_model = CustomGAT(**model_cfg).to(device)

        results = test(
            graphs=test_graphs,
            model=test_model,
            model_path=model_path,
            batch_size=CONFIG["batch_size"],
            device=device,
            return_dataframe=True,
        )

        # Annotate and accumulate metrics
        row = results["metrics"].copy()
        row.insert(0, "species", species)
        all_metrics.append(row)

        preds_path = os.path.join(out_dir, f"predictions_{species}.csv")
        results["probs"].to_csv(preds_path, index=False)
        print(f"  Predictions saved: {preds_path}")

    # ------------------------------------------------------------------ also evaluate on train species (sanity check)
    print_separator(f"Sanity check — train species: {CONFIG['train_species']}", char="-")
    sanity_graphs, _ = build_dataset(
        train_genes, train_labels, k=CONFIG["k"], vocab=torch.load(vocab_path)
    )
    sanity_model = CustomGAT(**torch.load(config_path)).to(device)
    sanity_results = test(
        graphs=sanity_graphs,
        model=sanity_model,
        model_path=model_path,
        batch_size=CONFIG["batch_size"],
        device=device,
        return_dataframe=False,
    )
    row = sanity_results["metrics"].copy()
    row.insert(0, "species", CONFIG["train_species"] + "_train_sanity")
    all_metrics.insert(0, row)

    # ------------------------------------------------------------------ combined metrics table
    print_separator("Summary")
    combined = pd.concat(all_metrics, ignore_index=True)
    metrics_path = os.path.join(out_dir, "metrics_all_species.csv")
    combined.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved: {metrics_path}")
    print("\n" + combined.to_string(index=False))

    # ------------------------------------------------------------------ MCC bar chart across species
    fig, ax = plt.subplots(figsize=(8, 4))
    species_labels = combined["species"].tolist()
    mcc_values     = combined["MCC"].tolist()
    colors = ["steelblue" if "_sanity" not in s else "lightgray" for s in species_labels]
    ax.bar(species_labels, mcc_values, color=colors, edgecolor="black")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("MCC")
    ax.set_title(f"Cross-species MCC  (trained on {CONFIG['train_species']})")
    ax.set_ylim(-1.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mcc_by_species.png"), dpi=120, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - total_start
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    print(f"All results saved in: {out_dir}")


if __name__ == "__main__":
    main()
