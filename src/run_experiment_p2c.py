"""
run_experiment_p2c.py  —  Phase 2c: multi-species (leave-one-out) training
Changes on top of Phase 2a+2b:
  For each test species, train on ALL remaining species combined.
  The vocabulary is built jointly from all training species so node
  identities are shared across organisms. Class weights are recomputed
  from the combined training label distribution.

All 7 available species are used: elegans, melanogaster, sapiens,
arabidopsis, saccharomyces, bacillus, maripaludis.
"""

import os
import json
import time
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
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

from seq_encoder_p2c import build_dataset, build_vocab, read_fasta
from gnn_models       import CustomGAT
from pipeline_v       import train, test

# ============================================================
# CONFIG
# ============================================================
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)

CONFIG = {
    # --- Data ---
    "data_dir":   os.path.join(_BASE_DIR, "data"),
    "all_species": [
        "elegans", "melanogaster", "sapiens",
        "arabidopsis", "saccharomyces", "bacillus", "maripaludis",
    ],
    # Species to evaluate; the rest form the training pool each time
    "test_species": ["melanogaster", "sapiens", "arabidopsis", "saccharomyces"],
    "k":            4,
    "bidirectional": True,

    # --- Model architecture ---
    "emb_dim":    128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout":    0.2,
    "pool":       "mean",

    # --- Training ---
    "batch_size":             64,
    "epoch_n":                100,
    "learning_rate":          1e-3,
    "use_weighted_loss":      True,
    "label_smoothing":        0.05,
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

    # Phase 1 fix carried forward
    "cross_species_threshold": 0.5,

    # --- Output ---
    "results_dir":     os.path.join(_BASE_DIR, "results"),
    "experiment_name": "p2c_multitrain",
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


def load_species_graphs(species, data_dir, k, vocab, bidirectional):
    fasta_path  = os.path.join(data_dir, f"{species}_genes.fasta")
    labels_path = os.path.join(data_dir, f"{species}_labels.txt")
    graphs, _   = build_dataset(fasta_path, labels_path, k=k,
                                 vocab=vocab, bidirectional=bidirectional)
    return graphs


def build_joint_vocab(train_species_list, data_dir, k):
    """Build a single vocabulary from all training species combined."""
    all_seqs = []
    for sp in train_species_list:
        fasta_path = os.path.join(data_dir, f"{sp}_genes.fasta")
        records    = read_fasta(fasta_path)
        all_seqs.extend([r[1] for r in records])
    return build_vocab(all_seqs, k)


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

    base_out_dir = os.path.join(CONFIG["results_dir"], CONFIG["experiment_name"])
    os.makedirs(base_out_dir, exist_ok=True)
    save_json(CONFIG, os.path.join(base_out_dir, "run_config.json"))

    all_species_metrics = []

    for test_species in CONFIG["test_species"]:
        print_separator(f"Leave-one-out: test = {test_species}", char="*")

        train_species_list = [s for s in CONFIG["all_species"] if s != test_species]
        print(f"  Training species: {train_species_list}")

        out_dir = os.path.join(base_out_dir, f"test_{test_species}")
        os.makedirs(out_dir, exist_ok=True)

        # ---- Build joint vocabulary from all training species ----
        print_separator("Building joint vocabulary", char="-")
        vocab = build_joint_vocab(train_species_list, CONFIG["data_dir"], CONFIG["k"])
        print(f"  Joint vocab: {len(vocab)} k-mers  (k={CONFIG['k']})")

        vocab_path  = os.path.join(out_dir, "vocab.pth")
        config_path = os.path.join(out_dir, "config.pth")
        model_path  = os.path.join(out_dir, "model.pt")
        torch.save(vocab, vocab_path)

        # ---- Load and concatenate all training species graphs ----
        print_separator("Loading training data", char="-")
        all_train_graphs = []
        for sp in train_species_list:
            sp_graphs = load_species_graphs(
                sp, CONFIG["data_dir"], CONFIG["k"], vocab, CONFIG["bidirectional"]
            )
            sp_counts = Counter(int(g.y) for g in sp_graphs)
            print(f"  {sp}: {len(sp_graphs)} graphs | "
                  f"essential rate: {sp_counts[1]/len(sp_graphs)*100:.1f}%")
            all_train_graphs.extend(sp_graphs)

        combined_counts = Counter(int(g.y) for g in all_train_graphs)
        print(f"\n  Combined training set: {len(all_train_graphs)} graphs")
        print(f"  Combined labels: {dict(combined_counts)}  "
              f"(essential rate: {combined_counts[1]/len(all_train_graphs)*100:.1f}%)")

        # ---- Build model ----
        model_config = {
            "vocab_size": len(vocab),
            "emb_dim":    CONFIG["emb_dim"],
            "hidden_dim": CONFIG["hidden_dim"],
            "num_layers": CONFIG["num_layers"],
            "dropout":    CONFIG["dropout"],
            "pool":       CONFIG["pool"],
        }
        torch.save(model_config, config_path)

        model    = CustomGAT(**model_config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model: CustomGAT  |  params: {n_params:,}")

        # ---- Train ----
        print_separator("Training", char="-")
        train_results = train(
            graphs=all_train_graphs,
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
        print("Training history saved.")

        # ---- Test on held-out species ----
        print_separator(f"Testing on held-out: {test_species}", char="-")
        test_graphs = load_species_graphs(
            test_species, CONFIG["data_dir"], CONFIG["k"],
            torch.load(vocab_path), CONFIG["bidirectional"]
        )
        test_counts = Counter(int(g.y) for g in test_graphs)
        print(f"  {len(test_graphs)} graphs | "
              f"essential rate: {test_counts[1]/len(test_graphs)*100:.1f}%")

        test_model = CustomGAT(**torch.load(config_path)).to(device)
        results = test(
            graphs=test_graphs,
            model=test_model,
            model_path=model_path,
            batch_size=CONFIG["batch_size"],
            device=device,
            return_dataframe=True,
            threshold=CONFIG["cross_species_threshold"],
        )

        row = results["metrics"].copy()
        row.insert(0, "species", test_species)
        all_species_metrics.append(row)

        preds_path = os.path.join(out_dir, f"predictions_{test_species}.csv")
        results["probs"].to_csv(preds_path, index=False)
        print(f"  Predictions saved: {preds_path}")

        metrics_path = os.path.join(out_dir, f"metrics_{test_species}.csv")
        results["metrics"].to_csv(metrics_path, index=False)

    # ------------------------------------------------------------------ combined summary
    print_separator("Overall Summary")
    combined = pd.concat(all_species_metrics, ignore_index=True)
    summary_path = os.path.join(base_out_dir, "metrics_all_species.csv")
    combined.to_csv(summary_path, index=False)
    print(f"\nMetrics saved: {summary_path}")
    print("\n" + combined.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(combined["species"].tolist(), combined["MCC"].tolist(),
           color="steelblue", edgecolor="black")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("MCC")
    ax.set_title("Cross-species MCC  (leave-one-out multi-species training)")
    ax.set_ylim(-1.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(base_out_dir, "mcc_by_species.png"), dpi=120, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - total_start
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    print(f"All results saved in: {base_out_dir}")


if __name__ == "__main__":
    main()
