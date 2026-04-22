import os
import torch
import random
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster / SLURM
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from collections import Counter


def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        use_weighted_loss=True,          # 1.3: class-weighted CE replaces WeightedRandomSampler
        label_smoothing=0.05,            # 1.3: mild smoothing to reduce over-confidence
        use_scheduler=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        use_gradient_clipping=True,
        clip_value=1.0,
        model_path="model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_split=0.2,
        random_seed=111,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
        cm_log_interval=5,               # print confusion matrix every N epochs
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = model.to(device)

    # === 3.4: Stratified train/val split (prevents skewed val set with rare positives) ===
    all_labels = [int(g.y) for g in graphs]
    train_idx, val_idx = train_test_split(
        list(range(len(graphs))),
        test_size=val_split,
        stratify=all_labels,
        random_state=random_seed,
    )
    trainset = [graphs[i] for i in train_idx]
    valset   = [graphs[i] for i in val_idx]

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False)

    train_label_count = Counter(int(g.y) for g in trainset)
    print(f"\n{len(graphs)} graphs split (stratified): "
          f"Train={len(trainset)}, Val={len(valset)}")
    print(f"  Train class counts: {dict(train_label_count)}")

    # === 1.3: Class-weighted loss (no WeightedRandomSampler — avoids double-correction) ===
    if use_weighted_loss:
        n     = len(trainset)
        n_neg = train_label_count[0]
        n_pos = train_label_count[1]
        w = torch.tensor(
            [n / (2.0 * n_neg), n / (2.0 * n_pos)],
            dtype=torch.float,
            device=device,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
        print(f"  Class weights — non-essential: {w[0]:.3f}, essential: {w[1]:.3f}")
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # === Optimizer ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # === 1.1: Scheduler on MCC (mode='max', pass MCC directly — no sign inversion) ===
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",                  # higher MCC is better
            patience=scheduler_patience,
            factor=scheduler_factor,
            min_lr=1e-6,
        )

    # === Logging containers ===
    train_losses,     val_losses     = [], []
    train_mccs,       val_mccs       = [], []
    train_aucs,       val_aucs       = [], []
    train_accuracies, val_accuracies = [], []

    best_val_mcc      = -1.0    # MCC ∈ [-1, 1]; safe sentinel
    best_threshold    = 0.5
    best_epoch        = 0
    early_stopping_counter = 0

    print(f"\n########## Training for {epoch_n} epochs on {device}...")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()

        # ------------------------------------------------------------------ train
        model.train()
        total_loss = 0
        y_true_tr, y_pred_tr, y_prob_tr = [], [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out  = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()

            probs_np = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            y_prob_tr.extend(probs_np)
            y_pred_tr.extend((probs_np >= 0.5).astype(int))
            y_true_tr.extend(batch.y.detach().cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true_tr, y_pred_tr)
        train_mcc = matthews_corrcoef(y_true_tr, y_pred_tr)
        train_auc = (roc_auc_score(y_true_tr, y_prob_tr)
                     if len(set(y_true_tr)) > 1 else float("nan"))
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_mccs.append(train_mcc)
        train_aucs.append(train_auc)

        # ------------------------------------------------------------------ validate
        model.eval()
        y_true_val, y_prob_val = [], []
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out   = model(batch)
                total_val_loss += criterion(out, batch.y).item()
                probs_np = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                y_prob_val.extend(probs_np)
                y_true_val.extend(batch.y.cpu().numpy())

        y_prob_val_arr = np.array(y_prob_val)
        y_true_val_arr = np.array(y_true_val)

        # === 1.2: Threshold sweep — pick threshold that maximises val MCC ===
        best_thr_epoch, best_mcc_epoch = 0.5, -2.0
        for thr in np.linspace(0.05, 0.95, 91):
            preds_thr = (y_prob_val_arr >= thr).astype(int)
            mcc_thr   = matthews_corrcoef(y_true_val_arr, preds_thr)
            if mcc_thr > best_mcc_epoch:
                best_mcc_epoch  = mcc_thr
                best_thr_epoch  = thr

        y_pred_val_arr = (y_prob_val_arr >= best_thr_epoch).astype(int)
        val_loss = total_val_loss / len(val_loader)
        val_acc  = accuracy_score(y_true_val_arr, y_pred_val_arr)
        val_mcc  = best_mcc_epoch
        val_auc  = (roc_auc_score(y_true_val_arr, y_prob_val_arr)
                    if len(set(y_true_val)) > 1 else float("nan"))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_mccs.append(val_mcc)
        val_aucs.append(val_auc)

        # === 1.1: Scheduler steps on MCC ===
        if use_scheduler and scheduler is not None:
            scheduler.step(val_mcc)

        dt = time.time() - t0
        print(
            f"- Epoch [{epoch:03d}/{epoch_n}] "
            f"Train Loss: {avg_train_loss:.4f} | Train MCC: {train_mcc:.3f} | Train AUC: {train_auc:.3f} "
            f"| Val Loss: {val_loss:.4f} | Val MCC: {val_mcc:.3f} | Val AUC: {val_auc:.3f} "
            f"| Thr: {best_thr_epoch:.2f} | ES: {early_stopping_counter}/{early_stopping_patience} "
            f"| Time: {dt:.1f}s"
        )

        # 3.5: Periodic confusion matrix
        if epoch % cm_log_interval == 0:
            cm = confusion_matrix(y_true_val_arr, y_pred_val_arr)
            print(f"  Val Confusion Matrix (thr={best_thr_epoch:.2f}):\n{cm}")

        # === 1.1: Early stopping on val MCC ===
        improvement = val_mcc - best_val_mcc
        if improvement > early_stopping_min_delta:
            best_val_mcc   = val_mcc
            best_threshold = float(best_thr_epoch)
            best_epoch     = epoch
            early_stopping_counter = 0

            # Save both weights and the best threshold together
            torch.save(
                {"state_dict": model.state_dict(), "threshold": best_threshold},
                model_path,
            )
            print(f"+ New best MCC: {best_val_mcc:.4f} @ thr={best_threshold:.2f} (epoch {epoch})")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"XXXXXXXXXX Early stopping triggered at epoch {epoch}!")
                break

    print("=================================================")
    print(f"Best val MCC: {best_val_mcc:.4f} @ threshold={best_threshold:.2f}, epoch {best_epoch}")
    print("=================================================")

    results = {
        "train_losses":     train_losses,
        "train_accuracies": train_accuracies,
        "train_mccs":       train_mccs,
        "train_aucs":       train_aucs,
        "val_losses":       val_losses,
        "val_accuracies":   val_accuracies,
        "val_mccs":         val_mccs,
        "val_aucs":         val_aucs,
        "best_val_mcc":     best_val_mcc,
        "best_threshold":   best_threshold,
        "best_epoch":       best_epoch,
    }

    # === Learning curves (saved to disk, not displayed) ===
    epochs_range = range(1, len(train_mccs) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses,   label="Val Loss",   linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title("Loss Curve")

    axes[1].plot(epochs_range, train_mccs, label="Train MCC", linewidth=2)
    axes[1].plot(epochs_range, val_mccs,   label="Val MCC",   linewidth=2)
    axes[1].axvline(x=best_epoch, color="r", linestyle="--", alpha=0.7,
                    label=f"Best Epoch {best_epoch}")
    axes[1].axhline(y=best_val_mcc, color="r", linestyle=":", alpha=0.5)
    axes[1].text(best_epoch + 0.5, best_val_mcc + 0.01,
                 f"Best MCC: {best_val_mcc:.4f}", color="red",
                 fontweight="bold", fontsize=9)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MCC")
    axes[1].legend(); axes[1].set_title("MCC per Epoch")

    axes[2].plot(epochs_range, train_aucs, label="Train AUC", linewidth=2)
    axes[2].plot(epochs_range, val_aucs,   label="Val AUC",   linewidth=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC")
    axes[2].legend(); axes[2].set_title("AUC per Epoch (reference)")

    plt.tight_layout()
    plot_path = model_path.replace(".pt", "_learning_curves.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Learning curves saved to: {plot_path}")

    return results


def test(
        graphs,
        model,
        model_path,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_dataframe=True,
        threshold=None,     # manual override; if None, uses threshold saved in checkpoint
):
    # === Load weights + threshold ===
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        saved_threshold = checkpoint.get("threshold", 0.5)
    else:
        # backwards-compatible: plain state dict from old pipeline.py
        model.load_state_dict(checkpoint)
        saved_threshold = 0.5

    thr = threshold if threshold is not None else saved_threshold
    print(f"  Decision threshold: {thr:.4f}")

    model = model.to(device)
    model.eval()

    test_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    criterion   = torch.nn.CrossEntropyLoss()

    y_true, y_prob, gene_ids_all = [], [], []
    total_loss = 0

    print(f"########## Testing on {len(graphs)} samples...")

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch)
            total_loss += criterion(out, batch.y).item()
            probs_np = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_true.extend(batch.y.cpu().numpy())
            y_prob.extend(probs_np)
            gene_ids_all.extend(batch.gene_id)

    y_prob_arr = np.array(y_prob)
    y_true_arr = np.array(y_true)
    y_pred_arr = (y_prob_arr >= thr).astype(int)

    # === Metrics ===
    test_acc  = accuracy_score(y_true_arr, y_pred_arr)
    test_auc  = roc_auc_score(y_true_arr, y_prob_arr)
    test_mcc  = matthews_corrcoef(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    recall    = recall_score(y_true_arr,    y_pred_arr, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel().tolist()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics_dict = {
        "#Sample":     len(graphs),
        "Threshold":   round(thr, 4),
        "TN":          int(tn),
        "FP":          int(fp),
        "FN":          int(fn),
        "TP":          int(tp),
        "Sensitivity": round(sn, 4),
        "Specificity": round(sp, 4),
        "Precision":   round(precision, 4),
        "Recall":      round(recall, 4),
        "Accuracy":    round(test_acc, 4),
        "AUC":         round(test_auc, 4),
        "MCC":         round(test_mcc, 4),
    }

    print("---------- Test Results ----------")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v}")
    print("----------------------------------")

    metrics_df = pd.DataFrame([metrics_dict])
    results = {"metrics": metrics_df}

    if return_dataframe:
        df = pd.DataFrame({
            "gene_id":         gene_ids_all,
            "true_label":      y_true_arr,
            "predicted_label": y_pred_arr,
            "prob_class_1":    y_prob_arr,
            "prob_class_0":    1 - y_prob_arr,
        })
        df["confidence"] = df[["prob_class_1", "prob_class_0"]].max(axis=1)
        df["is_correct"]  = df["true_label"] == df["predicted_label"]

        results["probs"]        = df
        results["sorted_probs"] = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    print("########## Testing completed.")
    return results
