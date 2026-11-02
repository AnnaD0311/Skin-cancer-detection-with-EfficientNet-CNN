#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BINARY Training: melanoma (mel) vs non-melanoma  —  HAM10000
=================================================================
Translated script of "Stage 1" from section 13 of the notebook
`Classification_leasion_hierarquico.ipynb`.

What it does
------------
1. (Optional) Organizes HAM10000 into Datasets/training|validation/<class>/
   with an 80/20 split BY LESION (prevents data leakage).
2. Loads images in uint8 and converts labels to binary (1 = mel).
3. Builds an EfficientNet (B0/B4) with a sigmoid head and trains it in two phases:
   head (frozen backbone) -> fine-tuning (last N layers, low lr).
4. Evaluates on validation: AUC, AP, sensitivity, specificity, precision, F1,
   confusion matrix, threshold sweep.
5. Creates a run folder with EVERYTHING that occurred:
      training_summary.txt       <- complete summary (config, environment, data, epochs,
                                     events, metrics, warnings, files)
      console.log                <- copy of the console output
      epoch_history.csv          <- metrics per epoch
      validation_predictions.csv <- P(mel) per validation image
      metrics.json               <- final and phase metrics
      training_curves.png / evaluation_stage1.png
      <name>_stage1.keras        <- trained model (+ <name>_stage1.json with metadata)

Examples
--------
  # dataset already organized
  python binary_training_mel.py --dataset-dir Datasets

  # first organize HAM10000 (CSV + images in --base-dir) and train
  python binary_training_mel.py --base-dir /path/to/HAM10000

  # quick pipeline test (30 img/class, B0, 1 epoch, no fine-tuning)
  python binary_training_mel.py --dataset-dir Datasets --quick

  # prioritize not missing melanomas (higher threshold with sensitivity >= 85%)
  python binary_training_mel.py --dataset-dir Datasets --thr-mode sensitivity --min-sensitivity 0.85

For educational use only — not intended for clinical diagnosis.
"""

import argparse
import datetime as dt
import gc
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")  # No windows: Graphs are saved as files
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, GlobalMaxPooling2D,
                                     RandomFlip, RandomRotation, RandomZoom)
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from preprocessing import preprocess_pipeline
from preprocessing import preprocess_pipeline


# ======================================================================================
# General utilities
# ======================================================================================
def fmt_dur(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Tee:
    """Duplicates console output to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def make_event_logger(events, t0):
    """Returns a function that logs (and displays) timestamped events."""
    def event(msg):
        line = f"[{fmt_dur(time.time() - t0)}] {msg}"
        events.append(line)
        print(line, flush=True)
    return event


def count_params(weights):
    return int(sum(int(np.prod(tuple(w.shape))) for w in weights))


def get_lr(model):
    lr = model.optimizer.learning_rate
    for fn in (lambda: float(lr.numpy()),
               lambda: float(tf.keras.backend.get_value(lr)),
               lambda: float(lr)):
        try:
            return fn()
        except Exception:
            continue
    return float("nan")


def parse_thr_mode(s):
    """'f1' | 'sensitivity' | number (fixed threshold)."""
    try:
        return float(s)
    except ValueError:
        if s in ("f1", "sensitivity"):
            return s
        raise argparse.ArgumentTypeError("thr-mode must be 'f1', 'sensitivity', or a number (e.g., 0.5)")


# ======================================================================================
# Dataset: organization into folders and uploading
# ======================================================================================
def prepare_dataset(base_dir, dataset_dir, seed=42):
    """Copy the HAM10000 images to Datasets/training|validation/<dx>/ (using an 80/20 split by lesion)."""
    from sklearn.model_selection import GroupShuffleSplit

    base_dir, dataset_dir = Path(base_dir), Path(dataset_dir)
    metadata = pd.read_csv(base_dir / "HAM10000_metadata.csv")
    paths = {f.stem: f for f in base_dir.rglob("*.jpg") if dataset_dir not in f.parents}
    metadata["path"] = metadata["image_id"].map(paths)
    print(f"Found images: {metadata['path'].notna().sum()} of {len(metadata)}")
    metadata = metadata.dropna(subset=["path"]).reset_index(drop=True)
    assert len(metadata) > 0, "No images found: check --base-dir (CSV + images)."

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(metadata, groups=metadata["lesion_id"]))
    for split, idx in [("training", train_idx), ("validation", val_idx)]:
        for _, row in tqdm(metadata.iloc[idx].iterrows(), total=len(idx), desc=split):
            dest = dataset_dir / split / row["dx"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(row["path"], dest / row["path"].name)
    print("Dataset organized into", dataset_dir)


def count_images(folder, categories):
    return {c: len(os.listdir(os.path.join(folder, c))) for c in categories}


def load_image(path, img_size=224, to_rgb=False, pad=False, preprocessing_method="baseline"):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if preprocessing_method != "baseline":
        img_rgb = preprocess_pipeline(img_rgb, target_size=(img_size, img_size), method=preprocessing_method)
        return img_rgb if to_rgb else cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if pad:  # cuadrar con margen negro sin deformar la lesión
        h, w = img.shape[:2]
        s = max(h, w)
        top, left = (s - h) // 2, (s - w) // 2
        img = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_split(folder, categories, img_size, to_rgb, pad, limit=None, preprocessing_method="baseline"):
    X, Y, files, failed = [], [], [], 0
    for idx, c in enumerate(categories):
        cdir = os.path.join(folder, c)
        names = sorted(f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if limit:
            names = names[:limit]
        for f in tqdm(names, desc=f"{os.path.basename(folder)}/{c}", leave=False):
            img = load_image(os.path.join(cdir, f), img_size, to_rgb, pad, preprocessing_method=preprocessing_method)
            if img is None:
                failed += 1
                continue
            X.append(img)
            Y.append(idx)
            files.append(f"{c}/{f}")
    return np.asarray(X, dtype=np.uint8), np.asarray(Y, dtype=np.int64), files, failed


# ======================================================================================
# Model
# ======================================================================================
def make_augmentation():
    # No color changes: color is clinically relevant information.
    return Sequential([RandomFlip("horizontal_and_vertical"), RandomRotation(0.2), RandomZoom(0.1)],
                      name="data_augmentation")


def compile_binary(model, lr):
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def build_binary_model(variant, img_size, augment, pretrained, head_lr):
    Base = {"B0": EfficientNetB0, "B4": EfficientNetB4}[variant]
    base = Base(include_top=False, weights=pretrained, input_shape=(img_size, img_size, 3))
    base.trainable = False  # Phase 1: Learn only the head.

    inputs = Input(shape=(img_size, img_size, 3))
    x = make_augmentation()(inputs) if augment else inputs
    x = base(x, training=False)
    x = GlobalMaxPooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    compile_binary(model, head_lr)
    return model, base


def unfreeze_top_layers(model, base, n_layers, lr):
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False  # retains ImageNet statistics
    compile_binary(model, lr)  # You must recompile after changing `trainable`.


def get_class_weights(y, mode):
    assert mode in ("none", "sqrt", "balanced")
    if mode == "none":
        return None
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    if mode == "sqrt":
        cw = np.sqrt(cw)
    return {int(c): float(w) for c, w in zip(classes, cw)}


# ======================================================================================
# Training log (callback that populates the .txt summary)
# ======================================================================================
class PhaseLogger(tf.keras.callbacks.Callback):
    """Store metrics by epoch, including timing, learning rate (LR) used, best epochs, and LR reductions."""

    def __init__(self, phase, rows, event_fn, epochs_planned):
        super().__init__()
        self.phase, self.rows, self.event, self.planned = phase, rows, event_fn, epochs_planned
        self.best_val_loss, self.best_epoch, self.n = np.inf, None, 0
        self.lr_start, self._lr_epoch, self._t = None, None, None

    def on_train_begin(self, logs=None):
        self.lr_start = get_lr(self.model)
        self.event(f"[{self.phase}] home | initial lr={self.lr_start:.2e} | "
                   f"trainable parameters={count_params(self.model.trainable_weights):,} | "
                   f"max epochs.={self.planned}")

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
        self._lr_epoch = get_lr(self.model)  # lr used for training during THIS period

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items() if k not in ("lr", "learning_rate")}
        self.n += 1
        self.rows.append({"phase": self.phase, "epoch": len(self.rows) + 1, "epoch_phase": epoch + 1,
                          "lr": self._lr_epoch, "seconds": round(time.time() - self._t, 1), **logs})
        v = logs.get("val_loss")
        if v is not None and v < self.best_val_loss:
            self.best_val_loss, self.best_epoch = v, epoch + 1
            self.event(f"[{self.phase}] epoch {epoch + 1}: new best val_loss={v:.4f} "
                       f"(val_auc={logs.get('val_auc', float('nan')):.4f})")
        lr_now = get_lr(self.model)
        if lr_now < self._lr_epoch * 0.999:
            self.event(f"[{self.phase}] ReduceLROnPlateau after the epoch {epoch + 1}: "
                       f"lr {self._lr_epoch:.2e} -> {lr_now:.2e}")


def fit_phase(model, name, X, Y, Xv, Yv, epochs, batch_size, class_weight, rows, event):
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)
    logger = PhaseLogger(name, rows, event, epochs)
    t0 = time.time()
    model.fit(X, Y, validation_data=(Xv, Yv), epochs=epochs, batch_size=batch_size,
              class_weight=class_weight, callbacks=[es, rlr, logger], verbose=2)
    stopped_early = logger.n < epochs
    if stopped_early:
        event(f"[{name}] EarlyStopping: stopped after {logger.n} de {epochs} epochs (patience=5); "
              f"best epoch={logger.best_epoch} (val_loss={logger.best_val_loss:.4f}), "
              f"weights adjusted to that period")
    else:
        event(f"[{name}] completed {epochs} epochs without triggering EarlyStopping"
              f"(best epoch={logger.best_epoch}, val_loss={logger.best_val_loss:.4f})")
    return dict(fase=name, epocas_max=epochs, epocas_ejecutadas=logger.n, early_stopping=stopped_early,
                mejor_epoca=logger.best_epoch, mejor_val_loss=float(logger.best_val_loss),
                lr_inicial=logger.lr_start, lr_final=get_lr(model), segundos=round(time.time() - t0, 1))


# ======================================================================================
# Metrics and charts (stage 1)
# ======================================================================================
def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return dict(thr=float(thr), sensitivity=sens, specificity=spec, precision=prec, f1=f1,
                balanced_acc=(sens + spec) / 2, accuracy=(tp + tn) / len(y), tp=tp, fp=fp, fn=fn, tn=tn)


def choose_threshold(y, p, mode="f1", min_sensitivity=0.85):
    if isinstance(mode, (int, float)):
        return float(mode)  # fixed thershold
    ts = np.linspace(0.01, 0.99, 99)
    ms = [binary_metrics(y, p, t) for t in ts]
    if mode == "f1":
        return float(ts[int(np.argmax([m["f1"] for m in ms]))])
    if mode == "sensitivity":
        ok = [t for t, m in zip(ts, ms) if m["sensitivity"] >= min_sensitivity]
        return float(max(ok)) if ok else float(ts[0])  # the highest one that still meets the sensitivity requirement
    raise ValueError("thr_mode: 'f1', 'sensitivity' or a number")


def plot_history(df, path, title):
    panels = [(c, t) for c, t in [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")] if c in df.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    boundaries = df.index[df["phase"] != df["fase"].shift()].tolist()[1:]  # phase changes
    for ax, (col, label) in zip(axes, panels):
        ax.plot(df["epoch"], df[col], label=f"train_{col}")
        ax.plot(df["epoch"], df[f"val_{col}"], label=f"val_{col}")
        for b in boundaries:
            ax.axvline(df.loc[b, "epoch"] - 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_title(f"{label}"); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stage1(y, p, thr, path, title):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    m = binary_metrics(y, p, thr)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].scatter([1 - m["specificity"]], [m["sensitivity"]], color="red", zorder=3, label=f"threshold {thr:.2f}")
    axes[0].set(xlabel="1 − specificity", ylabel="sensitivity (recall of mel)",
                title=f"ROC (AUC = {roc_auc_score(y, p):.3f})")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label="chance (prevalence)")
    axes[1].scatter([m["sensitivity"]], [m["precition"]], color="red", zorder=3, label=f"threshold {thr:.2f}")
    axes[1].set(xlabel="recall (mel)", ylabel="precisión (mel)",
                title=f"Precition-Recall (AP = {average_precision_score(y, p):.3f})")
    axes[1].legend(loc="upper right")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
                xticklabels=["no-mel", "mel"], yticklabels=["no-mel", "mel"])
    axes[2].set(xlabel="Predicted", ylabel="Real", title=f"confusion matrix (threshold {thr:.2f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ======================================================================================
# Construction of the .txt summary
# ======================================================================================
def _df_txt(df, **kw):
    sci = {c: (lambda v: f"{v:.2e}") for c in ("lr", "lr_inicial", "lr_final") if c in df.columns}
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}", formatters=sci, **kw)


def build_report(ctx):
    L = []

    def section(title):
        L.extend(["", "=" * 86, title, "=" * 86])

    def kv(k, v):
        L.append(f"{k:<34}: {v}")

    a = ctx.get("args", {})
    L.append("SUMMARY OF BINARY TRAINING — melanoma (MEL) vs. non-melanoma")
    L.append("For educational use only — not intended for clinical diagnosis.")

    # ---------------------------------------------------------------- 1. Executive summary
    section("1. EXECUTIVE SUMMARY")
    kv("Run name", a.get("name"))
    kv("Status", ctx.get("status"))
    kv("Start time", ctx.get("start"))
    kv("End time", ctx.get("end"))
    kv("Total duration", fmt_dur(ctx.get("elapsed", 0)))
    phases = ctx.get("phases", [])
    if phases:
        kv("Epochs executed (total)", sum(p["epocas_ejecutadas"] for p in phases))
    ev = ctx.get("eval")
    if ev:
        kv("AUC (validation)", f"{ev['auc']:.4f}" if ev["auc"] is not None else "n/a")
        kv("Average Precision (validation)", f"{ev['ap']:.4f}" if ev["ap"] is not None else "n/a")
        kv(f"Chosen threshold ({ev['thr_mode']})", f"{ev['thr']:.2f}")
        m = ev["at_thr"]
        kv("Sensitivity (mel recall)", f"{m['sensitivity']:.4f}  ({m['tp']} of {m['tp'] + m['fn']} melanomas detected)")
        kv("Specificity", f"{m['specificity']:.4f}")
        kv("Precision (mel)", f"{m['precision']:.4f}")
        kv("F1 (mel)", f"{m['f1']:.4f}")
        kv("False negatives / false positives", f"{m['fn']} / {m['fp']}")
    if ctx.get("error"):
        L.append("")
        L.append("!! EXECUTION DID NOT FINISH CORRECTLY — see section 9 (error).")

    # ---------------------------------------------------------------- 2. Environment
    section("2. ENVIRONMENT")
    for k, v in ctx.get("env", {}).items():
        kv(k, v)
    kv("Command", ctx.get("cmd"))

    # ---------------------------------------------------------------- 3. Configuration
    section("3. CONFIGURATION")
    for k, v in a.items():
        kv(k, v)

    # ---------------------------------------------------------------- 4. Data
    d = ctx.get("dataset")
    section("4. DATA")
    if d:
        kv("Training folder", d["train_folder"])
        kv("Validation folder", d["val_folder"])
        kv("Classes (alphabetical order)", ", ".join(d["categories"]))
        L.append("")
        L.append("Images per class on disk (before loading):")
        tab = pd.DataFrame({"class": d["categories"],
                            "training": [d["counts_train"][c] for c in d["categories"]],
                            "validation": [d["counts_val"][c] for c in d["categories"]]})
        L.append(tab.to_string(index=False))
        L.append("")
        kv("Images loaded (train / val)", f"{d['n_train']} / {d['n_val']}")
        kv("Unreadable images skipped", f"train={d['failed_train']}, val={d['failed_val']}")
        kv("Melanomas train", f"{d['mel_train']}  ({100 * d['mel_train'] / max(d['n_train'], 1):.1f} %)")
        kv("Melanomas val", f"{d['mel_val']}  ({100 * d['mel_val'] / max(d['n_val'], 1):.1f} %)")
        kv("Non-melanoma train / val", f"{d['n_train'] - d['mel_train']} / {d['n_val'] - d['mel_val']}")
        kv("X_train shape / dtype", f"{d['shape_train']} / {d['dtype']}")
        kv("Arrays memory", f"{d['mem_mb']:.0f} MB")
        kv("Load time", fmt_dur(d["load_seconds"]))
        kv("Preprocessing", a.get("preprocessing"))
        kv("Class weights (mode)", a.get("class_weights"))
        kv("Class weights (values)", ctx.get("class_weights"))
    else:
        L.append("(dataset was not loaded)")

    # ---------------------------------------------------------------- 5. Model
    section("5. MODEL")
    mo = ctx.get("model")
    if mo:
        kv("Architecture", f"EfficientNet{a.get('variant')} + GlobalMaxPool + Dense(256) + BN + Dropout(0.5) + Dense(1, sigmoid)")
        kv("Backbone weights", a.get("pretrained"))
        kv("Total parameters", f"{mo['params_total']:,}")
        kv("Trainable in phase 1 (head)", f"{mo['trainable_head']:,}")
        if mo.get("trainable_ft") is not None:
            kv("Trainable in phase 2 (fine-tuning)", f"{mo['trainable_ft']:,}")
        L.append("")
        L.append("model.summary():")
        L.extend(mo["summary"])
    else:
        L.append("(model was not built)")

    # ---------------------------------------------------------------- 6. Training
    section("6. TRAINING")
    if phases:
        L.append("Phases:")
        L.append(_df_txt(pd.DataFrame(phases)))
        # derived warnings
        L.append("")
    hist = ctx.get("history")
    if hist is not None and len(hist):
        L.append("Metrics per epoch (lr = learning rate used in that epoch):")
        L.append(_df_txt(hist))
    L.append("")
    L.append("Event timeline:")
    L.extend(ctx.get("events", []) or ["(no events)"])

    # ---------------------------------------------------------------- 7. Evaluation
    section("7. VALIDATION EVALUATION (final model)")
    if ev:
        kv("Threshold", f"{ev['thr']:.2f}  (mode: {ev['thr_mode']}"
            + (f", minimum sensitivity {a.get('min_sensitivity')}" if ev['thr_mode'] == 'sensitivity' else "") + ")")
        kv("val_loss / val_accuracy / val_auc (keras)", ", ".join(f"{k}={v:.4f}" for k, v in ev["keras_eval"].items()))
        L.append("")
        m = ev["at_thr"]
        L.append("Confusion matrix (rows = actual, columns = predicted):")
        L.append(f"{'':>12}{'pred non-mel':>15}{'pred mel':>12}")
        L.append(f"{'actual non-mel':>15}{m['tn']:>12}{m['fp']:>12}")
        L.append(f"{'actual mel':>15}{m['fn']:>12}{m['tp']:>12}")
        L.append("")
        L.append("Metrics at chosen threshold:")
        L.append(_df_txt(pd.DataFrame([m])))
        L.append("")
        L.append("Metrics with threshold 0.5 (reference):")
        L.append(_df_txt(pd.DataFrame([ev["at_05"]])))
        L.append("")
        L.append("Classification report (chosen threshold):")
        L.append(ev["report"])
        L.append("P(mel) distribution in validation:")
        L.append(_df_txt(pd.DataFrame(ev["prob_stats"])))
        L.append("")
        L.append("Threshold sweep (sensitivity / false alarms tradeoff):")
        L.append(_df_txt(pd.DataFrame(ev["sweep"])))
    else:
        L.append("(evaluation was not reached)")

    # ---------------------------------------------------------------- 8. Files
    section("8. GENERATED FILES")
    L.extend(ctx.get("files", []) or ["(none)"])

    # ---------------------------------------------------------------- 9. Notes / warnings / error
    section("9. AUTOMATIC NOTES AND WARNINGS")
    for w in ctx.get("warnings", []):
        L.append(f"- {w}")
    L.append("- The threshold and early stopping are decided using the SAME validation set used for evaluation: "
              "the metrics are somewhat optimistic.")
    if ctx.get("error"):
        L.extend(["", "ERROR / INTERRUPTION:", ctx["error"]])

    return "\n".join(L) + "\n"


def derive_warnings(ctx):
    w = []
    phases = ctx.get("phases", [])
    d = ctx.get("dataset")
    if d and d["mel_val"] < 50:
        w.append(f"There are only {d['mel_val']} melanomas in validation: metrics are highly unstable.")
    if d and (d["failed_train"] or d["failed_val"]):
        w.append(f"Unreadable images were skipped (train={d['failed_train']}, val={d['failed_val']}).")
    if len(phases) == 2 and phases[1]["mejor_val_loss"] >= phases[0]["mejor_val_loss"]:
        w.append("Fine-tuning did NOT improve the best val_loss from phase 1 "
                 f"({phases[1]['mejor_val_loss']:.4f} >= {phases[0]['mejor_val_loss']:.4f}); "
                 "the final model comes from phase 2.")
    hist = ctx.get("history")
    if hist is not None and len(hist) and {"auc", "val_auc"} <= set(hist.columns):
        last = hist.iloc[-1]
        if last["auc"] - last["val_auc"] > 0.10:
            w.append(f"Possible overfitting: train AUC {last['auc']:.3f} vs val {last['val_auc']:.3f} in the last epoch.")
    ev = ctx.get("eval")
    if ev and ev["at_thr"]["sensitivity"] < 0.5:
        w.append(f"Low sensitivity ({ev['at_thr']['sensitivity']:.2f}): the model misses many melanomas.")
    if ev and ev["auc"] is not None and ev["auc"] < 0.7:
        w.append(f"Low AUC ({ev['auc']:.3f}): the model has poor discrimination.")
    if not w:
        w.append("No relevant automatic warnings.")
    return w


# ======================================================================================
# Main execution
# ======================================================================================
def run_training(args, run_dir, ctx, event):
    files = ctx["files"]
    tf.keras.utils.set_random_seed(args.seed)

    # ------------------------------------------------ dataset
    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "training").exists():
        if args.base_dir:
            event(f"Organizing dataset in {dataset_dir} ...")
            prepare_dataset(args.base_dir, dataset_dir, seed=args.seed)
        else:
            raise FileNotFoundError(f"{dataset_dir}/training does not exist. Use --base-dir to organize it "
                                    f"or --dataset-dir with the training/validation/<class> structure.")
    train_folder, val_folder = str(dataset_dir / "training"), str(dataset_dir / "validation")
    categories = sorted(os.listdir(train_folder))
    if "mel" not in categories:
        raise ValueError(f"No 'mel' folder found in {train_folder}. Classes found: {categories}")
    mel_idx = categories.index("mel")
    counts_train, counts_val = count_images(train_folder, categories), count_images(val_folder, categories)
    event(f"Classes: {categories} | train={sum(counts_train.values())} val={sum(counts_val.values())} images on disk")

    t_load = time.time()
    event(f"Loading images ({args.img_size}x{args.img_size}, rgb={args.rgb}, pad={args.pad}, "
          f"preprocessing={args.preprocessing}, limit_per_class={args.limit_per_class}) ...")
    X_tr, Y_tr, _, failed_tr = load_split(train_folder, categories, args.img_size, args.rgb, args.pad, 
                                          args.limit_per_class, preprocessing_method=args.preprocessing)
    X_va, Y_va, files_va, failed_va = load_split(val_folder, categories, args.img_size, args.rgb, args.pad, 
                                                  args.limit_per_class, preprocessing_method=args.preprocessing)
    Y1_tr, Y1_va = (Y_tr == mel_idx).astype(np.int64), (Y_va == mel_idx).astype(np.int64)  # 1 = melanoma
    ctx["dataset"] = dict(train_folder=train_folder, val_folder=val_folder, categories=categories,
                          counts_train=counts_train, counts_val=counts_val,
                          n_train=len(Y1_tr), n_val=len(Y1_va),
                          mel_train=int(Y1_tr.sum()), mel_val=int(Y1_va.sum()),
                          failed_train=failed_tr, failed_val=failed_va,
                          shape_train=tuple(X_tr.shape), dtype=str(X_tr.dtype),
                          mem_mb=(X_tr.nbytes + X_va.nbytes) / 1e6, load_seconds=time.time() - t_load)
    event(f"Data loaded: train {X_tr.shape} ({int(Y1_tr.sum())} mel) | val {X_va.shape} ({int(Y1_va.sum())} mel) "
          f"| unreadable train={failed_tr} val={failed_va}")

    cw = get_class_weights(Y1_tr, args.class_weights)
    ctx["class_weights"] = {("mel" if k == 1 else "non-mel"): round(v, 3) for k, v in cw.items()} if cw else "none"
    if cw:
        event(f"Class weights ({args.class_weights}): {ctx['class_weights']}")

    # ------------------------------------------------ model
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(args.seed)
    pretrained = None if args.pretrained == "none" else args.pretrained
    event(f"Building EfficientNet{args.variant} (weights={args.pretrained}) ...")
    model, base = build_binary_model(args.variant, args.img_size, args.augment, pretrained, args.head_lr)
    summary_lines = []
    model.summary(print_fn=lambda s, **kw: summary_lines.append(s))
    ctx["model"] = dict(params_total=model.count_params(), trainable_head=count_params(model.trainable_weights),
                        trainable_ft=None, summary=summary_lines)

    # ------------------------------------------------ training
    rows, phases = [], ctx["phases"]
    phases.append(fit_phase(model, "phase1_head", X_tr, Y1_tr, X_va, Y1_va, args.epochs_head,
                            args.batch_size, cw, rows, event))
    if args.fine_tune:
        unfreeze_top_layers(model, base, args.ft_layers, args.ft_lr)
        ctx["model"]["trainable_ft"] = count_params(model.trainable_weights)
        event(f"Fine-tuning: unfreezing the last {args.ft_layers} layers of the backbone (BatchNorm frozen), "
              f"lr={args.ft_lr:.0e}")
        phases.append(fit_phase(model, "phase2_finetuning", X_tr, Y1_tr, X_va, Y1_va, args.epochs_ft,
                                args.batch_size, cw, rows, event))
    hist = pd.DataFrame(rows)
    pref = ["phase", "epoch", "epoch_phase", "loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc", "lr", "seconds"]
    hist = hist[[c for c in pref if c in hist.columns] + [c for c in hist.columns if c not in pref]]
    ctx["history"] = hist
    hist.to_csv(run_dir / "epoch_history.csv", index=False)
    files.append("epoch_history.csv — metrics per epoch")
    plot_history(hist, run_dir / "training_curves.png", f"{args.name} — Stage 1 (mel vs non-mel)")
    files.append("training_curves.png — accuracy / loss / AUC curves (gray line = phase change)")

    # ------------------------------------------------ evaluation
    event("Evaluating on validation ...")
    p_mel = model.predict(X_va, batch_size=64, verbose=0).ravel()
    keras_eval = dict(zip(["val_loss", "val_accuracy", "val_auc"],
                          model.evaluate(X_va, Y1_va, batch_size=64, verbose=0)))
    thr = choose_threshold(Y1_va, p_mel, args.thr_mode, args.min_sensitivity)
    m_thr, m_05 = binary_metrics(Y1_va, p_mel, thr), binary_metrics(Y1_va, p_mel, 0.5)
    try:
        auc, ap = float(roc_auc_score(Y1_va, p_mel)), float(average_precision_score(Y1_va, p_mel))
        plot_stage1(Y1_va, p_mel, thr, run_dir / "stage1_evaluation.png", f"{args.name} — Stage 1")
        files.append("stage1_evaluation.png — ROC curve, Precision-Recall curve, and confusion matrix")
    except ValueError as e:  # e.g., validation set without melanomas (with a very low --limit-per-class)
        auc = ap = None
        event(f"WARNING: could not calculate AUC/AP ({e})")
    pred = (p_mel >= thr).astype(int)
    report = classification_report(Y1_va, pred, labels=[0, 1], target_names=["non-mel", "mel"], zero_division=0)
    prob_stats = [dict(group=g, n=int((Y1_va == v).sum()), mean=float(p_mel[Y1_va == v].mean()) if (Y1_va == v).any() else float("nan"),
                       median=float(np.median(p_mel[Y1_va == v])) if (Y1_va == v).any() else float("nan"),
                       p10=float(np.percentile(p_mel[Y1_va == v], 10)) if (Y1_va == v).any() else float("nan"),
                       p90=float(np.percentile(p_mel[Y1_va == v], 90)) if (Y1_va == v).any() else float("nan"))
                  for g, v in [("non-mel (actual)", 0), ("mel (actual)", 1)]]
    sweep = [binary_metrics(Y1_va, p_mel, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)]
    ctx["eval"] = dict(auc=auc, ap=ap, thr=thr, thr_mode=str(args.thr_mode), at_thr=m_thr, at_05=m_05,
                       keras_eval={k: float(v) for k, v in keras_eval.items()},
                       report=report, prob_stats=prob_stats, sweep=sweep)
    event(f"Threshold={thr:.2f} ({args.thr_mode}) | AUC={auc if auc is None else round(auc, 3)} | "
          f"sensitivity={m_thr['sensitivity']:.3f} | specificity={m_thr['specificity']:.3f} | "
          f"precision={m_thr['precision']:.3f}")

    # ------------------------------------------------ save artifacts
    pd.DataFrame({"file": files_va, "original_class": [categories[i] for i in Y_va],
                  "y_actual_mel": Y1_va, "p_mel": p_mel, "pred_mel": pred}).to_csv(
        run_dir / "validation_predictions.csv", index=False)
    files.append("validation_predictions.csv — P(mel) and prediction per validation image")

    model_path = run_dir / f"{args.name}_stage1.keras"
    model.save(model_path)
    files.append(f"{model_path.name} — final trained model")
    meta = dict(thr=thr, img_size=args.img_size, to_rgb=args.rgb, pad=args.pad, 
                variant=args.variant, preprocessing=args.preprocessing)
    (run_dir / f"{args.name}_stage1.json").write_text(json.dumps(meta, indent=2))
    files.append(f"{args.name}_stage1.json — metadata (threshold and preprocessing)")

    (run_dir / "metrics.json").write_text(json.dumps(
        dict(config={k: str(v) for k, v in vars(args).items()}, phases=phases,
             evaluation={k: v for k, v in ctx["eval"].items() if k not in ("report",)}),
        indent=2, default=float))
    files.append("metrics.json — final metrics, phases, and configuration")


def parse_args():
    p = argparse.ArgumentParser(description="Binary training mel vs non-mel (HAM10000) with .txt summary",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--base-dir", default=None, help="Folder with HAM10000_metadata.csv + images (only if organizing the dataset)")
    g.add_argument("--dataset-dir", default=None, help="Folder with training/ and validation/ (defaults to <base-dir>/Datasets or ./Datasets)")
    g.add_argument("--limit-per-class", type=int, default=None, help="Max. images per class (quick runs)")
    g = p.add_argument_group("preprocessing")
    g.add_argument("--img-size", type=int, default=224)
    g.add_argument("--rgb", action=argparse.BooleanOptionalAction, default=True, help="Convert BGR -> RGB")
    g.add_argument("--pad", action=argparse.BooleanOptionalAction, default=False, help="Pad with black border before resizing")
    g.add_argument("--preprocessing", choices=["baseline", "hair_removal", "color_constancy", "contrast_focused"],
                   default="baseline", help="Preprocessing technique to apply from preprocessing.py")
    g = p.add_argument_group("model / training")
    g.add_argument("--variant", choices=["B0", "B4"], default="B4")
    g.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    g.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--class-weights", choices=["none", "sqrt", "balanced"], default="sqrt")
    g.add_argument("--epochs-head", type=int, default=15)
    g.add_argument("--fine-tune", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--epochs-ft", type=int, default=15)
    g.add_argument("--ft-layers", type=int, default=30)
    g.add_argument("--head-lr", type=float, default=5e-4)
    g.add_argument("--ft-lr", type=float, default=1e-5)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("threshold")
    g.add_argument("--thr-mode", type=parse_thr_mode, default="f1", help="'f1', 'sensitivity', or a fixed number (e.g. 0.5)")
    g.add_argument("--min-sensitivity", type=float, default=0.85, help="Used if thr-mode='sensitivity'")
    g = p.add_argument_group("output")
    g.add_argument("--name", default="B1_binary_B4_224", help="Run name")
    g.add_argument("--output-dir", default="runs", help="Folder where runs/<name>_<date>/ is created")
    g.add_argument("--quick", action="store_true", help="Quick test: 30 img/class, B0, 1 epoch, no fine-tuning")
    args = p.parse_args()

    if args.quick:
        args.limit_per_class, args.variant, args.epochs_head, args.fine_tune = 30, "B0", 1, False
        if args.name == p.get_default("name"):
            args.name = "QUICK_TEST_binary"
    if args.dataset_dir is None:
        args.dataset_dir = str(Path(args.base_dir) / "Datasets") if args.base_dir else "Datasets"
    return args


def main():
    args = parse_args()
    started = dt.datetime.now()
    run_dir = Path(args.output_dir) / f"{args.name}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log = open(run_dir / "console.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, console_log)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        build = tf.sysconfig.get_build_info()
        cuda = f"CUDA {build.get('cuda_version', 'n/a')} / cuDNN {build.get('cudnn_version', 'n/a')}"
    except Exception:
        cuda = "n/a"
    import sklearn
    env = {"Computer": platform.node(), "System": platform.platform(), "Python": platform.python_version(),
           "TensorFlow": tf.__version__, "Keras": getattr(tf.keras, "__version__", "n/a"), "NumPy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "OpenCV": cv2.__version__,
           "GPU(s)": ", ".join(g.name for g in gpus) if gpus else "none (CPU)", "TF Build": cuda}

    t0 = time.time()
    events = []
    event = make_event_logger(events, t0)
    ctx = dict(args=vars(args).copy(), cmd=" ".join(sys.argv), env=env, start=f"{started:%Y-%m-%d %H:%M:%S}",
               events=events, files=[], phases=[], status="IN PROGRESS", run_dir=str(run_dir))
    ctx["args"]["dataset_dir"] = args.dataset_dir
    ctx["args"]["thr_mode"] = str(args.thr_mode)

    exit_code = 0
    try:
        event(f"Run '{args.name}' — results in {run_dir}")
        run_training(args, run_dir, ctx, event)
        ctx["status"] = "COMPLETED"
    except BaseException as e:  # includes Ctrl+C: partial summary is still written
        ctx["status"] = "INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "ERROR"
        ctx["error"] = traceback.format_exc()
        event(f"{ctx['status']}: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        ctx["elapsed"] = time.time() - t0
        ctx["end"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
        ctx["warnings"] = derive_warnings(ctx)
        ctx["files"] += ["training_summary.txt — this file", "console.log — complete console output"]
        report_path = run_dir / "training_summary.txt"
        report_path.write_text(build_report(ctx), encoding="utf-8")
        print(f"\nSummary written to: {report_path}")
        sys.stdout = sys.__stdout__
        console_log.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()