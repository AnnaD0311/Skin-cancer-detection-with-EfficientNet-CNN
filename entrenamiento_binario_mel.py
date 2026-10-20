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
matplotlib.use("Agg")  # sin ventanas: los gráficos se guardan en archivos
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
# Utilidades generales
# ======================================================================================
def fmt_dur(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Tee:
    """Duplica lo que se imprime en consola hacia un archivo de log."""

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
    """Devuelve una función que registra (y muestra) eventos con marca de tiempo."""
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
    """'f1' | 'sensitivity' | número (umbral fijo)."""
    try:
        return float(s)
    except ValueError:
        if s in ("f1", "sensitivity"):
            return s
        raise argparse.ArgumentTypeError("thr-mode debe ser 'f1', 'sensitivity' o un número (p. ej. 0.5)")


# ======================================================================================
# Dataset: organización en carpetas y carga
# ======================================================================================
def prepare_dataset(base_dir, dataset_dir, seed=42):
    """Copia las imágenes del HAM10000 a Datasets/training|validation/<dx>/ (split 80/20 por lesión)."""
    from sklearn.model_selection import GroupShuffleSplit

    base_dir, dataset_dir = Path(base_dir), Path(dataset_dir)
    metadata = pd.read_csv(base_dir / "HAM10000_metadata.csv")
    paths = {f.stem: f for f in base_dir.rglob("*.jpg") if dataset_dir not in f.parents}
    metadata["path"] = metadata["image_id"].map(paths)
    print(f"Imágenes encontradas: {metadata['path'].notna().sum()} de {len(metadata)}")
    metadata = metadata.dropna(subset=["path"]).reset_index(drop=True)
    assert len(metadata) > 0, "Ninguna imagen encontrada: revisa --base-dir (CSV + imágenes)."

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(metadata, groups=metadata["lesion_id"]))
    for split, idx in [("training", train_idx), ("validation", val_idx)]:
        for _, row in tqdm(metadata.iloc[idx].iterrows(), total=len(idx), desc=split):
            dest = dataset_dir / split / row["dx"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(row["path"], dest / row["path"].name)
    print("Dataset organizado en", dataset_dir)


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
# Modelo
# ======================================================================================
def make_augmentation():
    # Sin cambios de color: el color es información clínica relevante.
    return Sequential([RandomFlip("horizontal_and_vertical"), RandomRotation(0.2), RandomZoom(0.1)],
                      name="data_augmentation")


def compile_binary(model, lr):
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def build_binary_model(variant, img_size, augment, pretrained, head_lr):
    Base = {"B0": EfficientNetB0, "B4": EfficientNetB4}[variant]
    base = Base(include_top=False, weights=pretrained, input_shape=(img_size, img_size, 3))
    base.trainable = False  # fase 1: solo aprende la cabeza

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
            layer.trainable = False  # conserva las estadísticas de ImageNet
    compile_binary(model, lr)  # hay que recompilar tras cambiar trainable


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
# Registro del entrenamiento (callback que alimenta el resumen .txt)
# ======================================================================================
class PhaseLogger(tf.keras.callbacks.Callback):
    """Guarda métricas por época, tiempos, lr usado, mejores épocas y reducciones de lr."""

    def __init__(self, phase, rows, event_fn, epochs_planned):
        super().__init__()
        self.phase, self.rows, self.event, self.planned = phase, rows, event_fn, epochs_planned
        self.best_val_loss, self.best_epoch, self.n = np.inf, None, 0
        self.lr_start, self._lr_epoch, self._t = None, None, None

    def on_train_begin(self, logs=None):
        self.lr_start = get_lr(self.model)
        self.event(f"[{self.phase}] inicio | lr inicial={self.lr_start:.2e} | "
                   f"parámetros entrenables={count_params(self.model.trainable_weights):,} | "
                   f"épocas máx.={self.planned}")

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
        self._lr_epoch = get_lr(self.model)  # lr con el que se entrena ESTA época

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items() if k not in ("lr", "learning_rate")}
        self.n += 1
        self.rows.append({"fase": self.phase, "epoca": len(self.rows) + 1, "epoca_fase": epoch + 1,
                          "lr": self._lr_epoch, "segundos": round(time.time() - self._t, 1), **logs})
        v = logs.get("val_loss")
        if v is not None and v < self.best_val_loss:
            self.best_val_loss, self.best_epoch = v, epoch + 1
            self.event(f"[{self.phase}] época {epoch + 1}: nuevo mejor val_loss={v:.4f} "
                       f"(val_auc={logs.get('val_auc', float('nan')):.4f})")
        lr_now = get_lr(self.model)
        if lr_now < self._lr_epoch * 0.999:
            self.event(f"[{self.phase}] ReduceLROnPlateau tras la época {epoch + 1}: "
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
        event(f"[{name}] EarlyStopping: detenido tras {logger.n} de {epochs} épocas (paciencia=5); "
              f"mejor época={logger.best_epoch} (val_loss={logger.best_val_loss:.4f}), "
              f"pesos restaurados a esa época")
    else:
        event(f"[{name}] completadas las {epochs} épocas sin activar EarlyStopping "
              f"(mejor época={logger.best_epoch}, val_loss={logger.best_val_loss:.4f})")
    return dict(fase=name, epocas_max=epochs, epocas_ejecutadas=logger.n, early_stopping=stopped_early,
                mejor_epoca=logger.best_epoch, mejor_val_loss=float(logger.best_val_loss),
                lr_inicial=logger.lr_start, lr_final=get_lr(model), segundos=round(time.time() - t0, 1))


# ======================================================================================
# Métricas y gráficos (etapa 1)
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
        return float(mode)  # umbral fijo
    ts = np.linspace(0.01, 0.99, 99)
    ms = [binary_metrics(y, p, t) for t in ts]
    if mode == "f1":
        return float(ts[int(np.argmax([m["f1"] for m in ms]))])
    if mode == "sensitivity":
        ok = [t for t, m in zip(ts, ms) if m["sensitivity"] >= min_sensitivity]
        return float(max(ok)) if ok else float(ts[0])  # el más alto que aún cumple la sensibilidad
    raise ValueError("thr_mode: 'f1', 'sensitivity' o un número")


def plot_history(df, path, title):
    panels = [(c, t) for c, t in [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")] if c in df.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    boundaries = df.index[df["fase"] != df["fase"].shift()].tolist()[1:]  # cambios de fase
    for ax, (col, label) in zip(axes, panels):
        ax.plot(df["epoca"], df[col], label=f"train_{col}")
        ax.plot(df["epoca"], df[f"val_{col}"], label=f"val_{col}")
        for b in boundaries:
            ax.axvline(df.loc[b, "epoca"] - 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("Época"); ax.set_ylabel(label); ax.set_title(f"{label}"); ax.legend()
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
    axes[0].scatter([1 - m["specificity"]], [m["sensitivity"]], color="red", zorder=3, label=f"umbral {thr:.2f}")
    axes[0].set(xlabel="1 − especificidad", ylabel="sensibilidad (recall de mel)",
                title=f"ROC (AUC = {roc_auc_score(y, p):.3f})")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label="azar (prevalencia)")
    axes[1].scatter([m["sensitivity"]], [m["precision"]], color="red", zorder=3, label=f"umbral {thr:.2f}")
    axes[1].set(xlabel="recall (mel)", ylabel="precisión (mel)",
                title=f"Precision-Recall (AP = {average_precision_score(y, p):.3f})")
    axes[1].legend(loc="upper right")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
                xticklabels=["no-mel", "mel"], yticklabels=["no-mel", "mel"])
    axes[2].set(xlabel="Predicho", ylabel="Real", title=f"Matriz de confusión (umbral {thr:.2f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ======================================================================================
# Construcción del resumen .txt
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
    L.append("RESUMEN DEL ENTRENAMIENTO BINARIO — melanoma (mel) vs no-melanoma")
    L.append("Uso exclusivamente educativo — no sirve para diagnóstico clínico.")

    # ---------------------------------------------------------------- 1. Resumen ejecutivo
    section("1. RESUMEN EJECUTIVO")
    kv("Nombre de la ejecución", a.get("name"))
    kv("Estado", ctx.get("status"))
    kv("Inicio", ctx.get("start"))
    kv("Fin", ctx.get("end"))
    kv("Duración total", fmt_dur(ctx.get("elapsed", 0)))
    phases = ctx.get("phases", [])
    if phases:
        kv("Épocas ejecutadas (total)", sum(p["epocas_ejecutadas"] for p in phases))
    ev = ctx.get("eval")
    if ev:
        kv("AUC (validación)", f"{ev['auc']:.4f}" if ev["auc"] is not None else "n/d")
        kv("Average Precision (validación)", f"{ev['ap']:.4f}" if ev["ap"] is not None else "n/d")
        kv(f"Umbral elegido ({ev['thr_mode']})", f"{ev['thr']:.2f}")
        m = ev["at_thr"]
        kv("Sensibilidad (recall de mel)", f"{m['sensitivity']:.4f}  ({m['tp']} de {m['tp'] + m['fn']} melanomas detectados)")
        kv("Especificidad", f"{m['specificity']:.4f}")
        kv("Precisión (mel)", f"{m['precision']:.4f}")
        kv("F1 (mel)", f"{m['f1']:.4f}")
        kv("Falsos negativos / falsos positivos", f"{m['fn']} / {m['fp']}")
    if ctx.get("error"):
        L.append("")
        L.append("!! LA EJECUCIÓN NO TERMINÓ CORRECTAMENTE — ver sección 9 (error).")

    # ---------------------------------------------------------------- 2. Entorno
    section("2. ENTORNO")
    for k, v in ctx.get("env", {}).items():
        kv(k, v)
    kv("Comando", ctx.get("cmd"))

    # ---------------------------------------------------------------- 3. Configuración
    section("3. CONFIGURACIÓN")
    for k, v in a.items():
        kv(k, v)

    # ---------------------------------------------------------------- 4. Datos
    d = ctx.get("dataset")
    section("4. DATOS")
    if d:
        kv("Carpeta de entrenamiento", d["train_folder"])
        kv("Carpeta de validación", d["val_folder"])
        kv("Clases (orden alfabético)", ", ".join(d["categories"]))
        L.append("")
        L.append("Imágenes por clase en disco (antes de cargar):")
        tab = pd.DataFrame({"clase": d["categories"],
                            "entrenamiento": [d["counts_train"][c] for c in d["categories"]],
                            "validacion": [d["counts_val"][c] for c in d["categories"]]})
        L.append(tab.to_string(index=False))
        L.append("")
        kv("Imágenes cargadas (train / val)", f"{d['n_train']} / {d['n_val']}")
        kv("Imágenes ilegibles omitidas", f"train={d['failed_train']}, val={d['failed_val']}")
        kv("Melanomas  train", f"{d['mel_train']}  ({100 * d['mel_train'] / max(d['n_train'], 1):.1f} %)")
        kv("Melanomas  val", f"{d['mel_val']}  ({100 * d['mel_val'] / max(d['n_val'], 1):.1f} %)")
        kv("No-melanoma train / val", f"{d['n_train'] - d['mel_train']} / {d['n_val'] - d['mel_val']}")
        kv("Forma X_train / dtype", f"{d['shape_train']} / {d['dtype']}")
        kv("Memoria de los arrays", f"{d['mem_mb']:.0f} MB")
        kv("Tiempo de carga", fmt_dur(d["load_seconds"]))
        kv("Pre-procesamiento", a.get("preprocessing"))
        kv("Pre-procesamiento", a.get("preprocessing"))
        kv("Pesos de clase (modo)", a.get("class_weights"))
        kv("Pesos de clase (valores)", ctx.get("class_weights"))
    else:
        L.append("(no se llegó a cargar el dataset)")

    # ---------------------------------------------------------------- 5. Modelo
    section("5. MODELO")
    mo = ctx.get("model")
    if mo:
        kv("Arquitectura", f"EfficientNet{a.get('variant')} + GlobalMaxPool + Dense(256) + BN + Dropout(0.5) + Dense(1, sigmoid)")
        kv("Pesos del backbone", a.get("pretrained"))
        kv("Parámetros totales", f"{mo['params_total']:,}")
        kv("Entrenables en fase 1 (cabeza)", f"{mo['trainable_head']:,}")
        if mo.get("trainable_ft") is not None:
            kv("Entrenables en fase 2 (fine-tuning)", f"{mo['trainable_ft']:,}")
        L.append("")
        L.append("model.summary():")
        L.extend(mo["summary"])
    else:
        L.append("(no se llegó a construir el modelo)")

    # ---------------------------------------------------------------- 6. Entrenamiento
    section("6. ENTRENAMIENTO")
    if phases:
        L.append("Fases:")
        L.append(_df_txt(pd.DataFrame(phases)))
        # advertencias derivadas
        L.append("")
    hist = ctx.get("history")
    if hist is not None and len(hist):
        L.append("Métricas por época (lr = tasa de aprendizaje usada en esa época):")
        L.append(_df_txt(hist))
    L.append("")
    L.append("Cronología de eventos:")
    L.extend(ctx.get("events", []) or ["(sin eventos)"])

    # ---------------------------------------------------------------- 7. Evaluación
    section("7. EVALUACIÓN EN VALIDACIÓN (modelo final)")
    if ev:
        kv("Umbral", f"{ev['thr']:.2f}  (modo: {ev['thr_mode']}"
            + (f", sensibilidad mínima {a.get('min_sensitivity')}" if ev['thr_mode'] == 'sensitivity' else "") + ")")
        kv("val_loss / val_accuracy / val_auc (keras)", ", ".join(f"{k}={v:.4f}" for k, v in ev["keras_eval"].items()))
        L.append("")
        m = ev["at_thr"]
        L.append("Matriz de confusión (filas = real, columnas = predicho):")
        L.append(f"{'':>12}{'pred no-mel':>14}{'pred mel':>12}")
        L.append(f"{'real no-mel':>12}{m['tn']:>14}{m['fp']:>12}")
        L.append(f"{'real mel':>12}{m['fn']:>14}{m['tp']:>12}")
        L.append("")
        L.append("Métricas en el umbral elegido:")
        L.append(_df_txt(pd.DataFrame([m])))
        L.append("")
        L.append("Métricas con umbral 0.5 (referencia):")
        L.append(_df_txt(pd.DataFrame([ev["at_05"]])))
        L.append("")
        L.append("Informe de clasificación (umbral elegido):")
        L.append(ev["report"])
        L.append("Distribución de P(mel) en validación:")
        L.append(_df_txt(pd.DataFrame(ev["prob_stats"])))
        L.append("")
        L.append("Barrido de umbrales (compromiso sensibilidad / falsos alarmes):")
        L.append(_df_txt(pd.DataFrame(ev["sweep"])))
    else:
        L.append("(no se llegó a evaluar)")

    # ---------------------------------------------------------------- 8. Archivos
    section("8. ARCHIVOS GENERADOS")
    L.extend(ctx.get("files", []) or ["(ninguno)"])

    # ---------------------------------------------------------------- 9. Notas / avisos / error
    section("9. NOTAS Y AVISOS AUTOMÁTICOS")
    for w in ctx.get("warnings", []):
        L.append(f"- {w}")
    L.append("- El umbral y el early stopping se deciden con la MISMA validación en la que se evalúa: "
              "los números son algo optimistas.")
    if ctx.get("error"):
        L.extend(["", "ERROR / INTERRUPCIÓN:", ctx["error"]])

    return "\n".join(L) + "\n"


def derive_warnings(ctx):
    w = []
    phases = ctx.get("phases", [])
    d = ctx.get("dataset")
    if d and d["mel_val"] < 50:
        w.append(f"Solo hay {d['mel_val']} melanomas en validación: métricas muy inestables.")
    if d and (d["failed_train"] or d["failed_val"]):
        w.append(f"Se omitieron imágenes ilegibles (train={d['failed_train']}, val={d['failed_val']}).")
    if len(phases) == 2 and phases[1]["mejor_val_loss"] >= phases[0]["mejor_val_loss"]:
        w.append("El fine-tuning NO mejoró el mejor val_loss de la fase 1 "
                 f"({phases[1]['mejor_val_loss']:.4f} >= {phases[0]['mejor_val_loss']:.4f}); "
                 "el modelo final proviene de la fase 2.")
    hist = ctx.get("history")
    if hist is not None and len(hist) and {"auc", "val_auc"} <= set(hist.columns):
        last = hist.iloc[-1]
        if last["auc"] - last["val_auc"] > 0.10:
            w.append(f"Posible sobreajuste: AUC train {last['auc']:.3f} vs val {last['val_auc']:.3f} en la última época.")
    ev = ctx.get("eval")
    if ev and ev["at_thr"]["sensitivity"] < 0.5:
        w.append(f"Sensibilidad baja ({ev['at_thr']['sensitivity']:.2f}): el modelo deja pasar muchos melanomas.")
    if ev and ev["auc"] is not None and ev["auc"] < 0.7:
        w.append(f"AUC bajo ({ev['auc']:.3f}): el modelo discrimina poco.")
    if not w:
        w.append("Sin avisos automáticos relevantes.")
    return w


# ======================================================================================
# Ejecución principal
# ======================================================================================
def run_training(args, run_dir, ctx, event):
    files = ctx["files"]
    tf.keras.utils.set_random_seed(args.seed)

    # ------------------------------------------------ dataset
    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "training").exists():
        if args.base_dir:
            event(f"Organizando dataset en {dataset_dir} ...")
            prepare_dataset(args.base_dir, dataset_dir, seed=args.seed)
        else:
            raise FileNotFoundError(f"No existe {dataset_dir}/training. Usa --base-dir para organizarlo "
                                    f"o --dataset-dir con la estructura training/validation/<clase>.")
    train_folder, val_folder = str(dataset_dir / "training"), str(dataset_dir / "validation")
    categories = sorted(os.listdir(train_folder))
    if "mel" not in categories:
        raise ValueError(f"No hay carpeta 'mel' en {train_folder}. Clases encontradas: {categories}")
    mel_idx = categories.index("mel")
    counts_train, counts_val = count_images(train_folder, categories), count_images(val_folder, categories)
    event(f"Clases: {categories} | train={sum(counts_train.values())} val={sum(counts_val.values())} imágenes en disco")

    t_load = time.time()
    event(f"Cargando imágenes ({args.img_size}x{args.img_size}, rgb={args.rgb}, pad={args.pad}, "
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
    event(f"Datos cargados: train {X_tr.shape} ({int(Y1_tr.sum())} mel) | val {X_va.shape} ({int(Y1_va.sum())} mel) "
          f"| ilegibles train={failed_tr} val={failed_va}")

    cw = get_class_weights(Y1_tr, args.class_weights)
    ctx["class_weights"] = {("mel" if k == 1 else "no-mel"): round(v, 3) for k, v in cw.items()} if cw else "ninguno"
    if cw:
        event(f"Pesos de clase ({args.class_weights}): {ctx['class_weights']}")

    # ------------------------------------------------ modelo
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(args.seed)
    pretrained = None if args.pretrained == "none" else args.pretrained
    event(f"Construyendo EfficientNet{args.variant} (pesos={args.pretrained}) ...")
    model, base = build_binary_model(args.variant, args.img_size, args.augment, pretrained, args.head_lr)
    summary_lines = []
    model.summary(print_fn=lambda s, **kw: summary_lines.append(s))
    ctx["model"] = dict(params_total=model.count_params(), trainable_head=count_params(model.trainable_weights),
                        trainable_ft=None, summary=summary_lines)

    # ------------------------------------------------ entrenamiento
    rows, phases = [], ctx["phases"]
    phases.append(fit_phase(model, "fase1_cabeza", X_tr, Y1_tr, X_va, Y1_va, args.epochs_head,
                            args.batch_size, cw, rows, event))
    if args.fine_tune:
        unfreeze_top_layers(model, base, args.ft_layers, args.ft_lr)
        ctx["model"]["trainable_ft"] = count_params(model.trainable_weights)
        event(f"Fine-tuning: se descongelan las últimas {args.ft_layers} capas del backbone (BatchNorm congelada), "
              f"lr={args.ft_lr:.0e}")
        phases.append(fit_phase(model, "fase2_finetuning", X_tr, Y1_tr, X_va, Y1_va, args.epochs_ft,
                                args.batch_size, cw, rows, event))
    hist = pd.DataFrame(rows)
    pref = ["fase", "epoca", "epoca_fase", "loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc", "lr", "segundos"]
    hist = hist[[c for c in pref if c in hist.columns] + [c for c in hist.columns if c not in pref]]
    ctx["history"] = hist
    hist.to_csv(run_dir / "historial_epocas.csv", index=False)
    files.append("historial_epocas.csv — métricas por época")
    plot_history(hist, run_dir / "curvas_entrenamiento.png", f"{args.name} — Etapa 1 (mel vs no-mel)")
    files.append("curvas_entrenamiento.png — curvas accuracy / loss / AUC (línea gris = cambio de fase)")

    # ------------------------------------------------ evaluación
    event("Evaluando en validación ...")
    p_mel = model.predict(X_va, batch_size=64, verbose=0).ravel()
    keras_eval = dict(zip(["val_loss", "val_accuracy", "val_auc"],
                          model.evaluate(X_va, Y1_va, batch_size=64, verbose=0)))
    thr = choose_threshold(Y1_va, p_mel, args.thr_mode, args.min_sensitivity)
    m_thr, m_05 = binary_metrics(Y1_va, p_mel, thr), binary_metrics(Y1_va, p_mel, 0.5)
    try:
        auc, ap = float(roc_auc_score(Y1_va, p_mel)), float(average_precision_score(Y1_va, p_mel))
        plot_stage1(Y1_va, p_mel, thr, run_dir / "evaluacion_etapa1.png", f"{args.name} — Etapa 1")
        files.append("evaluacion_etapa1.png — curva ROC, Precision-Recall y matriz de confusión")
    except ValueError as e:  # p. ej. validación sin melanomas (con --limit-per-class muy bajo)
        auc = ap = None
        event(f"AVISO: no se pudo calcular AUC/AP ({e})")
    pred = (p_mel >= thr).astype(int)
    report = classification_report(Y1_va, pred, labels=[0, 1], target_names=["no-mel", "mel"], zero_division=0)
    prob_stats = [dict(grupo=g, n=int((Y1_va == v).sum()), media=float(p_mel[Y1_va == v].mean()) if (Y1_va == v).any() else float("nan"),
                       mediana=float(np.median(p_mel[Y1_va == v])) if (Y1_va == v).any() else float("nan"),
                       p10=float(np.percentile(p_mel[Y1_va == v], 10)) if (Y1_va == v).any() else float("nan"),
                       p90=float(np.percentile(p_mel[Y1_va == v], 90)) if (Y1_va == v).any() else float("nan"))
                  for g, v in [("no-mel (real)", 0), ("mel (real)", 1)]]
    sweep = [binary_metrics(Y1_va, p_mel, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)]
    ctx["eval"] = dict(auc=auc, ap=ap, thr=thr, thr_mode=str(args.thr_mode), at_thr=m_thr, at_05=m_05,
                       keras_eval={k: float(v) for k, v in keras_eval.items()},
                       report=report, prob_stats=prob_stats, sweep=sweep)
    event(f"Umbral={thr:.2f} ({args.thr_mode}) | AUC={auc if auc is None else round(auc, 3)} | "
          f"sensibilidad={m_thr['sensitivity']:.3f} | especificidad={m_thr['specificity']:.3f} | "
          f"precisión={m_thr['precision']:.3f}")

    # ------------------------------------------------ guardar artefactos
    pd.DataFrame({"archivo": files_va, "clase_original": [categories[i] for i in Y_va],
                  "y_real_mel": Y1_va, "p_mel": p_mel, "pred_mel": pred}).to_csv(
        run_dir / "predicciones_validacion.csv", index=False)
    files.append("predicciones_validacion.csv — P(mel) y predicción por imagen de validación")

    model_path = run_dir / f"{args.name}_stage1.keras"
    model.save(model_path)
    files.append(f"{model_path.name} — modelo final entrenado")
    meta = dict(thr=thr, img_size=args.img_size, to_rgb=args.rgb, pad=args.pad, 
                variant=args.variant, preprocessing=args.preprocessing)
    meta = dict(thr=thr, img_size=args.img_size, to_rgb=args.rgb, pad=args.pad, 
                variant=args.variant, preprocessing=args.preprocessing)
    (run_dir / f"{args.name}_stage1.json").write_text(json.dumps(meta, indent=2))
    files.append(f"{args.name}_stage1.json — metadatos (umbral y pre-procesado)")

    (run_dir / "metricas.json").write_text(json.dumps(
        dict(config={k: str(v) for k, v in vars(args).items()}, fases=phases,
             evaluacion={k: v for k, v in ctx["eval"].items() if k not in ("report",)}),
        indent=2, default=float))
    files.append("metricas.json — métricas finales, fases y configuración")


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento binario mel vs no-mel (HAM10000) con resumen .txt",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("datos")
    g.add_argument("--base-dir", default=None, help="Carpeta con HAM10000_metadata.csv + imágenes (solo si hay que organizar el dataset)")
    g.add_argument("--dataset-dir", default=None, help="Carpeta con training/ y validation/ (por defecto <base-dir>/Datasets o ./Datasets)")
    g.add_argument("--limit-per-class", type=int, default=None, help="Máx. imágenes por clase (pruebas rápidas)")
    g = p.add_argument_group("pre-procesado")
    g.add_argument("--img-size", type=int, default=224)
    g.add_argument("--rgb", action=argparse.BooleanOptionalAction, default=True, help="Convertir BGR -> RGB")
    g.add_argument("--pad", action=argparse.BooleanOptionalAction, default=False, help="Cuadrar con margen negro antes de redimensionar")
    g.add_argument("--preprocessing", choices=["baseline", "hair_removal", "color_constancy", "contrast_focused"],
                   default="baseline", help="Técnica de pre-procesamiento a aplicar de preprocessing.py")
    g = p.add_argument_group("modelo / entrenamiento")
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
    g = p.add_argument_group("umbral")
    g.add_argument("--thr-mode", type=parse_thr_mode, default="f1", help="'f1', 'sensitivity' o un número fijo (p. ej. 0.5)")
    g.add_argument("--min-sensitivity", type=float, default=0.85, help="Usado si thr-mode='sensitivity'")
    g = p.add_argument_group("salida")
    g.add_argument("--name", default="B1_binario_B4_224", help="Nombre de la ejecución")
    g.add_argument("--output-dir", default="runs", help="Carpeta donde se crea runs/<name>_<fecha>/")
    g.add_argument("--quick", action="store_true", help="Prueba rápida: 30 img/clase, B0, 1 época, sin fine-tuning")
    args = p.parse_args()

    if args.quick:
        args.limit_per_class, args.variant, args.epochs_head, args.fine_tune = 30, "B0", 1, False
        if args.name == p.get_default("name"):
            args.name = "TESTE_RAPIDO_binario"
    if args.dataset_dir is None:
        args.dataset_dir = str(Path(args.base_dir) / "Datasets") if args.base_dir else "Datasets"
    return args


def main():
    args = parse_args()
    started = dt.datetime.now()
    run_dir = Path(args.output_dir) / f"{args.name}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log = open(run_dir / "consola.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, console_log)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        build = tf.sysconfig.get_build_info()
        cuda = f"CUDA {build.get('cuda_version', 'n/d')} / cuDNN {build.get('cudnn_version', 'n/d')}"
    except Exception:
        cuda = "n/d"
    import sklearn
    env = {"Equipo": platform.node(), "Sistema": platform.platform(), "Python": platform.python_version(),
           "TensorFlow": tf.__version__, "Keras": getattr(tf.keras, "__version__", "n/d"), "NumPy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "OpenCV": cv2.__version__,
           "GPU(s)": ", ".join(g.name for g in gpus) if gpus else "ninguna (CPU)", "Build TF": cuda}

    t0 = time.time()
    events = []
    event = make_event_logger(events, t0)
    ctx = dict(args=vars(args).copy(), cmd=" ".join(sys.argv), env=env, start=f"{started:%Y-%m-%d %H:%M:%S}",
               events=events, files=[], phases=[], status="EN CURSO", run_dir=str(run_dir))
    ctx["args"]["dataset_dir"] = args.dataset_dir
    ctx["args"]["thr_mode"] = str(args.thr_mode)

    exit_code = 0
    try:
        event(f"Ejecución '{args.name}' — resultados en {run_dir}")
        run_training(args, run_dir, ctx, event)
        ctx["status"] = "COMPLETADO"
    except BaseException as e:  # incluye Ctrl+C: igualmente se escribe el resumen parcial
        ctx["status"] = "INTERRUMPIDO" if isinstance(e, KeyboardInterrupt) else "ERROR"
        ctx["error"] = traceback.format_exc()
        event(f"{ctx['status']}: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        ctx["elapsed"] = time.time() - t0
        ctx["end"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
        ctx["warnings"] = derive_warnings(ctx)
        ctx["files"] += ["resumen_entrenamiento.txt — este archivo", "consola.log — salida de consola completa"]
        report_path = run_dir / "resumen_entrenamiento.txt"
        report_path.write_text(build_report(ctx), encoding="utf-8")
        print(f"\nResumen escrito en: {report_path}")
        sys.stdout = sys.__stdout__
        console_log.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()