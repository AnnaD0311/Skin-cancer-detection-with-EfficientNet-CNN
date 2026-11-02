# Skin Cancer Detection with EfficientNet (HAM10000)

Classification of dermatoscopic images from the **HAM10000** dataset with an ImageNet-pretrained **EfficientNet-B4**, in two tasks:

- **Binary:** melanoma (`mel`) vs. non-melanoma.
- **Multiclass:** the 7 HAM10000 diagnoses (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

Each task is trained with **four image-preprocessing pipelines** (Baseline, DullRazor, Shades of Gray, Contrast-Focused) at **two input resolutions** (128×128 and 224×224). That gives 16 controlled experiments, which are compared statistically on a validation set and then checked on an untouched test set.

> For educational and research use only. This is not a clinical diagnostic tool.

---

## How the work is split: Python scripts vs. notebook

The heavy GPU work (building the dataset and training the CNNs) is done in **standalone Python scripts** rather than in notebooks. We could then launch them **remotely on a computer with a more powerful GPU**, inside **`tmux`** sessions, so training kept running after we disconnected from SSH. Each script writes everything it produces (model, metrics, predictions, logs, figures) to its own run folder, so the results could be copied back and analysed offline.

All **data analysis** (dataset exploration, balancing checks, comparison of the runs, statistics and the final test-set evaluation) was done in **`HAM10000_unified_analysis.ipynb`**.

```bash
# Example: training session on the remote machine
tmux new -s skin
python3 entrenamiento_binario_mel.py --name Exp_DullRazor --dataset-dir Datasets_safe --preprocessing hair_removal
# Ctrl+B, D to detach; later: tmux attach -t skin
```

---

## Pipeline overview

```
HAM10000_metadata.csv + HAM10000_images_part_{1,2}/
            │
            ▼
 balance_dataset_new.py        lesion-level 70/15/15 split (no leakage), resize, save
            │
            ▼
 Datasets_safe/{training,validation,test}/<class>/*.jpg  + *_metadata.csv
            │
            ├──► entrenamiento_binario_mel.py        (mel vs. non-mel, sigmoid head)
            └──► sparse_categorical_crossentropy.py  (7 classes, softmax head)
                     │   both call preprocessing.py when images are loaded
                     ▼
 runs_dataleakage_fixed/{Binary,Multiple}/resolution{128x128,224x224}/Exp_<pipeline>_<date>/
            │
            ▼
 HAM10000_unified_analysis.ipynb  (Section 11: validation comparison + independent test)
```

### 1. Dataset preparation — `balance_dataset_new.py`

This script replaces the first balancing approach (`balance_dataset.py`), which caused **data leakage** (see [Data-leakage fix](#data-leakage-fix)).

1. Reads `HAM10000_metadata.csv` and checks that the required columns exist, that there are no missing or duplicate IDs, and that every listed image is on disk.
2. **Splits by `lesion_id`** (not by `image_id`) with two `GroupShuffleSplit`s: 70% training, then the remaining 30% split 50/50 into validation and test. All images of the same lesion therefore land in the same subset.
3. **Checks** that no `lesion_id` is shared between any two subsets, and stops with `DATA LEAKAGE DETECTED` if one is.
4. Saves every image (optionally preprocessed, `baseline` by default = resize only) as `output_dir/<split>/<dx>/<image_id>.jpg`, plus `training_metadata.csv`, `validation_metadata.csv` and `test_metadata.csv` for reproducibility.
5. Optional `--balance-training`: adds augmented copies (90° rotations and flips) **only to the training set** until each minority class reaches `--target-count`. Validation and test are never augmented. Classes above the target keep all their images (no undersampling).
6. Refuses to write into a non-empty output folder, so different experiments cannot get mixed.

The final experiments used `Datasets_safe` (no offline augmentation; imbalance handled with class weights during training):

| Split      | Images | Lesions | akiec | bcc | bkl | df | mel | nv    | vasc |
|------------|-------:|--------:|------:|----:|----:|---:|----:|------:|-----:|
| training   | 7,002  | 5,229   | 207   | 381 | 741 | 89 | 770 | 4,718 | 96   |
| validation | 1,503  | 1,120   | 67    | 74  | 172 | 18 | 162 | 988   | 22   |
| test       | 1,510  | 1,121   | 53    | 59  | 186 | 8  | 181 | 999   | 24   |

```bash
python3 balance_dataset_new.py --output-dir Datasets_safe
python3 balance_dataset_new.py --output-dir Datasets_balanced_safe --balance-training   # optional variant
```

### 2. Image preprocessing — `preprocessing.py`

These are the four pipelines compared in the project, based on Aakaou et al. (2026), *Contrast-Focused Preprocessing*. `preprocess_pipeline(img, target_size, method)` applies one of them and then resizes with `INTER_AREA`:

| `--preprocessing`  | Name in the analysis | What it does |
|--------------------|----------------------|--------------|
| `baseline`         | Baseline             | Resize only (control). |
| `hair_removal`     | DullRazor            | Black-Hat morphological transform (9×9 kernel) on the grayscale image → threshold → hair mask → **Telea inpainting**. |
| `color_constancy`  | Shades of Gray       | Estimates the illuminant with a Minkowski p-norm (p = 6) and rescales the RGB channels to remove colour casts from different cameras and lighting. |
| `contrast_focused` | Contrast-Focused     | DullRazor followed by **CLAHE** on the L channel of the LAB colour space (clip limit 2.0, 8×8 tiles). |

The training scripts apply the chosen pipeline **when the images are loaded**, the same way for training and validation. The notebook uses the same function on the test set. Do not combine `--preprocessing X` with a dataset already built with `balance_dataset_new.py --method X`: the transformation would be applied twice.

### 3. Training scripts

Both scripts share the same structure, model and training schedule. Only the output head, the loss and the evaluation differ.

**Model**
```
Input (128 or 224, RGB, uint8)
 → online augmentation: RandomFlip(h+v), RandomRotation(0.2), RandomZoom(0.1)   [no colour changes]
 → EfficientNet-B4 (ImageNet weights, include_top=False)
 → GlobalMaxPooling2D → Dense(256, ReLU) → BatchNorm → Dropout(0.5)
 → Dense(1, sigmoid)           [binary]
   Dense(7, softmax)           [multiclass]
```

**Two-phase training** (Adam, batch 32, seed 42):
1. **Head phase:** backbone frozen, `lr = 5e-4`, up to 15 epochs.
2. **Fine-tuning:** the last 30 backbone layers are unfrozen (BatchNorm stays frozen to keep the ImageNet statistics), `lr = 1e-5`, up to 15 epochs.

Both phases use `EarlyStopping(val_loss, patience=5, restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.3, patience=2)`. Class imbalance is handled with **square-root-balanced class weights** (`--class-weights sqrt`).

**Outputs:** each run creates `<output-dir>/<name>_<YYYYmmdd_HHMMSS>/` with:
- the trained model (`.keras`) plus a `.json` with its preprocessing metadata (and the decision threshold, for the binary model);
- `metrics.json`, `epoch_history.csv` and `validation_predictions.csv` (per-image probabilities);
- training curves and evaluation figures;
- `console.log` and a detailed `training_summary.txt` (environment, configuration, data, per-epoch metrics, event timeline, warnings). The summary is written even if the run crashes or is interrupted with Ctrl+C.

#### `entrenamiento_binario_mel.py` — melanoma vs. non-melanoma
- Loads the 7-class folders and turns the labels into binary (`1 = mel`).
- Loss: `binary_crossentropy`. Metrics: accuracy and AUC.
- On validation it reports AUC, average precision, sensitivity, specificity, precision, F1, the confusion matrix, the distribution of P(mel) per true class and a threshold sweep (0.05–0.95).
- Decision threshold `--thr-mode`: `f1` (maximise F1, the default), `sensitivity` (highest threshold that still reaches `--min-sensitivity`), or a fixed number.

#### `sparse_categorical_crossentropy.py` — 7-class classifier
- Keeps the integer labels 0–6 (classes in alphabetical order).
- Loss: `sparse_categorical_crossentropy`.
- On validation it reports accuracy, a per-class classification report and the 7×7 confusion matrix.

```bash
# One run per pipeline and resolution (16 in total)
python3 entrenamiento_binario_mel.py      --name Exp_Baseline    --dataset-dir Datasets_safe --preprocessing baseline     --img-size 224
python3 sparse_categorical_crossentropy.py --name Exp_MC_Baseline --dataset-dir Datasets_safe --preprocessing baseline     --img-size 224
#   ... repeated with --preprocessing hair_removal | color_constancy | contrast_focused and --img-size 128

python3 entrenamiento_binario_mel.py --dataset-dir Datasets_safe --quick   # quick check: 30 img/class, B0, 1 epoch
```

### 4. Analysis — `HAM10000_unified_analysis.ipynb`

Sections 1–10 contain the exploratory work: metadata exploration, example images, preprocessing visualisation, the analysis of the first balancing approach (where the leakage was found in §7.5) and the first training results.

**Section 11 — "Results after the data-leakage fix: binary and multiclass classifiers"** is where the final results presented in class were produced. It reads the 16 runs in `runs_dataleakage_fixed/`.

| Cell  | What it does |
|-------|--------------|
| 11.1  | Helpers: discovers the runs, infers the pipeline and resolution of each one, computes the metrics, and implements a **lesion-level bootstrap** (1,000 resamples of whole lesions, because images of the same lesion are not independent). |
| 11.2  | **Validity audit:** 16/16 runs found; 0 lesions shared between splits; every run evaluated on the **same 1,503 validation images**; inside each task and resolution, the only configuration difference is `--preprocessing`. |
| 11.3  | **Binary comparison:** AUC, average precision, and the pre-declared clinical operating point (**highest specificity with sensitivity ≥ 85%**), with 95% CIs and **paired differences vs. Baseline**. Differences whose interval contains 0 are labelled *within noise*. |
| 11.4  | **Multiclass comparison:** selected by **macro-F1** (so that `nv`, about two thirds of the data, cannot dominate), reported with balanced accuracy, melanoma recall, the % of melanomas predicted as `nv`, per-class recall heatmaps and row-normalised confusion matrices. |
| 11.5  | **128×128 vs. 224×224**, paired, for each pipeline. |
| 11.6  | **Binary head vs. multiclass P(mel)** as melanoma detectors, using the same operating-point rule. |
| 11.7  | Before vs. after the leakage fix (descriptive only: the validation sets are different). |
| 11.8  | One-line summary of the selected pipeline per task and resolution. |
| 11.9  | **Independent test set, multiclass 224×224:** loads each `.keras` model, checks that its class order, resolution and preprocessing match, and evaluates it on the 1,510 test images. |
| 11.10 | **Independent test set, binary 224×224**, using the F1 threshold saved in each model's `.json`. |
| 11.11 | **Binary test at the pre-declared operating point:** the threshold is chosen on **validation** (sensitivity ≥ 85%), frozen, and applied to test, with lesion-level bootstrap CIs and paired differences vs. Baseline. |

The cells must be run in order: 11.1 → 11.2 → 11.3/11.4 → … → 11.9 → 11.10 → 11.11.

---

## Main results

### Validation (1,503 images, 162 melanomas)

**Binary — specificity at sensitivity ≥ 85%**

| Pipeline         | AUC 128 | Spec. 128 | AUC 224 | Spec. 224 |
|------------------|--------:|----------:|--------:|----------:|
| Baseline         | 0.813   | 0.591     | 0.826   | 0.600     |
| DullRazor        | 0.818   | 0.615     | **0.842** | **0.677** |
| Shades of Gray   | 0.805   | 0.570     | 0.810   | 0.607     |
| Contrast-Focused | 0.782   | 0.532     | 0.830   | 0.597     |

**Multiclass — macro-F1 / balanced accuracy / melanoma recall**

| Pipeline         | 128×128               | 224×224                   |
|------------------|-----------------------|---------------------------|
| Baseline         | 0.476 / 0.476 / 0.395 | 0.498 / 0.513 / **0.519** |
| DullRazor        | 0.459 / 0.465 / 0.395 | 0.518 / 0.539 / 0.457     |
| Shades of Gray   | **0.489** / 0.481 / 0.389 | **0.523 / 0.549** / 0.426 |
| Contrast-Focused | 0.449 / 0.461 / 0.395 | 0.511 / 0.535 / 0.512     |

- The selection rules chose **DullRazor** for the binary task and **Shades of Gray** for the multiclass task at both resolutions. However, **none of their advantages over Baseline is distinguishable from bootstrap noise** on validation.
- **Resolution matters more than preprocessing:** 8 of 20 paired 128→224 differences exclude 0 (e.g. multiclass Baseline melanoma recall +0.123, binary Contrast-Focused AUC +0.048).
- **Binary vs. multiclass as a melanoma detector:** the multiclass P(mel) reaches a similar or slightly higher AUC than the dedicated binary head, and 0 of 8 specificity differences are distinguishable.

### Independent test set (224×224, 1,510 images, 181 melanomas)

**Multiclass**

| Pipeline         | Accuracy | Macro-F1  | Balanced acc. | Mel recall |
|------------------|---------:|----------:|--------------:|-----------:|
| Baseline         | 0.699    | 0.462     | 0.504         | 0.381      |
| DullRazor        | 0.723    | 0.483     | 0.521         | 0.370      |
| Shades of Gray   | 0.723    | **0.510** | **0.539**     | 0.315      |
| Contrast-Focused | 0.711    | 0.471     | 0.503         | 0.348      |

**Binary, threshold frozen from validation (sensitivity ≥ 85%) — §11.11**

| Pipeline         | ROC-AUC [95% CI]     | Sensitivity | Specificity [95% CI]     | FN | FP  |
|------------------|----------------------|------------:|--------------------------|---:|----:|
| Baseline         | 0.822 [0.788, 0.854] | 0.856       | 0.634 [0.603, 0.662]     | 26 | 487 |
| DullRazor        | **0.840** [0.808, 0.874] | 0.851   | **0.671** [0.640, 0.700] | 27 | 437 |
| Shades of Gray   | 0.815 [0.775, 0.853] | 0.856       | 0.607 [0.578, 0.636]     | 26 | 522 |
| Contrast-Focused | 0.813 [0.779, 0.845] | 0.867       | 0.603 [0.574, 0.631]     | 24 | 527 |

- The selections hold on test: **Shades of Gray** has the best multiclass macro-F1 and **DullRazor** is the best binary melanoma detector.
- At the clinical operating point, DullRazor's specificity gain over Baseline (**+0.038 [+0.011, +0.065]**) is now distinguishable from noise, and Contrast-Focused is significantly worse than Baseline (−0.030 [−0.057, −0.002]).
- The thresholds chosen on validation kept a sensitivity of ~85% on test, so the operating point generalises.
- With the F1 thresholds saved in the models (§11.10) every pipeline ends up at a different threshold (0.32–0.52), so the test sensitivities from that cell are not comparable. That is why §11.11 exists.

---

## Data-leakage fix

The first approach (`balance_dataset.py`) built a balanced set of 1,000 images per class (undersampling `nv`, augmenting the minority classes) **from the whole dataset before splitting**. Because HAM10000 contains several images of the same lesion, and because augmented copies are near-duplicates of their source, **20.3% of the balanced training images came from validation lesions** and **42.1% of the validation images also appeared in training** (as originals or augmented copies). See notebook §7.5 and the `leakage_before_after.png` figure.

`balance_dataset_new.py` fixes this by **splitting by `lesion_id` first** and only then (optionally) augmenting the training set. Every run in `runs_dataleakage_fixed/` uses that split.

---

## Repository structure

| File | Role |
|------|------|
| `balance_dataset_new.py` | Leakage-free lesion-level split (70/15/15) and optional training-only augmentation. **Used for the final results.** |
| `preprocessing.py` | The four preprocessing pipelines (Baseline, DullRazor, Shades of Gray, Contrast-Focused). |
| `entrenamiento_binario_mel.py` | Binary EfficientNet training (mel vs. non-mel). |
| `sparse_categorical_crossentropy.py` | 7-class EfficientNet training. |
| `HAM10000_unified_analysis.ipynb` | All data analysis; Section 11 has the final results. |
| `balance_dataset.py` | *Legacy.* First balancing approach (1,000 img/class, split after balancing → data leakage). |
| `Classification_leasion_melhorado.ipynb` | *Legacy.* Base notebook shown to the teacher at the start of the project. |
| `Classification_leasion_preprocessamento*.ipynb`, `Classification_leasion_hierarquico.ipynb`, `data_analysis.ipynb` | *Legacy.* Earlier experiments (preprocessing, hierarchical model, first data analysis). The training scripts were derived from the hierarchical notebook. |

Not tracked by git (see `.gitignore`): the HAM10000 images, the `Datasets*` folders, the `runs*` folders, `models/`, `Remoto/` and the `.csv` files.

### Note on output file names

The runs in `runs_dataleakage_fixed/` were produced before the scripts were translated to English, so their files have Spanish names (`metricas.json`, `predicciones_validacion.csv`, `historial_epocas.csv`, `resumen_entrenamiento.txt`, `consola.log`). The current scripts write English names (`metrics.json`, `validation_predictions.csv`, `epoch_history.csv`, …). The loaders in Section 11 of the notebook accept both versions, so old and new runs can be analysed together.
