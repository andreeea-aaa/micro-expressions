# Deepfake Detection

A deepfake detection pipeline that combines eye gaze and micro-expression analysis on face videos. The methodology uses OpenFace-extracted facial features, organizes data by person, and runs Random Forest–based classification on Celeb-DF and DFD datasets.

---

## Overview

1. **Download** the required video datasets.
2. **Process** videos with OpenFace (Docker) to get per-frame facial features (gaze, action units, landmarks).
3. **Organize** outputs by person ID so real/fake videos are grouped per identity.
4. **Run analysis**: eye gaze only, micro-expressions only, threshold tests, grid search, or the full combined pipeline.

---

## 1. Datasets

Download and extract the following datasets (Kaggle):

| Dataset | Link | Description |
|--------|------|-------------|
| **Celeb-DF v2** | [Kaggle – Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) | Celeb deepfake dataset |
| **Deep Fake Detection (DFD)** | [Kaggle – DFD](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset) | Google DFD original/manipulated sequences |
---

## 2. OpenFace (Docker)

Videos must be processed with OpenFace to obtain CSV files with gaze, action units, and landmarks.

- **Setup**: Follow the [OpenFace Docker wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker) to install and run OpenFace in Docker.
- **Automation**: The script `openface.py` runs OpenFace in Docker on your video folders and copies the resulting CSVs out.

Before running `openface.py`, set the paths and container name at the top of the file:

- `REAL_DIR` / `FAKE_DIR`: local folders containing real and fake `.mp4` files.
- `OUTPUT_DIR`: where to save the processed CSV files.
- `CONTAINER_NAME`: your OpenFace Docker container name.

The script copies real/fake folders into the container, runs `FeatureExtraction`, then copies the `processed/*.csv` files back to `OUTPUT_DIR/{real,fake}/`.

---

## 3. Organize Data by Person

After OpenFace processing, run `organize.py` (in the project root) to group videos by person ID.

**Expected naming:**

- **Real (Celeb-real)**: `id{X}_{####}.mp4` → person X.
- **Fake (Celeb-synthesis)**: `id{A}_id{B}_{####}.mp4` → person **B** (second ID is the identity in the video).

**Output layout:**

```
Celeb-DF/
├── real/
│   ├── person_00/
│   │   ├── id0_0000.csv
│   │   └── ...
│   ├── person_01/
│   └── ...
└── fake/
    ├── person_00/
    │   ├── id1_id0_0000.csv   # person 0 in a video of person 1
    │   └── ...
    └── ...
```

**Usage:**

```bash
python organize.py --input archive-2 --output archive-2-organized
# Optional: preview without copying
python organize.py --input archive-2 --output archive-2-organized --dry-run
```

- `--input`: folder containing `Celeb-real` and `Celeb-synthesis` (or your real/fake folders).
- `--output`: destination for the person-based structure above.

Adapt the script if your dataset uses different folder names (e.g. DFD real/fake directories); the same idea applies: one folder per person under `real/` and `fake/`.

---

## 4. Pipeline Order

1. **Download** Celeb-DF v2 and DFD from the Kaggle links above.
2. **OpenFace:** Set up Docker per [OpenFace Docker wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker), then run `cisis/openface.py` (after editing paths and container name).
3. **Organize:** Run `organize.py` so data is in `real/person_XX/` and `fake/person_XX/` with the correct naming.
4. **Analysis:**
   - **Gaze only:** `python eyegaze.py`
   - **Micro-expressions only:** python micro_expressions.py
   - **Threshold test:** `python threshold_test.py`
   - **Grid search:** `python grid_search.py`
   - **Full pipeline (gaze + micro-expressions):** `python full_pipeline.py`

---

## 5. Dependencies

Typical environment:

- Python 3.x
- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `matplotlib`

---
