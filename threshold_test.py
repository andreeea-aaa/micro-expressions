import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fps = 30
gaze_features = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']

def train_gaze_rf(train_real, train_fake):
    li = []
    for files, lab in [(train_real, 0), (train_fake, 1)]:
        for f in files:
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                if all(c in df.columns for c in gaze_features):
                    temp = df[gaze_features].copy()
                    temp['label'] = lab
                    li.append(temp)
            except:
                continue
    if not li:
        return None
    data = pd.concat(li, axis=0, ignore_index=True)
    model = RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_split=10,
        min_samples_leaf=1, max_features='sqrt', random_state=42
    )
    model.fit(data[gaze_features], data['label'])
    return model

def collect_scores(base_real, base_fake, name):
    print(f"\n>>> Processing: {name}...")
    folders = sorted([d for d in os.listdir(base_real) if os.path.isdir(os.path.join(base_real, d))])
    y_true, gaze_scores = [], []

    for subj in folders:
        f_real = glob.glob(os.path.join(base_real, subj, "*.csv"))
        f_fake = glob.glob(os.path.join(base_fake, subj, "*.csv"))
        if len(f_real) < 2 or len(f_fake) < 2:
            continue

        tr_real, te_real = train_test_split(f_real, train_size=0.7, random_state=42)
        tr_fake, te_fake = train_test_split(f_fake, train_size=0.3, random_state=42)

        rf = train_gaze_rf(tr_real, tr_fake)
        if not rf:
            continue

        for f, lab in [(f, 0) for f in te_real] + [(f, 1) for f in te_fake]:
            try:
                y_true.append(lab)
                df_test = pd.read_csv(f)
                df_test.columns = df_test.columns.str.strip()
                if all(c in df_test.columns for c in gaze_features):
                    g_score = np.mean(rf.predict(df_test[gaze_features]))
                    gaze_scores.append(g_score)
                else:
                    gaze_scores.append(0.0)
            except:
                continue

    if not y_true or not gaze_scores:
        print(f"Could not process data for {name}")
        return None

    return {'name': name, 'y_true': y_true, 'gaze_scores': gaze_scores}

path_celeb_real = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/real_celebDF"
path_celeb_fake = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/fake_celebDF"
path_dfd_real   = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/real_DFD"
path_dfd_fake   = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/fake_DFD"

res_dfd   = collect_scores(path_dfd_real,   path_dfd_fake,   "DFD")
res_celeb = collect_scores(path_celeb_real, path_celeb_fake, "Celeb-DF")

thresholds = [0.01, 0.20, 0.30, 0.40, 0.50]
threshold_labels = ['1%', '20%', '30%', '40%', '50%']

print("\n" + "="*85)
print(f"{'Threshold':<10} | {'DFD':^35} | {'Celeb-DF':^35}")
print(f"{'(τ)':<10} | {'Acc%':<8}{'TPR':<8}{'FPR':<8}{'FakeRate%':<10} | {'Acc%':<8}{'TPR':<8}{'FPR':<8}{'FakeRate%':<10}")
print("-" * 85)

for thresh, label in zip(thresholds, threshold_labels):
    row = f"{label:<10} |"
    for res in [res_dfd, res_celeb]:
        if res is None:
            row += f" {'N/A':<34} |"
            continue
            
        y_true  = res['y_true']
        scores  = res['gaze_scores']
        y_pred  = [1 if s >= thresh else 0 for s in scores]
        
        acc  = accuracy_score(y_true, y_pred) * 100
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
        fake_rate = sum(y_pred) / len(y_pred) * 100
        
        row += f" {acc:<8.1f}{tpr:<8.3f}{fpr:<8.3f}{fake_rate:<10.1f}|"
    print(row)
print("="*85)