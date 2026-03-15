import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

fps = 30
gaze_features = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "lines.linewidth": 2
})

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
        n_estimators=250, 
        max_depth=10, 
        min_samples_split=10, 
        min_samples_leaf=1, 
        max_features='sqrt', 
        random_state=42
    )
    model.fit(data[gaze_features], data['label'])
    return model

def process_dataset(base_real, base_fake, name):
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

    y_pred_bin = [1 if s >= 0.5 else 0 for s in gaze_scores]
    
    res = {
        'name': name,
        'auc': roc_auc_score(y_true, gaze_scores),
        'acc': accuracy_score(y_true, y_pred_bin),
        'prec': precision_score(y_true, y_pred_bin, zero_division=0),
        'recall': recall_score(y_true, y_pred_bin, zero_division=0),
        'f1': f1_score(y_true, y_pred_bin, zero_division=0),
        'fpr': None, 
        'tpr': None
    }
    
    res['fpr'], res['tpr'], _ = roc_curve(y_true, gaze_scores)
    return res

path_celeb_real = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/real_celebDF"
path_celeb_fake = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/fake_celebDF"
path_dfd_real = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/real_DFD"
path_dfd_fake = "/Users/andreeabrad/StudioCode/micro-expressions/datasets/fake_DFD"

res_celeb = process_dataset(path_celeb_real, path_celeb_fake, "Celeb-DF")
res_dfd = process_dataset(path_dfd_real, path_dfd_fake, "DFD")

print("\n" + "="*85)
print(f"{'DATASET':<15} | {'ACC':<8} | {'AUC':<8} | {'PRECISION':<10} | {'RECALL':<8} | {'F1':<8}")
print("-" * 85)
for r in [res_celeb, res_dfd]:
    if r:
        print(f"{r['name']:<15} | {r['acc']:.4f} | {r['auc']:.4f} | {r['prec']:.4f}    | {r['recall']:.4f} | {r['f1']:.4f}")
print("="*85)

plt.figure(figsize=(10, 8))
for r in [res_celeb, res_dfd]:
    if r and r['fpr'] is not None:
        plt.plot(r['fpr'], r['tpr'], label=f"{r['name']} (AUC = {r['auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Eye Gaze Only')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()