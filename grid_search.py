import os
import glob
import pandas as pd
import numpy as np
from scipy.ndimage import label
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from itertools import product
import time

fps = 30
au_cols = ['au12', 'au6', 'au15', 'au1', 'au4']
gaze_features = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']

me_weights = {
    'mean_intensity': 4.0,
    'std_intensity': 2.0,
    'frequency': 1.0,
    'max_intensity': 2.0,
    'count': 5.0
}

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "lines.linewidth": 2
})

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

def get_xy(df, idx):
    return df[f'x_{idx}'], df[f'y_{idx}']

def compute_au12(df):
    x48, y48 = get_xy(df, 48); x54, y54 = get_xy(df, 54)
    x0, _ = get_xy(df, 0); x16, _ = get_xy(df, 16)
    fw = np.abs(x16 - x0).replace(0, np.nan)
    mw = np.abs(x54 - x48) / fw
    _, y30 = get_xy(df, 30)
    lift = -((y48 + y54) / 2 - y30) / fw
    return 0.6 * mw + 0.4 * lift

def compute_au6(df):
    _, y27 = get_xy(df, 27); _, y30 = get_xy(df, 30)
    nl = np.abs(y30 - y27).replace(0, np.nan)
    _, y37 = get_xy(df, 37); _, y38 = get_xy(df, 38)
    _, y40 = get_xy(df, 40); _, y41 = get_xy(df, 41)
    left = ((y40 + y41) / 2) - ((y37 + y38) / 2)
    _, y43 = get_xy(df, 43); _, y44 = get_xy(df, 44)
    _, y46 = get_xy(df, 46); _, y47 = get_xy(df, 47)
    right = ((y46 + y47) / 2) - ((y43 + y44) / 2)
    return -((left + right) / 2) / nl

def compute_au15(df):
    _, y27 = get_xy(df, 27); _, y30 = get_xy(df, 30)
    nl = np.abs(y30 - y27).replace(0, np.nan)
    _, y48 = get_xy(df, 48); _, y54 = get_xy(df, 54)
    return ((y48 + y54) / 2 - y30) / nl

def compute_au1(df):
    _, y21 = get_xy(df, 21); _, y22 = get_xy(df, 22)
    _, y27 = get_xy(df, 27); _, y30 = get_xy(df, 30)
    return (y27 - (y21 + y22) / 2) / np.abs(y30 - y27).replace(0, np.nan)

def compute_au4(df):
    _, y21 = get_xy(df, 21); _, y22 = get_xy(df, 22)
    _, y27 = get_xy(df, 27); _, y30 = get_xy(df, 30)
    return ((y21 + y22) / 2 - y27) / np.abs(y30 - y27).replace(0, np.nan)

def z_normalize(df, col):
    smooth = df[col].rolling(3, center=True).mean()
    baseline = smooth.rolling(int(2 * fps), min_periods=1).median()
    delta = smooth - baseline
    return (delta - delta.mean()) / delta.std() if delta.std() != 0 else delta * 0

def extract_events(active_mask, df, peak_cols):
    events = []
    labeled, num = label(active_mask)
    for i in range(1, num + 1):
        idx = np.where(labeled == i)[0]
        if 0.1 <= (len(idx) / fps) <= 0.5:
            events.append(float(df.loc[idx, peak_cols].max().max()))
    return events

def get_microexpression_features(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df['au12'] = compute_au12(df); df['au6'] = compute_au6(df)
        df['au15'] = compute_au15(df); df['au1'] = compute_au1(df); df['au4'] = compute_au4(df)
        for au in au_cols: df[f'{au}_z'] = z_normalize(df, au)
        h = (df['au12_z'] > 2.0) & (df['au6_z'] > 1.5)
        s = (df['au15_z'] > 2.0) & ((df['au1_z'] > 1.5) | (df['au4_z'] > 1.5))
        peaks = extract_events(h, df, ['au12_z', 'au6_z']) + extract_events(s, df, ['au15_z', 'au1_z', 'au4_z'])
        return {
            'mean_intensity': np.mean(peaks) if peaks else 0.0,
            'max_intensity': np.max(peaks) if peaks else 0.0,
            'std_intensity': np.std(peaks) if peaks else 0.0,
            'count': len(peaks),
            'frequency': len(peaks) / (len(df) / fps) if len(df) > 0 else 0.0
        }
    except: return None

def train_gaze_rf(train_real, train_fake, rf_params):
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
            except: continue
    if not li: return None
    data = pd.concat(li, axis=0, ignore_index=True)
    model = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=42)
    model.fit(data[gaze_features], data['label'])
    return model

def learn_me_baseline(train_files):
    feats = [get_microexpression_features(f) for f in train_files]
    feats = [f for f in feats if f]
    if not feats: return None
    return {k: {'mean': np.mean([f[k] for f in feats]), 'std': np.std([f[k] for f in feats])} for k in feats[0].keys()}

def compute_me_distance(features, baseline):
    if not features or not baseline: return 10.0
    dist = 0.0
    for k, w in me_weights.items():
        if k in features:
            mu, sig = baseline[k]['mean'], max(0.1, baseline[k]['std'])
            dist += (abs(features[k] - mu) / sig) * w
    return dist

def process_dataset_with_params(base_real, base_fake, name, rf_params):
    folders = sorted([d for d in os.listdir(base_real) if os.path.isdir(os.path.join(base_real, d))])
    
    y_true, s1_scores, s2_scores, combined_scores = [], [], [], []

    for subj in folders:
        f_real = glob.glob(os.path.join(base_real, subj, "*.csv"))
        f_fake = glob.glob(os.path.join(base_fake, subj, "*.csv"))
        if len(f_real) < 2 or len(f_fake) < 2: continue

        tr_real, te_real = train_test_split(f_real, train_size=0.7, random_state=42)
        tr_fake, te_fake = train_test_split(f_fake, train_size=0.3, random_state=42)

        rf = train_gaze_rf(tr_real, tr_fake, rf_params)
        me_base = learn_me_baseline(tr_real)
        if not rf or not me_base: continue

        dists_tr = [compute_me_distance(get_microexpression_features(f), me_base) for f in tr_real]
        me_thresh = max(2.0, np.percentile(dists_tr, 95)) if dists_tr else 3.0

        for f, lab in [(f, 0) for f in te_real] + [(f, 1) for f in te_fake]:
            y_true.append(lab)
            df_test = pd.read_csv(f)
            df_test.columns = df_test.columns.str.strip()

            # STAGE 1: Pure gaze-based classification (INDEPENDENT)
            gaze_score = np.mean(rf.predict(df_test[gaze_features]))
            s1_scores.append(gaze_score)

            # STAGE 2: Pure microexpression-based classification (INDEPENDENT)
            me_feat = get_microexpression_features(f)
            dist = compute_me_distance(me_feat, me_base)
            # Convert distance to probability: higher distance = more likely fake
            me_score = 1 / (1 + np.exp(-(dist / me_thresh)))
            s2_scores.append(me_score)
            
            # COMBINED: Hybrid approach (your original Stage 2 logic)
            if gaze_score > 0.5:
                combined_score = gaze_score
            else:
                combined_score = me_score
            combined_scores.append(combined_score)

    if not y_true:
        return None

    # Calculate metrics for all three approaches
    y_pred_s1 = [1 if s >= 0.5 else 0 for s in s1_scores]
    y_pred_s2 = [1 if s >= 0.5 else 0 for s in s2_scores]
    y_pred_combined = [1 if s >= 0.5 else 0 for s in combined_scores]
    
    res = {
        'name': name,
        # Stage 1: Gaze-only metrics
        'auc_s1': roc_auc_score(y_true, s1_scores),
        'acc_s1': accuracy_score(y_true, y_pred_s1),
        'prec_s1': precision_score(y_true, y_pred_s1, zero_division=0),
        'recall_s1': recall_score(y_true, y_pred_s1, zero_division=0),
        'f1_s1': f1_score(y_true, y_pred_s1, zero_division=0),
        # Stage 2: Microexpression-only metrics
        'auc_s2': roc_auc_score(y_true, s2_scores),
        'acc_s2': accuracy_score(y_true, y_pred_s2),
        'prec_s2': precision_score(y_true, y_pred_s2, zero_division=0),
        'recall_s2': recall_score(y_true, y_pred_s2, zero_division=0),
        'f1_s2': f1_score(y_true, y_pred_s2, zero_division=0),
        # Combined approach metrics
        'auc_combined': roc_auc_score(y_true, combined_scores),
        'acc_combined': accuracy_score(y_true, y_pred_combined),
        'prec_combined': precision_score(y_true, y_pred_combined, zero_division=0),
        'recall_combined': recall_score(y_true, y_pred_combined, zero_division=0),
        'f1_combined': f1_score(y_true, y_pred_combined, zero_division=0),
        # Improvement metrics (Combined vs Stage 1)
        'auc_improvement': roc_auc_score(y_true, combined_scores) - roc_auc_score(y_true, s1_scores),
        'acc_improvement': accuracy_score(y_true, y_pred_combined) - accuracy_score(y_true, y_pred_s1),
        'params': rf_params
    }
    
    return res

def run_grid_search(base_real, base_fake, dataset_name):
    print(f"\n{'='*100}")
    print(f"RUNNING GRID SEARCH FOR {dataset_name}")
    print(f"{'='*100}\n")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    total_experiments = len(all_combinations)
    print(f"Total experiments to run: {total_experiments}\n")
    
    results = []
    
    for idx, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))
        
        print(f"Experiment {idx}/{total_experiments}")
        print(f"  n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
              f"min_samples_split={params['min_samples_split']}, min_samples_leaf={params['min_samples_leaf']}, "
              f"max_features={params['max_features']}, bootstrap={params['bootstrap']}")
        
        start_time = time.time()
        result = process_dataset_with_params(base_real, base_fake, dataset_name, params)
        elapsed = time.time() - start_time
        
        if result:
            print(f"  Stage 1 (Gaze)     => Acc: {result['acc_s1']:.4f}, AUC: {result['auc_s1']:.4f}, F1: {result['f1_s1']:.4f}")
            print(f"  Stage 2 (Micro-exp)=> Acc: {result['acc_s2']:.4f}, AUC: {result['auc_s2']:.4f}, F1: {result['f1_s2']:.4f}")
            print(f"  Combined (Hybrid)  => Acc: {result['acc_combined']:.4f}, AUC: {result['auc_combined']:.4f}, F1: {result['f1_combined']:.4f}")
            print(f"  Improvement (vs S1)=> Acc: {result['acc_improvement']:+.4f}, AUC: {result['auc_improvement']:+.4f}")
            print(f"  Time: {elapsed:.2f}s\n")
            results.append(result)
        else:
            print(f"  Failed to process this configuration\n")
    
    return results

def display_results_summary(results, dataset_name):
    print(f"\n{'='*140}")
    print(f"SUMMARY OF ALL EXPERIMENTS FOR {dataset_name}")
    print(f"{'='*140}\n")
    
    # Create DataFrame for easy analysis
    results_data = []
    for r in results:
        row = {
            'n_estimators': r['params']['n_estimators'],
            'max_depth': r['params']['max_depth'],
            'min_samples_split': r['params']['min_samples_split'],
            'min_samples_leaf': r['params']['min_samples_leaf'],
            'max_features': r['params']['max_features'],
            'bootstrap': r['params']['bootstrap'],
            # Stage 1 (Gaze) metrics
            'AUC_Gaze': r['auc_s1'],
            'Acc_Gaze': r['acc_s1'],
            'Prec_Gaze': r['prec_s1'],
            'Rec_Gaze': r['recall_s1'],
            'F1_Gaze': r['f1_s1'],
            # Stage 2 (Microexpression) metrics
            'AUC_MicroExp': r['auc_s2'],
            'Acc_MicroExp': r['acc_s2'],
            'Prec_MicroExp': r['prec_s2'],
            'Rec_MicroExp': r['recall_s2'],
            'F1_MicroExp': r['f1_s2'],
            # Combined metrics
            'AUC_Combined': r['auc_combined'],
            'Acc_Combined': r['acc_combined'],
            'Prec_Combined': r['prec_combined'],
            'Rec_Combined': r['recall_combined'],
            'F1_Combined': r['f1_combined'],
            # Improvements
            'AUC_Improvement': r['auc_improvement'],
            'Acc_Improvement': r['acc_improvement']
        }
        results_data.append(row)
    
    df_results = pd.DataFrame(results_data)
    
    # Analysis 1: Compare individual stages
    print(f"\n{'='*140}")
    print(f"COMPARISON: STAGE 1 (GAZE-ONLY) vs STAGE 2 (MICROEXPRESSION-ONLY)")
    print(f"{'='*140}\n")
    
    print(f"Average Performance Across All Experiments:")
    print(f"  Stage 1 (Gaze)          => AUC: {df_results['AUC_Gaze'].mean():.4f}, Acc: {df_results['Acc_Gaze'].mean():.4f}, F1: {df_results['F1_Gaze'].mean():.4f}")
    print(f"  Stage 2 (Microexp)      => AUC: {df_results['AUC_MicroExp'].mean():.4f}, Acc: {df_results['Acc_MicroExp'].mean():.4f}, F1: {df_results['F1_MicroExp'].mean():.4f}")
    print(f"  Combined (Hybrid)       => AUC: {df_results['AUC_Combined'].mean():.4f}, Acc: {df_results['Acc_Combined'].mean():.4f}, F1: {df_results['F1_Combined'].mean():.4f}")
    
    # Analysis 2: When does combining help?
    combined_improves = df_results[(df_results['AUC_Improvement'] > 0) & 
                                    (df_results['Acc_Improvement'] > 0)]
    
    print(f"\n{'='*140}")
    print(f"EXPERIMENTS WHERE COMBINED APPROACH IMPROVES OVER GAZE-ONLY ({len(combined_improves)}/{len(df_results)})")
    print(f"{'='*140}\n")
    
    if len(combined_improves) > 0:
        # Sort by total improvement
        combined_improves = combined_improves.copy()
        combined_improves['Total_Improvement'] = combined_improves['AUC_Improvement'] + combined_improves['Acc_Improvement']
        combined_improves = combined_improves.sort_values('Total_Improvement', ascending=False)
        
        print("TOP 10 CONFIGURATIONS WHERE COMBINED APPROACH HELPS:")
        print("-" * 140)
        for idx, row in combined_improves.head(10).iterrows():
            print(f"n_estimators={row['n_estimators']}, max_depth={row['max_depth']}, "
                  f"min_samples_split={row['min_samples_split']}, min_samples_leaf={row['min_samples_leaf']}, "
                  f"max_features={row['max_features']}")
            print(f"  Gaze-only     => AUC: {row['AUC_Gaze']:.4f}, Acc: {row['Acc_Gaze']:.4f}, F1: {row['F1_Gaze']:.4f}")
            print(f"  MicroExp-only => AUC: {row['AUC_MicroExp']:.4f}, Acc: {row['Acc_MicroExp']:.4f}, F1: {row['F1_MicroExp']:.4f}")
            print(f"  Combined      => AUC: {row['AUC_Combined']:.4f}, Acc: {row['Acc_Combined']:.4f}, F1: {row['F1_Combined']:.4f}")
            print(f"  Improvement   => AUC: +{row['AUC_Improvement']:.4f}, Acc: +{row['Acc_Improvement']:.4f}\n")
    else:
        print("No experiments found where Combined approach improves over Gaze-only.\n")
    
    # Analysis 3: Best overall performers
    df_results_sorted = df_results.sort_values('AUC_Combined', ascending=False)
    
    print("\n" + "="*140)
    print("TOP 10 CONFIGURATIONS BY COMBINED AUC:")
    print("-" * 140)
    for idx, row in df_results_sorted.head(10).iterrows():
        print(f"n_estimators={row['n_estimators']}, max_depth={row['max_depth']}, "
              f"min_samples_split={row['min_samples_split']}, min_samples_leaf={row['min_samples_leaf']}, "
              f"max_features={row['max_features']}")
        print(f"  Gaze-only     => AUC: {row['AUC_Gaze']:.4f}, Acc: {row['Acc_Gaze']:.4f}, F1: {row['F1_Gaze']:.4f}")
        print(f"  MicroExp-only => AUC: {row['AUC_MicroExp']:.4f}, Acc: {row['Acc_MicroExp']:.4f}, F1: {row['F1_MicroExp']:.4f}")
        print(f"  Combined      => AUC: {row['AUC_Combined']:.4f}, Acc: {row['Acc_Combined']:.4f}, F1: {row['F1_Combined']:.4f}")
        print(f"  Δ (vs Gaze)   => AUC: {row['AUC_Improvement']:+.4f}, Acc: {row['Acc_Improvement']:+.4f}\n")
    
    print("\n" + "="*140)
    print("BEST PARAMETERS BY EACH APPROACH:")
    print("="*140)
    
    # Best for each approach
    best_gaze_idx = df_results['AUC_Gaze'].idxmax()
    best_microexp_idx = df_results['AUC_MicroExp'].idxmax()
    best_combined_idx = df_results['AUC_Combined'].idxmax()
    
    print("\nBest Gaze-Only Configuration:")
    row = df_results.loc[best_gaze_idx]
    print(f"  n_estimators={row['n_estimators']}, max_depth={row['max_depth']}, "
          f"min_samples_split={row['min_samples_split']}, min_samples_leaf={row['min_samples_leaf']}, "
          f"max_features={row['max_features']}")
    print(f"  => AUC: {row['AUC_Gaze']:.4f}, Acc: {row['Acc_Gaze']:.4f}, F1: {row['F1_Gaze']:.4f}")
    
    print("\nBest MicroExpression-Only Configuration:")
    row = df_results.loc[best_microexp_idx]
    print(f"  n_estimators={row['n_estimators']}, max_depth={row['max_depth']}, "
          f"min_samples_split={row['min_samples_split']}, min_samples_leaf={row['min_samples_leaf']}, "
          f"max_features={row['max_features']}")
    print(f"  => AUC: {row['AUC_MicroExp']:.4f}, Acc: {row['Acc_MicroExp']:.4f}, F1: {row['F1_MicroExp']:.4f}")
    
    print("\nBest Combined Configuration:")
    row = df_results.loc[best_combined_idx]
    print(f"  n_estimators={row['n_estimators']}, max_depth={row['max_depth']}, "
          f"min_samples_split={row['min_samples_split']}, min_samples_leaf={row['min_samples_leaf']}, "
          f"max_features={row['max_features']}")
    print(f"  => AUC: {row['AUC_Combined']:.4f}, Acc: {row['Acc_Combined']:.4f}, F1: {row['F1_Combined']:.4f}")
    
    return df_results

# Main execution
path_celeb_real = "/Users/andreeabrad/StudioCode/eyes_notebook/real_celebDF"
path_celeb_fake = "/Users/andreeabrad/StudioCode/eyes_notebook/fake_celebDF"
path_dfd_real = "/Users/andreeabrad/StudioCode/eyes_notebook/real"
path_dfd_fake = "/Users/andreeabrad/StudioCode/eyes_notebook/fake"

# Choose which dataset to run (or run both)
print("Starting hyperparameter grid search...")
print(f"Total parameter combinations: {len(list(product(*param_grid.values())))}")

# Run grid search for Celeb-DF
results_celeb = run_grid_search(path_celeb_real, path_celeb_fake, "Celeb-DF")
df_celeb = display_results_summary(results_celeb, "Celeb-DF")

# Save results to CSV
df_celeb.to_csv('/Users/andreeabrad/StudioCode/eyes_notebook/celeb_df_independent_stages_results.csv', index=False)
print(f"\nResults saved to celeb_df_independent_stages_results.csv")

# Run grid search for DFD
results_dfd = run_grid_search(path_dfd_real, path_dfd_fake, "DFD")
df_dfd = display_results_summary(results_dfd, "DFD")

# Save results to CSV
df_dfd.to_csv('/Users/andreeabrad/StudioCode/eyes_notebook/dfd_independent_stages_results.csv', index=False)
print(f"\nResults saved to dfd_independent_stages_results.csv")

print("\n" + "="*140)
print("GRID SEARCH COMPLETE!")
print("="*140)
