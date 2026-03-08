import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION: HARDCODED FILE PATHS
# ==========================================
MAPPING_FILE = "./mapping/gene_info_table.csv"

# --- Classification Tasks ---
LABEL_TF = "./labels/tf_regulatory_range.csv"
LABEL_DOSAGE = "./labels/dosage_sensitivity_TFs.csv"
LABEL_BIV_NO_METH = "./labels/bivalent_vs_no_methyl.csv"
LABEL_BIV_LYS4 = "./labels/bivalent_vs_lys4_only.csv"
LABEL_N1_TARGET = "./labels/n1_target.csv"         
LABEL_N1_NETWORK = "./labels/n1_network.csv"       

# --- Prioritization Tasks (Time-Split) ---
DEE_KNOWN = "./labels/DEE_2022_09.txt"
DEE_NEW = "./labels/DEE_2025_03_vs_2022_09.txt"
CP_KNOWN = "./labels/CP-2022-09.txt"
CP_NEW = "./labels/CP-2025_06_vs_2022_09.txt"

# --- Baseline Results ---
BASELINE_CLASS = "./baseline/classification_results_pubmed_2022.csv"
BASELINE_PRIOR = "./baseline/prioritization_results_pubmed_2022.csv"
# ==========================================

def evaluate_task(X, y, task_name, tag):
    """Replicates 5-Fold CV for binary classification."""
    print(f"Evaluating Classification: {task_name} (N={len(y)})...")
    cv = StratifiedKFold(n_splits=5)
    
    roc_auc_lr, roc_auc_rf, roc_auc_mlp = [], [], []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Logistic Regression
        lr_model = LogisticRegression(class_weight='balanced', max_iter=500)
        lr_model.fit(X_train, y_train)
        roc_auc_lr.append(auc(*roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])[:2]))

        # Random Forest
        rf_model = RandomForestClassifier(random_state=2023, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        roc_auc_rf.append(auc(*roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])[:2]))

        # 10-Layer Deep Neural Network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256, 256, 128, 128, 64, 64, 32, 32, 16), max_iter=500, random_state=2023)
        mlp_model.fit(X_train_scaled, y_train)
        roc_auc_mlp.append(auc(*roc_curve(y_test, mlp_model.predict_proba(X_test_scaled)[:, 1])[:2]))

    mean_auc_lr = np.mean(roc_auc_lr)
    mean_auc_rf = np.mean(roc_auc_rf)
    mean_auc_mlp = np.mean(roc_auc_mlp)
    
    return [
        {'Task': task_name, 'Model': f'{tag} (LR)', 'ROC_AUC': mean_auc_lr},
        {'Task': task_name, 'Model': f'{tag} (RF)', 'ROC_AUC': mean_auc_rf},
        {'Task': task_name, 'Model': f'{tag} (Deep MLP)', 'ROC_AUC': mean_auc_mlp}
    ]

def calculate_prioritization_metrics(predictions_dict, new_genes):
    """Calculates Median Rank and Fold Enrichment @ 1% for new disease genes."""
    sorted_genes = sorted(predictions_dict.keys(), key=lambda g: predictions_dict[g], reverse=True)
    
    ranks = []
    for i, g in enumerate(sorted_genes):
        if g in new_genes:
            ranks.append(i + 1)
            
    mr = np.median(ranks) if ranks else 0
    
    total_genes = len(sorted_genes)
    total_new = len(new_genes)
    top_1_pct_cutoff = max(1, int(0.01 * total_genes))
    
    hits_in_top_1 = sum(1 for r in ranks if r <= top_1_pct_cutoff)
    expected_hits = 0.01 * total_new
    fe_1 = hits_in_top_1 / expected_hits if expected_hits > 0 else 0
    
    return mr, fe_1

def evaluate_time_split(emb_dict, known_path, new_path, task_name, tag):
    """Trains on Known genes vs Background, evaluates ranks of New genes."""
    if not os.path.exists(known_path) or not os.path.exists(new_path):
        return []
        
    print(f"Evaluating Time-Split Prioritization: {task_name}...")
    
    with open(known_path, 'r') as f:
        known_genes = set(line.strip() for line in f if line.strip())
    with open(new_path, 'r') as f:
        new_genes = set(line.strip() for line in f if line.strip())
        
    universe = list(emb_dict.keys())
    known_set = known_genes & set(universe)
    new_set = new_genes & set(universe)
    
    if not known_set or not new_set:
        print(f"  -> Skipping {task_name}: Insufficient overlap with embeddings.")
        return []

    # Train on Knowns (1) vs everything else (0)
    X = np.array([emb_dict[g] for g in universe])
    y = np.array([1 if g in known_set else 0 for g in universe])
    
    results = []
    
    # 1. LR
    lr = LogisticRegression(class_weight='balanced', max_iter=500)
    lr.fit(X, y)
    probs_lr = dict(zip(universe, lr.predict_proba(X)[:, 1]))
    mr_lr, fe_lr = calculate_prioritization_metrics(probs_lr, new_set)
    results.append({'Task': task_name, 'Model': f'{tag} (LR)', 'Median_Rank': mr_lr, 'FE_1%': fe_lr})
    
    # 2. RF
    rf = RandomForestClassifier(n_estimators=100, random_state=2023, class_weight='balanced')
    rf.fit(X, y)
    probs_rf = dict(zip(universe, rf.predict_proba(X)[:, 1]))
    mr_rf, fe_rf = calculate_prioritization_metrics(probs_rf, new_set)
    results.append({'Task': task_name, 'Model': f'{tag} (RF)', 'Median_Rank': mr_rf, 'FE_1%': fe_rf})
    
    # 3. Deep MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 256, 128, 128, 64, 64, 32, 32, 16), max_iter=500, random_state=2023)
    mlp.fit(X_scaled, y)
    probs_mlp = dict(zip(universe, mlp.predict_proba(X_scaled)[:, 1]))
    mr_mlp, fe_mlp = calculate_prioritization_metrics(probs_mlp, new_set)
    results.append({'Task': task_name, 'Model': f'{tag} (Deep MLP)', 'Median_Rank': mr_mlp, 'FE_1%': fe_mlp})
    
    return results

def get_genes_from_row(df, class_name):
    """Parses stringified lists from pickle-to-csv conversion."""
    try:
        val = df[df['gene_id'] == class_name]['label'].values[0]
        if isinstance(val, str):
            return [g.strip() for g in val.strip('[]').replace("'", "").replace('"', '').split(',') if g.strip()]
        return val 
    except IndexError:
        return []

def main(args):
    print(f"Loading custom embeddings from {args.embeddings}...")
    df_emb_raw = pd.read_csv(args.embeddings)
    
    # 1. Create a dictionary using the original Gene Symbols (for CP/DEE Prioritization)
    emb_dict_symbols = df_emb_raw.set_index('gene').T.to_dict('list')
    for k in emb_dict_symbols:
        emb_dict_symbols[k] = np.array(emb_dict_symbols[k])
    
    # 2. Translate to Ensembl IDs (for Hugging Face Classification Tasks)
    if os.path.exists(MAPPING_FILE):
        print(f"Translating gene symbols using {MAPPING_FILE}...")
        mapping_df = pd.read_csv(MAPPING_FILE)
        merged_df = pd.merge(df_emb_raw, mapping_df[['ensembl_id', 'gene_name']], left_on='gene', right_on='gene_name', how='inner')
        merged_df = merged_df.drop(columns=['gene', 'gene_name']).rename(columns={'ensembl_id': 'gene'})
        df_emb_ensembl = merged_df.drop_duplicates(subset=['gene'])
    else:
        df_emb_ensembl = df_emb_raw
        
    emb_dict_ensembl = df_emb_ensembl.set_index('gene').T.to_dict('list')
    for k in emb_dict_ensembl:
        emb_dict_ensembl[k] = np.array(emb_dict_ensembl[k])

    class_results = []
    prior_results = []
    np.random.seed(2023)

    # =========================================================
    # 1. RUN CLASSIFICATION TASKS (Using emb_dict_ensembl)
    # =========================================================
    if os.path.exists(LABEL_TF):
        df = pd.read_csv(LABEL_TF)
        long_genes, short_genes = get_genes_from_row(df, 'long_range'), get_genes_from_row(df, 'short_range')
        x_long, x_short = [emb_dict_ensembl[g] for g in long_genes if g in emb_dict_ensembl], [emb_dict_ensembl[g] for g in short_genes if g in emb_dict_ensembl]
        if x_long and x_short:
            class_results.extend(evaluate_task(np.concatenate((x_long, x_short)), np.concatenate((np.ones(len(x_long)), np.zeros(len(x_short)))), 'Long/Short Range TF', args.tag))

    if os.path.exists(LABEL_DOSAGE):
        df = pd.read_csv(LABEL_DOSAGE)
        s_genes, i_genes = get_genes_from_row(df, 'Dosage-sensitive TFs'), get_genes_from_row(df, 'Dosage-insensitive TFs')
        x_s, x_i = [emb_dict_ensembl[g] for g in s_genes if g in emb_dict_ensembl], [emb_dict_ensembl[g] for g in i_genes if g in emb_dict_ensembl]
        if x_s and x_i:
            class_results.extend(evaluate_task(np.concatenate((x_s, x_i)), np.concatenate((np.ones(len(x_s)), np.zeros(len(x_i)))), 'Dosage Sensitivity', args.tag))

    if os.path.exists(LABEL_BIV_NO_METH):
        df = pd.read_csv(LABEL_BIV_NO_METH)
        b_genes, n_genes = get_genes_from_row(df, 'bivalent'), get_genes_from_row(df, 'no_methylation')
        x_b, x_n = [emb_dict_ensembl[g] for g in b_genes if g in emb_dict_ensembl], [emb_dict_ensembl[g] for g in n_genes if g in emb_dict_ensembl]
        if x_b and x_n:
            class_results.extend(evaluate_task(np.concatenate((x_b, x_n)), np.concatenate((np.ones(len(x_b)), np.zeros(len(x_n)))), 'Bivalent vs Non-methylated', args.tag))

    if os.path.exists(LABEL_BIV_LYS4):
        df = pd.read_csv(LABEL_BIV_LYS4)
        b_genes, l_genes = get_genes_from_row(df, 'bivalent'), get_genes_from_row(df, 'lys4_only')
        x_b, x_l = [emb_dict_ensembl[g] for g in b_genes if g in emb_dict_ensembl], [emb_dict_ensembl[g] for g in l_genes if g in emb_dict_ensembl]
        if x_b and x_l:
            class_results.extend(evaluate_task(np.concatenate((x_b, x_l)), np.concatenate((np.ones(len(x_b)), np.zeros(len(x_l)))), 'Bivalent vs Lys-4', args.tag))

    if os.path.exists(LABEL_N1_TARGET):
        df = pd.read_csv(LABEL_N1_TARGET)
        a_genes, non_genes = get_genes_from_row(df, 'n1_activated'), get_genes_from_row(df, 'n1_nontarget')
        x_a, x_non = [emb_dict_ensembl[g] for g in a_genes if g in emb_dict_ensembl], [emb_dict_ensembl[g] for g in non_genes if g in emb_dict_ensembl]
        if x_a and x_non:
            class_results.extend(evaluate_task(np.concatenate((x_a, x_non)), np.concatenate((np.ones(len(x_a)), np.zeros(len(x_non)))), 'N1 Target', args.tag))

    if os.path.exists(LABEL_N1_NETWORK):
        df = pd.read_csv(LABEL_N1_NETWORK)
        c_genes, p_genes = get_genes_from_row(df, 'n1_central'), get_genes_from_row(df, 'n1_peripheral')
        x_c, x_p = [emb_dict_ensembl[g] for g in c_genes if g in emb_dict_ensembl], [emb_dict_ensembl[g] for g in p_genes if g in emb_dict_ensembl]
        if x_c and x_p:
            class_results.extend(evaluate_task(np.concatenate((x_c, x_p)), np.concatenate((np.ones(len(x_c)), np.zeros(len(x_p)))), 'N1 Network', args.tag))

    # =========================================================
    # 2. RUN PRIORITIZATION TASKS (Using emb_dict_symbols)
    # =========================================================
    prior_results.extend(evaluate_time_split(emb_dict_symbols, DEE_KNOWN, DEE_NEW, 'DEE Prioritization', args.tag))
    prior_results.extend(evaluate_time_split(emb_dict_symbols, CP_KNOWN, CP_NEW, 'CP Prioritization', args.tag))

    # =========================================================
    # SAVING & PLOTTING
    # =========================================================
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Save and Plot Classification ---
    if class_results:
        # Load baseline if it exists, combine with new results
        class_dfs = []
        if os.path.exists(BASELINE_CLASS):
            class_dfs.append(pd.read_csv(BASELINE_CLASS))
        class_dfs.append(pd.DataFrame(class_results))
        class_df = pd.concat(class_dfs, ignore_index=True)
        
        # Save ONLY the new results output to the output dir
        pd.DataFrame(class_results).to_csv(os.path.join(args.out_dir, f"classification_results_{args.tag}.csv"), index=False)
        
        class_df['Embedding'] = class_df['Model'].apply(lambda x: x.rsplit(' (', 1)[0])
        class_df['Classifier'] = class_df['Model'].apply(lambda x: x.rsplit(' (', 1)[1].replace(')', ''))

        # Dynamically build the palette: newly evaluated model gets Red, any baseline gets Blue
        custom_palette = {args.tag: '#de2d26'}
        for emb in class_df['Embedding'].unique():
            if emb != args.tag:
                custom_palette[emb] = '#3182bd'

        sns.set_style("whitegrid")
        g1 = sns.catplot(data=class_df, x='Classifier', y='ROC_AUC', hue='Embedding', col='Task', 
                         kind='bar', palette=custom_palette, col_wrap=3, height=3.5, aspect=1.2, sharey=True)
        g1.set(ylim=(0.4, 1.0))
        g1.fig.suptitle(f'Classification ROC-AUC - {args.tag} vs Baseline', y=1.05)
        
        # Force x-axis labels on all subplots
        for ax in g1.axes.flat:
            ax.tick_params(labelbottom=True)
            
        plt.savefig(os.path.join(args.out_dir, f"classification_plot_{args.tag}.png"), dpi=300, bbox_inches='tight')

    # --- Save and Plot Prioritization ---
    if prior_results:
        # Load baseline if it exists, combine with new results
        prior_dfs = []
        if os.path.exists(BASELINE_PRIOR):
            prior_dfs.append(pd.read_csv(BASELINE_PRIOR))
        prior_dfs.append(pd.DataFrame(prior_results))
        prior_df = pd.concat(prior_dfs, ignore_index=True)
        
        # Save ONLY the new results output to the output dir
        pd.DataFrame(prior_results).to_csv(os.path.join(args.out_dir, f"prioritization_results_{args.tag}.csv"), index=False)
        
        prior_df['Embedding'] = prior_df['Model'].apply(lambda x: x.rsplit(' (', 1)[0])
        prior_df['Classifier'] = prior_df['Model'].apply(lambda x: x.rsplit(' (', 1)[1].replace(')', ''))

        melted_prior = prior_df.melt(id_vars=['Task', 'Model', 'Embedding', 'Classifier'], 
                                     value_vars=['Median_Rank', 'FE_1%'], 
                                     var_name='Metric', value_name='Score')

        # Dynamically build the palette for the models: new evaluated model gets Red, baseline gets Blue
        prior_palette = {args.tag: '#de2d26'}
        for emb in prior_df['Embedding'].unique():
            if emb != args.tag:
                prior_palette[emb] = '#3182bd'

        # Map hue to 'Embedding' so Baseline and New are side-by-side
        g2 = sns.catplot(data=melted_prior, x='Classifier', y='Score', hue='Embedding', dodge=True,
                         col='Task', row='Metric', kind='bar', palette=prior_palette, 
                         sharey='row', height=4, aspect=1.2)
        
        g2.fig.suptitle(f'Time-Split Prioritization Metrics - {args.tag} vs Baseline', y=1.05)
        
        # Force x-axis labels on all subplots
        for ax in g2.axes.flat:
            ax.tick_params(labelbottom=True)
            
        plt.savefig(os.path.join(args.out_dir, f"prioritization_plot_{args.tag}.png"), dpi=300, bbox_inches='tight')

    # =========================================================
    # TERMINAL OUTPUT SUMMARY
    # =========================================================
    print(f"\nAll evaluations complete! Results saved to {args.out_dir}")
    
    if class_results:
        print("\n" + "="*60)
        print(f"FINAL CLASSIFICATION RESULTS: {args.tag}")
        print("="*60)
        formatted_class_df = pd.DataFrame(class_results)
        formatted_class_df['ROC_AUC'] = formatted_class_df['ROC_AUC'].map('{:.3f}'.format)
        print(formatted_class_df.to_string(index=False))

    if prior_results:
        print("\n" + "="*60)
        print(f"FINAL PRIORITIZATION RESULTS: {args.tag}")
        print("="*60)
        formatted_prior_df = pd.DataFrame(prior_results)
        formatted_prior_df['FE_1%'] = formatted_prior_df['FE_1%'].map('{:.2f}'.format)
        formatted_prior_df['Median_Rank'] = formatted_prior_df['Median_Rank'].map('{:.0f}'.format)
        print(formatted_prior_df.to_string(index=False))
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate custom embeddings against baseline.")
    parser.add_argument("--embeddings", required=True, help="CSV containing custom embeddings.")
    parser.add_argument("--tag", required=True, help="Tag for this run.")
    parser.add_argument("--out_dir", default="./outputs", help="Directory to save plots and CSVs.")
    args = parser.parse_args()
    main(args)