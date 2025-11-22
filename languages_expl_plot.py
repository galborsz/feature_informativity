import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests
import shutil

# Define inventories
inventories = ["HC", "SPE", "JFH"]

# Load all data
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Load pb_languages_formatted.csv to get language families and organize by language
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv")

print("\n" + "=" * 60)
print("Creating Language × Feature System MDL Heatmap")
print("=" * 60)

# Collect average MDL per language for each inventory
language_mdl_data = []

for inv in inventories:
    print(f"\nProcessing inventory: {inv}")
    
    for language, lang_data in all_data[inv].items():
        if "min_lengths" in lang_data:
            min_lengths = lang_data["min_lengths"]
            
            # Calculate average MDL for this language
            mdl_values_lang = list(min_lengths.values())
            if mdl_values_lang:
                avg_mdl = np.mean(mdl_values_lang)
                
                # Get language family from pb_languages
                lang_rows = pb_languages[pb_languages['language'] == language]
                if not lang_rows.empty:
                    family = lang_rows.iloc[0]['family']
                else:
                    family = 'Unknown'
                
                language_mdl_data.append({
                    'language': language,
                    'family': family,
                    'inventory': inv,
                    'avg_mdl': avg_mdl
                })

# Convert to DataFrame
df_lang_mdl = pd.DataFrame(language_mdl_data)

print(f"\nCollected MDL data for {len(df_lang_mdl)} language-inventory combinations")
print(f"Unique languages: {df_lang_mdl['language'].nunique()}")
print(f"Feature systems: {df_lang_mdl['inventory'].unique()}")

# Pivot to create matrix: rows = languages, columns = inventories
heatmap_data = df_lang_mdl.pivot_table(
    index='language',
    columns='inventory',
    values='avg_mdl',
    aggfunc='first'  # In case of duplicates, use first value
)

# Reorder columns to match inventory order
heatmap_data = heatmap_data[inventories]

print(f"\nHeatmap shape: {heatmap_data.shape}")
print(f"Rows (languages): {heatmap_data.shape[0]}")
print(f"Columns (feature systems): {heatmap_data.shape[1]}")
print(f"Data not normalized (correlation distance is scale-invariant)")

# Get family information for reference (but not in labels)
language_to_family = {}
for lang in heatmap_data.index:
    lang_rows = pb_languages[pb_languages['language'] == lang]
    if not lang_rows.empty:
        language_to_family[lang] = lang_rows.iloc[0]['family']
    else:
        language_to_family[lang] = 'Unknown'

# Perform hierarchical clustering
print("\nPerforming hierarchical clustering on MDL values...")
# Drop rows with any NaN values
heatmap_data_clean = heatmap_data.dropna()
print(f"Languages with complete data (before filtering): {len(heatmap_data_clean)}")

# Identify rows with zero or near-zero variance (correlation undefined for constant rows)
variance_threshold = 1e-10
const_mask = heatmap_data_clean.std(axis=1) <= variance_threshold
heatmap_data_const = heatmap_data_clean[const_mask]
heatmap_data_var = heatmap_data_clean[~const_mask]

print(f"Languages with constant MDL across all systems: {len(heatmap_data_const)}")
print(f"Languages with variable MDL: {len(heatmap_data_var)}")

if len(heatmap_data_const) > 0:
    print(f"  Constant languages: {', '.join(heatmap_data_const.index.tolist()[:10])}{'...' if len(heatmap_data_const) > 10 else ''}")

# Compute pairwise correlation distance (1 - Pearson r) for variable rows only
# This clusters by pattern shape, not absolute magnitude
print("Computing correlation distance matrix for variable languages...")
D = pdist(heatmap_data_var.values, metric='correlation')

# Check for NaN or inf values in distance matrix
if np.any(~np.isfinite(D)):
    print(f"Warning: Found {np.sum(~np.isfinite(D))} non-finite values in distance matrix")
    print("Attempting to handle by replacing non-finite values with max distance...")
    D = np.where(np.isfinite(D), D, np.nanmax(D[np.isfinite(D)]))

# Compute linkage matrix using average linkage (pattern-based clustering)
print("Computing linkage with average method...")
row_linkage = linkage(D, method='average')

# Evaluate different numbers of clusters using silhouette score
print("\nEvaluating silhouette scores for K=5 to K=12...")
k_values = range(2, 9)
silhouette_scores = []

for k in k_values:
    cluster_labels_test = fcluster(row_linkage, k, criterion='maxclust')
    # Use correlation distance for silhouette score calculation
    sil_score = silhouette_score(heatmap_data_var.values, cluster_labels_test, metric='correlation')
    silhouette_scores.append(sil_score)
    print(f"  K={k}: Silhouette Score = {sil_score:.4f}")

# Plot silhouette scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(k_values)

# Add value labels on points
for k, score in zip(k_values, silhouette_scores):
    ax.text(k, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=300, bbox_inches='tight')
print("\n✓ Silhouette scores plot saved as: silhouette_scores.png")
plt.close()

# Cut dendrogram to form exactly 8 clusters (for variable languages)
n_clusters_var = 2
cluster_labels_var = fcluster(row_linkage, n_clusters_var, criterion='maxclust')

# Create a mapping of languages to cluster IDs for variable rows
lang_to_cluster_var = dict(zip(heatmap_data_var.index, cluster_labels_var))

print(f"Number of clusters (for variable languages): {n_clusters_var}")
print(f"Clustering method: Correlation distance + Average linkage (pattern-based)")

# Group languages by cluster (including constant rows as a separate category)
clusters_dict = {}
for lang, cluster_id in lang_to_cluster_var.items():
    if cluster_id not in clusters_dict:
        clusters_dict[cluster_id] = []
    clusters_dict[cluster_id].append(lang)

# Add constant languages as a special cluster
if len(heatmap_data_const) > 0:
    clusters_dict['constant'] = heatmap_data_const.index.tolist()

# Print cluster composition
print("\nCluster composition:")
for cluster_id in sorted(clusters_dict.keys(), key=lambda x: (isinstance(x, str), x)):
    langs = sorted(clusters_dict[cluster_id])
    print(f"  {'Constant (all MDL equal)' if cluster_id == 'constant' else f'Cluster {cluster_id}'}: {len(langs)} languages - {', '.join(langs[:5])}{'...' if len(langs) > 5 else ''}")

# Create output directory for cluster plots
cluster_dir = "language_clusters"
if os.path.exists(cluster_dir):
    # Clean existing folder
    shutil.rmtree(cluster_dir)
os.makedirs(cluster_dir)

# Generate separate heatmap for each category
print("\nGenerating separate heatmaps for each cluster...")

# Sort cluster keys: integers first (1-8), then 'constant' at the end
sorted_cluster_ids = sorted([k for k in clusters_dict.keys() if k != 'constant']) + (['constant'] if 'constant' in clusters_dict else [])

for cluster_id in sorted_cluster_ids:
    langs_in_cluster = sorted(clusters_dict[cluster_id])
    cluster_data = heatmap_data_clean.loc[langs_in_cluster]
    
    # Create figure for this cluster
    fig_height = max(6, len(langs_in_cluster) * 0.4)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Create heatmap
    sns.heatmap(
        cluster_data,
        cmap='RdBu_r',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Average MDL'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Feature System', fontsize=12, fontweight='bold')
    ax.set_ylabel('Language', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['HC', 'SPE', 'JFH'], rotation=45, ha='right', fontsize=10)
    
    # Set title based on cluster type
    if cluster_id == 'constant':
        ax.set_title(f'Constant (all MDL equal) - {len(langs_in_cluster)} languages', 
                     fontsize=12, fontweight='bold', pad=15)
        safe_name = 'constant'
    else:
        ax.set_title(f'Cluster {cluster_id} ({len(langs_in_cluster)} languages)', 
                     fontsize=12, fontweight='bold', pad=15)
        safe_name = f'cluster_{cluster_id:02d}'
    
    plt.tight_layout()
    
    # Save cluster heatmap with descriptive name
    filename = os.path.join(cluster_dir, f'{safe_name}_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

print(f"\n✓ Category heatmaps saved in folder: {cluster_dir}")

# Define statistical test functions
def permutation_pvalue(x, y, n_perm=5000, seed=0):
    """Two-sample Monte-Carlo permutation test using median difference as test statistic."""
    rng = np.random.default_rng(seed)
    x = np.array(x)
    y = np.array(y)
    obs_diff = abs(np.median(x) - np.median(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x_perm = pooled[:n_x]
        y_perm = pooled[n_x:]
        if abs(np.median(x_perm) - np.median(y_perm)) >= obs_diff:
            count += 1
    return (count + 1) / (n_perm + 1)

def rank_biserial_unpaired(x, y):
    """Compute rank-biserial effect size for unpaired samples."""
    scores = []
    for xi in x:
        for yj in y:
            if xi > yj:
                scores.append(1.0)
            elif xi == yj:
                scores.append(0.5)
            else:
                scores.append(0.0)
    A = np.mean(scores)
    return 2 * A - 1  # in [-1,1]

def p_to_stars(p):
    """Map corrected p-value to significance stars."""
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return ''

# Generate violin plots for each cluster
print("\nGenerating violin plots for each cluster...")

# Calculate global y-axis range for all MDL values across all clusters
all_mdl_values = []
for feature_system in inventories:
    all_mdl_values.extend(heatmap_data_clean[feature_system].dropna().values)
y_min = np.min(all_mdl_values)
y_max = np.max(all_mdl_values)
y_margin = (y_max - y_min) * 0.05  # Add 5% margin
y_axis_range = (1.5, 3.4) #(y_min - y_margin, y_max + y_margin)

for cluster_id in sorted_cluster_ids:
    langs_in_cluster = sorted(clusters_dict[cluster_id])
    cluster_data = heatmap_data_clean.loc[langs_in_cluster]
    
    # Prepare data for violin plot: reshape to long format
    violin_data_list = []
    for feature_system in inventories:
        mdl_values = cluster_data[feature_system].values
        for mdl_val in mdl_values:
            violin_data_list.append({
                'Feature System': feature_system,
                'Average MDL': mdl_val
            })
    
    violin_df = pd.DataFrame(violin_data_list)
    
    # Extract data per feature system for statistical tests
    hc_data = cluster_data['HC'].values
    spe_data = cluster_data['SPE'].values
    jfh_data = cluster_data['JFH'].values
    
    # Perform pairwise Monte-Carlo permutation tests
    pairs_to_test = [
        ('HC', 'SPE', hc_data, spe_data, 0, 1),
        ('SPE', 'JFH', spe_data, jfh_data, 1, 2),
        ('HC', 'JFH', hc_data, jfh_data, 0, 2)
    ]
    
    pvals_raw = []
    effect_sizes = []
    for inv1, inv2, data1, data2, _, _ in pairs_to_test:
        p_val = permutation_pvalue(data1, data2, n_perm=10000, seed=42)
        r_pair = rank_biserial_unpaired(data1, data2)
        pvals_raw.append(p_val)
        effect_sizes.append(r_pair)
    
    # Apply Benjamini-Hochberg FDR correction
    pvals_array = np.array(pvals_raw)
    rejected, pvals_corrected, _, _ = multipletests(pvals_array, alpha=0.01, method='fdr_bh')
    
    # Create figure for violin plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create violin plot with median line
    sns.violinplot(data=violin_df, x='Feature System', y='Average MDL', ax=ax, 
                   palette='Set2', inner=None, linewidth=1.5)
    
    # Add median lines for each feature system
    medians = [np.median(cluster_data[inv].values) for inv in inventories]
    for i, median_val in enumerate(medians):
        ax.hlines(median_val, i - 0.4, i + 0.4, colors='darkred', linewidth=2.5, label='Median' if i == 0 else '')
    
    # Set consistent y-axis range across all clusters
    ax.set_ylim(y_axis_range)
    
    # Set labels and title
    ax.set_xlabel('Feature System', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average MDL', fontsize=12, fontweight='bold')
    
    # Set title based on cluster type
    if cluster_id == 'constant':
        ax.set_title(f'Constant (all MDL equal) - {len(langs_in_cluster)} languages', 
                     fontsize=12, fontweight='bold', pad=15)
        safe_name = 'constant'
    else:
        ax.set_title(f'Cluster {cluster_id} ({len(langs_in_cluster)} languages)', 
                     fontsize=12, fontweight='bold', pad=15)
        safe_name = f'cluster_{cluster_id:02d}'
    
    # Add p-value annotation box if sample size is sufficient
    if len(hc_data) > 2 and len(spe_data) > 2 and len(jfh_data) > 2:
        annotation_text = '' # Pairwise Comparisons\n(corrected p-values)\n
        for i, (inv1, inv2, data1, data2, pos1, pos2) in enumerate(pairs_to_test):
            p_adj = pvals_corrected[i]
            stars = p_to_stars(p_adj)
            annotation_text += f'{inv1} vs {inv2}: {p_adj:.3f} {stars}\n'
        
        annotation_text = annotation_text.rstrip()
        ax.text(0.98, 0.97, annotation_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))
    
    plt.tight_layout()
    
    # Save violin plot with descriptive name
    filename = os.path.join(cluster_dir, f'{safe_name}_violin.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistical results for this cluster
    if cluster_id == 'constant':
        print(f"\n  Constant cluster statistics:")
    else:
        print(f"\n  Cluster {cluster_id} statistics:")
    
    for i, (inv1, inv2, data1, data2, _, _) in enumerate(pairs_to_test):
        p_adj = pvals_corrected[i]
        r_pair = effect_sizes[i]
        stars = p_to_stars(p_adj)
        print(f"    {inv1} vs {inv2}: p={p_adj:.4g}, r={r_pair:.3f} {stars}")
    
    print(f"  Saved: {filename}")

print(f"✓ Violin plots saved in folder: {cluster_dir}")

# Print summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print("\nAverage MDL by Feature System:")
for inv in inventories:
    inv_data = heatmap_data[inv].dropna()
    print(f"\n  {inv}:")
    print(f"    Mean: {inv_data.mean():.4f}")
    print(f"    Median: {inv_data.median():.4f}")
    print(f"    Min: {inv_data.min():.4f}")
    print(f"    Max: {inv_data.max():.4f}")
    print(f"    Std Dev: {inv_data.std():.4f}")

print("\nLanguage-wise statistics (average across all feature systems):")
heatmap_data['Mean_MDL'] = heatmap_data[inventories].mean(axis=1)
heatmap_data_sorted = heatmap_data.sort_values('Mean_MDL', ascending=False)

print("\nTop 10 languages with highest average MDL:")
for idx, (lang, row) in enumerate(heatmap_data_sorted.head(10).iterrows(), 1):
    family = language_to_family.get(lang, 'Unknown')
    print(f"  {idx:2d}. {lang:20s} ({family:20s}): {row['Mean_MDL']:.4f}")

print("\nTop 10 languages with lowest average MDL:")
for idx, (lang, row) in enumerate(heatmap_data_sorted.tail(10).iloc[::-1].iterrows(), 1):
    family = language_to_family.get(lang, 'Unknown')
    print(f"  {idx:2d}. {lang:20s} ({family:20s}): {row['Mean_MDL']:.4f}")

print("\nClustering dendrogram interpretation:")
print("  - Languages are ordered by hierarchical clustering (Correlation distance + Average linkage)")
print("  - Clustering is pattern-based: groups languages with similar MDL profiles")
print("  - Ignores absolute magnitude differences, focuses on relative patterns across HC/SPE/JFH")
print("  - Data is split into 5 clusters based on dendrogram structure")
print("  - Each cluster shown in a separate heatmap for readability")

print("\n" + "=" * 60)

# ============================================================
# GLOBAL VIOLIN PLOT: All languages, three feature systems
# ============================================================
print("\nGenerating global violin plot across all languages...")

# Prepare data for global violin plot: reshape to long format
global_violin_data_list = []
for inv in inventories:
    mdl_values = heatmap_data[inv].dropna().values
    for mdl_val in mdl_values:
        global_violin_data_list.append({
            'Feature System': inv,
            'Average MDL': mdl_val
        })

global_violin_df = pd.DataFrame(global_violin_data_list)

# Extract data per feature system for statistical tests
hc_global = heatmap_data['HC'].dropna().values
spe_global = heatmap_data['SPE'].dropna().values
jfh_global = heatmap_data['JFH'].dropna().values

# Perform pairwise Monte-Carlo permutation tests
pairs_global = [
    ('HC', 'SPE', hc_global, spe_global, 0, 1),
    ('SPE', 'JFH', spe_global, jfh_global, 1, 2),
    ('HC', 'JFH', hc_global, jfh_global, 0, 2)
]

pvals_raw_global = []
effect_sizes_global = []
for inv1, inv2, data1, data2, _, _ in pairs_global:
    p_val = permutation_pvalue(data1, data2, n_perm=5000, seed=42)
    r_pair = rank_biserial_unpaired(data1, data2)
    pvals_raw_global.append(p_val)
    effect_sizes_global.append(r_pair)

# Apply Benjamini-Hochberg FDR correction
pvals_array_global = np.array(pvals_raw_global)
rejected_global, pvals_corrected_global, _, _ = multipletests(pvals_array_global, alpha=0.05, method='fdr_bh')

# Create global violin plot
fig, ax = plt.subplots(figsize=(10, 7))

# Create violin plot with median line
sns.violinplot(data=global_violin_df, x='Feature System', y='Average MDL', ax=ax, 
               palette='Set2', inner=None, linewidth=1.5)

# Add median lines for each feature system
medians_global = [np.median(hc_global), np.median(spe_global), np.median(jfh_global)]
for i, median_val in enumerate(medians_global):
    ax.hlines(median_val, i - 0.4, i + 0.4, colors='darkred', linewidth=2.5, label='Median' if i == 0 else '')

# Set labels and title
ax.set_xlabel('Feature System', fontsize=12, fontweight='bold')
ax.set_ylabel('Average MDL', fontsize=12, fontweight='bold')
# ax.set_title(f'All Languages ({len(heatmap_data)} languages) - Feature System Comparison', 
#              fontsize=14, fontweight='bold', pad=15)

# Add p-value annotation box if sample size is sufficient
if len(hc_global) > 2 and len(spe_global) > 2 and len(jfh_global) > 2:
    # Find y-axis range for adjustment if needed
    max_y_global = max([np.max(hc_global), np.max(spe_global), np.max(jfh_global)])
    min_y_global = min([np.min(hc_global), np.min(spe_global), np.min(jfh_global)])
    data_range_global = max_y_global - min_y_global
    
    annotation_text_global = ''
    for i, (inv1, inv2, data1, data2, pos1, pos2) in enumerate(pairs_global):
        p_adj = pvals_corrected_global[i]
        stars = p_to_stars(p_adj)
        annotation_text_global += f'{inv1} vs {inv2}: {p_adj:.3f} {stars}\n'
    
    annotation_text_global = annotation_text_global.rstrip()
    
    # Set y-axis with some margin for readability
    ax.set_ylim(min_y_global - data_range_global * 0.05, 
                max_y_global + data_range_global * 0.05)
    
    # Add annotation box
    ax.text(0.98, 0.97, annotation_text_global, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))

plt.tight_layout()
plt.savefig('global_violin_all_languages.png', dpi=300, bbox_inches='tight')
plt.close()

# Print global statistical results
print("\n" + "=" * 60)
print("Global Analysis: All Languages, All Feature Systems")
print("=" * 60)

print("\nSample sizes:")
print(f"  HC: {len(hc_global)} languages")
print(f"  SPE: {len(spe_global)} languages")
print(f"  JFH: {len(jfh_global)} languages")

print("\nMedians and IQRs:")
for inv, data in [('HC', hc_global), ('SPE', spe_global), ('JFH', jfh_global)]:
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    print(f"  {inv}: Median = {median:.4f}, IQR = [{q1:.4f}, {q3:.4f}] (range: {iqr:.4f})")

print("\nMonte-Carlo permutation tests (raw p-values):")
for i, (inv1, inv2, data1, data2, _, _) in enumerate(pairs_global):
    print(f"  {inv1} vs {inv2}: p = {pvals_raw_global[i]:.4f}")

print("\nBenjamini-Hochberg FDR-corrected p-values:")
for i, (inv1, inv2, data1, data2, _, _) in enumerate(pairs_global):
    p_adj = pvals_corrected_global[i]
    r_pair = effect_sizes_global[i]
    stars = p_to_stars(p_adj)
    print(f"  {inv1} vs {inv2}: p_corrected = {p_adj:.4f}, r = {r_pair:.3f} {stars}")

print(f"\n✓ Global violin plot saved as: global_violin_all_languages.png")
print("\n" + "=" * 60)

# ============================================================
# RANDOM FOREST + SHAP: Phoneme Presence → Language Clusters
# ============================================================

print("\n" + "=" * 60)
print("Random Forest + SHAP: Predicting Language Clusters from Phoneme Presence")
print("=" * 60)

print("\nReconstucting cluster assignments...")

# Use existing cluster data
print(f"Clusters created:")
for cluster_id in sorted(clusters_dict.keys(), key=lambda x: (isinstance(x, str), x)):
    count = len(clusters_dict[cluster_id])
    print(f"  {'Constant' if cluster_id == 'constant' else f'Cluster {cluster_id}'}: {count} languages")

# Build dataset: languages × phoneme presence + cluster label
print("\nBuilding phoneme presence dataset...")

# Collect all unique phonemes across all languages
all_phonemes_rf = set()
for lang in heatmap_data_clean.index:
    lang_rows = pb_languages[pb_languages['language'] == lang]
    if not lang_rows.empty:
        inventory_str = lang_rows.iloc[0]["core inventory"]
        inventory_str = inventory_str.strip('[]')
        phonemes = [p.strip().strip('\'"') for p in inventory_str.split(',')]
        phonemes = [p for p in phonemes if p and p != ""]
        all_phonemes_rf.update(phonemes)

all_phonemes_rf = sorted(list(all_phonemes_rf))
print(f"Total unique phonemes: {len(all_phonemes_rf)}")

# Create language-level dataset
language_phoneme_data = []

for language in heatmap_data_clean.index:
    # Find which cluster this language belongs to
    cluster_id = None
    for cid, langs in clusters_dict.items():
        if language in langs:
            cluster_id = cid
            break
    
    if cluster_id is None:
        continue
    
    # Get phoneme inventory for this language
    lang_rows = pb_languages[pb_languages['language'] == language]
    if lang_rows.empty:
        continue
    
    inventory_str = lang_rows.iloc[0]["core inventory"]
    inventory_str = inventory_str.strip('[]')
    phonemes_in_lang = set(p.strip().strip('\'"') for p in inventory_str.split(','))
    phonemes_in_lang = set(p for p in phonemes_in_lang if p and p != "")
    
    # Create binary feature vector
    row = {'language': language, 'cluster': cluster_id}
    for phoneme in all_phonemes_rf:
        row[phoneme] = 1 if phoneme in phonemes_in_lang else 0
    
    language_phoneme_data.append(row)

df_lang_phonemes = pd.DataFrame(language_phoneme_data)

print(f"\nDataset shape: {df_lang_phonemes.shape}")
print(f"  Languages: {len(df_lang_phonemes)}")
print(f"  Phoneme features: {len(all_phonemes_rf)}")
print(f"  Cluster distribution:")
for cid in sorted(df_lang_phonemes['cluster'].unique(), key=lambda x: (isinstance(x, str), x)):
    count = (df_lang_phonemes['cluster'] == cid).sum()
    print(f"    {'Constant' if cid == 'constant' else f'Cluster {cid}'}: {count} languages")

# Prepare X and y for Random Forest
X_lang_rf = df_lang_phonemes[[col for col in df_lang_phonemes.columns if col not in ['language', 'cluster']]]
y_lang_rf = df_lang_phonemes['cluster']

# Convert all cluster labels to strings for consistent encoding
y_lang_rf = y_lang_rf.astype(str)

# Encode cluster labels (convert to numeric)
from sklearn.preprocessing import LabelEncoder
le_cluster = LabelEncoder()
y_lang_rf_encoded = le_cluster.fit_transform(y_lang_rf)

print(f"\nCluster encoding: {dict(zip(le_cluster.classes_, le_cluster.transform(le_cluster.classes_)))}")

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
from sklearn.ensemble import RandomForestClassifier
rf_cluster_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

rf_cluster_model.fit(X_lang_rf, y_lang_rf_encoded)

# Evaluate on training data
train_score = rf_cluster_model.score(X_lang_rf, y_lang_rf_encoded)
print(f"Training accuracy: {train_score:.4f}")

# Get predictions
y_pred_encoded = rf_cluster_model.predict(X_lang_rf)
y_pred = le_cluster.inverse_transform(y_pred_encoded)

# Print per-class accuracy
print("\nPer-cluster training accuracy:")
for cluster_id in sorted(le_cluster.classes_, key=lambda x: (isinstance(x, str), x)):
    idx = le_cluster.transform([cluster_id])[0]
    mask = y_lang_rf_encoded == idx
    if mask.sum() > 0:
        acc = (y_pred_encoded[mask] == idx).mean()
        print(f"  {'Constant' if cluster_id == 'constant' else f'Cluster {cluster_id}'}: {acc:.4f}")

# Compute SHAP values
print("\nComputing SHAP values for cluster prediction...")
import shap
explainer_cluster = shap.TreeExplainer(rf_cluster_model)
shap_values_cluster = explainer_cluster(X_lang_rf)

# Get base value
base_value_cluster = explainer_cluster.expected_value
if isinstance(base_value_cluster, np.ndarray):
    base_value_cluster = base_value_cluster[0]
print(f"Base value (expected prediction): {base_value_cluster:.4f}")

# Generate SHAP summary bar plot
print("\nGenerating SHAP summary bar plot...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_cluster, X_lang_rf, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_summary_bar_cluster_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_summary_bar_cluster_prediction.png")

# # Generate SHAP beeswarm summary plot
# print("\nGenerating SHAP beeswarm summary plot...")
# fig, ax = plt.subplots(figsize=(12, 10))
# shap.summary_plot(shap_values_cluster, X_lang_rf, show=False)
# plt.tight_layout()
# plt.savefig('shap_summary_beeswarm_cluster_prediction.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("  Saved: shap_summary_beeswarm_cluster_prediction.png")

# # Print top impactful phonemes
# print("\nTop 15 phonemes by mean absolute SHAP value:")
# mean_abs_shap_cluster = np.mean(np.abs(shap_values_cluster.values), axis=0)
# top_indices_cluster = np.argsort(mean_abs_shap_cluster)[-15:][::-1]
# for rank, idx in enumerate(top_indices_cluster, 1):
#     phoneme = X_lang_rf.columns[idx]
#     print(f"  {rank:2d}. {phoneme:8s}: {mean_abs_shap_cluster[idx]:.6f}")

# Summary
print("\n" + "=" * 60)
print("Random Forest + SHAP Cluster Prediction Complete")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  Training Accuracy: {train_score:.4f}")
print(f"  Number of languages: {len(df_lang_phonemes)}")
print(f"  Number of phoneme features: {len(all_phonemes_rf)}")
print(f"  Number of clusters: {len(le_cluster.classes_)}")

print("\n" + "=" * 60)
