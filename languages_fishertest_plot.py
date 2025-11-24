import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import unicodedata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, fisher_exact
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
print("\n[OK] Silhouette scores plot saved as: silhouette_scores.png")
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
    fig, ax = plt.subplots(figsize=(9, fig_height))
    
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

print(f"\n[OK] Category heatmaps saved in folder: {cluster_dir}")

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
    
    # Calculate scale factor based on cluster size
    # Find the maximum cluster size to normalize
    max_cluster_size = max(len(clusters_dict[cid]) for cid in sorted_cluster_ids)
    cluster_size = len(langs_in_cluster)
    # Scale violin width from 0.3 to 1.0 based on cluster size
    violin_scale = 0.3 + (cluster_size / max_cluster_size) * 0.7
    
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
    # Use scale parameter to make violin width proportional to cluster size
    sns.violinplot(data=violin_df, x='Feature System', y='Average MDL', ax=ax, 
                   palette='Set2', inner=None, linewidth=1.5, scale_hue=False, width=violin_scale)
    
    # Add individual sample points
    sns.stripplot(data=violin_df, x='Feature System', y='Average MDL', ax=ax,
                  color='black', alpha=0.4, size=4, jitter=True)
    
    # Add median lines for each feature system with width matching violin scale
    medians = [np.median(cluster_data[inv].values) for inv in inventories]
    median_width = violin_scale * 0.8  # Scale median line width proportionally
    for i, median_val in enumerate(medians):
        ax.hlines(median_val, i - (violin_scale/2), i + (violin_scale/2), colors='darkred', linewidth=2.5, label='Median' if i == 0 else '')
    
    # Set consistent y-axis range across all clusters
    ax.set_ylim(1.5, 3.4)
    
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

print(f"[OK] Violin plots saved in folder: {cluster_dir}")

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

# Set y-axis with fixed range
ax.set_ylim(1.5, 3.4)

# Create violin plot with median line
sns.violinplot(data=global_violin_df, x='Feature System', y='Average MDL', ax=ax, 
               palette='Set2', inner=None, linewidth=1.5, scale='width')

# Add individual sample points
sns.stripplot(data=global_violin_df, x='Feature System', y='Average MDL', ax=ax,
              color='black', alpha=0.4, size=4, jitter=True)

# Add median lines for each feature system with standard width for global plot
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

print(f"\n[OK] Global violin plot saved as: global_violin_all_languages.png")
print("\n" + "=" * 60)

# ============================================================
# LOGISTIC REGRESSION WITH LASSO (L1): Phoneme Presence → Language Clusters
# ============================================================

print("\n" + "=" * 60)
print("Chi-Squared Test: Phoneme Presence vs. Language Clusters")
print("=" * 60)

print("\nBuilding phoneme presence dataset...")

# Collect all unique phonemes across all languages
all_phonemes = set()
for lang in heatmap_data_clean.index:
    lang_rows = pb_languages[pb_languages['language'] == lang]
    if not lang_rows.empty:
        inventory_str = lang_rows.iloc[0]["core inventory"]
        # Remove brackets and split by commas, then clean quotes
        inventory_str = inventory_str.strip('[]').strip()
        phonemes = [p.strip().strip('\'"') for p in inventory_str.split(',')]
        phonemes = [p for p in phonemes if p and p != ""]
        all_phonemes.update(phonemes)

all_phonemes = sorted(list(all_phonemes))
print(f"Total unique phonemes: {len(all_phonemes)}")

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
    # Remove brackets and split by commas, then clean quotes
    inventory_str = inventory_str.strip('[]').strip()
    phonemes_in_lang = set(p.strip().strip('\'"') for p in inventory_str.split(','))
    phonemes_in_lang = set(p for p in phonemes_in_lang if p and p != "")
    
    # Create binary feature vector
    row = {'language': language, 'cluster': cluster_id}
    for phoneme in all_phonemes:
        row[phoneme] = 1 if phoneme in phonemes_in_lang else 0
    
    language_phoneme_data.append(row)

df_lang = pd.DataFrame(language_phoneme_data)

print(f"\nDataset shape: {df_lang.shape}")
print(f"  Languages: {len(df_lang)}")
print(f"  Phoneme features: {len(all_phonemes)}")
print(f"  Cluster distribution:")
for cid in sorted(df_lang['cluster'].unique(), key=lambda x: (isinstance(x, str), x)):
    count = (df_lang['cluster'] == cid).sum()
    print(f"    {'Constant' if cid == 'constant' else f'Cluster {cid}'}: {count} languages")

# Perform contingency table analysis for each phoneme
print("\nPerforming contingency table analysis for each phoneme...")

# Filter out constant cluster
df_lang_filtered = df_lang[df_lang['cluster'] != 'constant'].copy()
cluster_ids_filtered = sorted([c for c in df_lang_filtered['cluster'].unique() if c != 'constant'])

print(f"Number of clusters (excluding constant): {len(cluster_ids_filtered)}")
for cid in cluster_ids_filtered:
    count = (df_lang_filtered['cluster'] == cid).sum()
    print(f"  Cluster {cid}: {count} languages")

if len(cluster_ids_filtered) < 1:
    print("\nWarning: No clusters available after filtering.")
    contingency_results = []
elif len(cluster_ids_filtered) == 2:
    # Apply Fisher's exact test for 2 clusters
    print("\n[OK] Exactly 2 clusters detected. Applying Fisher's exact test...")
    
    cluster_1, cluster_2 = cluster_ids_filtered[0], cluster_ids_filtered[1]
    print(f"Comparing Cluster {cluster_1} vs Cluster {cluster_2}")
    
    contingency_results = []
    
    for phoneme in all_phonemes:
        # Get languages in each cluster
        cluster_1_langs = df_lang_filtered[df_lang_filtered['cluster'] == cluster_1]
        cluster_2_langs = df_lang_filtered[df_lang_filtered['cluster'] == cluster_2]
        
        # Build 2x2 contingency table
        c1_has = (cluster_1_langs[phoneme] == 1).sum()
        c1_no = (cluster_1_langs[phoneme] == 0).sum()
        c2_has = (cluster_2_langs[phoneme] == 1).sum()
        c2_no = (cluster_2_langs[phoneme] == 0).sum()
        
        contingency_table_2x2 = [[c1_has, c1_no], [c2_has, c2_no]]
        
        # Perform Fisher's exact test
        try:
            oddsratio, p_value = fisher_exact(contingency_table_2x2)
        except:
            oddsratio = np.nan
            p_value = np.nan
        
        # Calculate proportions
        c1_total = c1_has + c1_no
        c2_total = c2_has + c2_no
        c1_proportion_str = f"{c1_has}/{c1_total} = {(c1_has / c1_total * 100):.1f}%" if c1_total > 0 else "N/A"
        c2_proportion_str = f"{c2_has}/{c2_total} = {(c2_has / c2_total * 100):.1f}%" if c2_total > 0 else "N/A"
        
        # Store results
        result_dict = {'phoneme': phoneme}
        result_dict[f'cluster_{cluster_1}_has'] = c1_has
        result_dict[f'cluster_{cluster_1}_no'] = c1_no
        result_dict[f'cluster_{cluster_2}_has'] = c2_has
        result_dict[f'cluster_{cluster_2}_no'] = c2_no
        result_dict[f'proportion_cluster_{cluster_1}'] = c1_proportion_str
        result_dict[f'proportion_cluster_{cluster_2}'] = c2_proportion_str
        result_dict['oddsratio'] = oddsratio
        result_dict['p_value'] = p_value
        
        contingency_results.append(result_dict)
    
else:
    # More than 2 clusters: create per-cluster contingency counts (no Fisher's exact test)
    print(f"\n[OK] More than 2 clusters detected ({len(cluster_ids_filtered)} clusters). Skipping Fisher's exact test.")
    print("Creating per-cluster contingency table with counts and proportions...")
    
    contingency_results = []
    
    for phoneme in all_phonemes:
        result_dict = {'phoneme': phoneme}
        
        # For each cluster, compute counts and proportions
        for cid in cluster_ids_filtered:
            cluster_langs = df_lang_filtered[df_lang_filtered['cluster'] == cid]
            has_phoneme = (cluster_langs[phoneme] == 1).sum()
            no_phoneme = (cluster_langs[phoneme] == 0).sum()
            total = has_phoneme + no_phoneme
            proportion_str = f"{has_phoneme}/{total} = {(has_phoneme / total * 100):.1f}%" if total > 0 else "N/A"
            
            result_dict[f'cluster_{cid}_has'] = has_phoneme
            result_dict[f'cluster_{cid}_no'] = no_phoneme
            result_dict[f'proportion_cluster_{cid}'] = proportion_str
        
        contingency_results.append(result_dict)

# Convert results to DataFrame
df_contingency_results = pd.DataFrame(contingency_results)

# If 2 clusters (Fisher's test case), sort by p-value and add significance columns
if len(cluster_ids_filtered) == 2:
    df_contingency_results = df_contingency_results.sort_values('p_value').reset_index(drop=True)
    df_contingency_results['significant_0.001'] = df_contingency_results['p_value'].apply(lambda p: 'Yes' if p < 0.001 else 'No')
    df_contingency_results['significant_0.01'] = df_contingency_results['p_value'].apply(lambda p: 'Yes' if p < 0.01 else 'No')
    df_contingency_results['significant_0.05'] = df_contingency_results['p_value'].apply(lambda p: 'Yes' if p < 0.05 else 'No')

# Determine output filename and save based on number of clusters
if len(cluster_ids_filtered) == 2:
    csv_filename = 'fisher_exact_phoneme_cluster_results.csv'
else:
    csv_filename = 'phoneme_cluster_contingency.csv'

df_contingency_results.to_csv(csv_filename, index=False)
print(f"\n[OK] Results saved as: {csv_filename}")

# Print summary
print("\n" + "=" * 60)
if len(cluster_ids_filtered) == 2:
    print("Fisher's Exact Test Results Summary")
else:
    print("Contingency Table Analysis Summary")
print("=" * 60)

print(f"\nTotal phonemes analyzed: {len(df_contingency_results)}")

# Print first few rows as sample
print("\nFirst 10 phonemes (sample):")
print(df_contingency_results.head(10).to_string(index=False))

if len(cluster_ids_filtered) == 2:
    # Print Fisher's test results for 2-cluster case
    df_contingency_results_sorted = df_contingency_results.sort_values('p_value')
    print(f"\nSignificant at α=0.05: {(df_contingency_results_sorted['p_value'] < 0.05).sum()}")
    print(f"Significant at α=0.01: {(df_contingency_results_sorted['p_value'] < 0.01).sum()}")
    print(f"Significant at α=0.001: {(df_contingency_results_sorted['p_value'] < 0.001).sum()}")
    
    print("\n" + "=" * 60)
    print("Top 20 phonemes by Fisher's exact test p-value (most significant):")
    print("=" * 60)
    for idx, row in df_contingency_results_sorted.head(20).iterrows():
        print(f"  {row['phoneme']:12s}: p={row['p_value']:.4g}, OR={row['oddsratio']:8.3f}")

print("\n" + "=" * 60)
if len(cluster_ids_filtered) == 2:
    print("Fisher's Exact Test Analysis Complete")
else:
    print("Contingency Table Analysis Complete")
print("=" * 60)
print(f"\nResults saved to: {csv_filename}")
print("\n" + "=" * 60)

# ============================================================
# TOP 15 PHONEMES: Feature Description Analysis
# ============================================================

if len(cluster_ids_filtered) == 2:
    print("\n" + "=" * 60)
    print("Analyzing Top 15 Significant Phonemes: Feature Descriptions")
    print("=" * 60)
    
    # Get candidate phonemes by p-value and filter to only those with descriptions in all inventories
    all_candidates = df_contingency_results['phoneme'].tolist()
    valid_phonemes = []
    
    for phoneme in all_candidates:
        # Check if this phoneme has descriptions in all feature systems
        has_descriptions_in_all = True
        
        for inv in inventories:
            data_inv = all_data[inv]
            found_in_inv = False
            
            # Check if phoneme exists in any language's min_descriptions for this inventory
            for language, lang_data in data_inv.items():
                if "min_descriptions" in lang_data:
                    min_descriptions = lang_data["min_descriptions"]
                    if phoneme in min_descriptions:
                        found_in_inv = True
                        break
            
            if not found_in_inv:
                has_descriptions_in_all = False
                break
        
        if has_descriptions_in_all:
            valid_phonemes.append(phoneme)
        
        # Stop once we have 15 valid phonemes
        if len(valid_phonemes) >= 15:
            break
    
    top_15_phonemes = valid_phonemes[:15]
    print(f"\nTop 15 phonemes by significance (with descriptions in all inventories): {top_15_phonemes}")
    print(f"(Note: {len(all_candidates) - len(valid_phonemes)} phonemes were skipped due to missing descriptions)")
    
    # Extract min_lengths for these phonemes from each inventory
    # phoneme -> {inventory -> {language -> avg_min_length_of_features}}
    phoneme_features = {}
    
    for phoneme in top_15_phonemes:
        phoneme_features[phoneme] = {}
        
        for inv in inventories:
            phoneme_features[phoneme][inv] = {}
            
            # Load the data for this inventory
            data_inv = all_data[inv]
            
            # For each language, get the features describing this phoneme and their min_lengths
            for language, lang_data in data_inv.items():
                if "min_descriptions" in lang_data and "min_lengths" in lang_data:
                    min_descriptions = lang_data["min_descriptions"]
                    min_lengths = lang_data["min_lengths"]
                    
                    # Get the features describing this phoneme
                    if phoneme in min_descriptions:
                        feature_desc = min_descriptions[phoneme]
                        
                        # Extract all unique features from feature_desc (which is a list of descriptions/features)
                        features = set()
                        for mindesc in feature_desc:
                            for feat in mindesc:
                                features.add(feat.strip('+-'))
                        
                        features = list(features)  # Convert back to list
                        
                        # Get min_lengths for each feature and compute average
                        feature_lengths = []
                        for feature in features:
                            if feature in min_lengths:
                                feature_lengths.append(min_lengths[feature])
                        
                        # Store average min_length for this phoneme in this language
                        if feature_lengths:
                            avg_mdl = np.mean(feature_lengths)
                            phoneme_features[phoneme][inv][language] = avg_mdl
    
    # Calculate average MDL for each phoneme per inventory
    phoneme_mdl_stats = []
    
    for phoneme in top_15_phonemes:
        for inv in inventories:
            mdl_values = list(phoneme_features[phoneme][inv].values())
            
            if mdl_values:
                avg_mdl = np.mean(mdl_values)
                std_mdl = np.std(mdl_values)
                n_langs = len(mdl_values)
            else:
                avg_mdl = 0
                std_mdl = 0
                n_langs = 0
            
            phoneme_mdl_stats.append({
                'phoneme': phoneme,
                'inventory': inv,
                'avg_mdl': avg_mdl,
                'std_mdl': std_mdl,
                'n_langs': n_langs
            })
    
    df_phoneme_mdl = pd.DataFrame(phoneme_mdl_stats)
    
    print("\nAverage feature count (MDL) by phoneme and inventory:")
    print(df_phoneme_mdl.to_string(index=False))
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for grouped bar chart
    phoneme_list = top_15_phonemes
    x = np.arange(len(phoneme_list))
    width = 0.25
    
    for i, inv in enumerate(inventories):
        inv_data = df_phoneme_mdl[df_phoneme_mdl['inventory'] == inv]
        # Ensure data is in the same order as phoneme_list
        mdl_values = [inv_data[inv_data['phoneme'] == ph]['avg_mdl'].values[0] if len(inv_data[inv_data['phoneme'] == ph]) > 0 else 0 for ph in phoneme_list]
        
        ax.bar(x + i * width, mdl_values, width, label=inv, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Phoneme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average MDL', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Significant Phonemes: Average Feature Description Length by Feature System', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'/{ph}/' for ph in phoneme_list], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('top_15_phonemes_feature_mdl.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Histogram saved as: top_15_phonemes_feature_mdl.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary: Average Feature Count per Phoneme")
    print("=" * 60)
    
    for phoneme in top_15_phonemes:
        print(f"\n/{phoneme}/:")
        for inv in inventories:
            inv_stats = df_phoneme_mdl[(df_phoneme_mdl['phoneme'] == phoneme) & (df_phoneme_mdl['inventory'] == inv)]
            if len(inv_stats) > 0:
                row = inv_stats.iloc[0]
                print(f"  {inv}: avg={row['avg_mdl']:.2f} ± {row['std_mdl']:.2f} (n={int(row['n_langs'])} languages)")
    
    print("\n" + "=" * 60)
    print("Top 15 Phonemes Feature Analysis Complete")
    print("=" * 60)

