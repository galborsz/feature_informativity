import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import shutil
from scipy.stats import rankdata

# Define inventories
inventories = ["HC", "SPE", "JFH"]

# Load all data
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Load pb_languages_formatted.csv to get language families and organize by language
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv", encoding='utf-8')

print("\n" + "=" * 60)
print("Creating Language × Feature System MDL Heatmap")
print("=" * 60)

# Define function to compute weighted average MDL
def compute_weighted_avg_mdl(allsegments, min_lengths, min_descriptions):
    """Compute average minimal description length for a set of phonemes."""
    total_avg_length = 0
    feature_count = 0
    
    for phoneme in allsegments:
        if phoneme in min_descriptions:
            feature_descriptions = min_descriptions[phoneme]
            unique_features = {item.strip('+-') for sublist in feature_descriptions for item in sublist}
            for feature in unique_features:
                if feature in min_lengths:
                    total_avg_length += min_lengths[feature]
                    feature_count += 1
    
    if feature_count > 0:
        return total_avg_length / feature_count
    return None

# Collect average MDL per language for each inventory using weighted average
language_mdl_data = []

for inv in inventories:
    print(f"\nProcessing inventory: {inv}")
    
    for language, lang_data in all_data[inv].items():
        if "min_descriptions" in lang_data and "min_lengths" in lang_data:
            min_descriptions = lang_data["min_descriptions"]
            min_lengths = lang_data["min_lengths"]
            
            # Get all phonemes for this language
            allsegments = list(min_descriptions.keys())
            
            # Compute weighted average MDL
            weighted_avg_mdl = compute_weighted_avg_mdl(allsegments, min_lengths, min_descriptions)
            
            if weighted_avg_mdl is not None:
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
                    'avg_mdl': weighted_avg_mdl
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

# Create clusters using if-else statements based on JFH MDL
print("\nCreating clusters based on JFH MDL comparison...")

# Cluster 1: JFH average MDL is larger than both HC and SPE
# Cluster 2: all other languages
clusters_dict = {'1': [], '2': []}

for language in heatmap_data.index:
    hc_mdl = heatmap_data.loc[language, 'HC']
    spe_mdl = heatmap_data.loc[language, 'SPE']
    jfh_mdl = heatmap_data.loc[language, 'JFH']
    
    # Check if JFH is larger than both HC and SPE
    if jfh_mdl > hc_mdl and jfh_mdl > spe_mdl:
        clusters_dict['1'].append(language)
    else:
        clusters_dict['2'].append(language)

# Create a mapping of languages to cluster IDs
lang_to_cluster_var = {}
for cluster_id, langs in clusters_dict.items():
    for lang in langs:
        lang_to_cluster_var[lang] = int(cluster_id)

n_clusters_var = 2
print(f"Number of clusters: {n_clusters_var}")
print(f"Clustering method: If-else based on JFH MDL > HC MDL AND JFH MDL > SPE MDL")

# Print cluster composition
print("\nCluster composition:")
for cluster_id in sorted(clusters_dict.keys()):
    langs = sorted(clusters_dict[cluster_id])
    print(f"  Cluster {cluster_id}: {len(langs)} languages - {', '.join(langs[:5])}{'...' if len(langs) > 5 else ''}")

# Create output directory for cluster plots
cluster_dir = "language_clusters"
if os.path.exists(cluster_dir):
    # Clean existing folder
    shutil.rmtree(cluster_dir)
os.makedirs(cluster_dir)

# Generate separate heatmap for each category
print("\nGenerating separate heatmaps for each cluster...")

# Sort cluster keys: integers first
sorted_cluster_ids = sorted([k for k in clusters_dict.keys()])

for cluster_id in sorted_cluster_ids:
    langs_in_cluster = sorted(clusters_dict[cluster_id])
    cluster_data = heatmap_data.loc[langs_in_cluster]
    
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
    ax.set_title(f'Cluster {cluster_id} ({len(langs_in_cluster)} languages)', 
                    fontsize=12, fontweight='bold', pad=15)
    safe_name = f'cluster_{cluster_id}'
    
    plt.tight_layout()
    
    # Save cluster heatmap with descriptive name
    filename = os.path.join(cluster_dir, f'{safe_name}_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")

print(f"\n[OK] Category heatmaps saved in folder: {cluster_dir}")

# Define statistical test functions
def permutation_pvalue(x, y, n_perm=5000, seed=0, median=True):
    """Two-sample Monte-Carlo permutation test using median difference as test statistic."""
    rng = np.random.default_rng(seed)
    x = np.array(x)
    y = np.array(y)
    if median:
        obs_diff = abs(np.median(x) - np.median(y))
    else:
        obs_diff = abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x_perm = pooled[:n_x]
        y_perm = pooled[n_x:]

        if median:
            stat_diff = abs(np.median(x_perm) - np.median(y_perm))
        else:
            stat_diff = abs(np.mean(x_perm) - np.mean(y_perm))
        
        if stat_diff >= obs_diff:
            count += 1
    return (count + 1) / (n_perm + 1)

def paired_signflip_test(diffs, nperm=10000, seed=123, two_sided=True):
    diffs = np.asarray(diffs, dtype=np.float64)
    obs = np.median(diffs)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    perm_medians = np.empty(nperm, dtype=np.float64)
    for i in range(nperm):
        signs = rng.choice([-1, 1], size=n)
        perm_medians[i] = np.median(signs * diffs)
    if two_sided:
        count = np.sum(np.abs(perm_medians) >= abs(obs))
    else:
        count = np.sum(perm_medians >= obs)
    print(f"Observed median difference: {obs}")
    print(f"Permutation medians (first 10): {perm_medians[:10]}")
    print(f"Number of permutations with median >= observed: {count} out of {nperm}")
    return (count + 1) / (nperm + 1)

def rank_biserial_paired(x, y):
    """Compute paired rank-biserial correlation (Wilcoxon effect size)."""
    
    diffs = np.asarray(x) - np.asarray(y)
    
    # Remove zero differences
    nonzero = diffs != 0
    diffs = diffs[nonzero]
    
    N = len(diffs)
    if N < 1:
        return np.nan
    
    # Rank absolute differences
    abs_diffs = np.abs(diffs)
    ranks = rankdata(abs_diffs)
    
    # Sum of ranks for positive differences
    R_plus = np.sum(ranks[diffs > 0])
    
    # Rank-biserial correlation
    r = (2 * R_plus) / (N * (N + 1)) - 1
    
    return r

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
    all_mdl_values.extend(heatmap_data[feature_system].values)
    
y_min = np.min(all_mdl_values)
y_max = np.max(all_mdl_values)
y_margin = (y_max - y_min) * 0.05  # Add 5% margin
y_axis_range = (1.5, 3.4) #(y_min - y_margin, y_max + y_margin)

for cluster_id in sorted_cluster_ids:
    langs_in_cluster = sorted(clusters_dict[cluster_id])
    cluster_data = heatmap_data.loc[langs_in_cluster]
    
    # Calculate scale factor based on cluster size
    # Find the maximum cluster size to normalize
    max_cluster_size = max(len(clusters_dict[cid]) for cid in sorted_cluster_ids)
    cluster_size = len(langs_in_cluster)
    # Scale violin width from 0.3 to 1.0 based on cluster size
    violin_scale = 0.3 + (cluster_size / max_cluster_size) * 0.7
    
    # Prepare data for violin plot: reshape to long format (cluster_data already has weighted averages)
    violin_data_list = []
    for inv in inventories:
        mdl_values = cluster_data[inv].values
        for mdl_val in mdl_values:
            violin_data_list.append({
                'Feature System': inv,
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
        p_val = permutation_pvalue(data1, data2, n_perm=5000, seed=42)
        r_pair = rank_biserial_paired(data1, data2)
        pvals_raw.append(p_val)
        effect_sizes.append(r_pair)
    
    # Apply Benjamini-Hochberg FDR correction
    pvals_array = np.array(pvals_raw)
    rejected, pvals_corrected, _, _ = multipletests(pvals_array, alpha=0.05, method='fdr_bh')
    
    # Create figure for violin plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create violin plot with median line
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
    ax.set_title(f'Cluster {cluster_id} ({len(langs_in_cluster)} languages)', 
                 fontsize=12, fontweight='bold', pad=15)
    safe_name = f'cluster_{cluster_id}'
    
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

print("\n" + "=" * 60)

# ============================================================
# GLOBAL VIOLIN PLOT: All languages, three feature systems
# ============================================================
print("\nGenerating global violin plot across all languages...")

# Prepare data for global violin plot using weighted average MDL: reshape to long format
global_violin_data_list = []
for inv in inventories:
    print(f"\nProcessing inventory: {inv}")
    
    data_inv = all_data[inv]
    
    for language, lang_data in data_inv.items():
        if language not in heatmap_data.index:
            continue
            
        if "min_descriptions" in lang_data and "min_lengths" in lang_data:
            min_descriptions = lang_data["min_descriptions"]
            min_lengths = lang_data["min_lengths"]
            
            # Get all phonemes for this language
            allsegments = list(min_descriptions.keys())
            
            # Compute weighted average MDL
            weighted_avg_mdl = compute_weighted_avg_mdl(allsegments, min_lengths, min_descriptions)
            
            if weighted_avg_mdl is not None:
                global_violin_data_list.append({
                    'Feature System': inv,
                    'Average MDL': weighted_avg_mdl
                })

global_violin_df = pd.DataFrame(global_violin_data_list)

# Extract data per feature system for statistical tests
hc_global = global_violin_df[global_violin_df['Feature System'] == 'HC']['Average MDL'].values
spe_global = global_violin_df[global_violin_df['Feature System'] == 'SPE']['Average MDL'].values
jfh_global = global_violin_df[global_violin_df['Feature System'] == 'JFH']['Average MDL'].values

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
    r_pair = rank_biserial_paired(data1, data2)
    pvals_raw_global.append(p_val)
    effect_sizes_global.append(r_pair)

# Apply Benjamini-Hochberg FDR correction
pvals_array_global = np.array(pvals_raw_global)
rejected_global, pvals_corrected_global, _, _ = multipletests(pvals_array_global, alpha=0.05, method='fdr_bh')

# Create global violin plot
fig, ax = plt.subplots(figsize=(10, 7))

# Set y-axis with fixed range
ax.set_ylim(1.5, 3.4)

# Set y-axis ticks with 0.5 interval and one decimal place
ax.set_yticks([2.0, 2.5, 3.0])
ax.set_yticklabels(['2.0', '2.5', '3.0'], fontsize=12)

# Create violin plot with median line
sns.violinplot(x=global_violin_df[global_violin_df['Average MDL'] < 2.7]['Feature System'], y=global_violin_df[global_violin_df['Average MDL'] < 2.7]['Average MDL'], ax=ax, color=".8",
                #alpha=0.8, palette='Set2',
                inner=None, linewidth=3, scale='width')

# Add individual sample points
sns.stripplot(data=global_violin_df, x='Feature System', y='Average MDL', ax=ax, alpha=0.7,
              palette=['#1f77b4', '#ff7f0e', '#2ca02c'], size=6, jitter=True)

# Add median lines for each feature system with standard width for global plot
medians_global = [np.median(hc_global), np.median(spe_global), np.median(jfh_global)]
for i, median_val in enumerate(medians_global):
    ax.hlines(median_val, i - 0.4, i + 0.4, colors='darkred', linewidth=2.5, label='Median' if i == 0 else '')

# Set labels and title
ax.set_xlabel('Feature System', fontsize=14) # , fontweight='bold'
ax.set_ylabel('Average Minimal Description Length (MDL)', fontsize=14) # , fontweight='bold'

# Set x-axis tick labels to be bigger and bold
ax.set_xticklabels(['HC', 'SPE', 'JFH'], fontsize=14, fontweight='bold')

# Add significance lines with stars connecting compared distributions
if len(hc_global) > 2 and len(spe_global) > 2 and len(jfh_global) > 2:
    # Calculate y positions for significance lines
    max_y = ax.get_ylim()[1]
    line_y_positions = [
        max_y - 0.13,  # HC vs SPE
        max_y - 0.26,  # SPE vs JFH
        max_y - 0.39   # HC vs JFH
    ]
    
    # Tail parameters
    tail_length = 0.015  # Shorter length of vertical tails at each end
    
    # Draw significance lines for each pairwise comparison
    for i, (inv1, inv2, data1, data2, pos1, pos2) in enumerate(pairs_global):
        p_adj = pvals_corrected_global[i]
        stars = p_to_stars(p_adj)
        line_y = line_y_positions[i]
        
        # Draw horizontal line connecting the two distributions
        ax.plot([pos1, pos2], [line_y, line_y], 'k-', linewidth=2)
        
        # Draw tails at the edges
        ax.plot([pos1, pos1], [line_y - tail_length, line_y], 'k-', linewidth=2)
        ax.plot([pos2, pos2], [line_y - tail_length, line_y], 'k-', linewidth=2)
        
        # Add stars above the line at the midpoint
        mid_x = (pos1 + pos2) / 2
        ax.text(mid_x, line_y + 0.015, stars, ha='center', va='bottom', 
                fontsize=14, fontweight='bold')

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
#  Fisher's exact test: Phoneme Presence vs. Language Clusters
# ============================================================

print("\n" + "=" * 60)
print("Fisher's Exact Test: Phoneme Presence vs. Language Clusters")
print("=" * 60)

print("\nBuilding phoneme presence dataset...")

# Collect all unique phonemes across all languages
all_phonemes = set()
for lang in heatmap_data.index:
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

for language in heatmap_data.index:
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
for cid in sorted(df_lang['cluster'].unique()):
    count = (df_lang['cluster'] == cid).sum()
    print(f"    Cluster {cid}: {count} languages")

# Perform contingency table analysis for each phoneme
print("\nPerforming contingency table analysis for each phoneme...")

# Use all clusters (no filtering out 'constant' anymore)
df_lang_filtered = df_lang.copy()
cluster_ids_filtered = sorted(df_lang_filtered['cluster'].unique())

print(f"Number of clusters (excluding constant): {len(cluster_ids_filtered)}")
for cid in cluster_ids_filtered:
    count = (df_lang_filtered['cluster'] == cid).sum()
    print(f"  Cluster {cid}: {count} languages")


print("\nApplying Fisher's exact test...")

cluster_1, cluster_2 = cluster_ids_filtered[0], cluster_ids_filtered[1]
print(f"Comparing Cluster {cluster_1} (JFH > HC & JFH > SPE) vs Cluster {cluster_2} (others)")

contingency_results = []
p_values_raw = []

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
    
    p_values_raw.append(p_value)
    
    # Calculate proportions
    c1_total = c1_has + c1_no
    c2_total = c2_has + c2_no
    c1_proportion_str = f"{c1_has}/{c1_total} = {(c1_has / c1_total * 100):.1f}%" if c1_total > 0 else "N/A"
    c2_proportion_str = f"{c2_has}/{c2_total} = {(c2_has / c2_total * 100):.1f}%" if c2_total > 0 else "N/A"
    
    # Store results (will add corrected p-value later)
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

# Apply Benjamini-Hochberg FDR correction to all p-values
p_values_array = np.array(p_values_raw)
rejected, p_values_corrected, _, _ = multipletests(p_values_array, alpha=0.05, method='fdr_bh')
    
# Add corrected p-values to results
for i, result_dict in enumerate(contingency_results):
    result_dict['p_value_corrected'] = p_values_corrected[i]

# Add list of languages containing each phoneme
for result_dict in contingency_results:
    phoneme = result_dict['phoneme']
    languages_with_phoneme = df_lang[df_lang[phoneme] == 1]['language'].tolist()
    result_dict['languages_with_phoneme'] = '; '.join(sorted(languages_with_phoneme))

# Convert results to DataFrame
df_contingency_results = pd.DataFrame(contingency_results)

df_contingency_results = df_contingency_results.sort_values('p_value_corrected').reset_index(drop=True)
df_contingency_results['significant_0.001'] = df_contingency_results['p_value_corrected'].apply(lambda p: 'Yes' if p < 0.001 else 'No')
df_contingency_results['significant_0.01'] = df_contingency_results['p_value_corrected'].apply(lambda p: 'Yes' if p < 0.01 else 'No')
df_contingency_results['significant_0.05'] = df_contingency_results['p_value_corrected'].apply(lambda p: 'Yes' if p < 0.05 else 'No')

# Determine output filename and save based on number of clusters
csv_filename = 'fisher_exact_phoneme_cluster_results.csv'

df_contingency_results.to_csv(csv_filename, index=False)
print(f"\n[OK] Results saved as: {csv_filename}")

# Print summary
print("\n" + "=" * 60)
print("Fisher's Exact Test Results Summary")
print("=" * 60)

print(f"\nTotal phonemes analyzed: {len(df_contingency_results)}")

# Print first few rows as sample
print("\nFirst 10 phonemes (sample):")
print(df_contingency_results.head(10).to_string(index=False))

# Print Fisher's test results
df_contingency_results_sorted = df_contingency_results.sort_values('p_value_corrected')
print(f"\nSignificant at α=0.05 (corrected): {(df_contingency_results_sorted['p_value_corrected'] < 0.05).sum()}")
print(f"Significant at α=0.01 (corrected): {(df_contingency_results_sorted['p_value_corrected'] < 0.01).sum()}")
print(f"Significant at α=0.001 (corrected): {(df_contingency_results_sorted['p_value_corrected'] < 0.001).sum()}")

print("\n" + "=" * 60)
print("Top 20 phonemes by Fisher's exact test p-value (corrected, most significant):")
print("=" * 60)
for idx, row in df_contingency_results_sorted.head(20).iterrows():
    print(f"  {row['phoneme']:12s}: p_corrected={row['p_value_corrected']:.4g}, OR={row['oddsratio']:8.3f}")

print("\n" + "=" * 60)
print("Fisher's Exact Test Analysis Complete")
print("=" * 60)
print(f"\nResults saved to: {csv_filename}")
print("\n" + "=" * 60)
    
# ============================================================
# Create individual plots for significant phonemes: Cluster × Phoneme Presence + MDL
# ============================================================
print("\n" + "=" * 60)
print("Creating Individual Cluster × Phoneme Presence Plots with MDL Subplots")
print("=" * 60)

# Create output directory for phoneme plots
phoneme_plots_dir = "phoneme_cluster_presence_plots"
if os.path.exists(phoneme_plots_dir):
    shutil.rmtree(phoneme_plots_dir)
os.makedirs(phoneme_plots_dir)

# Get cluster information
cluster_1, cluster_2 = cluster_ids_filtered[0], cluster_ids_filtered[1]

# Filter to only phonemes with significant p-values (α=0.05) using corrected p-values
significant_phonemes = df_contingency_results[df_contingency_results['p_value_corrected'] < 0.05]['phoneme'].tolist()
print(f"\nTotal phonemes with significant corrected p-values (α=0.05): {len(significant_phonemes)}")

# Extract MDL data for all phonemes across all inventories
phoneme_mdl_all = {}

for phoneme in significant_phonemes:
    phoneme_mdl_all[phoneme] = {}
    
    for inv in inventories:
        phoneme_mdl_all[phoneme][inv] = {}
        
        # Load the data for this inventory
        data_inv = all_data[inv]
        
        # For each language, get the features describing this phoneme and their min_lengths
        for language, lang_data in data_inv.items():
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
                
                features = list(features) 
                
                # Get min_lengths for each feature and compute average
                feature_lengths = []
                for feature in features:
                    if feature in min_lengths:
                        feature_lengths.append(min_lengths[feature])
                
                # Store average min_length for this phoneme in this language
                if feature_lengths:
                    avg_mdl = np.mean(feature_lengths) # average over features
                    phoneme_mdl_all[phoneme][inv][language] = avg_mdl

# Create individual plots for each significant phoneme
for phoneme in significant_phonemes:
    # Get data from contingency results
    phoneme_row = df_contingency_results[df_contingency_results['phoneme'] == phoneme]
    
    if len(phoneme_row) > 0:
        row = phoneme_row.iloc[0]
        
        # Calculate proportions based on cluster configuration
        c1_has = row[f'cluster_{cluster_1}_has']
        c1_no = row[f'cluster_{cluster_1}_no']
        c2_has = row[f'cluster_{cluster_2}_has']
        c2_no = row[f'cluster_{cluster_2}_no']
        
        c1_total = c1_has + c1_no
        c2_total = c2_has + c2_no
        c1_prop = c1_has / c1_total if c1_total > 0 else 0
        c2_prop = c2_has / c2_total if c2_total > 0 else 0
        
        category_1_label = f'Cluster {cluster_1}'
        category_2_label = f'Cluster {cluster_2}'
        
        # Create figure with two subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
        
        # ===== LEFT SUBPLOT: Cluster Presence Bar Plot =====
        categories = [category_1_label, category_2_label]
        proportions = [c1_prop, c2_prop]
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        
        bars = ax_left.bar(categories, proportions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, prop in zip(bars, proportions):
            height = bar.get_height()
            ax_left.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prop:.1%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add count labels below bars
        count_labels = [f'n={int(c1_has)}/{int(c1_total)}', f'n={int(c2_has)}/{int(c2_total)}']
        for i, (bar, label) in enumerate(zip(bars, count_labels)):
            ax_left.text(bar.get_x() + bar.get_width()/2., -0.08,
                    label,
                    ha='center', va='top', fontsize=11, transform=ax_left.get_xaxis_transform())
        
        # Set y-axis limits and labels
        ax_left.set_ylim(0, 1.15)
        ax_left.set_ylabel('Proportion with phoneme', fontsize=12, fontweight='bold')
        # ax_left.set_title(f'Cluster Presence', fontsize=12, fontweight='bold')
        ax_left.grid(True, alpha=0.3, axis='y')
        ax_left.set_axisbelow(True)
        
        # Add p-value and odds ratio annotation on left subplot (only for 2-cluster case)
        if n_clusters_var == 2 and 'p_value_corrected' in row and 'oddsratio' in row:
            p_val = row['p_value_corrected']
            or_val = row['oddsratio']
            stars = p_to_stars(p_val)
            annotation_text = f"p_corr = {p_val:.4f} {stars}\nOR = {or_val:.2f}"
            ax_left.text(0.98, 0.97, annotation_text, transform=ax_left.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
        
        # ===== RIGHT SUBPLOT: MDL Distribution by Feature System (Violin Plot) =====
        mdl_arrays_per_inv = []  # Store raw arrays for statistical testing
        
        # Prepare data for violin plot
        violin_mdl_data = []
        for inv in inventories:
            mdl_values = list(phoneme_mdl_all[phoneme][inv].values()) # Extract MDL values over languages
            mdl_arrays_per_inv.append(np.array(mdl_values))
            
            for mdl_val in mdl_values:
                violin_mdl_data.append({
                    'Feature System': inv,
                    'MDL': mdl_val
                })
        
        violin_mdl_df = pd.DataFrame(violin_mdl_data)
        
        # Perform pairwise Monte-Carlo permutation tests between feature systems
        hc_mdl_vals = mdl_arrays_per_inv[0]
        spe_mdl_vals = mdl_arrays_per_inv[1]
        jfh_mdl_vals = mdl_arrays_per_inv[2]
        
        pairwise_tests = [
            ('HC', 'SPE', hc_mdl_vals, spe_mdl_vals),
            ('SPE', 'JFH', spe_mdl_vals, jfh_mdl_vals),
            ('HC', 'JFH', hc_mdl_vals, jfh_mdl_vals)
        ]
        
        pvals_raw_mdl = []
        effect_sizes_mdl = []
        for inv1, inv2, data1, data2 in pairwise_tests:
            if len(data1) > 2 and len(data2) > 2:
                p_val = permutation_pvalue(data1, data2, n_perm=5000, seed=42)
                r_pair = rank_biserial_paired(data1, data2)
                pvals_raw_mdl.append(p_val)
                effect_sizes_mdl.append(r_pair)
            else:
                pvals_raw_mdl.append(np.nan)
                effect_sizes_mdl.append(np.nan)
        
        # Apply Benjamini-Hochberg FDR correction
        pvals_array_mdl = np.array(pvals_raw_mdl)
        valid_indices = ~np.isnan(pvals_array_mdl)
        if valid_indices.sum() > 0:
            pvals_valid = pvals_array_mdl[valid_indices]
            rejected_mdl, pvals_corrected_mdl_valid, _, _ = multipletests(pvals_valid, alpha=0.05, method='fdr_bh')
            pvals_corrected_mdl = np.full_like(pvals_array_mdl, np.nan)
            pvals_corrected_mdl[valid_indices] = pvals_corrected_mdl_valid
        else:
            pvals_corrected_mdl = pvals_array_mdl.copy()
        
        # Create violin plot
        sns.violinplot(data=violin_mdl_df, x='Feature System', y='MDL', ax=ax_right, 
                        palette=['#1f77b4', '#ff7f0e', '#2ca02c'], inner=None, linewidth=1.5)
        
        # Add individual sample points
        sns.stripplot(data=violin_mdl_df, x='Feature System', y='MDL', ax=ax_right,
                        color='black', alpha=0.4, size=4, jitter=True)
        
        # Add median lines for each feature system
        medians_mdl = []
        for inv in inventories:
            inv_data = violin_mdl_df[violin_mdl_df['Feature System'] == inv]
            if len(inv_data) > 0:
                median_val = inv_data['MDL'].median()
                medians_mdl.append(median_val)
            else:
                medians_mdl.append(0)
        
        for i, median_val in enumerate(medians_mdl):
            ax_right.hlines(median_val, i - 0.4, i + 0.4, colors='darkred', linewidth=2.5, label='Median' if i == 0 else '')
        
        # Set labels and limits
        ax_right.set_xlabel('Feature System', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Average MDL over phoneme features', fontsize=12, fontweight='bold')
        ax_right.grid(True, alpha=0.3, axis='y')
        ax_right.set_axisbelow(True)
        
        # Add p-value annotation box for MDL comparisons
        annotation_mdl_text = '' #'Pairwise MDL Comparisons\n(corrected p-values)\n'
        for idx, (inv1, inv2, _, _) in enumerate(pairwise_tests):
            p_adj = pvals_corrected_mdl[idx]
            if not np.isnan(p_adj):
                stars = p_to_stars(p_adj)
                r_val = effect_sizes_mdl[idx]
                annotation_mdl_text += f'{inv1} vs {inv2}: p={p_adj:.3f} {stars}, r={r_val:.2f}\n'
        
        annotation_mdl_text = annotation_mdl_text.rstrip()
        ax_right.text(0.02, 0.98, annotation_mdl_text, transform=ax_right.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1))
        
        # Main title
        fig.suptitle(f'Phoneme /{phoneme}/', 
                        fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save individual plot
        safe_phoneme_name = phoneme.replace('/', '_').replace(' ', '_')
        filename = os.path.join(phoneme_plots_dir, f'phoneme_{safe_phoneme_name}_cluster_presence.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

print(f"\n[OK] Individual phoneme plots saved in folder: {phoneme_plots_dir}")
print(f"Total plots created: {len(significant_phonemes)}")

# ============================================================
# Create individual plots for significant phonemes using heatmap_data MDL
# (Language-level average MDL for all feature systems)
# Grouped violin plots: For each feature system, show distributions with and without phoneme
# ============================================================
print("\n" + "=" * 60)
print("Creating Individual Phoneme Plots with Language-level MDL from heatmap_data")
print("=" * 60)

# Create output directory for phoneme plots with heatmap_data
phoneme_plots_dir_heatmap = "phoneme_cluster_presence_plots_heatmap_mdl"
if os.path.exists(phoneme_plots_dir_heatmap):
    shutil.rmtree(phoneme_plots_dir_heatmap)
os.makedirs(phoneme_plots_dir_heatmap)

# Create individual plots for each significant phoneme using heatmap_data MDL
for phoneme in significant_phonemes:
    # Get languages that contain this phoneme
    languages_with_phoneme = df_lang[df_lang[phoneme] == 1]['language'].tolist()
    
    # Prepare combined data for grouped violin plot
    combined_violin_data = []
    
    # ===== COLLECT DATA: With phoneme =====
    for inv in inventories:
        mdl_values_with = []
        for language in languages_with_phoneme:
            if language in heatmap_data.index:
                mdl_val = heatmap_data.loc[language, inv]
                mdl_values_with.append(mdl_val)
        
        # Add to combined dataframe
        for mdl_val in mdl_values_with:
            combined_violin_data.append({
                'Feature System': inv,
                'Phoneme Presence': f'With /{phoneme}/',
                'MDL': mdl_val
            })

        print(f"Collected {len(mdl_values_with)} MDL values for {inv} with phoneme /{phoneme}/")
    
    # ===== COLLECT DATA: Without phoneme (excluding this phoneme's features) =====
    for inv in inventories:
        # Get the data for this inventory
        data_inv = all_data[inv]
        mdl_values_excluding = []
        for language in languages_with_phoneme:    
            if language in data_inv:
                lang_data = data_inv[language]

                min_descriptions = lang_data["min_descriptions"]
                min_lengths = lang_data["min_lengths"]
                
                # Collect all phonemes except the current one
                allsegments_excluding = [pho for pho in min_descriptions.keys() if pho != phoneme]
                
                # Compute weighted average MDL excluding current phoneme
                weighted_avg_mdl_excluding = compute_weighted_avg_mdl(allsegments_excluding, min_lengths, min_descriptions)
                
                # Store weighted average MDL for this language excluding current phoneme
                if weighted_avg_mdl_excluding is not None:
                    mdl_values_excluding.append(weighted_avg_mdl_excluding)
            
        # Add to combined dataframe
        for mdl_val in mdl_values_excluding:
            combined_violin_data.append({
                'Feature System': inv,
                'Phoneme Presence': f'Without /{phoneme}/',
                'MDL': mdl_val
            })
        print(f"Collected {len(mdl_values_excluding)} MDL values for {inv} without phoneme /{phoneme}/")
    
    # Convert to DataFrame
    combined_violin_df = pd.DataFrame(combined_violin_data)
    
    # ===== PERFORM STATISTICAL TESTS: With vs Without for each feature system =====
    pvals_raw_presence = []
    effect_sizes_presence = []
    
    for feature_system in inventories:
        with_presence_data = combined_violin_df[
            (combined_violin_df['Feature System'] == feature_system) & 
            (combined_violin_df['Phoneme Presence'] == f'With /{phoneme}/')
        ]['MDL'].values
        
        without_presence_data = combined_violin_df[
            (combined_violin_df['Feature System'] == feature_system) & 
            (combined_violin_df['Phoneme Presence'] == f'Without /{phoneme}/')
        ]['MDL'].values
        
        # Perform permutation test
        if len(with_presence_data) > 2 and len(without_presence_data) > 2:
            p_val = permutation_pvalue(with_presence_data, without_presence_data, n_perm=5000, seed=42)
            r_pair = rank_biserial_paired(with_presence_data, without_presence_data)
            pvals_raw_presence.append(p_val)
            effect_sizes_presence.append(r_pair)
        else:
            pvals_raw_presence.append(np.nan)
            effect_sizes_presence.append(np.nan)
    
    # Apply Benjamini-Hochberg FDR correction
    pvals_array_presence = np.array(pvals_raw_presence)
    valid_indices_presence = ~np.isnan(pvals_array_presence)
    if valid_indices_presence.sum() > 0:
        pvals_valid_presence = pvals_array_presence[valid_indices_presence]
        rejected_presence, pvals_corrected_presence_valid, _, _ = multipletests(pvals_valid_presence, alpha=0.05, method='fdr_bh')
        pvals_corrected_presence = np.full_like(pvals_array_presence, np.nan)
        pvals_corrected_presence[valid_indices_presence] = pvals_corrected_presence_valid
    else:
        pvals_corrected_presence = pvals_array_presence.copy()
    
    # Create single grouped violin plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create grouped violin plot with hue for 'Phoneme Presence'
    sns.violinplot(data=combined_violin_df, x='Feature System', y='MDL', 
                   hue='Phoneme Presence', ax=ax, 
                   palette=['#1f77b4', '#ff7f0e'], inner=None, linewidth=1.5,
                   split=False)
    
    # Add individual sample points
    sns.stripplot(data=combined_violin_df, x='Feature System', y='MDL', 
                  hue='Phoneme Presence', ax=ax,
                  color='black', alpha=0.3, size=3, jitter=True, dodge=True,
                  legend=False)
    
    # Calculate and add median lines for each combination
    for feature_idx, feature_system in enumerate(inventories):
        # Median with phoneme
        with_data = combined_violin_df[
            (combined_violin_df['Feature System'] == feature_system) & 
            (combined_violin_df['Phoneme Presence'] == f'With /{phoneme}/')
        ]
        if len(with_data) > 0:
            median_with = with_data['MDL'].median()
            # Position slightly to the left within the violin
            ax.hlines(median_with, feature_idx - 0.2, feature_idx - 0.05, 
                     colors='darkred', linewidth=2.5)
        
        # Median without phoneme
        without_data = combined_violin_df[
            (combined_violin_df['Feature System'] == feature_system) & 
            (combined_violin_df['Phoneme Presence'] == f'Without /{phoneme}/')
        ]
        if len(without_data) > 0:
            median_without = without_data['MDL'].median()
            # Position slightly to the right within the violin
            ax.hlines(median_without, feature_idx + 0.05, feature_idx + 0.2, 
                     colors='darkred', linewidth=2.5)
    
    # Customize plot
    ax.set_xlabel('Feature System', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average MDL', fontsize=12, fontweight='bold')
    ax.set_title(f'Phoneme /{phoneme}/', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.set_ylim(1.5, 3.4)
    
    # # Add p-value annotation box for presence comparisons
    # annotation_presence_text = '' #'With vs Without /{}/:\n(corrected p-values)\n'.format(phoneme)
    # for feature_idx, feature_system in enumerate(inventories):
    #     p_adj = pvals_corrected_presence[feature_idx]
    #     if not np.isnan(p_adj):
    #         stars = p_to_stars(p_adj)
    #         r_val = effect_sizes_presence[feature_idx]
    #         annotation_presence_text += f'{feature_system}: p={p_adj:.3f} {stars}, r={r_val:.2f}\n'
    
    # annotation_presence_text = annotation_presence_text.rstrip()
    # ax.text(0.02, 0.98, annotation_presence_text, transform=ax.transAxes,
    #         fontsize=9, verticalalignment='top', horizontalalignment='left',
    #         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=1))
    
    # Customize legend
    ax.legend(title='Phoneme Presence', fontsize=10, title_fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    # Save individual plot
    safe_phoneme_name = phoneme.replace('/', '_').replace(' ', '_')
    filename = os.path.join(phoneme_plots_dir_heatmap, f'phoneme_{safe_phoneme_name}_mdl_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # ===== COMPUTE MEDIAN DIFFERENCES: Without vs With =====
    # Compute per-language differences: MDL with phoneme vs without phoneme
    language_diffs = []
    
    for language in languages_with_phoneme:
        if language in heatmap_data.index:
            # For each feature system, compute the difference
            for inv in inventories:
                # MDL with phoneme (from heatmap_data - language-level average)
                mdl_with = heatmap_data.loc[language, inv]
                
                # MDL without phoneme (computed from all_data)
                lang_data = all_data[inv][language]
                min_descriptions = lang_data["min_descriptions"]
                min_lengths = lang_data["min_lengths"]
                
                # Collect all phonemes except the current one
                allsegments_excluding = [pho for pho in min_descriptions.keys() if pho != phoneme]
                
                # Compute weighted average MDL excluding current phoneme
                mdl_without = compute_weighted_avg_mdl(allsegments_excluding, min_lengths, min_descriptions)
                
                if mdl_without is not None:
                    # Compute difference: with - without
                    diff = mdl_with - mdl_without
                    language_diffs.append({
                        'language': language,
                        'Feature System': inv,
                        'Difference (With - Without)': diff
                    })
    
    # Convert to DataFrame
    diffs_df = pd.DataFrame(language_diffs)
    
    # Perform paired sign-flip test for each feature system
    print(f"\n  Paired sign-flip tests for /{phoneme}/:")
    
    for inv in inventories:
        inv_diffs = diffs_df[diffs_df['Feature System'] == inv]['Difference (With - Without)'].values
        
        if len(inv_diffs) > 1:
            p_val_signflip = paired_signflip_test(inv_diffs, nperm=10000, seed=123, two_sided=True)
            obs_median = np.median(inv_diffs)
            stars = p_to_stars(p_val_signflip)
            print(f"    {inv}: median_diff = {obs_median:.4f}, p_signflip = {p_val_signflip:.4f} {stars}")
    
    # Create violin plot showing per-language differences
    fig_diff, ax_diff = plt.subplots(figsize=(10, 7))

    ax_diff.set_ylim(-0.015, 0.015)
    
    # Create violin plot
    sns.violinplot(data=diffs_df, x='Feature System', y='Difference (With - Without)', ax=ax_diff,
                   palette=['#1f77b4', '#ff7f0e', '#2ca02c'], inner=None, linewidth=1.5)
    
    # Add individual sample points
    sns.stripplot(data=diffs_df, x='Feature System', y='Difference (With - Without)', ax=ax_diff,
                  color='black', alpha=0.4, size=4, jitter=True)
    
    # Add median lines for each feature system
    for idx, inv in enumerate(inventories):
        inv_diffs = diffs_df[diffs_df['Feature System'] == inv]['Difference (With - Without)'].values
        if len(inv_diffs) > 0:
            median_diff = np.median(inv_diffs)
            ax_diff.hlines(median_diff, idx - 0.4, idx + 0.4, colors='darkred', linewidth=2.5, 
                          label='Median' if idx == 0 else '')
    
    # Add zero line
    ax_diff.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    ax_diff.set_ylabel('MDL Difference (With - Without)', fontsize=12, fontweight='bold')
    ax_diff.set_xlabel('Feature System', fontsize=12, fontweight='bold')
    ax_diff.set_title(f'Per-Language MDL Differences for /{phoneme}/', fontsize=13, fontweight='bold')
    ax_diff.grid(True, alpha=0.3, axis='y')
    ax_diff.set_axisbelow(True)
    
    # Add annotation with sample sizes and test results
    annotation_diff_text = ''
    for inv in inventories:
        inv_diffs = diffs_df[diffs_df['Feature System'] == inv]['Difference (With - Without)'].values
        if len(inv_diffs) > 0:
            p_val = paired_signflip_test(inv_diffs, nperm=10000, seed=123, two_sided=True)
            stars = p_to_stars(p_val)
            annotation_diff_text += f'{inv} (n={len(inv_diffs)}): p={p_val:.3f} {stars}\n'
    
    annotation_diff_text = annotation_diff_text.rstrip()
    ax_diff.text(0.98, 0.97, annotation_diff_text, transform=ax_diff.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save median difference plot
    safe_phoneme_name = phoneme.replace('/', '_').replace(' ', '_')
    filename_diff = os.path.join(phoneme_plots_dir_heatmap, f'phoneme_{safe_phoneme_name}_per_language_diffs.png')
    plt.savefig(filename_diff, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n[OK] Individual phoneme plots with heatmap_data MDL saved in folder: {phoneme_plots_dir_heatmap}")
print(f"Total plots created: {len(significant_phonemes)}")

# ============================================================
# Phoneme Co-occurrence Analysis: Correlation Matrix
# ============================================================
print("\n" + "=" * 60)
print("Phoneme Co-occurrence Analysis")
print("=" * 60)

# Use the binary phoneme data from df_lang (languages × phonemes matrix)
phoneme_columns = [col for col in df_lang.columns if col != 'language' and col != 'cluster']
df_phoneme_matrix = df_lang[phoneme_columns].copy()
df_phoneme_matrix = df_phoneme_matrix[significant_phonemes]

print(f"\nAnalyzing co-occurrence patterns for {len(significant_phonemes)} phonemes across {len(df_lang)} languages")

# Calculate Pearson correlation between all phonemes
phoneme_correlation = df_phoneme_matrix.corr(method='pearson')

print(f"Correlation matrix shape: {phoneme_correlation.shape}")
print(f"\nTop 10 strongest co-occurrence correlations:")

# Extract upper triangle of correlation matrix (to avoid duplicates and self-correlations)
# Convert to long format
corr_pairs = []
for i, phoneme1 in enumerate(phoneme_correlation.columns):
    for j, phoneme2 in enumerate(phoneme_correlation.columns):
        if i < j:  # Upper triangle only
            corr_value = phoneme_correlation.loc[phoneme1, phoneme2]
            corr_pairs.append({
                'phoneme_1': phoneme1,
                'phoneme_2': phoneme2,
                'correlation': corr_value
            })

# Sort by absolute correlation value (strongest associations)
corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

for idx, pair in enumerate(corr_pairs_sorted[:10], 1):
    print(f"  {idx:2d}. /{pair['phoneme_1']}/ <-> /{pair['phoneme_2']}/: r = {pair['correlation']:+.3f}")

# Find strongest positive and negative correlations
print(f"\nTop 5 positive correlations (phonemes that tend to co-occur):")
positive_corrs = [p for p in corr_pairs_sorted if p['correlation'] > 0]
for idx, pair in enumerate(positive_corrs[:5], 1):
    print(f"  {idx}. /{pair['phoneme_1']}/ <-> /{pair['phoneme_2']}/: r = {pair['correlation']:+.3f}")

print(f"\nTop 5 negative correlations (phonemes that tend to exclude each other):")
negative_corrs = sorted([p for p in corr_pairs_sorted if p['correlation'] < 0], 
                        key=lambda x: x['correlation'])  # Most negative first
for idx, pair in enumerate(negative_corrs[:5], 1):
    print(f"  {idx}. /{pair['phoneme_1']}/ <-> /{pair['phoneme_2']}/: r = {pair['correlation']:+.3f}")

# Calculate some basic co-occurrence statistics
print(f"\nCo-occurrence statistics:")
print(f"  Mean correlation: {phoneme_correlation.values[np.triu_indices_from(phoneme_correlation.values, k=1)].mean():.3f}")
print(f"  Median correlation: {np.median(phoneme_correlation.values[np.triu_indices_from(phoneme_correlation.values, k=1)]):.3f}")
print(f"  Std Dev of correlations: {phoneme_correlation.values[np.triu_indices_from(phoneme_correlation.values, k=1)].std():.3f}")

# Create visualization: heatmap of correlations for significant phonemes
print(f"\nGenerating correlation heatmap for {len(significant_phonemes)} significant phonemes...")

corr_subset = phoneme_correlation

# Getting the Upper Triangle of the co-relation matrix
mask = np.triu(corr_subset)

# Create figure
fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap
sns.heatmap(corr_subset, cmap='RdBu_r', center=0, annot=False, fmt='.2f', mask=mask,
            square=True, linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Pearson Correlation'},
            vmin=-1, vmax=1, ax=ax)

# Customize labels
ax.set_xticklabels([f'/{ph}/' for ph in significant_phonemes], rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([f'/{ph}/' for ph in significant_phonemes], rotation=0, fontsize=9)

ax.set_title(f'Phoneme Co-occurrence Correlation Matrix\n({len(significant_phonemes)} Significant Phonemes)', 
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('phoneme_cooccurrence_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Correlation heatmap saved as: phoneme_cooccurrence_correlation.png")

# Save correlation results to CSV
df_corr_results = pd.DataFrame(corr_pairs)
df_corr_results = df_corr_results.sort_values(by='correlation', key=lambda x: abs(x), ascending=False).reset_index(drop=True)
df_corr_results.to_csv('phoneme_cooccurrence_correlation_results.csv', index=False)
print(f"[OK] Correlation results saved as: phoneme_cooccurrence_correlation_results.csv")
