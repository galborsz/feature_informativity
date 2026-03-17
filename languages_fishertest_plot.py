"""
Phonemic Inventory Analysis: Fisher's Exact Test and Linguistic Patterns
Analyzes relationships between phoneme presence and language clusters across feature systems.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from scipy.stats import fisher_exact, rankdata
from statsmodels.stats.multitest import multipletests

# ====================== SETUP AND CONFIGURATION ======================

# Journal-compliant style matching randomreal_plot.py
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})
sns.set_style("white")

# Define inventories and colors
inventories = ["HC", "SPE", "JFH"]
inventory_colors = {
    "HC": "#1f77b4",      # Blue
    "SPE": "#ff7f0e",     # Orange
    "JFH": "#2ca02c"      # Green
}

# ====================== DATA LOADING ======================

print("\n" + "=" * 70)
print("PHONEMIC INVENTORY ANALYSIS")
print("=" * 70)

# Load all data
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Filter to keep only languages present in ALL inventories
language_sets = {inv: set(all_data[inv].keys()) for inv in inventories}
common_languages = set.intersection(*language_sets.values())

print(f"\nLanguage counts before filtering:")
for inv in inventories:
    print(f"  {inv}: {len(language_sets[inv])} languages")
print(f"\nCommon languages across all inventories: {len(common_languages)}")

# Keep only common languages in all_data
for inv in inventories:
    all_data[inv] = {lang: all_data[inv][lang] for lang in common_languages}

# Load pb_languages data
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv", encoding='utf-8')

# ====================== UTILITY FUNCTIONS ======================

def compute_avg_mdl(allsegments, min_lengths, min_descriptions):
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


def permutation_pvalue(x, y, n_perm=5000, seed=0, median=True):
    """Two-sample Monte-Carlo permutation test using median or mean difference as test statistic."""
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


def paired_signflip_test(diffs, nperm=10000, seed=123, two_sided=True):
    """
    Paired sign-flip test: tests whether median of differences is significantly different from 0.
    
    Parameters:
    -----------
    diffs : array-like
        Array of paired differences
    nperm : int
        Number of permutations
    seed : int
        Random seed
    two_sided : bool
        If True, perform two-sided test
    
    Returns:
    --------
    p_value : float
        P-value for the test
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    
    # Observed test statistic: median of absolute differences
    obs_median = np.abs(np.median(diffs))
    
    # Count how many permutations give a median at least as extreme
    count = 0
    for _ in range(nperm):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diffs = diffs * signs
        perm_median = np.abs(np.median(perm_diffs))
        
        if perm_median >= obs_median:
            count += 1
    
    p_value = (count + 1) / (nperm + 1)
    return p_value


def cohen_h(p1, p2):
    """
    Compute Cohen's h effect size for the difference between two proportions.
    
    Parameters:
    -----------
    p1, p2 : float
        Proportions (between 0 and 1)
    
    Returns:
    --------
    h : float
        Cohen's h effect size
    """
    # Avoid domain errors by clipping to valid range
    p1 = np.clip(p1, 0, 1)
    p2 = np.clip(p2, 0, 1)
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

# ====================== DATA PREPARATION ======================

print("\n" + "=" * 70)
print("Creating Language × Feature System MDL Heatmap")
print("=" * 70)

# Collect average MDL per language for each inventory
language_mdl_data = []

for inv in inventories:
    print(f"\nProcessing inventory: {inv}")
    
    for language, lang_data in all_data[inv].items():
        if "min_descriptions" in lang_data and "min_lengths" in lang_data:
            min_descriptions = lang_data["min_descriptions"]
            min_lengths = lang_data["min_lengths"]
            
            # Get all phonemes for this language
            allsegments = list(min_descriptions.keys())
            
            # Compute average MDL
            avg_mdl = compute_avg_mdl(allsegments, min_lengths, min_descriptions)
            
            if avg_mdl is not None:
                # Get language family from pb_languages
                lang_rows = pb_languages[pb_languages['language'] == language]
                family = lang_rows.iloc[0]['family'] if not lang_rows.empty else 'Unknown'
                
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
print(f"Feature systems: {df_lang_mdl['inventory'].unique().tolist()}")

# Create heatmap data: pivot to rows=languages, columns=feature systems
heatmap_data = df_lang_mdl.pivot_table(
    index='language',
    columns='inventory',
    values='avg_mdl',
    aggfunc='first'
)
heatmap_data = heatmap_data[inventories]

print(f"\nHeatmap shape: {heatmap_data.shape}")
print(f"  Rows (languages): {heatmap_data.shape[0]}")
print(f"  Columns (feature systems): {heatmap_data.shape[1]}")

# Get language family information
language_to_family = {}
for lang in heatmap_data.index:
    lang_rows = pb_languages[pb_languages['language'] == lang]
    language_to_family[lang] = lang_rows.iloc[0]['family'] if not lang_rows.empty else 'Unknown'

# ====================== CLUSTERING ======================

print("\n" + "=" * 70)
print("Creating Language Clusters")
print("=" * 70)

# Cluster 1: JFH MDL > HC MDL AND JFH MDL > SPE MDL
# Cluster 2: All other languages
clusters_dict = {'1': [], '2': []}

for language in heatmap_data.index:
    hc_mdl = heatmap_data.loc[language, 'HC']
    spe_mdl = heatmap_data.loc[language, 'SPE']
    jfh_mdl = heatmap_data.loc[language, 'JFH']
    
    if jfh_mdl > hc_mdl and jfh_mdl > spe_mdl:
        clusters_dict['1'].append(language)
    else:
        clusters_dict['2'].append(language)

print(f"\nCluster composition:")
print(f"  Clustering method: JFH MDL > HC MDL AND JFH MDL > SPE MDL")
for cluster_id in sorted(clusters_dict.keys()):
    langs = sorted(clusters_dict[cluster_id])
    print(f"  Cluster {cluster_id}: {len(langs)} languages")

# ====================== VIOLIN PLOT ANALYSIS ======================

print("\n" + "=" * 70)
print("Generating Feature System Distribution Violin Plots")
print("=" * 70)

cluster_dir = "language_clusters"
if os.path.exists(cluster_dir):
    shutil.rmtree(cluster_dir)
os.makedirs(cluster_dir)

# Prepare data for overall violin plot across all languages
violin_data_overall = []
for inv in inventories:
    for lang in heatmap_data.index:
        mdl_val = heatmap_data.loc[lang, inv]
        violin_data_overall.append({
            'Feature System': inv,
            'Average MDL': mdl_val,
            'Language': lang
        })

violin_df_overall = pd.DataFrame(violin_data_overall)

# Extract data per feature system
hc_data = heatmap_data['HC'].values
spe_data = heatmap_data['SPE'].values
jfh_data = heatmap_data['JFH'].values

# Perform pairwise permutation tests
pairs_to_test = [
    ('HC', 'SPE', 0, 1, hc_data, spe_data),
    ('HC', 'JFH', 0, 2, hc_data, jfh_data),
    ('SPE', 'JFH', 1, 2, spe_data, jfh_data)
]

pvals_raw = []
effect_sizes = []
for inv1, inv2, idx1, idx2, data1, data2 in pairs_to_test:
    p_val = permutation_pvalue(data1, data2, n_perm=5000, seed=42)
    r_pair = rank_biserial_paired(data1, data2)
    pvals_raw.append(p_val)
    effect_sizes.append(r_pair)

# Apply FDR correction
pvals_array = np.array(pvals_raw)
_, pvals_corrected, _, _ = multipletests(pvals_array, alpha=0.05, method='fdr_bh')

# Create violin plot
fig, ax = plt.subplots(figsize=(11, 7))

ax.set_ylim(1.4, 3.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])
ax.set_yticklabels(['1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=16)

# Violin plot
sns.violinplot(x='Feature System', y='Average MDL', data=violin_df_overall, ax=ax,
               palette=[inventory_colors['HC'], inventory_colors['SPE'], inventory_colors['JFH']],
               linewidth=1.1, inner=None, alpha=0.65, edgecolor='black')

# Strip plot for individual points
sns.stripplot(data=violin_df_overall, x='Feature System', y='Average MDL', ax=ax,
              color='black', alpha=0.4, size=5, jitter=True)

# Add median lines
medians = [np.median(hc_data), np.median(spe_data), np.median(jfh_data)]
for i, median_val in enumerate(medians):
    ax.hlines(median_val, i - 0.4, i + 0.4, colors='red', linewidth=2.5)

# Add significance lines connecting distributions
# Get y-axis range for proportional spacing
y_min_ax, y_max_ax = ax.get_ylim()
y_range = y_max_ax - y_min_ax

# Collect all data to find maximum
all_data_overall = np.concatenate([hc_data, spe_data, jfh_data])
max_y = np.max(all_data_overall)

# Spacing parameters proportional to y-axis range
spacing_offset = y_range * 0.08   # First line 8% above data
line_spacing = y_range * 0.075     # Space between lines 3.5% of y-range
tail_length = y_range * 0.015     # Tail length proportional to range

line_height_offset = spacing_offset

for (inv1, inv2, idx1, idx2, data1, data2), p_corr, effect in zip(pairs_to_test, pvals_corrected, effect_sizes):
    if p_corr < 0.05:
        line_y = max_y + line_height_offset
        
        # Draw horizontal line connecting the two positions
        ax.plot([idx1, idx2], [line_y, line_y], 'k-', linewidth=2)
        
        # Draw tails at the ends
        ax.plot([idx1, idx1], [line_y - tail_length, line_y], 'k-', linewidth=2)
        ax.plot([idx2, idx2], [line_y - tail_length, line_y], 'k-', linewidth=2)
        
        # Add significance stars at the midpoint
        mid_x = (idx1 + idx2) / 2
        stars = p_to_stars(p_corr)
        ax.text(mid_x, line_y + tail_length, stars, ha='center', va='bottom', 
                fontsize=20, fontweight='bold')
        
        # Update offset for next line
        line_height_offset += line_spacing

ax.set_xlabel('Feature System', fontsize=16)
ax.set_ylabel('Average Minimal Description Length', fontsize=16)
ax.set_xticklabels(['HC', 'SPE', 'JFH'], fontsize=16, fontweight='bold')
# ax.set_title(f'Feature System MDL Distributions ({len(heatmap_data)} languages)', 
#              fontsize=18, pad=15)
ax.tick_params(labelsize=16)
ax.grid(True, alpha=0.25, axis='y')
ax.set_axisbelow(True)

# Store for later use in combined plot
overall_ax = ax
overall_fig = fig

plt.close()

# Print statistics
print(f"\nFeature system pairwise comparisons (FDR-corrected):")
for i, (inv1, inv2, idx1, idx2, data1, data2) in enumerate(pairs_to_test):
    print(f"  {inv1} vs {inv2}: p = {pvals_corrected[i]:.4g}, r = {effect_sizes[i]:.3f} {p_to_stars(pvals_corrected[i])}")

# ====================== CLUSTER VIOLIN PLOTS ======================

print("\n" + "=" * 70)
print("Generating Combined Cluster and Feature System Plots")
print("=" * 70)

# Create a GridSpec layout: left column for overall, right column for clusters (2 rows)
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1], height_ratios=[1, 1], 
                       hspace=0.15, wspace=0.08)

# Left: overall distribution spanning both rows
ax_overall = fig.add_subplot(gs[:, 0])
ax_cluster_1 = fig.add_subplot(gs[0, 1])
ax_cluster_2 = fig.add_subplot(gs[1, 1])

axes_all = [ax_overall, ax_cluster_1, ax_cluster_2]
cluster_list = sorted(clusters_dict.keys())

# Plot overall distribution on left
ax_overall.set_ylim(1.4, 3.5)
ax_overall.spines['top'].set_visible(False)
ax_overall.spines['right'].set_visible(False)

ax_overall.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])
ax_overall.set_yticklabels(['1.5', '2.0', '2.5', '3.0', '3.5'], fontsize=16)

sns.violinplot(x='Feature System', y='Average MDL', data=violin_df_overall, ax=ax_overall,
               palette=[inventory_colors['HC'], inventory_colors['SPE'], inventory_colors['JFH']],
               linewidth=1.1, inner=None, alpha=0.65, edgecolor='black')

sns.stripplot(data=violin_df_overall, x='Feature System', y='Average MDL', ax=ax_overall,
              color='black', alpha=0.4, size=5, jitter=True)

medians = [np.median(hc_data), np.median(spe_data), np.median(jfh_data)]
for i, median_val in enumerate(medians):
    ax_overall.hlines(median_val, i - 0.4, i + 0.4, colors='red', linewidth=2.5)

y_min_ax, y_max_ax = ax_overall.get_ylim()
y_range = y_max_ax - y_min_ax
spacing_offset = y_range * 0.08
line_spacing = y_range * 0.075
tail_length = y_range * 0.015
line_height_offset = spacing_offset

for (inv1, inv2, idx1, idx2, data1, data2), p_corr, effect in zip(pairs_to_test, pvals_corrected, effect_sizes):
    if p_corr < 0.05:
        line_y = max_y + line_height_offset
        ax_overall.plot([idx1, idx2], [line_y, line_y], 'k-', linewidth=2)
        ax_overall.plot([idx1, idx1], [line_y - tail_length, line_y], 'k-', linewidth=2)
        ax_overall.plot([idx2, idx2], [line_y - tail_length, line_y], 'k-', linewidth=2)
        mid_x = (idx1 + idx2) / 2
        stars = p_to_stars(p_corr)
        ax_overall.text(mid_x, line_y + tail_length, stars, ha='center', va='bottom', 
                fontsize=16)
        line_height_offset += line_spacing

ax_overall.set_xlabel('Feature System', fontsize=16)
ax_overall.set_ylabel('Average Minimal Description Length', fontsize=16)
ax_overall.set_xticklabels(['HC', 'SPE', 'JFH'], fontsize=16, fontweight='bold')

# Add sample size to title
n_overall = len(heatmap_data)
ax_overall.set_title('All Languages', fontsize=18, fontweight='bold', pad=5)
# Add sample size as text in top right
ax_overall.text(0.55, 0.98, f'n = {n_overall}', transform=ax_overall.transAxes,
               ha='right', va='top', fontsize=14, fontweight='normal',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax_overall.tick_params(labelsize=16)
ax_overall.grid(True, alpha=0.25, axis='y')
ax_overall.set_axisbelow(True)

# Plot cluster distributions on right
for plot_idx, (cluster_id, ax) in enumerate(zip(cluster_list, [ax_cluster_1, ax_cluster_2])):
    langs_in_cluster = sorted(clusters_dict[cluster_id])
    cluster_data = heatmap_data.loc[langs_in_cluster]
    
    # Prepare data for violin plot
    violin_data_list = []
    for inv in inventories:
        for lang in langs_in_cluster:
            mdl_val = cluster_data.loc[lang, inv]
            violin_data_list.append({
                'Feature System': inv,
                'Average MDL': mdl_val,
                'Language': lang
            })
    
    violin_df = pd.DataFrame(violin_data_list)
    
    # Extract data per feature system
    hc_data_cluster = cluster_data['HC'].values
    spe_data_cluster = cluster_data['SPE'].values
    jfh_data_cluster = cluster_data['JFH'].values
    
    # Perform pairwise permutation tests
    pairs_to_test_cluster = [
        ('HC', 'SPE', 0, 1, hc_data_cluster, spe_data_cluster),
        ('HC', 'JFH', 0, 2, hc_data_cluster, jfh_data_cluster),
        ('SPE', 'JFH', 1, 2, spe_data_cluster, jfh_data_cluster)
    ]
    
    pvals_raw_cluster = []
    effect_sizes_cluster = []
    for inv1, inv2, idx1, idx2, data1, data2 in pairs_to_test_cluster:
        p_val = permutation_pvalue(data1, data2, n_perm=5000, seed=42)
        r_pair = rank_biserial_paired(data1, data2)
        pvals_raw_cluster.append(p_val)
        effect_sizes_cluster.append(r_pair)
    
    # Apply FDR correction
    pvals_array_cluster = np.array(pvals_raw_cluster)
    _, pvals_corrected_cluster, _, _ = multipletests(pvals_array_cluster, alpha=0.05, method='fdr_bh')
    
    # Format the violin plot
    ax.set_ylim(1.5, 3.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_yticks([2.0, 2.5, 3.0])
    ax.set_yticklabels(['2.0', '2.5', '3.0'], fontsize=14)
    
    # Violin plot with narrower width
    sns.violinplot(x='Feature System', y='Average MDL', data=violin_df, ax=ax,
                   palette=[inventory_colors['HC'], inventory_colors['SPE'], inventory_colors['JFH']],
                   linewidth=1.1, inner=None, alpha=0.65, edgecolor='black', width=0.75)
    
    # Strip plot for individual points
    sns.stripplot(data=violin_df, x='Feature System', y='Average MDL', ax=ax,
                  color='black', alpha=0.4, size=4, jitter=True)
    
    # Add median lines
    medians_cluster = [np.median(hc_data_cluster), np.median(spe_data_cluster), np.median(jfh_data_cluster)]
    for i, median_val in enumerate(medians_cluster):
        ax.hlines(median_val, i - 0.3, i + 0.3, colors='red', linewidth=3)
    
    # Add significance lines connecting distributions
    y_min_ax_c, y_max_ax_c = ax.get_ylim()
    y_range_c = y_max_ax_c - y_min_ax_c
    
    all_data_cluster = np.concatenate([hc_data_cluster, spe_data_cluster, jfh_data_cluster])
    max_y_cluster = np.max(all_data_cluster)
    
    spacing_offset_c = y_range_c * 0.08
    line_spacing_c = y_range_c * 0.12
    tail_length_cluster = y_range_c * 0.015
    
    line_height_offset_cluster = spacing_offset_c
    
    for (inv1, inv2, idx1, idx2, data1, data2), p_corr_cluster, effect_cluster in zip(pairs_to_test_cluster, pvals_corrected_cluster, effect_sizes_cluster):
        if p_corr_cluster < 0.05:
            line_y = max_y_cluster + line_height_offset_cluster
            
            ax.plot([idx1, idx2], [line_y, line_y], 'k-', linewidth=2)
            ax.plot([idx1, idx1], [line_y - tail_length_cluster, line_y], 'k-', linewidth=2)
            ax.plot([idx2, idx2], [line_y - tail_length_cluster, line_y], 'k-', linewidth=2)
            
            mid_x = (idx1 + idx2) / 2
            stars = p_to_stars(p_corr_cluster)
            ax.text(mid_x, line_y + tail_length_cluster, stars, ha='center', va='bottom', 
                    fontsize=16)
            
            line_height_offset_cluster += line_spacing_c
    
    ax.set_xlabel('Feature System', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    ax.set_xticklabels(['HC', 'SPE', 'JFH'], fontsize=16, fontweight='bold')
    
    # Add sample size to title
    n_cluster = len(langs_in_cluster)
    ax.set_title(f'Group {cluster_id}', fontsize=18, fontweight='bold', pad=5)
    # Add sample size as text in top right
    ax.text(0.55, 0.98, f'n = {n_cluster}', transform=ax.transAxes,
           ha='right', va='top', fontsize=14, fontweight='normal',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_axisbelow(True)
    
    # Set x-axis limits to make violins closer together
    ax.set_xlim(-0.5, 2.5)
    
    # Hide x-axis label for top cluster plot
    if plot_idx == 0:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    # Print statistics
    print(f"\n  Cluster {cluster_id} pairwise comparisons (FDR-corrected):")
    for i, (inv1, inv2, idx1, idx2, data1, data2) in enumerate(pairs_to_test_cluster):
        print(f"    {inv1} vs {inv2}: p = {pvals_corrected_cluster[i]:.4g}, r = {effect_sizes_cluster[i]:.3f} {p_to_stars(pvals_corrected_cluster[i])}")

plt.tight_layout()
filename_base = os.path.join(cluster_dir, 'combined_distributions_and_clusters')

# Save in multiple formats
for fmt in ['png', 'pdf', 'tif']:
    filename = f'{filename_base}.{fmt}'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format=fmt)
    print(f"  Saved: {filename}")

plt.close()

# ====================== FISHER'S EXACT TEST ANALYSIS ======================

print("\n" + "=" * 70)
print("Fisher's Exact Test: Phoneme Presence vs. Language Clusters")
print("=" * 70)

# Collect all unique phonemes
all_phonemes = set()
for lang in heatmap_data.index:
    lang_rows = pb_languages[pb_languages['language'] == lang]
    if not lang_rows.empty:
        inventory_str = lang_rows.iloc[0]["core inventory"]
        inventory_str = inventory_str.strip('[]').strip()
        phonemes = set(p.strip().strip('\'"') for p in inventory_str.split(','))
        all_phonemes.update(p for p in phonemes if p)

all_phonemes = sorted(list(all_phonemes))
print(f"\nTotal unique phonemes: {len(all_phonemes)}")

# Create binary phoneme presence matrix
language_phoneme_data = []

for language in heatmap_data.index:
    # Determine cluster
    cluster_id = None
    for cid, langs in clusters_dict.items():
        if language in langs:
            cluster_id = int(cid)
            break
    
    # Get phoneme inventory
    lang_rows = pb_languages[pb_languages['language'] == language]
    if lang_rows.empty:
        continue
    
    inventory_str = lang_rows.iloc[0]["core inventory"]
    inventory_str = inventory_str.strip('[]').strip()
    phonemes_in_lang = set(p.strip().strip('\'"') for p in inventory_str.split(','))
    phonemes_in_lang = set(p for p in phonemes_in_lang if p)
    
    # Create row
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

# Perform Fisher's exact test
print("\nPerforming Fisher's exact test for each phoneme...")

cluster_1, cluster_2 = sorted(df_lang['cluster'].unique())
contingency_results = []
p_values_raw = []

for phoneme in all_phonemes:
    # Build 2x2 contingency table
    cluster_1_langs = df_lang[df_lang['cluster'] == cluster_1]
    cluster_2_langs = df_lang[df_lang['cluster'] == cluster_2]
    
    c1_has = (cluster_1_langs[phoneme] == 1).sum()
    c1_no = (cluster_1_langs[phoneme] == 0).sum()
    c2_has = (cluster_2_langs[phoneme] == 1).sum()
    c2_no = (cluster_2_langs[phoneme] == 0).sum()
    
    contingency_table = [[c1_has, c1_no], [c2_has, c2_no]]
    
    # Perform Fisher's exact test
    try:
        oddsratio, p_value = fisher_exact(contingency_table)
    except:
        oddsratio, p_value = np.nan, 1.0
    
    p_values_raw.append(p_value)
    
    # Calculate proportions
    c1_total = c1_has + c1_no
    c2_total = c2_has + c2_no
    c1_prop_value = (c1_has / c1_total) if c1_total > 0 else 0
    c2_prop_value = (c2_has / c2_total) if c2_total > 0 else 0
    c1_prop = f"{c1_has}/{c1_total} = {(c1_has / c1_total * 100):.1f}%" if c1_total > 0 else "N/A"
    c2_prop = f"{c2_has}/{c2_total} = {(c2_has / c2_total * 100):.1f}%" if c2_total > 0 else "N/A"
    
    # Calculate Cohen's h effect size
    h = cohen_h(c1_prop_value, c2_prop_value)
    
    contingency_results.append({
        'phoneme': phoneme,
        f'cluster_{cluster_1}_has': c1_has,
        f'cluster_{cluster_1}_no': c1_no,
        f'cluster_{cluster_2}_has': c2_has,
        f'cluster_{cluster_2}_no': c2_no,
        f'proportion_cluster_{cluster_1}': c1_prop,
        f'proportion_cluster_{cluster_2}': c2_prop,
        'oddsratio': oddsratio,
        'p_value': p_value,
        'cohens_h': h
    })

# Apply FDR correction
p_values_array = np.array(p_values_raw)
_, p_values_corrected, _, _ = multipletests(p_values_array, alpha=0.05, method='fdr_bh')

for i, result in enumerate(contingency_results):
    result['p_value_corrected'] = p_values_corrected[i]

# Convert to DataFrame and sort
df_contingency = pd.DataFrame(contingency_results)
df_contingency = df_contingency.sort_values('p_value_corrected').reset_index(drop=True)

# Save results
csv_filename = 'fisher_exact_phoneme_cluster_results.csv'
df_contingency.to_csv(csv_filename, index=False)

print(f"\nFisher's Exact Test Summary:")
print(f"  Total phonemes analyzed: {len(df_contingency)}")
print(f"  Significant at α=0.05 (FDR-corrected): {(df_contingency['p_value_corrected'] < 0.05).sum()}")
print(f"  Significant at α=0.01 (FDR-corrected): {(df_contingency['p_value_corrected'] < 0.01).sum()}")
print(f"  Significant at α=0.001 (FDR-corrected): {(df_contingency['p_value_corrected'] < 0.001).sum()}")

print(f"\nTop 20 significant phonemes (ranked by Cohen's h effect size):")
sig_phonemes_df = df_contingency[df_contingency['p_value_corrected'] < 0.05].copy()
sig_phonemes_df['cohens_h_abs'] = sig_phonemes_df['cohens_h'].abs()
sig_phonemes_df = sig_phonemes_df.sort_values('cohens_h_abs', ascending=False)
for idx, row in sig_phonemes_df.head(20).iterrows():
    print(f"  {row['phoneme']:12s}: p_corrected={row['p_value_corrected']:.4g}, Cohen's h={row['cohens_h']:8.3f}, OR={row['oddsratio']:8.3f}")

print(f"\n✓ Fisher's exact test results saved to: {csv_filename}")

# ====================== INDIVIDUAL PHONEME PLOTS ======================

print("\n" + "=" * 70)
print("Creating Individual Phoneme Plots with Language-level MDL")
print("=" * 70)

# Create output directory for phoneme plots
phoneme_plots_dir_heatmap = "phoneme_cluster_presence_plots_heatmap_mdl"
if os.path.exists(phoneme_plots_dir_heatmap):
    shutil.rmtree(phoneme_plots_dir_heatmap)
os.makedirs(phoneme_plots_dir_heatmap)

# Create a set of significant phonemes from Fisher's test, sorted by Cohen's h effect size
sig_phonemes_df = df_contingency[df_contingency['p_value_corrected'] < 0.05].copy()
# Filter to only positive Cohen's h values
sig_phonemes_df = sig_phonemes_df[sig_phonemes_df['cohens_h'] > 0].copy()
sig_phonemes_df['cohens_h_abs'] = sig_phonemes_df['cohens_h'].abs()
sig_phonemes_df = sig_phonemes_df.sort_values('cohens_h_abs', ascending=False)
# Reverse the list so largest effect size appears at the top of the plot
significant_phonemes_list = list(reversed(sig_phonemes_df['phoneme'].tolist()))
print(f"\nSignificant phonemes (Fisher's exact test, p < 0.05, positive Cohen's h): {len(significant_phonemes_list)}")
print(f"Sorted by Cohen's h effect size (largest effect size at top)")

# Collect data for all significant phonemes
all_phoneme_stats = []
phoneme_plot_data = {}  # Store data for combined plot

for phoneme in all_phonemes:
    # Get languages that contain this phoneme
    languages_with_phoneme = df_lang[df_lang[phoneme] == 1]['language'].tolist()
    
    if len(languages_with_phoneme) == 0:
        continue
    
    # ===== COMPUTE MEDIAN DIFFERENCES: Without vs With =====
    language_diffs = []
    
    for language in languages_with_phoneme:
        if language in heatmap_data.index:
            for inv in inventories:
                mdl_with = heatmap_data.loc[language, inv]
                
                lang_data = all_data[inv][language]
                min_descriptions = lang_data["min_descriptions"]
                min_lengths = lang_data["min_lengths"]
                
                allsegments_excluding = [pho for pho in min_descriptions.keys() if pho != phoneme]
                mdl_without = compute_avg_mdl(allsegments_excluding, min_lengths, min_descriptions)
                
                if mdl_without is not None:
                    diff = mdl_with - mdl_without
                    language_diffs.append({
                        'language': language,
                        'Feature System': inv,
                        'Difference (With - Without)': diff
                    })
    
    diffs_df = pd.DataFrame(language_diffs)
    
    if len(diffs_df) == 0:
        continue
    
    # Get Fisher's exact test results for this phoneme
    fisher_row = df_contingency[df_contingency['phoneme'] == phoneme]
    fisher_p_corrected = fisher_row['p_value_corrected'].iloc[0] if not fisher_row.empty else np.nan
    fisher_oddsratio = fisher_row['oddsratio'].iloc[0] if not fisher_row.empty else np.nan
    fisher_cohens_h = fisher_row['cohens_h'].iloc[0] if not fisher_row.empty else np.nan
    
    # Perform paired sign-flip test
    signflip_results = {}
    for inv in inventories:
        inv_diffs = diffs_df[diffs_df['Feature System'] == inv]['Difference (With - Without)'].values
        if len(inv_diffs) > 1:
            p_val_signflip = paired_signflip_test(inv_diffs, nperm=5000, seed=123, two_sided=True) # TODO: nperm=10000
            obs_median = np.median(inv_diffs)
            signflip_results[inv] = {'p_value': p_val_signflip, 'median': obs_median}
        else:
            signflip_results[inv] = {'p_value': np.nan, 'median': np.nan}
    
    # Collect statistics for all phonemes
    for inv in inventories:
        inv_diffs = diffs_df[diffs_df['Feature System'] == inv]['Difference (With - Without)'].values
        median_diff = np.median(inv_diffs) if len(inv_diffs) > 0 else np.nan
        p_signflip = signflip_results[inv]['p_value']
        
        all_phoneme_stats.append({
            'phoneme': phoneme,
            'feature_system': inv,
            'n_languages_with': len(languages_with_phoneme),
            'median_mdl_difference': median_diff,
            'pval_signflip_median_vs_zero': p_signflip,
            'fisher_exact_p_corrected': fisher_p_corrected,
            'fisher_exact_oddsratio': fisher_oddsratio,
            'fisher_exact_cohens_h': fisher_cohens_h
        })
    
    # Store data if phoneme is significant
    if phoneme in significant_phonemes_list:
        phoneme_plot_data[phoneme] = {
            'diffs_df': diffs_df.copy(),
            'signflip_results': signflip_results,
            'fisher_p': fisher_p_corrected,
            'fisher_or': fisher_oddsratio,
            'fisher_h': fisher_cohens_h,
            'languages_with': languages_with_phoneme
        }

# ====================== CREATE COMBINED FIGURE ======================

if len(significant_phonemes_list) > 0:
    print(f"\nCreating horizontal bar plot with {len(significant_phonemes_list)} significant phonemes...")
    
    # Prepare data for bar plot
    bar_plot_data = []
    
    for phoneme in significant_phonemes_list: # CHANGED
        diffs_df = phoneme_plot_data[phoneme]['diffs_df']
        
        for inv in inventories:
            inv_diffs = diffs_df[diffs_df['Feature System'] == inv]['Difference (With - Without)'].values
            if len(inv_diffs) > 0:
                median_diff = np.median(inv_diffs)
                bar_plot_data.append({
                    'Phoneme': phoneme,
                    'Feature System': inv,
                    'Median MDL Difference': median_diff
                })
    
    df_bar = pd.DataFrame(bar_plot_data)
    
    # Calculate figure height based on number of phonemes
    n_phonemes = len(significant_phonemes_list)
    fig_height = max(8, n_phonemes * 0.4)
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Create positions for grouped bars (butterfly/diverging plot with separate subplots)
    y_positions = np.arange(n_phonemes)
    bar_height = 0.25
    
    # Organize data by phoneme and feature system
    phoneme_data_dict = {}
    for phoneme in significant_phonemes_list:
        phoneme_data_dict[phoneme] = {}
        for inv in inventories:
            inv_data = df_bar[df_bar['Feature System'] == inv]
            phoneme_data = inv_data[inv_data['Phoneme'] == phoneme]
            if not phoneme_data.empty:
                phoneme_data_dict[phoneme][inv] = phoneme_data['Median MDL Difference'].values[0]
            else:
                phoneme_data_dict[phoneme][inv] = 0
    
    # Create figure with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, fig_height), sharey=True, gridspec_kw={'wspace': 0.1})
    
    # Plot left subplot (negative values, reversed x-axis)
    for idx, inv in enumerate(inventories):
        y_pos = []
        values = []
        
        for phoneme_idx, phoneme in enumerate(significant_phonemes_list):
            value = phoneme_data_dict[phoneme].get(inv, 0)
            # Only plot negative values on the left
            if value < 0:
                y_pos.append(phoneme_idx)
                values.append(value)
        
        if y_pos:
            offset = (idx - 1) * bar_height
            colors = [inventory_colors[inv] for _ in values]
            ax_left.barh([p + offset for p in y_pos], values, bar_height, 
                        label=inv, color=colors, alpha=0.8, edgecolor='black', linewidth=1.1)
    
    ax_left.set_ylabel('')  # Remove y-axis label
    ax_left.set_xlabel('')  # Remove individual x-axis label
    ax_left.grid(True, alpha=0.3, axis='x')
    ax_left.set_axisbelow(True)
    ax_left.tick_params(axis='x', labelsize=16)
    
    # Plot right subplot (positive values)
    for idx, inv in enumerate(inventories):
        y_pos = []
        values = []
        
        for phoneme_idx, phoneme in enumerate(significant_phonemes_list):
            value = phoneme_data_dict[phoneme].get(inv, 0)
            # Only plot positive values on the right
            if value > 0:
                y_pos.append(phoneme_idx)
                values.append(value)
        
        if y_pos:
            offset = (idx - 1) * bar_height
            colors = [inventory_colors[inv] for _ in values]
            ax_right.barh([p + offset for p in y_pos], values, bar_height, 
                         label=inv, color=colors, alpha=0.8, edgecolor='black', linewidth=1.1)
    
    ax_right.set_ylabel('')
    ax_right.set_xlabel('')  # Remove individual x-axis label
    ax_right.grid(True, alpha=0.3, axis='x')
    ax_right.set_axisbelow(True)
    ax_right.tick_params(axis='x', labelsize=16)
    
    # Hide y-axis ticks from both subplots
    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels([])
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels([])
    
    # Set x-axis tick labels: fewer ticks and hide the 0
    ax_left.set_xticks([-0.008, -0.006, -0.004, -0.002])
    ax_right.set_xticks([0.002, 0.004, 0.006, 0.008])
    
    # Set symmetric x-axis ranges
    ax_left.set_xlim(-0.008, 0)
    ax_right.set_xlim(0, 0.008)
    
    # Add significance stars based on paired sign-flip test p-values
    # Add stars positioned at the end of each bar
    for phoneme_idx, phoneme in enumerate(significant_phonemes_list):
        # Plot stars for each inventory based on sign-flip test p-value
        for inv_idx, inv in enumerate(inventories):
            value = phoneme_data_dict[phoneme].get(inv, 0)
            if value != 0 and phoneme in phoneme_plot_data:
                p_val = phoneme_plot_data[phoneme]['signflip_results'][inv]['p_value']
                stars = p_to_stars(p_val)
                if stars:
                    offset = (inv_idx - 1) * bar_height -0.04  # Adjust offset for star position
                    y_pos = phoneme_idx + offset
                    
                    if value < 0:
                        # Left subplot: position star slightly left of bar end
                        ax_left.text(value - 0.00025, y_pos, stars, ha='right', va='center', 
                                   fontsize=16, fontweight='bold', zorder=5)
                    else:
                        # Right subplot: position star slightly right of bar end
                        ax_right.text(value + 0.00025, y_pos, stars, ha='left', va='center', 
                                    fontsize=16, fontweight='bold', zorder=5)
    
    # Add phoneme labels between the two plots (moved a bit to the left)
    fig.canvas.draw()  # Ensure layout is computed
    left_pos = ax_left.get_position()
    right_pos = ax_right.get_position()
    middle_x = (left_pos.x1 + right_pos.x0) / 2
    # middle_x -= 0.01  # Move slightly to the left
    
    for idx, phoneme in enumerate(significant_phonemes_list):
        fig.text(middle_x, left_pos.y0 + (idx + 0.5) * (left_pos.height / n_phonemes), 
                f'/{phoneme}/', ha='center', va='center', fontsize=16, fontweight='bold')
    
        # Add shared x-axis label near the plots (slightly above the bottom)
        fig.text(0.5, 0.02, 'Median Δ Minimal Description Length (present − absent)', 
            ha='center', va='bottom', fontsize=16)
    
    # Add legend to right plot only
    ax_right.legend(title='Feature System', loc='best', fontsize=16, title_fontsize=16)
    
    # Add zero lines
    ax_left.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax_right.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    filename_base = os.path.join(phoneme_plots_dir_heatmap, 'significant_phonemes_mdl_contribution')
    
    # Save in multiple formats
    for fmt in ['png', 'pdf', 'tif']:
        filename = f'{filename_base}.{fmt}'
        plt.savefig(filename, dpi=300, bbox_inches='tight', format=fmt)
        print(f"  Saved: {filename}")
    
    plt.close()

# Save all phoneme statistics to CSV
df_phoneme_stats = pd.DataFrame(all_phoneme_stats)
df_phoneme_stats = df_phoneme_stats.sort_values(['fisher_exact_p_corrected', 'phoneme', 'feature_system'])
csv_phoneme_stats_filename = 'phoneme_mdl_statistics.csv'
df_phoneme_stats.to_csv(csv_phoneme_stats_filename, index=False)

print(f"\n✓ Combined phoneme figure saved in: {phoneme_plots_dir_heatmap}")
print(f"✓ All phoneme statistics saved to: {csv_phoneme_stats_filename}")
print(f"  - Total phonemes: {df_phoneme_stats['phoneme'].nunique()}")
print(f"  - Phonemes with plots (significant): {len(significant_phonemes_list)}")
print(f"  - Phoneme-feature system combinations: {len(df_phoneme_stats)}")

# ====================== COMPLETION ======================

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
