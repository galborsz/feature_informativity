import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import sys
from statsmodels.stats.multitest import multipletests
import seaborn as sns

# === Journal-compliant style ===
plt.rcParams.update({
    'font.family': 'Times New Roman',      # or 'serif' if you use Doulos SIL
    'font.size': 10,                       # base size
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,                # thicker lines
    'axes.linewidth': 1.0,
})

# For even better control, you can also set it per-axis
sns.set_style("white")   # or "ticks"

# Define inventories
inventories = ["HC", "SPE", "JFH"]
# Define colors for each feature system
inventory_colors = {
    "HC": "#1f77b4",      # Blue
    "SPE": "#ff7f0e",     # Orange
    "JFH": "#2ca02c"      # Green
}
random_grey = "#808080"  # Grey for random/null distributions

# Load all data first
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Filter to keep only languages present in ALL inventories
language_sets = {inv: set(all_data[inv].keys()) for inv in inventories}
common_languages = set.intersection(*language_sets.values())

# Keep only common languages in all_data
for inv in inventories:
    all_data[inv] = {lang: all_data[inv][lang] for lang in common_languages}

# Load pb_languages data
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv", encoding='utf-8')

# Configuration
NUM_SAMPLES = 1000  

print("\n" + "-" * 60)
print("Generating random samples and computing informativity...")
print("-" * 60)

# Step 1: Collect all unique phonemes from pb_languages_formatted.csv to create phoneme pool
all_phonemes = set()
for _, row in pb_languages.iterrows():
    inventory_str = row["core inventory"]
    # Parse the inventory string which is in Python list format: ['a', 'b', ...]
    # Remove brackets and split by comma
    inventory_str = inventory_str.strip('[]')
    phonemes = [p.strip().strip('\'"') for p in inventory_str.split(',')]
    for phoneme in phonemes:
        if phoneme and phoneme != "":
            all_phonemes.add(phoneme)

phoneme_pool = list(all_phonemes)
print(f"Total unique phonemes in pool: {len(phoneme_pool)}")

def readinventory(filename):
    """Read phoneme inventory and store in a dictionary."""
    featdict = {}
    allsegments = set()

    lines = [line.strip() for line in open(f'feature_sets/{filename}_features.txt')]
    fields = lines[0].split()
    for f in fields:
        featdict[f] = {}
        featdict[f]['name'] = f # name of the feature
        featdict[f]['+'] = set() # phonemes with a + for that feature
        featdict[f]['-'] = set() # phonemes with a - for that feature
    for i in range(1, len(lines)):
        thisline = lines[i]
        if len(thisline) == 0:
            continue
        linefields = thisline.split()
        if len(linefields)!= len(fields) + 1 :
            print(f"Field length mismatch on line {i+1}")
            quit()
        phoneme = linefields[0]
        allsegments |= {phoneme}
        for j in range(1,len(linefields)):
            if linefields[j] == '+' or linefields[j] == '-':
                featdict[fields[j-1]][linefields[j]] |= {phoneme}

    return featdict, allsegments

def reccheck(fd, basefeats, basemodes, feats, modes, correct, baseindex, current_base):
    """
    Start with an empty set of features and keep adding features one by one with different starting phonemes, generating all possible unique feature combinations.
    Check if the generated feature combinations are natural classes for the given phoneme.
    
    Optimizations:
    - Pass current_base to avoid recalculating intersection each time
    - Use tuple for feats to enable hashing and memoization
    """

    def store_feats(fd, feats, modes):
        """Store features for one solution in dictionary indexed by length."""
        global solutions
        length = len(feats)
        if length not in solutions:
            solutions[length] = []
        thissol = []
        for idx, feat in enumerate(feats):
            thissol.append(modes[idx] + fd[feat]['name'])
        solutions[length].append('[' + ','.join(thissol) + ']')
        
    global maxlen
    if len(feats) > maxlen: # Bound the search (max: total amount of features)
        return
    
    # Check if current combination is a solution
    if current_base == correct: # New solution
        store_feats(fd, feats, modes) # if proposed feature combination is a natural class, store solution
        if len(feats) < maxlen:
            maxlen = len(feats)
    
    numelem = len(basefeats)
    # This for loop iterates over all possible indeces and generates all possible feature combinations
    for i in range(baseindex, numelem):  # Add one feature
        if basefeats[i] not in feats:    # If we didn't add this already
            # Calculate new base once
            new_base = current_base & fd[basefeats[i]][basemodes[i]]
            if new_base:  # Only recurse if there are still phonemes in the set
                reccheck(fd, basefeats, basemodes, feats + [basefeats[i]], modes + [basemodes[i]], correct, i + 1, new_base)
    return

def get_general_info_natural_classes(natural_classes, keys):
    """Get descriptive information for the given natural classes - Optimized"""

    min_lengths = {} # store the length of the minimal description where each feature is included
    min_lengths_phonemes = {}
    avg_lengths = {key: [0,0] for key in keys} # store the average lengths of all descriptions per feature
    min_descriptions = {} # store the minimal descriptions of each phoneme
    count_phoneme = {} # The number of times the feature is included in the minimal description of a phoneme
    count_lengths = {} # Count of minimal descriptions for various lengths

    for phoneme, sublists in natural_classes.items():
        # Pre-parse all sublists once
        parsed_sublists = []
        for sublist in sublists:
            parsed = sublist.strip("[]").split(',')
            parsed_sublists.append(parsed)
            
            # Process features in this sublist
            for value in parsed:
                value = value.strip('+-')  # Combined strip for efficiency
                
                # Update min_lengths
                if value in min_lengths:
                    min_lengths[value] = min(min_lengths[value], len(parsed))
                else:
                    min_lengths[value] = len(parsed)

                # Update avg_lengths
                if value in avg_lengths:
                    avg_lengths[value][0] += len(parsed)
                    avg_lengths[value][1] += 1
            
            # Update min_lengths_phonemes
            if phoneme in min_lengths_phonemes:
                min_lengths_phonemes[phoneme] = min(min_lengths_phonemes[phoneme], len(parsed))
            else: 
                min_lengths_phonemes[phoneme] = len(parsed)
        
        # Get minimal descriptions for this phoneme
        min_len = min_lengths_phonemes[phoneme]
        min_descriptions[phoneme] = [parsed for parsed in parsed_sublists if len(parsed) == min_len]
        
        # Count features in minimal descriptions
        for sublist in min_descriptions[phoneme]:
            for value in sublist:
                value = value.strip('+-')
                count_phoneme[value] = count_phoneme.get(value, 0) + 1
            
            sublist_len = len(sublist)
            count_lengths[sublist_len] = count_lengths.get(sublist_len, 0) + 1
                        
    avg_lengths = {k: v[0] / v[1] if v[1] != 0 else 0 for k, v in avg_lengths.items()}
    
    return min_lengths, min_descriptions, count_phoneme, avg_lengths, count_lengths


def process_phoneme_inventory(allsegments, fd, features):
    """Process a phoneme inventory and return natural classes per phoneme."""
    natural_classes_perphoneme = {}
    global solutions, maxlen
    
    for phoneme in allsegments:
        testset = {phoneme}
        base = allsegments
        feats, modes = [], []

        # Find all features that describe this phoneme
        for feat in features:
            if testset <= fd[feat]['+']:
                base = base & fd[feat]['+']
                feats.append(feat)
                modes.append('+')
            elif testset <= fd[feat]['-']:
                base = base & fd[feat]['-']
                feats.append(feat)
                modes.append('-')

        solutions = {}
        # Check if we have a valid natural class
        if base == testset: 
            maxlen = len(feats)
            reccheck(fd, feats, modes, [], [], base, 0, allsegments)
            
            # Store only the solutions
            if phoneme not in natural_classes_perphoneme:
                natural_classes_perphoneme[phoneme] = []
            
            for s in solutions.values():
                natural_classes_perphoneme[phoneme].extend(s)
    
    return natural_classes_perphoneme

# Function to store features for one solution (thread-safe version)
def store_feats(solutions_dict, maxlen_ref, fd, feats, modes):
    length_feats = len(feats)
    if length_feats not in solutions_dict:
        solutions_dict[length_feats] = []
    thissol = []
    for idx, feat in enumerate(feats):
        thissol.append(modes[idx] + fd[feat]["name"])
    solutions_dict[length_feats].append("[" + ",".join(thissol) + "]")

# Function to compute average MDL
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

# Function to compute random sample MDL (for parallel execution)
def compute_random_sample_mdl(inventory_size, phoneme_pool, featdict, features):
    # Randomly sample phonemes from the pool
    sampled_phonemes = set(random.sample(phoneme_pool, inventory_size))

    # Compute natural classes for the sampled phonemes
    natural_classes = process_phoneme_inventory(sampled_phonemes, featdict, features)
    
    # Get informativity information
    sample_min_lengths, sample_min_descriptions, _, _, _ = get_general_info_natural_classes(natural_classes, features)

    random_avg_mdl = compute_avg_mdl(sampled_phonemes, sample_min_lengths, sample_min_descriptions)
    
    if random_avg_mdl:
        return random_avg_mdl
            
    return None

# # Collect average MDL per language for each inventory (using simple mean like Plot 2)
# weighted_avg_mdl = {inv: {"Real": [], "Random": []} for inv in inventories}

# print(f"\nUsing ProcessPoolExecutor for parallel processing")

# # Process each inventory
# for inv in inventories:
#     print(f"\nProcessing inventory: {inv}")
    
#     # Read the feature dictionary for this inventory
#     featdict, all_segments_inv = readinventory(inv)
#     features = list(featdict.keys())
    
#     lang_count = 0
    
#     for language, lang_data in all_data[inv].items():
#         min_lengths = lang_data["min_lengths"]
#         min_descriptions = lang_data["min_descriptions"]
        
#         # Get the actual phoneme inventory from pb_languages_formatted.csv
#         lang_rows = pb_languages[pb_languages['language'] == language]
#         if lang_rows.empty:
#             continue  # Skip if language not found in CSV
        
#         inventory_str = lang_rows.iloc[0]["core inventory"]
#         # Parse the inventory string which is in Python list format: ['a', 'b', ...]
#         inventory_str = inventory_str.strip('[]')
#         allsegments_list = [p.strip().strip('\'"') for p in inventory_str.split(',')]
#         allsegments = {p for p in allsegments_list if p and p != ""}
#         inventory_size = len(allsegments)
        
#         # Compute real weighted average MDL
#         if min_lengths:
#             # mdl_values_lang = list(min_lengths.values())
#             # real_avg_mdl = np.mean(mdl_values_lang)
#             real_avg_mdl = compute_weighted_avg_mdl(allsegments, min_lengths, min_descriptions)
#             weighted_avg_mdl[inv]["Real"].append(real_avg_mdl)
#             lang_count += 1
            
#             # Generate NUM_SAMPLES random inventories and compute their average MDL in parallel
#             with ProcessPoolExecutor(max_workers=4) as executor:
#                 futures = []
#                 for sample_num in range(NUM_SAMPLES):
#                     future = executor.submit(compute_random_sample_mdl, 
#                                             inventory_size, phoneme_pool, featdict, features)
#                     futures.append(future)
                
#                 sample_avg_mdls = []
#                 for future in as_completed(futures):
#                     result = future.result()
#                     if result is not None:
#                         sample_avg_mdls.append(result)

#             # sample_avg_mdls = []
#             # for sample_num in range(NUM_SAMPLES):
#             #     sample_mdl = compute_random_sample_mdl(inventory_size, phoneme_pool, featdict, features)
#             #     if sample_mdl is not None:
#             #         sample_avg_mdls.append(sample_mdl)
            
#             # Compute mean of random samples
#             if sample_avg_mdls:
#                 mean_random_mdl = np.mean(sample_avg_mdls)
#                 weighted_avg_mdl[inv]["Random"].append(mean_random_mdl)
            
#             if lang_count % 10 == 0:
#                 print(f"  Processed {lang_count} languages...")
    
#     print(f"  Processed {lang_count} languages")
#     print(f"  Real samples: {len(avg_mdl[inv]['Real'])}")
#     print(f"  Random samples: {len(avg_mdl[inv]['Random'])}")

# # Save avg_mdl to JSON file
# output_filename = "avg_mdl_data.json"
# with open(output_filename, 'w') as f:
#     json.dump(avg_mdl, f, indent=4)
# print(f"\navg_mdl data saved to {output_filename}")

# Load avg_mdl data from JSON file
input_filename = "weighted_avg_mdl_data.json"
if os.path.exists(input_filename):
    with open(input_filename, 'r') as f:
        avg_mdl = json.load(f)
    print(f"\nLoaded avg_mdl data from {input_filename}")
else:
    print(f"\nWarning: {input_filename} not found, using computed data")

# Helper function to count frequencies (equivalent to Julia's count_frequencies)
def count_frequencies(values, bin_edges):
    counts = [0] * (len(bin_edges) - 1)
    for val in values:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= val < bin_edges[i + 1]:
                counts[i] += 1
                break
            elif i == len(bin_edges) - 2 and val == bin_edges[-1]:
                counts[i] += 1
                break
    return counts

# ====================== SEPARATE FIGURES FOR OBSERVED AND NULL DISTRIBUTIONS ======================
def plot_observed_distributions_figure(all_data_by_inv, seed=42):
    """
    Creates a figure with observed distributions:
    - Top row: HC and SPE side by side
    - Bottom row: JFH centered in the middle
    
    Args:
        all_data_by_inv: dict with keys ['HC', 'SPE', 'JFH'] containing {'Real': [...], 'Random': [...]}
        seed: Random seed for reproducibility
    """
    
    # Create figure with custom GridSpec layout
    # 2 rows, 4 columns: HC and SPE take up 2 cols each on top, JFH takes up 2 middle cols on bottom
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 4, figure=fig)
    
    ax_hc = fig.add_subplot(gs[0, 0:2])
    ax_spe = fig.add_subplot(gs[0, 2:4])
    ax_jfh = fig.add_subplot(gs[1, 1:3])
    
    axes_list = [ax_hc, ax_spe, ax_jfh]
    
    # First pass: calculate shared axis ranges and bin edges
    obs_x_min, obs_x_max = float('inf'), float('-inf')
    obs_y_max = float('-inf')
    
    for inv in inventories:
        real_data = np.array(all_data_by_inv[inv]["Real"])
        random_data = np.array(all_data_by_inv[inv]["Random"])
        
        all_values = np.concatenate([real_data, random_data])
        obs_x_min = min(obs_x_min, all_values.min())
        obs_x_max = max(obs_x_max, all_values.max())
        
        bin_width = (all_values.max() - all_values.min()) / 50
        bin_edges = np.arange(all_values.min(), all_values.max() + bin_width, bin_width)
        real_hist, _ = np.histogram(real_data, bins=bin_edges)
        random_hist, _ = np.histogram(random_data, bins=bin_edges)
        obs_y_max = max(obs_y_max, real_hist.max(), random_hist.max())
    
    # Calculate shared bin edges for all observed distributions
    shared_bin_width = (obs_x_max - obs_x_min) / 50
    shared_bin_edges = np.arange(obs_x_min, obs_x_max + shared_bin_width, shared_bin_width)
    
    # Process each inventory
    for plot_idx, inv in enumerate(inventories):
        real_data = np.array(all_data_by_inv[inv]["Real"])
        random_data = np.array(all_data_by_inv[inv]["Random"])
        
        inv_color = inventory_colors[inv]
        
        ax = axes_list[plot_idx]
        
        # ------------------- Panel: Observed Distributions -------------------
        real_hist, _ = np.histogram(real_data, bins=shared_bin_edges)
        random_hist, _ = np.histogram(random_data, bins=shared_bin_edges)
        x_vals = (shared_bin_edges[:-1] + shared_bin_edges[1:]) / 2
        
        ax.bar(x_vals, real_hist, width=shared_bin_width*0.92, alpha=0.65,
                   color=inv_color, label='Real', edgecolor='black', linewidth=1.1)
        ax.bar(x_vals, random_hist, width=shared_bin_width*0.92, alpha=0.65,
                   color=random_grey, label='Random', edgecolor='black', linewidth=1.1)
        
        obs_median_real = np.median(real_data)
        obs_median_random = np.median(random_data)
        
        ax.axvline(obs_median_real, color=inv_color, linestyle='--', linewidth=2.8)
        ax.axvline(obs_median_random, color=random_grey, linestyle='--', linewidth=2.8)
        
        # Add text annotations for medians positioned to the side
        # Position labels higher on the y-axis
        y_pos = obs_y_max * 0.9
        
        # Determine which median is left and which is right
        if obs_median_real < obs_median_random:
            # Real is on the left, Random is on the right
            ax.text(obs_median_real - (obs_x_max - obs_x_min) * 0.03, y_pos, 
                    f'Median\n{obs_median_real:.3f}', 
                    fontsize=14, color=inv_color, fontweight='bold', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
            ax.text(obs_median_random + (obs_x_max - obs_x_min) * 0.03, y_pos, 
                    f'Median\n{obs_median_random:.3f}', 
                    fontsize=14, color=random_grey, fontweight='bold', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            # Random is on the left, Real is on the right
            ax.text(obs_median_random - (obs_x_max - obs_x_min) * 0.03, y_pos, 
                    f'Median\n{obs_median_random:.3f}', 
                    fontsize=14, color=random_grey, fontweight='bold', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
            ax.text(obs_median_real + (obs_x_max - obs_x_min) * 0.03, y_pos, 
                    f'Median\n{obs_median_real:.3f}', 
                    fontsize=14, color=inv_color, fontweight='bold', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Set shared axis ranges with increased y-max to ensure nothing gets cut off
        ax.set_xlim(obs_x_min, obs_x_max)
        ax.set_ylim(0, obs_y_max * 1.25)
        
        ax.set_xlabel('Average Minimal Description Length', fontsize=16)
        ax.set_ylabel('Language Count', fontsize=16)
        ax.set_title(f'Feature system: {inv}', fontsize=18, fontweight='bold')
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.25, axis='y')
        
        # Clean appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def statistical_analysis(all_data_by_inv, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
        
    row_data = []  # Store permutation data for each inventory
    
    # First pass: calculate shared y-axis range
    null_y_max = float('-inf')
    
    for inv in inventories:
        real_data = np.array(all_data_by_inv[inv]["Real"])
        random_data = np.array(all_data_by_inv[inv]["Random"])
        
        obs_median_real = np.median(real_data)
        obs_median_random = np.median(random_data)
        obs_diff = obs_median_real - obs_median_random
        
        pooled = np.concatenate([real_data, random_data])
        n_real = len(real_data)
        perm_diffs = []
        
        for _ in range(n_perm):
            rng.shuffle(pooled)
            x_perm = pooled[:n_real]
            y_perm = pooled[n_real:]
            perm_diffs.append(np.median(x_perm) - np.median(y_perm))
        
        perm_diffs_counts, _ = np.histogram(perm_diffs, bins=40, density=True)
        null_y_max = max(null_y_max, perm_diffs_counts.max())
    
    # Process each inventory
    for row_idx, inv in enumerate(inventories):
        real_data = np.array(all_data_by_inv[inv]["Real"])
        random_data = np.array(all_data_by_inv[inv]["Random"])
        
        # ====================== Permutation test ======================
        obs_median_real = np.median(real_data)
        obs_median_random = np.median(random_data)
        obs_diff = obs_median_real - obs_median_random
        
        pooled = np.concatenate([real_data, random_data])
        n_real = len(real_data)
        perm_diffs = []
        
        for _ in range(n_perm):
            rng.shuffle(pooled)
            x_perm = pooled[:n_real]
            y_perm = pooled[n_real:]
            perm_diffs.append(np.median(x_perm) - np.median(y_perm))
        
        perm_diffs = np.array(perm_diffs)
        
        # Two-sided p-value (Monte-Carlo)
        count = np.sum(np.abs(perm_diffs) >= np.abs(obs_diff))
        p_value = (count + 1) / (n_perm + 1)
        
        row_data.append({
            'real': real_data,
            'random': random_data,
            'perm_diffs': perm_diffs,
            'obs_diff': obs_diff,
            'obs_median_real': obs_median_real,
            'obs_median_random': obs_median_random,
            'p_value': p_value
        })
        
    return row_data


# ====================== USAGE: CREATE SEPARATE FIGURES ======================
# Create separate figures for observed and null distributions
fig_obs = plot_observed_distributions_figure(avg_mdl, seed=42)
row_data = statistical_analysis(avg_mdl, n_perm=5000, seed=42)

# Save observed distributions at publication quality
fig_obs.savefig("mdl_observed_distributions.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
fig_obs.savefig("mdl_observed_distributions.tif", dpi=1200, bbox_inches='tight', pad_inches=0.05)
fig_obs.savefig("mdl_observed_distributions.png", dpi=1200, bbox_inches='tight', pad_inches=0.05)

plt.close(fig_obs)

# Print summary statistics
print("\n" + "="*70)
print("PERMUTATION TEST RESULTS - MINIMAL DESCRIPTION LENGTH ANALYSIS")
print("="*70)

for idx, inv in enumerate(inventories):
    data = row_data[idx]
    real = data['real']
    random = data['random']
    perm_diffs = data['perm_diffs']
    obs_diff = data['obs_diff']
    obs_median_real = data['obs_median_real']
    obs_median_random = data['obs_median_random']
    p_value = data['p_value']
    
    # Calculate additional statistics
    n_real = len(real)
    n_random = len(random)
    n_perm = len(perm_diffs)
    
    # Effect size: Cohen's d-like measure
    pooled_std = np.sqrt((np.std(real, ddof=1)**2 + np.std(random, ddof=1)**2) / 2)
    effect_size = obs_diff / pooled_std if pooled_std > 0 else 0
    
    # Confidence interval for the permutation difference distribution
    ci_lower = np.percentile(perm_diffs, 2.5)
    ci_upper = np.percentile(perm_diffs, 97.5)
    
    # Additional statistics
    mean_real = np.mean(real)
    mean_random = np.mean(random)
    std_real = np.std(real, ddof=1)
    std_random = np.std(random, ddof=1)
    
    print(f"\n{inv} Feature System:")
    print("-" * 70)
    print(f"  Sample Sizes:")
    print(f"    Real languages: n = {n_real}")
    print(f"    Random samples: n = {n_random}")
    print(f"\n  Central Tendency (Medians):")
    print(f"    Real median MDL:     {obs_median_real:.4f}")
    print(f"    Random median MDL:   {obs_median_random:.4f}")
    print(f"    Observed difference: {obs_diff:.4f}")
    print(f"\n  Central Tendency (Means):")
    print(f"    Real mean MDL:       {mean_real:.4f} (SD = {std_real:.4f})")
    print(f"    Random mean MDL:     {mean_random:.4f} (SD = {std_random:.4f})")
    print(f"\n  Permutation Test Details:")
    print(f"    Test name: Two-sided Monte-Carlo permutation test")
    print(f"    Test statistic: Difference in medians")
    print(f"    Number of permutations: {n_perm:,}")
    print(f"    Permutation dist. mean: {np.mean(perm_diffs):.4f}")
    print(f"    Permutation dist. SD:   {np.std(perm_diffs, ddof=1):.4f}")
    print(f"\n  Statistical Results:")
    print(f"    p-value (two-sided): {p_value:.4g}")
    print(f"    Effect size (Cohen's d-like): {effect_size:.4f}")
    print(f"    95% CI of perm. dist.: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\n  Interpretation:")
    if p_value < 0.001:
        sig = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig = "** (p < 0.01)"
    elif p_value < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "ns (p >= 0.05)"
    print(f"    Significance: {sig}")
    print(f"    Direction: {'Real > Random' if obs_diff > 0 else 'Real < Random' if obs_diff < 0 else 'No difference'}")

print("\n" + "="*70)
print("✓ Saved observed distributions: mdl_observed_distributions.pdf/tif/png")
print("✓ Saved null distributions: mdl_null_distributions.pdf/tif/png")
print("="*70 + "\n")


