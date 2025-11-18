import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from statsmodels.stats.multitest import multipletests

# Define inventories
inventories = ["HC", "SPE", "JFH"]
colors = ['red', 'green', 'blue']

# Load all data first
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Load pb_languages data
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv")

# Configuration
NUM_SAMPLES = 1000  # Number of random samples per language (matching Julia)

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

# Function to compute weighted average MDL (matching Julia's compute_avg_mdl)
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

# Function to compute random sample MDL (for parallel execution)
def compute_random_sample_mdl(inventory_size, phoneme_pool, featdict, features):
    # Randomly sample phonemes from the pool
    sampled_phonemes = set(random.sample(phoneme_pool, inventory_size))

    # Compute natural classes for the sampled phonemes
    natural_classes = process_phoneme_inventory(sampled_phonemes, featdict, features)
    
    # Get informativity information
    sample_min_lengths, sample_min_descriptions, _, _, _ = get_general_info_natural_classes(natural_classes, features)

    random_avg_mdl = compute_weighted_avg_mdl(sampled_phonemes, sample_min_lengths, sample_min_descriptions)
    
    if random_avg_mdl:
        return random_avg_mdl
            
    return None

# Collect average MDL per language for each inventory (using simple mean like Plot 2)
weighted_avg_mdl = {inv: {"Real": [], "Random": []} for inv in inventories}

print(f"\nUsing ThreadPoolExecutor for parallel processing")

# Process each inventory
for inv in inventories:
    print(f"\nProcessing inventory: {inv}")
    
    # Read the feature dictionary for this inventory
    featdict, all_segments_inv = readinventory(inv)
    features = list(featdict.keys())
    
    lang_count = 0
    
    for language, lang_data in all_data[inv].items():
        min_lengths = lang_data["min_lengths"]
        min_descriptions = lang_data["min_descriptions"]
        
        # Get the actual phoneme inventory from pb_languages_formatted.csv
        lang_rows = pb_languages[pb_languages['language'] == language]
        if lang_rows.empty:
            continue  # Skip if language not found in CSV
        
        inventory_str = lang_rows.iloc[0]["core inventory"]
        # Parse the inventory string which is in Python list format: ['a', 'b', ...]
        inventory_str = inventory_str.strip('[]')
        allsegments_list = [p.strip().strip('\'"') for p in inventory_str.split(',')]
        allsegments = {p for p in allsegments_list if p and p != ""}
        inventory_size = len(allsegments)
        
        # Compute real average using simple mean (same as Plot 2)
        if min_lengths:
            mdl_values_lang = list(min_lengths.values())
            real_avg_mdl = np.mean(mdl_values_lang)
            weighted_avg_mdl[inv]["Real"].append(real_avg_mdl)
            lang_count += 1
            
            # Generate NUM_SAMPLES random inventories and compute their average MDL in parallel
            # with ThreadPoolExecutor(max_workers=4) as executor:
            #     futures = []
            #     for sample_num in range(NUM_SAMPLES):
            #         future = executor.submit(compute_random_sample_mdl, 
            #                                inventory_size, phoneme_pool, featdict, features)
            #         futures.append(future)
                
            #     sample_avg_mdls = []
            #     for future in as_completed(futures):
            #         result = future.result()
            #         if result is not None:
            #             sample_avg_mdls.append(result)

            sample_avg_mdls = []
            for sample_num in range(NUM_SAMPLES):
                sample_mdl = compute_random_sample_mdl(inventory_size, phoneme_pool, featdict, features)
                if sample_mdl is not None:
                    sample_avg_mdls.append(sample_mdl)
            
            # Compute mean of random samples
            if sample_avg_mdls:
                mean_random_mdl = np.mean(sample_avg_mdls)
                weighted_avg_mdl[inv]["Random"].append(mean_random_mdl)
            
            if lang_count % 10 == 0:
                print(f"  Processed {lang_count} languages...")
    
    print(f"  Processed {lang_count} languages")
    print(f"  Real samples: {len(weighted_avg_mdl[inv]['Real'])}")
    print(f"  Random samples: {len(weighted_avg_mdl[inv]['Random'])}")

# Save weighted_avg_mdl to JSON file
output_filename = "weighted_avg_mdl_data.json"
with open(output_filename, 'w') as f:
    json.dump(weighted_avg_mdl, f, indent=4)
print(f"\nweighted_avg_mdl data saved to {output_filename}")

# # Load weighted_avg_mdl data from JSON file
# input_filename = "weighted_avg_mdl_data.json"
# if os.path.exists(input_filename):
#     with open(input_filename, 'r') as f:
#         weighted_avg_mdl = json.load(f)
#     print(f"\nLoaded weighted_avg_mdl data from {input_filename}")
# else:
#     print(f"\nWarning: {input_filename} not found, using computed data")

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

# Create bar plots for each inventory
for inv in inventories:
    if weighted_avg_mdl[inv]["Real"] or weighted_avg_mdl[inv]["Random"]:
        # Combine all values to determine bin range
        all_values_inv = weighted_avg_mdl[inv]["Real"] + weighted_avg_mdl[inv]["Random"]
        
        if all_values_inv:
            min_val = min(all_values_inv)
            max_val = max(all_values_inv)
            bin_width = (max_val - min_val) / 50
            bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
            
            # Count frequencies for both Real and Random
            real_counts = count_frequencies(weighted_avg_mdl[inv]["Real"], bin_edges)
            random_counts = count_frequencies(weighted_avg_mdl[inv]["Random"], bin_edges)
            
            x_vals = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
            
            # Calculate statistics
            real_median = np.median(weighted_avg_mdl[inv]["Real"]) if weighted_avg_mdl[inv]["Real"] else 0.0
            random_median = np.median(weighted_avg_mdl[inv]["Random"]) if weighted_avg_mdl[inv]["Random"] else 0.0
            
            # Determine color for this inventory
            color_idx = inventories.index(inv)
            color_real = colors[color_idx]
            color_random = 'gray'  # Use gray for random samples
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot bars
            plt.bar(x_vals, real_counts, width=bin_width, alpha=0.6, 
                   color=color_real, label='Real', edgecolor=color_real, linewidth=1.5)
            
            if weighted_avg_mdl[inv]["Random"]:
                plt.bar(x_vals, random_counts, width=bin_width, alpha=0.6,
                       color=color_random, label='Random', edgecolor=color_random, linewidth=1.5)
            
            # Add vertical lines for medians
            if weighted_avg_mdl[inv]["Real"]:
                plt.axvline(real_median, color=color_real, linestyle='--', linewidth=2,
                           label=f'Real Median: {real_median:.2f}')
            
            if weighted_avg_mdl[inv]["Random"]:
                plt.axvline(random_median, color=color_random, linestyle='--', linewidth=2,
                           label=f'Random Median: {random_median:.2f}')
            
            plt.xlabel('Average Minimal Description Length', fontsize=14)
            plt.ylabel('Language Count', fontsize=14)
            plt.title(f'Feature system: {inv}', fontsize=16)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tick_params(labelsize=12)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(f"mdl_distribution_{inv}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nPlot saved as mdl_distribution_{inv}.png")
            print(f"Statistics for {inv}:")
            print(f"  Real - Total languages: {len(weighted_avg_mdl[inv]['Real'])}")
            if weighted_avg_mdl[inv]["Real"]:
                print(f"  Real - Mean: {np.mean(weighted_avg_mdl[inv]['Real']):.2f}")
                print(f"  Real - Median: {real_median:.2f}")
            if weighted_avg_mdl[inv]["Random"]:
                print(f"  Random - Total samples: {len(weighted_avg_mdl[inv]['Random'])}")
                print(f"  Random - Mean: {np.mean(weighted_avg_mdl[inv]['Random']):.2f}")
                print(f"  Random - Median: {random_median:.2f}")

# ============================================================
# PLOT 3 DENSITY VERSION: Individual inventory density plots with Real and Random
# ============================================================

print("\n" + "-" * 60)
print("Creating density plot versions of Plot 3...")
print("-" * 60)

# Create individual density plots for each inventory
for inv in inventories:
    if weighted_avg_mdl[inv]["Real"] or weighted_avg_mdl[inv]["Random"]:
        # Determine color for this inventory
        color_idx = inventories.index(inv)
        color_real = colors[color_idx]
        color_random = 'gray'
        
        # Calculate statistics
        real_median = np.median(weighted_avg_mdl[inv]["Real"]) if weighted_avg_mdl[inv]["Real"] else 0.0
        random_median = np.median(weighted_avg_mdl[inv]["Random"]) if weighted_avg_mdl[inv]["Random"] else 0.0
        
        # Create density plot
        plt.figure(figsize=(12, 8))
        
        # Add Real data density (KDE only, no histogram)
        if weighted_avg_mdl[inv]["Real"]:
            from scipy.stats import gaussian_kde
            kde_real = gaussian_kde(weighted_avg_mdl[inv]["Real"])
            x_range = np.linspace(min(weighted_avg_mdl[inv]["Real"]), 
                                max(weighted_avg_mdl[inv]["Real"]), 100)
            plt.plot(x_range, kde_real(x_range), color=color_real, linewidth=2.5, label='Real')
            plt.fill_between(x_range, kde_real(x_range), alpha=0.4, color=color_real)
        
        # Add Random data density (KDE only, no histogram)
        if weighted_avg_mdl[inv]["Random"]:
            kde_random = gaussian_kde(weighted_avg_mdl[inv]["Random"])
            x_range_random = np.linspace(min(weighted_avg_mdl[inv]["Random"]), 
                                       max(weighted_avg_mdl[inv]["Random"]), 100)
            plt.plot(x_range_random, kde_random(x_range_random), color=color_random, linewidth=2.5, label='Random')
            plt.fill_between(x_range_random, kde_random(x_range_random), alpha=0.3, color=color_random)
        
        # Add vertical lines for medians
        if weighted_avg_mdl[inv]["Real"]:
            plt.axvline(real_median, color=color_real, linestyle='--', linewidth=2,
                       label=f'Real Median: {real_median:.2f}')
        
        if weighted_avg_mdl[inv]["Random"]:
            plt.axvline(random_median, color=color_random, linestyle='--', linewidth=2,
                       label=f'Random Median: {random_median:.2f}')
        
        plt.xlabel('Average Minimal Description Length', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Feature system: {inv}', fontsize=16)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=12)
        
        # Perform Monte-Carlo permutation test between Real and Random
        if weighted_avg_mdl[inv]["Real"] and weighted_avg_mdl[inv]["Random"]:
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
            
            p_real_random = permutation_pvalue(weighted_avg_mdl[inv]["Real"], 
                                               weighted_avg_mdl[inv]["Random"],
                                               n_perm=5000, seed=42)
            r_real_random = rank_biserial_unpaired(weighted_avg_mdl[inv]["Real"], 
                                                   weighted_avg_mdl[inv]["Random"])
            n_real = len(weighted_avg_mdl[inv]["Real"])
            n_random = len(weighted_avg_mdl[inv]["Random"])
            
            print(f"\n  Monte-Carlo permutation test for {inv} (Real vs Random):")
            print(f"    p-value = {p_real_random:.4g}")
            print(f"    effect size (rank-biserial) = {r_real_random:.3f}")
            print(f"    sample size (Real) = {n_real}")
            print(f"    sample size (Random) = {n_random}")
        
        # Save density plot
        plt.tight_layout()
        plt.savefig(f"mdl_distribution_{inv}_density.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nDensity plot saved as mdl_distribution_{inv}_density.png")

print("\n" + "-" * 60)
print("Creating violin plot comparing all three feature systems...")
print("-" * 60)

# Create violin plot comparing Real distributions across all three feature systems
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for violin plot
violin_data = [weighted_avg_mdl[inv]["Real"] for inv in inventories]

# Create violin plot
positions = [1, 2, 3]
parts = ax.violinplot(violin_data, positions=positions, widths=0.7, showmeans=False, showmedians=False)

# Color the violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor(colors[i])
    pc.set_linewidth(1.5)

# Color other parts
for partname in ('cbars', 'cmins', 'cmaxes'):
    if partname in parts:
        vp = parts[partname]
        vp.set_edgecolor(colors[i])
        vp.set_linewidth(1.5)

# Add median lines
for i, inv in enumerate(inventories):
    median_val = np.median(weighted_avg_mdl[inv]["Real"])
    ax.plot([positions[i] - 0.2, positions[i] + 0.2], [median_val, median_val], 
            color=colors[i], linewidth=2)

# Set up x-axis
ax.set_xticks(positions)
ax.set_xticklabels(inventories)
ax.set_xlabel('Feature System', fontsize=14)
ax.set_ylabel('Average Minimal Description Length', fontsize=14)
ax.set_title('Real Language Distributions by Feature System', fontsize=16)
ax.grid(True, axis='y', alpha=0.3)

# Calculate statistics and add significance brackets
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

def p_to_stars(p):
    """Map corrected p-value to significance stars."""
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return ''

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

# Collect raw p-values for all three pairwise comparisons
pairs_to_test = [
    ('HC', 'SPE', 0, 1),
    ('SPE', 'JFH', 1, 2),
    ('HC', 'JFH', 0, 2)
]

pvals_raw = []
effect_sizes = []
for inv1, inv2, pos1, pos2 in pairs_to_test:
    p_val = permutation_pvalue(weighted_avg_mdl[inv1]["Real"],
                               weighted_avg_mdl[inv2]["Real"],
                               n_perm=5000, seed=42)
    r_pair = rank_biserial_unpaired(weighted_avg_mdl[inv1]["Real"],
                                    weighted_avg_mdl[inv2]["Real"])
    pvals_raw.append(p_val)
    effect_sizes.append(r_pair)

# Apply Benjamini-Hochberg FDR correction
pvals_array = np.array(pvals_raw)
rejected, pvals_corrected, _, _ = multipletests(pvals_array, alpha=0.05, method='fdr_bh')

# Find max y value for positioning brackets
max_y = max([max(weighted_avg_mdl[inv]["Real"]) for inv in inventories])
data_range = max_y - min([min(weighted_avg_mdl[inv]["Real"]) for inv in inventories])
bracket_spacing = data_range * 0.08
tick_height = data_range * 0.02

# Draw brackets and print results for each comparison
for i, (inv1, inv2, pos1, pos2) in enumerate(pairs_to_test):
    p_adj = pvals_corrected[i]
    stars = p_to_stars(p_adj)
    r_pair = effect_sizes[i]
    
    y_bracket = max_y + (i + 1) * bracket_spacing
    ax.plot([positions[pos1], positions[pos2]], [y_bracket, y_bracket], 
            color='black', linewidth=2)
    ax.plot([positions[pos1], positions[pos1]], [y_bracket - tick_height, y_bracket + tick_height], 
            color='black', linewidth=2)
    ax.plot([positions[pos2], positions[pos2]], [y_bracket - tick_height, y_bracket + tick_height], 
            color='black', linewidth=2)
    ax.text((positions[pos1] + positions[pos2]) / 2, y_bracket + 0.05, stars, 
            ha='center', fontsize=14, fontweight='bold')
    
    print(f"\nMonte-Carlo permutation test - {inv1} vs {inv2}:")
    print(f"  Raw p-value = {pvals_raw[i]:.4g}")
    print(f"  Corrected p-value (BH) = {p_adj:.4g}")
    print(f"  Effect size (rank-biserial) = {r_pair:.3f}")
    print(f"  Significance: {stars if stars else 'ns (not significant)'}")

# Set y-axis limits to accommodate brackets
y_lower = min([min(weighted_avg_mdl[inv]["Real"]) for inv in inventories]) - data_range * 0.05
y_upper = max_y + 4 * bracket_spacing + data_range * 0.08
ax.set_ylim(y_lower, y_upper)

ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("real_distributions_violin.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nViolin plot saved as real_distributions_violin.png")

print("\n✓ All plots created successfully!")

