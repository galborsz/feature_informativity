import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

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
pb_languages = pd.read_csv("pb_languages_formatted.csv")

# Configuration
NUM_SAMPLES = 1  # Number of random samples per language (matching Julia)

print("\n" + "=" * 60)
print("Generating random samples and computing informativity...")
print("=" * 60)

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

# Function to read feature inventory (matching Julia's readinventory)
def readinventory(filename):
    featdict = {}
    allsegments = set()
    
    filepath = os.path.join("feature_sets", f"{filename}_features.txt")
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    fields = lines[0].split()
    
    for f in fields:
        featdict[f] = {
            "name": f,
            "+": set(),
            "-": set()
        }
    
    for i in range(1, len(lines)):
        thisline = lines[i]
        if len(thisline) == 0:
            continue
        linefields = thisline.split()
        if len(linefields) != len(fields) + 1:
            print(f"Field length mismatch on line {i+1}")
            exit()
        phoneme = linefields[0]
        allsegments.add(phoneme)
        for j in range(1, len(linefields)):
            if linefields[j] in ["+", "-"]:
                featdict[fields[j-1]][linefields[j]].add(phoneme)
    
    return featdict, allsegments

# Function to store features for one solution (thread-safe version)
def store_feats(solutions_dict, maxlen_ref, fd, feats, modes):
    length_feats = len(feats)
    if length_feats not in solutions_dict:
        solutions_dict[length_feats] = []
    thissol = []
    for idx, feat in enumerate(feats):
        thissol.append(modes[idx] + fd[feat]["name"])
    solutions_dict[length_feats].append("[" + ",".join(thissol) + "]")

# Recursive function to check natural classes (thread-safe version)
def reccheck(solutions_dict, maxlen_ref, fd, basefeats, basemodes, feats, modes, correct, baseindex, current_base):
    if len(feats) > maxlen_ref[0]:
        return
    
    # Check if current combination is a solution
    if current_base == correct:
        store_feats(solutions_dict, maxlen_ref, fd, feats, modes)
        if len(feats) < maxlen_ref[0]:
            maxlen_ref[0] = len(feats)
    
    numelem = len(basefeats)
    for i in range(baseindex, numelem):
        if basefeats[i] not in feats:
            new_base = current_base.intersection(fd[basefeats[i]][basemodes[i]])
            if new_base:
                reccheck(solutions_dict, maxlen_ref, fd, basefeats, basemodes, 
                        feats + [basefeats[i]], modes + [basemodes[i]], correct, i + 1, new_base)

# Process phoneme inventory and return natural classes per phoneme (thread-safe version)
def process_phoneme_inventory(allsegments, fd, features):
    natural_classes_perphoneme = {}
    
    for phoneme in allsegments:
        testset = {phoneme}
        base = allsegments.copy()
        feats = []
        modes = []
        
        # Find all features that describe this phoneme
        for feat in features:
            if testset.issubset(fd[feat]["+"]):
                base = base.intersection(fd[feat]["+"])
                feats.append(feat)
                modes.append("+")
            elif testset.issubset(fd[feat]["-"]):
                base = base.intersection(fd[feat]["-"])
                feats.append(feat)
                modes.append("-")
        
        # Use local variables instead of global
        local_solutions = {}
        local_maxlen = [len(feats)]
        
        if base == testset:
            reccheck(local_solutions, local_maxlen, fd, feats, modes, [], [], base, 0, allsegments)
            
            if phoneme not in natural_classes_perphoneme:
                natural_classes_perphoneme[phoneme] = []
            
            for s in local_solutions.values():
                natural_classes_perphoneme[phoneme].extend(s)
    
    return natural_classes_perphoneme

# Get descriptive information for natural classes
def get_general_info_natural_classes(natural_classes, keys):
    min_lengths = {}
    min_lengths_phonemes = {}
    avg_lengths = {key: [0, 0] for key in keys}
    min_descriptions = {}
    count_phoneme = {}
    count_lengths = {}
    
    for phoneme, sublists in natural_classes.items():
        parsed_sublists = []
        for sublist in sublists:
            parsed = [s.strip() for s in sublist.strip('[]').split(',')]
            parsed_sublists.append(parsed)
            
            # Process features in this sublist
            for value in parsed:
                value = value.lstrip('+-')
                
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
                value = value.lstrip('+-')
                count_phoneme[value] = count_phoneme.get(value, 0) + 1
            
            sublist_len = len(sublist)
            count_lengths[sublist_len] = count_lengths.get(sublist_len, 0) + 1
    
    avg_lengths = {k: v[0] / v[1] for k, v in avg_lengths.items() if v[1] != 0}
    
    return min_lengths, min_descriptions, count_phoneme, avg_lengths, count_lengths

# Function to compute weighted average MDL (matching Julia's compute_avg_mdl)
def compute_weighted_avg_mdl(allsegments, min_lengths, min_descriptions):
    total_avg_length = 0.0
    feature_count = 0
    
    for phoneme in allsegments:
        if phoneme in min_descriptions:
            feature_descriptions = min_descriptions[phoneme]
            
            # Get unique features: flatten all sublists and strip +/-
            unique_features = set()
            for sublist in feature_descriptions:
                for item in sublist:
                    feature_name = item.lstrip('+-')
                    unique_features.add(feature_name)
            
            # Add MDL for each unique feature
            for feature in unique_features:
                if feature in min_lengths:
                    total_avg_length += min_lengths[feature]
                    feature_count += 1
    
    if feature_count > 0:
        return total_avg_length / feature_count
    return None

# Function to compute simple average MDL (same as Plot 2 approach)
def compute_simple_avg_mdl(min_lengths):
    if min_lengths:
        mdl_values = list(min_lengths.values())
        return np.mean(mdl_values)
    return None

# Function to compute random sample MDL (for parallel execution)
def compute_random_sample_mdl(inventory_size, phoneme_pool, featdict, features):
    # Randomly sample phonemes from the pool
    sampled_phonemes = set(random.sample(phoneme_pool, min(inventory_size, len(phoneme_pool))))
    
    # Compute natural classes for the sampled phonemes
    natural_classes = process_phoneme_inventory(sampled_phonemes, featdict, features)
    
    # Get informativity information
    sample_min_lengths, _, _, _, _ = get_general_info_natural_classes(natural_classes, features)
    
    # Compute simple mean of MDL values for this random sample
    if sample_min_lengths:
        mdl_values_sample = list(sample_min_lengths.values())
        return np.mean(mdl_values_sample)
    return None

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
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for sample_num in range(NUM_SAMPLES):
                    future = executor.submit(compute_random_sample_mdl, 
                                           inventory_size, phoneme_pool, featdict, features)
                    futures.append(future)
                
                sample_avg_mdls = []
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        sample_avg_mdls.append(result)
            
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
                plt.bar(x_vals, random_counts, width=bin_width, alpha=0.5,
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

print("\n" + "=" * 60)
print("Creating density plot versions of Plot 3...")
print("=" * 60)

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
        
        # Add Real data density
        if weighted_avg_mdl[inv]["Real"]:
            plt.hist(weighted_avg_mdl[inv]["Real"], bins=30, density=True, alpha=0.4,
                    color=color_real, label='Real', edgecolor=color_real, linewidth=1.5)
            
            # Add KDE for smoother density
            from scipy.stats import gaussian_kde
            kde_real = gaussian_kde(weighted_avg_mdl[inv]["Real"])
            x_range = np.linspace(min(weighted_avg_mdl[inv]["Real"]), 
                                max(weighted_avg_mdl[inv]["Real"]), 100)
            plt.plot(x_range, kde_real(x_range), color=color_real, linewidth=2.5)
        
        # Add Random data density
        if weighted_avg_mdl[inv]["Random"]:
            plt.hist(weighted_avg_mdl[inv]["Random"], bins=30, density=True, alpha=0.3,
                    color=color_random, label='Random', edgecolor=color_random, linewidth=1.5)
            
            # Add KDE for smoother density
            kde_random = gaussian_kde(weighted_avg_mdl[inv]["Random"])
            x_range_random = np.linspace(min(weighted_avg_mdl[inv]["Random"]), 
                                       max(weighted_avg_mdl[inv]["Random"]), 100)
            plt.plot(x_range_random, kde_random(x_range_random), color=color_random, linewidth=2.5)
        
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
        
        # Perform Mann-Whitney U test between Real and Random
        if weighted_avg_mdl[inv]["Real"] and weighted_avg_mdl[inv]["Random"]:
            # Helper function to compute rank-biserial effect size for unpaired samples
            def rank_biserial_unpaired(x, y):
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
            
            statistic, p_real_random = mannwhitneyu(weighted_avg_mdl[inv]["Real"], 
                                                   weighted_avg_mdl[inv]["Random"], 
                                                   alternative='two-sided')
            r_real_random = rank_biserial_unpaired(weighted_avg_mdl[inv]["Real"], 
                                                  weighted_avg_mdl[inv]["Random"])
            n_real = len(weighted_avg_mdl[inv]["Real"])
            n_random = len(weighted_avg_mdl[inv]["Random"])
            
            print(f"\n  Mann-Whitney U test for {inv} (Real vs Random):")
            print(f"    p-value = {p_real_random:.4g}")
            print(f"    effect size (rank-biserial) = {r_real_random:.3f}")
            print(f"    sample size (Real) = {n_real}")
            print(f"    sample size (Random) = {n_random}")
        
        # Save density plot
        plt.tight_layout()
        plt.savefig(f"mdl_distribution_{inv}_density.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nDensity plot saved as mdl_distribution_{inv}_density.png")

print("\n✓ All plots created successfully!")