import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import os

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


def process_single_language(args):
    """Process a single language - for parallel execution."""
    idx, row, inv, phoneme_pool, num_samples = args
    
    # Load feature system (each process needs its own copy)
    fd, _ = readinventory(inv)
    features = list(fd.keys())
    
    language = row['language'].replace("/", " or ")
    print(f"Processing language {idx + 1}: {language} with {inv}")
    
    # Parse inventory
    inventory = row['core inventory']
    inventory = inventory.strip("[]").split(',')
    inventory = [phon.strip().replace("'", "") for phon in inventory]
    inventory = [phon for phon in inventory if phon != '']
    allsegments = set(inventory)
    
    # Process actual language
    natural_classes_perphoneme = process_phoneme_inventory(allsegments, fd, features)
    min_lengths, min_descriptions, count_phoneme, avg_lengths, count_lengths = get_general_info_natural_classes(
        natural_classes_perphoneme, features
    )
    
    real_avg_mdl = compute_avg_mdl(allsegments, min_lengths, min_descriptions)
    
    # Process random languages
    inventory_size = len(allsegments)
    sample_avg_lengths = []
    
    for sample_num in range(num_samples):
        sampled_phonemes = set(np.random.choice(phoneme_pool, size=inventory_size, replace=False))
        
        natural_classes_perphoneme_random = process_phoneme_inventory(sampled_phonemes, fd, features)
        min_lengths_r, min_descriptions_r, _, _, _ = get_general_info_natural_classes(
            natural_classes_perphoneme_random, features
        )
        
        random_avg_mdl = compute_avg_mdl(sampled_phonemes, min_lengths_r, min_descriptions_r)
        if random_avg_mdl is not None:
            sample_avg_lengths.append(random_avg_mdl)
    
    mean_random_mdl = np.mean(sample_avg_lengths) if sample_avg_lengths else None
    
    return {
        'language': language,
        'inventory': inv,
        'real_avg_mdl': real_avg_mdl,
        'random_avg_mdl': mean_random_mdl
    }


if __name__ == '__main__':
    # Configuration
    INVENTORY = ['SPE', 'JFH', 'HC']
    NUM_SAMPLES = 1  # Number of random samples per language
    NUM_WORKERS = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Using {NUM_WORKERS} parallel workers")

    # Read the language data
    languages_df = pd.read_csv('phonemic_inventories/pb_languages_formatted.csv')

    # Step 1: Collect all unique phonemes from all languages
    all_phonemes = set()
    for idx, row in languages_df.iterrows():
        inventory = row['core inventory']
        if pd.notna(inventory):
            phonemes = [p.strip().replace("'", "") for p in str(inventory).split(',')]
            all_phonemes.update(phonemes)

    phoneme_pool = list(all_phonemes)
    print(f"Total unique phonemes in pool: {len(phoneme_pool)}")

    # Prepare arguments for parallel processing
    # Create all (language, inventory) combinations
    tasks = []
    for inv in INVENTORY:
        for idx, row in languages_df.iterrows():
            tasks.append((idx, row, inv, phoneme_pool, NUM_SAMPLES))
    
    print(f"Processing {len(tasks)} tasks ({len(languages_df)} languages × {len(INVENTORY)} inventories)")
    
    # Process in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_single_language, tasks)
    
    # Organize results
    real_sample_avg_lengths = {inv: [] for inv in INVENTORY}
    
    results_by_inventory = {inv: {'Real': [], 'Random': []} for inv in INVENTORY}
    
    for result in results:
        inv = result['inventory']
        
        if result['real_avg_mdl'] is not None:
            real_sample_avg_lengths[inv].append(result['real_avg_mdl'])
            results_by_inventory[inv]['Real'].append(result['real_avg_mdl'])
            print(f"  {result['language']} ({inv}): Real avg MDL = {result['real_avg_mdl']:.4f}")
        
        if result['random_avg_mdl'] is not None:
            results_by_inventory[inv]['Random'].append(result['random_avg_mdl'])
            print(f"  {result['language']} ({inv}): Random avg MDL = {result['random_avg_mdl']:.4f}")
    
    # Plot results for each inventory
    for inv in INVENTORY:
        if results_by_inventory[inv]['Real'] or results_by_inventory[inv]['Random']:
            plt.figure(figsize=(10, 6))
            sns.histplot(results_by_inventory[inv], bins=50, kde=True)
            plt.xlabel('Average Minimal Description Length')
            plt.ylabel('Language Count')
            plt.title(f'Feature system: {inv}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'mdl_distribution_{inv}.png', dpi=300)
            plt.show()

    # Plot combined results
    plt.figure(figsize=(10, 6))
    sns.histplot(real_sample_avg_lengths, bins=50, kde=True)
    plt.xlabel('Average Minimal Description Length')
    plt.ylabel('Language Count')
    plt.title('All Feature Systems Combined')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mdl_distribution_combined.png', dpi=300)
    plt.show()
    
    print("\n✓ Processing complete!")