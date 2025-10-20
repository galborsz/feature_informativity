import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

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

def reccheck(fd, basefeats, basemodes, feats, modes, correct, baseindex):
    """
    Start with an empty set of features and keep adding features one by one with different starting phonemes, generating all possible unique feature combinations.
    Check if the generated feature combinations are natural classes for the given phoneme.
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
            
    def check_feats(fd, feats, modes, correct):
        """Check if proposed feature combination is a valid solution."""
        newbase = allsegments 
        for idx, feat in enumerate(feats):
            mode = modes[idx]
            newbase = newbase & fd[feat][mode]
        if newbase != correct:
            return False
        return True
        
    global maxlen
    if len(feats) > maxlen: # Bound the search (max: total amount of features)
        return
    if check_feats(fd, feats, modes, correct): # New solution
        store_feats(fd, feats, modes) # if proposed feature combination is a natural class, store solution
        if len(feats) < maxlen:
            maxlen = len(feats)
    numelem = len(basefeats)
    # This for loop iterates over all possible indeces and generates all possible feature combinations
    for i in range(baseindex, numelem):  # Add one feature
        if basefeats[i] not in feats:    # If we didn't add this already
            reccheck(fd, basefeats, basemodes, feats + [basefeats[i]], modes + [basemodes[i]], correct, i + 1)
    return

def get_general_info_natural_classes(natural_classes, keys):
    """Get descriptive information for the given natural classes"""

    min_lengths = {} # store the length of the minimal description where each feature is included
    min_lengths_phonemes = {}
    avg_lengths = {key: [0,0] for key in keys} # store the average lengths of all descriptions per feature

    for phoneme in natural_classes:
        for sublist in natural_classes[phoneme]:
            sublist = sublist.strip("[]").split(',')
            # Iterate through each unique value
            for value in sublist:
                value = value.strip('+') # remove + symbols
                value = value.strip('-') # remove - symbols
                # Check if the value already exists in the dictionary
                if value in min_lengths:
                    # If the length of the current sublist is smaller than the stored length,
                    # update the stored length
                    min_lengths[value] = min(min_lengths[value], len(sublist))
                else:
                    # If the value doesn't exist in the dictionary, add it with the length
                    min_lengths[value] = len(sublist)

                if value in avg_lengths:
                    avg_lengths[value][0] += len(sublist)
                    avg_lengths[value][1] += 1
            
            if phoneme in min_lengths_phonemes:
                min_lengths_phonemes[phoneme] = min(min_lengths_phonemes[phoneme], len(sublist))
            else: 
                min_lengths_phonemes[phoneme] = len(sublist)
                        
    avg_lengths = {k: v[0] / v[1] if v[1] != 0 else 0 for k, v in avg_lengths.items()}

    min_descriptions = {} # store the minimal descriptions of each phoneme
    # get all minimal descriptions per phoneme
    for phoneme in natural_classes:
        if phoneme not in min_descriptions:
            min_descriptions[phoneme] = []

        for sublist in natural_classes[phoneme]:
            sublist = sublist.strip("[]").split(',')
            if min_lengths_phonemes[phoneme] == len(sublist):
                min_descriptions[phoneme].append(sublist)

    count_phoneme = {} # The number of times the feature is included in the minimal description of a phoneme
    count_lengths = {} # Count of minimal descriptions for various lengths
    # count features in minimal descriptions          
    for phoneme in min_descriptions:
        for sublist in min_descriptions[phoneme]:
            for value in sublist:
                value = value.strip('+') # remove + symbol
                value = value.strip('-') # remove - symbol
                if value in count_phoneme:
                    count_phoneme[value] += 1
                else:
                    count_phoneme[value] = 1
    
            if len(sublist) in count_lengths:
                count_lengths[len(sublist)] += 1
            else:
                count_lengths[len(sublist)] = 1
    
    return min_lengths, min_descriptions, count_phoneme, avg_lengths, count_lengths


def normalize_phoneme(phoneme):
    """Normalize phoneme representation to ensure consistent string format"""
    return str(phoneme).encode('unicode-escape').decode('utf-8')

# Configuration
INVENTORY = 'SPE'
NUM_SAMPLES = 3

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


# Step 3: Load feature system
fd, allsegments = readinventory(INVENTORY)

# Step 4: Process each language and generate random samples
results = []

all_languages = {}
for idx, row in languages_df.iterrows():
    language = row['language'].replace("/", " or ")
    print(f"Processing language {idx + 1}/{len(languages_df)}: {language}")

    family = row['family']
    inventory = row['core inventory']
    inventory = inventory.strip("[]").split(',')
    inventory = [phon.strip().replace("'", "") for phon in inventory]
    inventory = [phon for phon in inventory if phon != '']
    allsegments = set(inventory) # set of all phonemes in the inventory


    # Get phonemes in this language's inventory
    inventory_size = len(allsegments)
    
    # Generate 100 random languages of the same size
    sample_avg_lengths = []

    for sample_num in range(NUM_SAMPLES):
        # Sample N phonemes from the pool
        sampled_phonemes = np.random.choice(phoneme_pool, size=inventory_size, replace=False)
        # convert sampled_phonemes (np.ndarray of numpy.str_) to a Python set of str
        sampled_phonemes = set({str(x) for x in sampled_phonemes})
        allsegments = sampled_phonemes
        print(f"Sampled phonemes: {allsegments}")

        features = [f for f in fd] # list of all features in the selected feature system

        minimal_natural_classes = []
        minimal_natural_classes_perphoneme = {}
        natural_classes = []
        natural_classes_perphoneme = {}
        for testset in allsegments: # iterate through all phonemes in the inventory
            base = allsegments
            feats, modes = [], [] # list with featres, list with signs for each feature
            phoneme = testset
            testset = {testset}

            # Iterate over all features to find:
                # base: list of phonemes that are described by the same features as the given test phoneme
                # feats: list of features that describe the given phoneme
                # modes: list with the respective signs of the features describing the given phoneme
            for feat in features:
                if testset <= fd[feat]['+']: # test whether testset is a subset of fd[feat]['+']
                    # fd[feat]['+']: set of phonemes that have the feature feat with sign + 
                    base = base & fd[feat]['+'] # returns intersection between the two sets (those elements that are in both sets)
                    feats.append(feat)
                    modes.append('+')
                elif testset <= fd[feat]['-']:
                    # fd[feat]['-']: set of phonemes that have the feature feat with sign - 
                    base = base & fd[feat]['-']
                    feats.append(feat)
                    modes.append('-')

            solutions = {}
            # Check if the procedure above has resulted in the phoneme being tested (i.e. we have the correct general feature description and it is a natural class)
            if base == testset: 
                maxlen = len(feats)
                reccheck(fd, feats, modes, [], [], base, 0)
                for s in solutions.values():
                    for a in s:
                        natural_classes.append(a) 
                        # phoneme = list(testset)[0]
                        if phoneme in natural_classes_perphoneme:
                            natural_classes_perphoneme[phoneme].append(a)
                        else: 
                            natural_classes_perphoneme[phoneme] = []
                minsol = min(solutions.keys())
                for s in solutions[minsol]:
                    minimal_natural_classes.append(s)
                    if phoneme in minimal_natural_classes_perphoneme:
                        minimal_natural_classes_perphoneme[phoneme].append(s)
                    else: 
                        minimal_natural_classes_perphoneme[phoneme] = []

        min_lengths, min_descriptions, count_phoneme, avg_lengths, count_lengths = get_general_info_natural_classes(natural_classes_perphoneme, list(fd.keys()))
        
        # Compute average avg_length for this sampled language
        total_avg_length = 0
        feature_count = 0
        print(f'Min desc keys: {list(min_descriptions.keys())}')
        print(f'Types: {type(list(min_descriptions.keys())[0])}, {type(phoneme)}')
        for phoneme in sampled_phonemes:
            if phoneme in min_descriptions:
                feature_descriptions = min_descriptions[phoneme]
                unique_features = set([item for sublist in feature_descriptions for item in sublist])
                for feature in unique_features:
                    feature = feature.strip('+').strip('-')
                    if feature in min_lengths:
                        total_avg_length += min_lengths[feature]
                        feature_count += 1
        print(f'Total avg length: {total_avg_length}, Feature count: {feature_count}')
        if feature_count > 0:
            avg_for_sample = total_avg_length / feature_count
            sample_avg_lengths.append(avg_for_sample)
            print(f"  Sample {sample_num + 1}/{NUM_SAMPLES}: avg MDL = {avg_for_sample:.4f}")

    # Average over all 100 samples for this language size
    if sample_avg_lengths:
        mean_avg_length = np.mean(sample_avg_lengths)
        results.append(mean_avg_length)
        print(f"Language {idx + 1} (size {inventory_size}): avg MDL = {mean_avg_length:.4f}")


# Step 5: Plot the distribution
plt.figure(figsize=(10, 6))
sns.histplot(results, bins=30, kde=True, stat='density')
plt.xlabel('Average Minimal Description Length')
plt.ylabel('Density of Languages')
plt.title(f'Distribution of Average MDL for Random Languages ({INVENTORY} features)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'random_languages_mdl_distribution_{INVENTORY}.png', dpi=300)
plt.show()

print(f"\nOverall statistics:")
print(f"Mean: {np.mean(results):.4f}")
print(f"Std: {np.std(results):.4f}")
print(f"Min: {np.min(results):.4f}")
print(f"Max: {np.max(results):.4f}")