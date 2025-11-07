import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the output folder if it doesn't exist
os.makedirs('plot2', exist_ok=True)

# Inventories and colors
inventories = ['HC', 'SPE', 'JFH']
colors = ['red', 'green', 'blue']

# Load all data first
all_data = {}
for inv in inventories:
    with open(f'data_all_languages_{inv}_features.json', 'r') as file:
        all_data[inv] = json.load(file)

# Get all languages (from the first inventory)
all_languages = list(all_data[inventories[0]].keys())

# Create a plot for each language
for language_to_plot in all_languages:
    # Collect all features and their min_lengths for the selected language
    all_features_data = {}
    for inv in inventories:
        if language_to_plot in all_data[inv]:
            all_features_data[inv] = all_data[inv][language_to_plot]['min_lengths']
        else:
            all_features_data[inv] = {}
    
    # Get union of all features across inventories
    all_features = set()
    for inv_data in all_features_data.values():
        all_features.update(inv_data.keys())
    all_features = sorted(all_features)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(all_features))
    width = 0.25
    
    # Plot bars for each inventory
    for i, inv in enumerate(inventories):
        min_lengths = [all_features_data[inv].get(feat, 0) for feat in all_features]
        ax.bar(x + i * width, min_lengths, width=width, color=colors[i], label=inv)
    
    # Add labels and legend
    ax.set_xlabel('Features')
    ax.set_ylabel('Informativity (Minimal Description Length)')
    ax.set_title(f'Feature Informativity for {language_to_plot}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plot2/{language_to_plot}.png', dpi=300, bbox_inches='tight')
    plt.close()

