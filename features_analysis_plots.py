import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Define inventories
inventories = ["HC", "SPE", "JFH"]

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


def collect_phoneme_data(phoneme):
    """Collect data for a single phoneme across all inventories."""
    swarm_data_by_inv = {}
    all_values = {}
    
    for inv in inventories:
        swarm_data_by_inv[inv] = []
        all_values[inv] = {}
        data_inv = all_data[inv]
        
        for language, lang_data in data_inv.items():
            if phoneme in lang_data["min_descriptions"]:
                feature_desc = lang_data["min_descriptions"][phoneme]
                min_lengths = lang_data["min_lengths"]
                
                # Extract all unique features
                features_set = set()
                for mindesc in feature_desc:
                    for feat in mindesc:
                        if len(feat) > 0:
                            features_set.add(feat.strip('+-'))
                            if feat.strip('+-') not in all_values[inv]:
                                all_values[inv][feat.strip('+-')] = feat[0]
                
                # Get MDL values for each feature
                for feature in features_set:
                    if feature in min_lengths:
                        mdl_val = min_lengths[feature]
                        swarm_data_by_inv[inv].append({
                            'language': language,
                            'feature': feature,
                            'MDL': mdl_val
                        })
    
    return swarm_data_by_inv, all_values


def prepare_grouped_data(df, phoneme, inv, all_values):
    """Prepare grouped data for plotting."""
    if len(df) == 0:
        return None
    
    grouped = df.groupby(['feature', 'MDL']).size().unstack(fill_value=0)
    
    # Ensure all MDL values (1, 2, 3) are present
    for mdl_val in [1, 2, 3]:
        if mdl_val not in grouped.columns:
            grouped[mdl_val] = 0
    
    grouped = grouped[[col for col in [1, 2, 3] if col in grouped.columns]]
    
    # Sort rows by total language count
    grouped['total'] = grouped.sum(axis=1)
    grouped = grouped.sort_values('total', ascending=True)
    grouped = grouped.drop('total', axis=1)
    
    return grouped


def create_combined_phoneme_plot(phonemes, output_dir="phoneme_feature_analysis_plots"):
    """Create a combined plot with feature systems in columns and phonemes in rows."""
    print(f"\nCreating combined plot for phonemes: {phonemes}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect data for all phonemes
    all_phoneme_data = {}
    all_phoneme_values = {}
    
    for phoneme in phonemes:
        print(f"  Collecting data for /{phoneme}/...")
        data, values = collect_phoneme_data(phoneme)
        all_phoneme_data[phoneme] = data
        all_phoneme_values[phoneme] = values
    
    # Create DataFrames
    dfs = {}
    for phoneme in phonemes:
        dfs[phoneme] = {}
        for inv in inventories:
            dfs[phoneme][inv] = pd.DataFrame(all_phoneme_data[phoneme][inv])
    
    # Find max y value across all plots for consistent scaling
    max_y = 0
    for phoneme in phonemes:
        for inv in inventories:
            df = dfs[phoneme][inv]
            if len(df) > 0:
                grouped = prepare_grouped_data(df, phoneme, inv, all_phoneme_values[phoneme])
                if grouped is not None:
                    max_y = max(max_y, grouped.sum(axis=1).max())
    
    max_y = max_y * 1.1  # Add 10% padding
    
    # Create figure with 3 rows (inventories) and 2 columns (phonemes)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    for row_idx, inv in enumerate(inventories):
        for col_idx, phoneme in enumerate(phonemes):
            ax = axes[row_idx, col_idx]
            df = dfs[phoneme][inv]
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            grouped = prepare_grouped_data(df, phoneme, inv, all_phoneme_values[phoneme])
            
            # Create stacked bar plot
            grouped.plot(kind='bar', stacked=True, ax=ax, 
                        color=["#ff0e0e", "#fe6d6d", "#f7b1b1"], 
                        width=0.8, edgecolor=None, linewidth=0.5, alpha=0.8, legend=False)
            
            # Add count annotations
            for bar_idx, feature in enumerate(grouped.index):
                cumulative_height = 0
                for mdl_val in [1, 2, 3]:
                    if mdl_val in grouped.columns:
                        count = grouped.loc[feature, mdl_val]
                        if count > 7:
                            height = count
                            ax.text(bar_idx, cumulative_height + height/2, str(int(count)), 
                                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                            cumulative_height += height
            
            # Set y-axis label only for leftmost column
            if col_idx == 0:
                ax.set_ylabel('Language Count', fontsize=16)
            else:
                ax.set_ylabel('')
            
            # Set x-axis label only for bottom row
            if row_idx == 2:
                feature_labels = [f"{all_phoneme_values[phoneme][inv].get(feat, '')}{feat}" 
                                 for feat in grouped.index]
                ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontweight='bold', fontsize=12)
                ax.set_xlabel('Feature', fontsize=16)
            else:
                feature_labels = [f"{all_phoneme_values[phoneme][inv].get(feat, '')}{feat}" 
                                 for feat in grouped.index]
                ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontweight='bold', fontsize=12)
                ax.set_xlabel('')
            
            # Set consistent y-axis range
            ax.set_ylim(0, max_y)
            
            # Title only for rightmost column (feature system name)
            if row_idx == 0:
                if col_idx == 0:
                    ax.set_title(f'/{phonemes[0]}/', fontsize=18, fontweight='bold')
                else:
                    ax.set_title(f'/{phonemes[1]}/', fontsize=18, fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
    
    # Add background rectangles (inventory colors for rows, phoneme colors for columns)
    inventory_colors_dict = {"HC": "#1f77b4", "SPE": "#ff7f0e", "JFH": "#2ca02c"}
    
    # Plot area coordinates (adjusted for tight_layout)
    plot_x_start = 0.08
    plot_x_end = 0.94
    plot_y_start = 0.08
    plot_y_end = 0.92
    plot_width = plot_x_end - plot_x_start  # ~0.86
    plot_height = plot_y_end - plot_y_start  # ~0.84
    
    # Column width and row height with spacing
    col_width = plot_width / len(phonemes)
    row_height = plot_height / len(inventories)
    col_spacing = 0.012  # White space between columns
    
    # # Add horizontal colored background rectangles for each inventory (rows)
    # for row_idx, inv in enumerate(inventories):
    #     row_y = plot_y_start + (len(inventories) - 1 - row_idx) * row_height
    #     rect = mpatches.Rectangle((plot_x_start, row_y), plot_width, row_height, 
    #                               facecolor=inventory_colors_dict[inv], 
    #                               edgecolor='none', alpha=0.15,
    #                               transform=fig.transFigure, zorder=-1)
    #     fig.patches.append(rect)
    
    # Add legend to the top-right plot (axes[0, 1])
    handles = [mpatches.Rectangle((0, 0), 1, 1, color=color) for color in ["#ff0e0e", "#fe6d6d", "#f7b1b1"]]
    # axes[0, 1].legend(handles, ['MDL=1', 'MDL=2', 'MDL=3'], loc='upper right', fontsize=10, ncol=1)
    
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    phoneme_names = '_'.join([p.replace('/', '_').replace(' ', '_') for p in phonemes])
    filename = os.path.join(output_dir, f'phonemes_{phoneme_names}_combined.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Plot saved as: {filename}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phoneme Feature Analysis: Combined Comparison")
    print("=" * 60)
    
    phonemes_to_analyze = ['l', 'ɔ']
    
    try:
        create_combined_phoneme_plot(phonemes_to_analyze)
    except Exception as e:
        print(f"  Error creating combined plot: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Feature Analysis Complete")
    print("=" * 60)
