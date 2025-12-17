import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from upsetplot import UpSet, from_indicators, plot

# Define inventories
inventories = ["HC", "SPE", "JFH"]

# Load all data
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Load pb_languages_formatted.csv to get language families
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv")

def create_phoneme_feature_analysis_plot(phoneme, output_dir="phoneme_feature_analysis_plots"):
    print(f"\nAnalyzing phoneme: /{phoneme}/")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect data for this phoneme across all inventories
    swarm_data_by_inv = {}
    lang_counts_by_inv = {}
    
    for inv in inventories:
        swarm_data_by_inv[inv] = []
        lang_counts_by_inv[inv] = 0
        
        # Load the data for this inventory
        data_inv = all_data[inv]
        
        # Iterate through all languages
        for language, lang_data in data_inv.items():
            min_descriptions = lang_data["min_descriptions"]
            min_lengths = lang_data["min_lengths"]
            
            # Check if this phoneme is described in this language
            if phoneme in min_descriptions:
                feature_desc = min_descriptions[phoneme]
                
                # Extract all unique features from feature_desc
                features_set = set()
                for mindesc in feature_desc:
                    for feat in mindesc:
                        features_set.add(feat.strip('+-'))
                
                features_list = list(features_set)
                
                # Get min_lengths for each feature
                for feature in features_list:
                    if feature in min_lengths:
                        mdl_val = min_lengths[feature]
                        swarm_data_by_inv[inv].append({
                            'language': language,
                            'feature': feature,
                            'MDL': mdl_val
                        })
                lang_counts_by_inv[inv] += 1
        print(f"  {inv}: Processed {lang_counts_by_inv[inv]} languages")
    
    # Check if we have data for this phoneme
    total_data_points = sum(len(data) for data in swarm_data_by_inv.values())
    if total_data_points == 0:
        print(f"  Warning: No data found for phoneme /{phoneme}/ across any feature system")
        return
    
    print(f"  Found data for phoneme /{phoneme}/:")
    for inv in inventories:
        n_points = len(swarm_data_by_inv[inv])
        if n_points > 0:
            df_inv = pd.DataFrame(swarm_data_by_inv[inv])
            n_features = df_inv['feature'].nunique()
            n_languages = df_inv['language'].nunique()
            print(f"    {inv}: {n_points} data points ({n_features} features, {n_languages} languages)")
    
    # Convert to DataFrames
    dfs = {}
    for inv in inventories:
        dfs[inv] = pd.DataFrame(swarm_data_by_inv[inv])
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, inv in enumerate(inventories):
        ax = axes[idx]
        df = dfs[inv]
        
        if len(df) == 0:
            ax.text(0.5, 0.5, f'No data for {inv}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{inv}: No Data', fontsize=12, fontweight='bold')
            continue
        
        # Create stacked bar plot: features on x-axis, language count on y-axis, stacked by MDL value
        # Group data by feature and MDL value, count languages
        grouped = df.groupby(['feature', 'MDL']).size().unstack(fill_value=0)
        
        # Ensure all MDL values (1, 2, 3) are present as columns
        for mdl_val in [1, 2, 3]:
            if mdl_val not in grouped.columns:
                grouped[mdl_val] = 0
        
        # Sort columns by MDL value
        grouped = grouped[[col for col in [1, 2, 3] if col in grouped.columns]]
        
        # Sort rows (features) by total language count in ascending order
        grouped['total'] = grouped.sum(axis=1)
        grouped = grouped.sort_values('total', ascending=True)
        grouped = grouped.drop('total', axis=1)
        
        # Create stacked bar plot
        grouped.plot(kind='bar', stacked=True, ax=ax, 
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                    width=0.8, edgecolor='black', linewidth=0.5)
        
        # Add count annotations on top of each stacked section
        for bar_idx, feature in enumerate(grouped.index):
            cumulative_height = 0
            for mdl_idx, mdl_val in enumerate([1, 2, 3]):
                if mdl_val in grouped.columns:
                    count = grouped.loc[feature, mdl_val]
                    if count > 0:
                        height = count
                        # Place text at the middle of each section
                        ax.text(bar_idx, cumulative_height + height/2, str(int(count)), 
                               ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                        cumulative_height += height
        
        # Customize subplot
        ax.set_xlabel('Feature', fontsize=11, fontweight='bold')
        ax.set_ylabel('Language Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{inv}\n(n={len(df)} points, {df["feature"].nunique()} features)', 
                    fontsize=11, fontweight='bold')
        
        # Customize legend
        ax.legend(title='MDL Value', labels=['MDL=1', 'MDL=2', 'MDL=3'], 
                 loc='upper left', fontsize=10, title_fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Rotate x-axis labels for readability with proper alignment
        ax.set_xticklabels(grouped.index, rotation=90, ha='right', fontsize=8)
    
    # Main title
    fig.suptitle(f'Phoneme /{phoneme}/ (Total Language Count: {lang_counts_by_inv["HC"]})', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85)
    
    # Save plot
    safe_phoneme_name = phoneme.replace('/', '_').replace(' ', '_')
    filename = os.path.join(output_dir, f'phoneme_{safe_phoneme_name}_feature_mdl_stacked_bars.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Plot saved as: {filename}")
    
    # Print detailed statistics
    print(f"\n  Detailed statistics for /{phoneme}/:")
    for inv in inventories:
        df = dfs[inv]
        if len(df) > 0:
            print(f"\n    {inv}:")
            print(f"      Total data points: {len(df)}")
            print(f"      Number of features: {df['feature'].nunique()}")
            print(f"      Number of languages: {df['language'].nunique()}")
            
            # Show distribution of MDL values
            mdl_dist = df['MDL'].value_counts().sort_index()
            print(f"      MDL value distribution:")
            for mdl_val in sorted(mdl_dist.index):
                count = mdl_dist[mdl_val]
                pct = (count / len(df)) * 100
                print(f"        MDL={int(mdl_val)}: {count} occurrences ({pct:.1f}%)")
            
            # Show top features by language count
            print(f"      Top 10 features by language count:")
            feature_lang_counts = df.groupby('feature')['language'].nunique().sort_values(ascending=False).head(10)
            for feat, count in feature_lang_counts.items():
                mdl_vals = df[df['feature'] == feat]['MDL'].unique()
                print(f"        {feat:15s}: {count} languages, MDL values: {sorted(mdl_vals)}")


def create_phoneme_upset_plot(phoneme, output_dir="phoneme_feature_analysis_plots"):
    """Create stacked bar plots showing feature combinations and their language counts by MDL value."""
    print(f"\nGenerating feature combination plots for phoneme: /{phoneme}/")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect data for this phoneme across all inventories
    phoneme_data_by_inv = {}
    
    for inv in inventories:
        phoneme_data_by_inv[inv] = {}
        
        # Load the data for this inventory
        data_inv = all_data[inv]
        
        # Iterate through all languages
        for language, lang_data in data_inv.items():
            min_descriptions = lang_data["min_descriptions"]
            min_lengths = lang_data["min_lengths"]
            
            # Check if this phoneme is described in this language
            if phoneme in min_descriptions:
                feature_desc = min_descriptions[phoneme]
                
                # Extract all unique features from feature_desc
                features_set = set()
                for mindesc in feature_desc:
                    for feat in mindesc:
                        features_set.add(feat.strip('+-'))
                
                # Get MDL values for each feature
                for feature in features_set:
                    if feature in min_lengths:
                        mdl_val = int(min_lengths[feature])
                        if language not in phoneme_data_by_inv[inv]:
                            phoneme_data_by_inv[inv][language] = {}
                        phoneme_data_by_inv[inv][language][feature] = mdl_val
    
    # Check if we have data
    total_data_points = sum(len(data) for data in phoneme_data_by_inv.values())
    if total_data_points == 0:
        print(f"  Warning: No data found for phoneme /{phoneme}/ across any feature system")
        return
    
    # Create feature combination plots for each inventory
    for inv in inventories:
        if len(phoneme_data_by_inv[inv]) == 0:
            print(f"  Skipping {inv} - no data")
            continue
        
        # Get all unique features for this phoneme in this inventory
        all_features = set()
        for lang_features in phoneme_data_by_inv[inv].values():
            all_features.update(lang_features.keys())
        
        all_features = sorted(list(all_features))
        
        print(f"  Creating {len(all_features)} feature combination plots for {inv}")
        
        # For each feature, create an UpSet-style stacked bar plot
        for primary_feature in all_features:
            
            # Get all languages that have this primary feature
            langs_with_primary = [lang for lang, feats in phoneme_data_by_inv[inv].items() 
                                 if primary_feature in feats]
            primary_mdl_per_lang = {lang: phoneme_data_by_inv[inv][lang][primary_feature] 
                                   for lang in langs_with_primary}
            
            # Get all other features
            other_features = [f for f in all_features if f != primary_feature]
            
            # Create bars for: primary feature alone + each combination with another feature
            bar_labels = [primary_feature]  # First bar is primary feature alone
            bar_data = {1: [], 2: [], 3: []}  # Count of languages per MDL value
            
            # First bar: primary feature alone (all languages with primary feature)
            for mdl_val in [1, 2, 3]:
                count = sum(1 for mdl in primary_mdl_per_lang.values() if mdl == mdl_val)
                bar_data[mdl_val].append(count)
            
            # Subsequent bars: primary feature + another feature (co-occurrence)
            for other_feature in other_features:
                # Get languages that have BOTH primary feature and other feature
                langs_with_both = [lang for lang in langs_with_primary 
                                  if other_feature in phoneme_data_by_inv[inv][lang]]
                
                # For these languages, use the MDL of the primary feature
                for mdl_val in [1, 2, 3]:
                    count = sum(1 for lang in langs_with_both 
                               if primary_mdl_per_lang[lang] == mdl_val)
                    bar_data[mdl_val].append(count)
                
                bar_labels.append(f"{primary_feature}\n+{other_feature}")
            
            # Create stacked bar plot
            x_pos = np.arange(len(bar_labels))
            bottom = np.zeros(len(bar_labels))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for MDL 1, 2, 3
            
            fig, ax = plt.subplots(figsize=(max(12, len(bar_labels) * 1.5), 7))
            
            for mdl_idx, mdl_val in enumerate([1, 2, 3]):
                values = np.array(bar_data[mdl_val])
                ax.bar(x_pos, values, bottom=bottom, label=f'MDL={mdl_val}', 
                       color=colors[mdl_idx], edgecolor='black', linewidth=0.7)
                
                # Add value labels on bars
                for i, v in enumerate(values):
                    if v > 0:
                        ax.text(i, bottom[i] + v/2, str(int(v)), ha='center', va='center', 
                               fontsize=9, fontweight='bold', color='white')
                
                bottom += values
            
            # Customize plot
            ax.set_xlabel('Feature Combinations', fontsize=12, fontweight='bold')
            ax.set_ylabel('Language Count', fontsize=12, fontweight='bold')
            ax.set_title(f'Feature Combinations: /{phoneme}/ in {inv}\nPrimary: {primary_feature}', 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(bar_labels, fontsize=10, rotation=45, ha='right')
            ax.legend(title='MDL Value', fontsize=10, title_fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save plot
            safe_phoneme_name = phoneme.replace('/', '_').replace(' ', '_')
            safe_feature_name = primary_feature.replace('/', '_').replace(' ', '_').replace('+', 'plus')
            filename = os.path.join(output_dir, f'phoneme_{safe_phoneme_name}_{inv}_upset_{safe_feature_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"  Feature combination plots complete for /{phoneme}/")


# Example usage: Create plots for multiple phonemes
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phoneme Feature Analysis: MDL Stacked Bars and UpSet Plots")
    print("=" * 60)
    
    # List of phonemes to analyze
    # You can modify this list or pass specific phonemes as arguments
    phonemes_to_analyze = ['tʃ', 'l', 'ɛ', 'ɔ', 'ẽ', 'ũ', 'õ']
    
    print(f"\nGenerating stacked bar plots for {len(phonemes_to_analyze)} phonemes...")
    
    for phoneme in phonemes_to_analyze:
        try:
            create_phoneme_feature_analysis_plot(phoneme)
        except Exception as e:
            print(f"  Error processing phoneme /{phoneme}/: {str(e)}")
    
    print(f"\nGenerating feature combination plots (UpSet-style) for {len(phonemes_to_analyze)} phonemes...")
    
    for phoneme in phonemes_to_analyze:
        try:
            create_phoneme_upset_plot(phoneme)
        except Exception as e:
            print(f"  Error creating feature combination plot for phoneme /{phoneme}/: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Feature Analysis Complete")
    print("=" * 60)
