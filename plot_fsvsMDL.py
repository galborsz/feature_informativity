import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Define feature system sizes (number of features in each system)
FEATURE_SYSTEM_SIZES = {
    'HC': 23,   # Hayes & Cziráky
    'SPE': 25,  # Sound Pattern of English
    'JFH': 15   # Jakobson, Fant, Halle
}

def load_data():
    """Load MDL data from all three feature systems."""
    data_frames = []
    
    for system_name in ['HC', 'SPE', 'JFH']:
        file_path = f'data_all_languages_{system_name}_features.json'
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract relevant information
        for lang_name, lang_data in data.items():
            for inventory_idx, inventory_data in enumerate(lang_data.get('inventories', [])):
                min_length = inventory_data.get('min_lengths', 0)
                
                data_frames.append({
                    'language': lang_name,
                    'feature_system': system_name,
                    'feature_count': FEATURE_SYSTEM_SIZES[system_name],
                    'mdl': min_length,
                    'inventory_index': inventory_idx
                })
    
    df = pd.DataFrame(data_frames)
    return df

def create_visualizations(df):
    """Create comprehensive visualizations comparing MDL across feature systems."""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Box plot with individual points
    ax1 = plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='feature_system', y='mdl', 
                order=['JFH', 'HC', 'SPE'], palette='Set2', ax=ax1)
    sns.stripplot(data=df, x='feature_system', y='mdl', 
                  order=['JFH', 'HC', 'SPE'], color='black', 
                  alpha=0.2, size=2, ax=ax1)
    ax1.set_xlabel('Feature System', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MDL (Minimum Description Length)', fontsize=12, fontweight='bold')
    ax1.set_title('MDL Distribution by Feature System', fontsize=14, fontweight='bold')
    
    # Add feature counts to x-axis labels
    ax1.set_xticklabels([f'JFH\n(15 features)', f'HC\n(23 features)', f'SPE\n(25 features)'])
    
    # 2. Violin plot
    ax2 = plt.subplot(2, 3, 2)
    sns.violinplot(data=df, x='feature_system', y='mdl', 
                   order=['JFH', 'HC', 'SPE'], palette='Set2', ax=ax2)
    ax2.set_xlabel('Feature System', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MDL (Minimum Description Length)', fontsize=12, fontweight='bold')
    ax2.set_title('MDL Density by Feature System', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([f'JFH\n(15 features)', f'HC\n(23 features)', f'SPE\n(25 features)'])
    
    # 3. Scatter plot: Feature Count vs MDL
    ax3 = plt.subplot(2, 3, 3)
    for system in ['JFH', 'HC', 'SPE']:
        system_data = df[df['feature_system'] == system]
        ax3.scatter(system_data['feature_count'], system_data['mdl'], 
                   label=system, alpha=0.3, s=20)
    ax3.set_xlabel('Number of Features in System', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MDL (Minimum Description Length)', fontsize=12, fontweight='bold')
    ax3.set_title('MDL vs Feature System Size', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mean MDL by system with error bars
    ax4 = plt.subplot(2, 3, 4)
    summary = df.groupby('feature_system')['mdl'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary = summary.reindex(['JFH', 'HC', 'SPE'])
    
    x_pos = [15, 23, 25]  # Actual feature counts
    ax4.errorbar(x_pos, summary['mean'], yerr=summary['se'], 
                fmt='o-', markersize=10, capsize=5, linewidth=2)
    ax4.set_xlabel('Number of Features in System', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean MDL', fontsize=12, fontweight='bold')
    ax4.set_title('Mean MDL ± SE by Feature System Size', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add system labels
    for x, system in zip(x_pos, ['JFH', 'HC', 'SPE']):
        ax4.text(x, summary.loc[system, 'mean'] - 0.5, system, 
                ha='center', va='top', fontweight='bold')
    
    # 5. Histogram comparison
    ax5 = plt.subplot(2, 3, 5)
    for system in ['JFH', 'HC', 'SPE']:
        system_data = df[df['feature_system'] == system]['mdl']
        ax5.hist(system_data, alpha=0.5, label=system, bins=30)
    ax5.set_xlabel('MDL (Minimum Description Length)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('MDL Distribution Comparison', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_stats = df.groupby('feature_system')['mdl'].agg([
        ('N', 'count'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).round(2)
    
    # Add feature count column
    summary_stats['Features'] = summary_stats.index.map(FEATURE_SYSTEM_SIZES)
    summary_stats = summary_stats[['Features', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max']]
    summary_stats = summary_stats.reindex(['JFH', 'HC', 'SPE'])
    
    table_data = []
    table_data.append(['System'] + list(summary_stats.columns))
    for idx, row in summary_stats.iterrows():
        table_data.append([idx] + [str(val) for val in row.values])
    
    table = ax6.table(cellText=table_data, cellLoc='center', 
                     loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary Statistics by Feature System', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('mdl_vs_feature_system_size.png', dpi=300, bbox_inches='tight')
    print("Saved: mdl_vs_feature_system_size.png")
    plt.show()

def statistical_analysis(df):
    """Perform statistical tests to compare MDL across feature systems."""
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: MDL vs Feature System Size")
    print("="*70)
    
    # Basic descriptive statistics
    print("\nDescriptive Statistics:")
    print("-" * 70)
    for system in ['JFH', 'HC', 'SPE']:
        system_data = df[df['feature_system'] == system]['mdl']
        feature_count = FEATURE_SYSTEM_SIZES[system]
        n_languages = df[df['feature_system'] == system]['language'].nunique()
        
        print(f"\n{system} ({feature_count} features):")
        print(f"  Total data points: {len(system_data)}")
        print(f"  Unique languages: {n_languages}")
        print(f"  Mean MDL: {system_data.mean():.2f}")
        print(f"  Median MDL: {system_data.median():.2f}")
        print(f"  Std Dev: {system_data.std():.2f}")
        print(f"  Range: [{system_data.min():.2f}, {system_data.max():.2f}]")
    
    # Normality tests
    print("\n" + "-" * 70)
    print("Normality Tests (Shapiro-Wilk):")
    print("-" * 70)
    
    normal_dist = True
    for system in ['JFH', 'HC', 'SPE']:
        system_data = df[df['feature_system'] == system]['mdl']
        stat, p_value = stats.shapiro(system_data[:5000])  # Limit sample size for test
        print(f"{system}: W={stat:.4f}, p={p_value:.4e}")
        if p_value < 0.05:
            print(f"  → Data is NOT normally distributed (p < 0.05)")
            normal_dist = False
        else:
            print(f"  → Data appears normally distributed (p >= 0.05)")
    
    # Choose appropriate test
    print("\n" + "-" * 70)
    if normal_dist:
        print("Using One-Way ANOVA (parametric test)")
    else:
        print("Using Kruskal-Wallis H-test (non-parametric test)")
    print("-" * 70)
    
    # Prepare data for tests
    jfh_data = df[df['feature_system'] == 'JFH']['mdl']
    hc_data = df[df['feature_system'] == 'HC']['mdl']
    spe_data = df[df['feature_system'] == 'SPE']['mdl']
    
    if normal_dist:
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(jfh_data, hc_data, spe_data)
        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4e}")
    else:
        # Kruskal-Wallis H-test
        h_stat, p_value = stats.kruskal(jfh_data, hc_data, spe_data)
        print(f"\nH-statistic: {h_stat:.4f}")
        print(f"p-value: {p_value:.4e}")
    
    if p_value < 0.001:
        print("\n*** HIGHLY SIGNIFICANT difference between feature systems (p < 0.001) ***")
    elif p_value < 0.01:
        print("\n** VERY SIGNIFICANT difference between feature systems (p < 0.01) **")
    elif p_value < 0.05:
        print("\n* SIGNIFICANT difference between feature systems (p < 0.05) *")
    else:
        print("\nNo significant difference between feature systems (p >= 0.05)")
    
    # Pairwise comparisons (post-hoc tests)
    print("\n" + "-" * 70)
    print("Pairwise Comparisons (Mann-Whitney U tests with Bonferroni correction):")
    print("-" * 70)
    
    pairs = [('JFH', 'HC'), ('JFH', 'SPE'), ('HC', 'SPE')]
    bonferroni_alpha = 0.05 / len(pairs)
    
    print(f"\nBonferroni-corrected significance level: {bonferroni_alpha:.4f}")
    
    for sys1, sys2 in pairs:
        data1 = df[df['feature_system'] == sys1]['mdl']
        data2 = df[df['feature_system'] == sys2]['mdl']
        
        stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        feat1 = FEATURE_SYSTEM_SIZES[sys1]
        feat2 = FEATURE_SYSTEM_SIZES[sys2]
        mean1 = data1.mean()
        mean2 = data2.mean()
        diff = mean2 - mean1
        
        print(f"\n{sys1} ({feat1} features) vs {sys2} ({feat2} features):")
        print(f"  Mean MDL: {mean1:.2f} vs {mean2:.2f} (difference: {diff:+.2f})")
        print(f"  U-statistic: {stat:.2f}")
        print(f"  p-value: {p:.4e}")
        
        if p < bonferroni_alpha:
            print(f"  → SIGNIFICANT difference (p < {bonferroni_alpha:.4f})")
        else:
            print(f"  → Not significant after Bonferroni correction")
    
    # Effect size (correlation between feature count and MDL)
    print("\n" + "-" * 70)
    print("Correlation Analysis: Feature Count vs MDL")
    print("-" * 70)
    
    correlation, p_corr = stats.pearsonr(df['feature_count'], df['mdl'])
    print(f"\nPearson correlation coefficient: {correlation:.4f}")
    print(f"p-value: {p_corr:.4e}")
    
    if abs(correlation) < 0.1:
        strength = "negligible"
    elif abs(correlation) < 0.3:
        strength = "weak"
    elif abs(correlation) < 0.5:
        strength = "moderate"
    elif abs(correlation) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if correlation > 0 else "negative"
    print(f"Interpretation: {strength} {direction} correlation")
    
    # Spearman correlation (non-parametric)
    spearman_corr, p_spearman = stats.spearmanr(df['feature_count'], df['mdl'])
    print(f"\nSpearman correlation coefficient: {spearman_corr:.4f}")
    print(f"p-value: {p_spearman:.4e}")
    
    print("\n" + "="*70)

def main():
    """Main execution function."""
    print("Loading data from all feature systems...")
    df = load_data()
    print(df)
    
    print(f"\nTotal data points loaded: {len(df)}")
    print(f"Unique languages: {df['language'].nunique()}")
    print(f"Feature systems: {df['feature_system'].unique()}")
    
    print("\nPerforming statistical analysis...")
    statistical_analysis(df)
    
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
