import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import seaborn as sns
import shap

# Define inventories
inventories = ["HC", "SPE", "JFH"]
colors = ['red', 'green', 'blue']

# Load all data first
all_data = {}
for inv in inventories:
    filename = f"data_all_languages_{inv}_features.json"
    with open(filename, 'r') as f:
        all_data[inv] = json.load(f)

# Load pb_languages_formatted.csv to get actual phoneme inventories
pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted.csv")

# ============================================================
# VIOLIN PLOT 3: Average MDL per language grouped by language family
# ============================================================

print("\n" + "=" * 60)
print("Creating violin plots by language family...")
print("=" * 60)

# Collect average MDL per language with family information
family_mdl_data = {inv: defaultdict(list) for inv in inventories}

for inv in inventories:
    print(f"\nProcessing inventory: {inv} for family violin plot")
    
    for language, lang_data in all_data[inv].items():
        if "min_lengths" in lang_data:
            min_lengths = lang_data["min_lengths"]
            
            # Get the language family from pb_languages_formatted.csv
            lang_rows = pb_languages[pb_languages['language'] == language]
            if lang_rows.empty:
                continue  # Skip if language not found in CSV
            
            family = lang_rows.iloc[0]['family']
            
            # Calculate average MDL for this language
            mdl_values_lang = list(min_lengths.values())
            if mdl_values_lang:
                avg_mdl = np.mean(mdl_values_lang)
                
                # Add to family data
                family_mdl_data[inv][family].append(avg_mdl)

# Filter families to only include those with at least 5 languages
# This makes the plot more readable and statistically meaningful
MIN_LANGUAGES_PER_FAMILY = 5

# First, find families that have >= MIN_LANGUAGES_PER_FAMILY in at least one inventory
all_families = set()
for inv in inventories:
    for family, mdl_values in family_mdl_data[inv].items():
        if len(mdl_values) >= MIN_LANGUAGES_PER_FAMILY:
            all_families.add(family)

if not all_families:
    print(f"  No families with >= {MIN_LANGUAGES_PER_FAMILY} languages, skipping...")
else:
    # Calculate median MDL across all inventories for sorting
    family_medians_combined = {}
    for family in all_families:
        all_mdl_values = []
        for inv in inventories:
            if family in family_mdl_data[inv]:
                all_mdl_values.extend(family_mdl_data[inv][family])
        if all_mdl_values:
            family_medians_combined[family] = np.median(all_mdl_values)
    
    # Sort families by combined median
    sorted_families = sorted(all_families, key=lambda f: family_medians_combined[f])
    
    print("\nCreating combined violin plot with all feature systems")
    print(f"  Families included: {len(sorted_families)}")
    
    # Create the combined violin plot
    fig, ax = plt.subplots(figsize=(max(14, len(sorted_families) * 1.2), 8))
    
    # Prepare data for violin plot
    spacing_between_families = 4.0  # Space between family groups
    spacing_within_family = 1.0     # Space between inventories within a family
    
    x_tick_positions = []
    x_tick_labels = []
    current_x = 0.0
    
    # Collect violin data and positions
    violin_data_list = []
    violin_positions = []
    
    for family_idx, family in enumerate(sorted_families):
        # Center position for this family group
        family_center = current_x + 1.5 * spacing_within_family
        x_tick_positions.append(family_center)
        x_tick_labels.append(family)
        
        # Add data for each inventory
        for inv_idx, inv in enumerate(inventories):
            x_pos = current_x + inv_idx * spacing_within_family
            
            if family in family_mdl_data[inv] and family_mdl_data[inv][family]:
                mdl_values = family_mdl_data[inv][family]
                violin_data_list.append(mdl_values)
                violin_positions.append(x_pos)
        
        current_x += spacing_between_families
    
    # Create violin plots
    parts = ax.violinplot(violin_data_list, positions=violin_positions, widths=0.7, 
                         showmeans=False, showmedians=False)
    
    # Color the violin plots
    color_counter = 0
    for family_idx, family in enumerate(sorted_families):
        current_x = sum(spacing_between_families if i > 0 else 0 for i in range(family_idx)) + family_idx * spacing_between_families
        for inv_idx, inv in enumerate(inventories):
            if family in family_mdl_data[inv] and family_mdl_data[inv][family]:
                if color_counter < len(parts['bodies']):
                    pc = parts['bodies'][color_counter]
                    pc.set_facecolor(colors[inv_idx])
                    pc.set_alpha(0.6)
                    pc.set_edgecolor(colors[inv_idx])
                    pc.set_linewidth(1.5)
                    color_counter += 1
    
    # Color other parts
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.5)
    
    # Add median lines with legend
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    
    current_x = 0.0
    for family_idx, family in enumerate(sorted_families):
        for inv_idx, inv in enumerate(inventories):
            x_pos = current_x + inv_idx * spacing_within_family
            
            if family in family_mdl_data[inv] and family_mdl_data[inv][family]:
                mdl_values = family_mdl_data[inv][family]
                median_val = np.median(mdl_values)
                ax.plot([x_pos - 0.2, x_pos + 0.2], [median_val, median_val], 
                       color=colors[inv_idx], linewidth=2)
                
                # Add to legend (only once per inventory)
                if inv not in legend_labels:
                    legend_handles.append(Line2D([0], [0], color=colors[inv_idx], linewidth=2))
                    legend_labels.append(inv)
        
        current_x += spacing_between_families
    
    # Set labels and ticks
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    ax.set_xlabel('Language Family', fontsize=14)
    ax.set_ylabel('Average Minimal Description Length (MDL)', fontsize=14)
    ax.set_title('Average MDL by Language Family (All Feature Systems)', fontsize=14)
    ax.legend(legend_handles, legend_labels, fontsize=12, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=9)
    
    # Save the combined plot
    plt.tight_layout()
    plt.savefig("mdl_by_family_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Combined violin plot saved as mdl_by_family_combined.png")
    
    # Print statistics for each family and inventory
    print("\n  Family statistics:")
    for family in sorted_families:
        print(f"\n    {family}:")
        for inv in inventories:
            if family in family_mdl_data[inv] and family_mdl_data[inv][family]:
                mdl_values = family_mdl_data[inv][family]
                n = len(mdl_values)
                median = np.median(mdl_values)
                mean = np.mean(mdl_values)
                print(f"      {inv}: n={n}, median={median:.2f}, mean={mean:.2f}")


# ============================================================
# HEATMAP: Pairwise differences in median MDL between feature systems
# ============================================================

print("\n" + "=" * 60)
print("Creating heatmap of median MDL differences by family...")
print("=" * 60)

# Prepare data: collect MDL values for each language by family and inventory
family_mdl_heatmap_data = {inv: defaultdict(list) for inv in inventories}

for inv in inventories:
    for language, lang_data in all_data[inv].items():
        if "min_lengths" in lang_data:
            min_lengths = lang_data["min_lengths"]
            
            # Get the language family from pb_languages_formatted.csv
            lang_rows = pb_languages[pb_languages['language'] == language]
            if lang_rows.empty:
                continue
            
            family = lang_rows.iloc[0]['family']
            
            # Calculate average MDL for this language
            mdl_values_lang = list(min_lengths.values())
            if mdl_values_lang:
                avg_mdl = np.mean(mdl_values_lang)
                family_mdl_heatmap_data[inv][family].append(avg_mdl)

# Calculate medians for each family and inventory
family_medians_mdl = {}
for family in set(f for inv_data in family_mdl_heatmap_data.values() for f in inv_data.keys()):
    family_medians_mdl[family] = {}
    for inv in inventories:
        if family in family_mdl_heatmap_data[inv]:
            family_medians_mdl[family][inv] = np.median(family_mdl_heatmap_data[inv][family])
        else:
            family_medians_mdl[family][inv] = np.nan

# Count languages per family
family_language_counts = defaultdict(int)
for inv in inventories:
    for family in family_mdl_heatmap_data[inv]:
        # Count unique languages in this family for this inventory
        family_language_counts[family] = len(family_mdl_heatmap_data[inv][family])

# Compute pairwise differences
differences_data = []
families_list = []

for family, medians in family_medians_mdl.items():
    # Only include families that have data for all three inventories AND more than 5 languages
    if not any(np.isnan(v) for v in medians.values()) and family_language_counts.get(family, 0) > 5:
        diff_jfh_hc = medians['JFH'] - medians['HC']
        diff_jfh_spe = medians['JFH'] - medians['SPE']
        diff_spe_hc = medians['SPE'] - medians['HC']
        
        differences_data.append({
            'family': family,
            'JFH - HC': diff_jfh_hc,
            'JFH - SPE': diff_jfh_spe,
            'SPE - HC': diff_spe_hc,
            'avg_diff': np.mean([diff_jfh_hc, diff_jfh_spe, diff_spe_hc])
        })
        families_list.append(family)

# Create DataFrame and sort by average absolute difference
diff_df = pd.DataFrame(differences_data)
diff_df = diff_df.sort_values('avg_diff', ascending=False)

# Create matrix for heatmap
heatmap_data = diff_df[['JFH - HC', 'JFH - SPE', 'SPE - HC']].values
family_labels = diff_df['family'].values

print(f"\nCreating heatmap with {len(family_labels)} families")
if len(family_labels) > 5:
    print(f"  Families included: {', '.join(family_labels[:5])}... (and {len(family_labels) - 5} more)")
else:
    print(f"  Families included: {', '.join(family_labels)}")

# Create figure for heatmap
fig, ax = plt.subplots(figsize=(10, max(6, len(family_labels) * 0.25)))

# Create heatmap
im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-np.max(np.abs(heatmap_data)), vmax=np.max(np.abs(heatmap_data)))

# Set ticks and labels
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['JFH - HC', 'JFH - SPE', 'SPE - HC'], fontsize=12)
ax.set_yticks(np.arange(len(family_labels)))
ax.set_yticklabels(family_labels, fontsize=9)

# Rotate x labels for readability
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Median Avg MDL Difference', rotation=270, labelpad=20, fontsize=11)

# Add title and labels
ax.set_xlabel('Feature System Comparison', fontsize=12)
ax.set_ylabel('Language Family', fontsize=12)
#ax.set_title('Pairwise Differences in Median MDL by Language Family\n(Blue = lower MDL, Red = higher MDL)\nSignificance (Monte-Carlo permutation + Benjamini-Hochberg FDR): * p<0.05, ** p<0.01, *** p<0.001', fontsize=13)

# Monte-Carlo permutation test with Benjamini-Hochberg FDR correction
def permutation_pvalue(x, y, n_perm=5000, seed=0):
    """Two-sample Monte-Carlo permutation test (two-sided) using median difference as test statistic."""
    rng = np.random.default_rng(seed)
    x = np.array(x)
    y = np.array(y)
    obs_diff = abs(np.median(x) - np.median(y))  # Use median difference as test statistic
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for perm_idx in range(n_perm):
        pooled_perm = pooled.copy()
        rng.shuffle(pooled_perm)
        x_perm = pooled_perm[:n_x]
        y_perm = pooled_perm[n_x:]
        if abs(np.median(x_perm) - np.median(y_perm)) >= obs_diff:
            count += 1
    return (count + 1) / (n_perm + 1)

# Collect all p-values and metadata for FDR correction
comparison_pairs = [
    ('JFH', 'HC', 0),      # JFH - HC at column 0
    ('JFH', 'SPE', 1),     # JFH - SPE at column 1
    ('SPE', 'HC', 2)       # SPE - HC at column 2
]

pvals_raw = []
pvals_metadata = []  # (family_idx, comparison_idx, inv1, inv2)

for i, family in enumerate(family_labels):
    for j, (inv1, inv2, col_idx) in enumerate(comparison_pairs):
        if family in family_mdl_heatmap_data[inv1] and family in family_mdl_heatmap_data[inv2]:
            data1 = family_mdl_heatmap_data[inv1][family]
            data2 = family_mdl_heatmap_data[inv2][family]
            
            if len(data1) > 0 and len(data2) > 0:
                p_val = permutation_pvalue(data1, data2, n_perm=5000, seed=42)
                pvals_raw.append(p_val)
                pvals_metadata.append((i, j))
            else:
                pvals_raw.append(np.nan)
                pvals_metadata.append((i, j))
        else:
            pvals_raw.append(np.nan)
            pvals_metadata.append((i, j))

# Apply Benjamini-Hochberg FDR correction
pvals_array = np.array(pvals_raw)
valid_mask = ~np.isnan(pvals_array)
valid_pvals = pvals_array[valid_mask]

if len(valid_pvals) > 0:
    rejected, pvals_corrected_valid, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')
    
    # Reconstruct full corrected p-values array (with NaNs in original positions)
    pvals_corrected = np.full_like(pvals_array, np.nan)
    pvals_corrected[valid_mask] = pvals_corrected_valid
else:
    pvals_corrected = pvals_array.copy()

# Function to map p-value to stars
def p_to_stars(p):
    if np.isnan(p):
        return ''
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return ''

# Add text annotations on cells with significance stars
pval_idx = 0
for i, family in enumerate(family_labels):
    for j in range(len(comparison_pairs)):
        value = heatmap_data[i, j]
        color = 'white' if np.abs(value) > np.max(np.abs(heatmap_data)) * 0.5 else 'black'
        
        # Get adjusted p-value for this cell
        p_adj = pvals_corrected[pval_idx] if pval_idx < len(pvals_corrected) else np.nan
        sig_stars = p_to_stars(p_adj)
        
        # Add text with value and significance stars
        text_label = f'{value:.3f}{sig_stars}'
        ax.text(j, i, text_label, ha='center', va='center', 
               color=color, fontsize=8, fontweight='bold' if sig_stars else 'normal')
        
        pval_idx += 1

# Tight layout and save
plt.tight_layout()
plt.savefig("mdl_differences_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n  Heatmap saved as mdl_differences_heatmap.png")

# Print summary statistics
print("\n  Difference summary statistics:")
print("\n    JFH - HC:")
print(f"      Mean: {diff_df['JFH - HC'].mean():.4f}")
print(f"      Median: {diff_df['JFH - HC'].median():.4f}")
print(f"      Min: {diff_df['JFH - HC'].min():.4f}, Max: {diff_df['JFH - HC'].max():.4f}")

print("\n    JFH - SPE:")
print(f"      Mean: {diff_df['JFH - SPE'].mean():.4f}")
print(f"      Median: {diff_df['JFH - SPE'].median():.4f}")
print(f"      Min: {diff_df['JFH - SPE'].min():.4f}, Max: {diff_df['JFH - SPE'].max():.4f}")

print("\n    SPE - HC:")
print(f"      Mean: {diff_df['SPE - HC'].mean():.4f}")
print(f"      Median: {diff_df['SPE - HC'].median():.4f}")
print(f"      Min: {diff_df['SPE - HC'].min():.4f}, Max: {diff_df['SPE - HC'].max():.4f}")

print("\n" + "=" * 60)
print("Heatmap analysis complete!")

# ============================================================
# SCATTER PLOT: Median MDL differences by family and comparison
# ============================================================

print("\nCreating scatter plot of median MDL differences by family...")

# Create scatter plot
fig, ax = plt.subplots(figsize=(max(10, len(family_labels) * 0.5), 8))

# Define colors for each comparison
comparison_colors = {
    'JFH - HC': '#1f77b4',   # blue
    'JFH - SPE': '#ff7f0e',  # orange
    'SPE - HC': '#2ca02c'    # green
}

# Plot each comparison as a scatter series
x_positions = np.arange(len(family_labels))
x_offset = 0.12  # Smaller offset between comparisons

for comp_idx, (comp_name, color) in enumerate(comparison_colors.items()):
    y_values = diff_df[comp_name].values
    x_scatter = x_positions + (comp_idx - 1) * x_offset
    
    # Plot dashed lines connecting the dots
    ax.plot(x_scatter, y_values, linestyle='--', color=color, alpha=0.5, linewidth=1.5)
    
    # Plot scatter points
    ax.scatter(x_scatter, y_values, s=80, alpha=0.7, color=color, label=comp_name, edgecolors='black', linewidth=0.5)

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Set labels and ticks (centered on x_positions so they align above the dot clusters)
ax.set_xticks(x_positions)
ax.set_xticklabels(family_labels, rotation=90)
ax.set_xlabel('Language Family', fontsize=14)
ax.set_ylabel('Median Avg MDL Difference', fontsize=14)
# ax.set_title('Median Avg MDL Differences by Language Family and Feature System Comparison', fontsize=15)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)
ax.tick_params(labelsize=10)

# Tight layout and save
plt.tight_layout()
plt.savefig("mdl_differences_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

print("  Scatter plot saved as mdl_differences_scatter.png")


# ============================================================
# CONSTRUCT PHONEME INVENTORIES FOR SHAP ANALYSIS
# ============================================================

# Extract phoneme inventories and language families from pb_languages CSV
print("\nConstructing phoneme inventories for SHAP analysis...")

pb_languages = pd.read_csv("phonemic_inventories/pb_languages_formatted_filtered.csv")

language_list = []
family_list = []
phoneme_inventories = {}

for _, row in pb_languages.iterrows():
    language = row['language']
    family = row['family']
    inventory_str = row['core inventory']
    
    # Parse inventory string (format: "['phoneme1', 'phoneme2', ...]")
    inventory_str = inventory_str.strip('[]')
    phonemes = [p.strip().strip('\'"') for p in inventory_str.split(',')]
    phonemes = [p for p in phonemes if p]  # Remove empty strings
    
    if phonemes:
        language_list.append(language)
        family_list.append(family)
        phoneme_inventories[language] = list(set(phonemes[0].split()))

# Collect all unique phonemes
all_phonemes = set()
for phonemes in phoneme_inventories.values():
    all_phonemes.update(phonemes)

all_phonemes = sorted(list(all_phonemes))
unique_families = sorted(list(set(family_list)))

print(f"Total languages: {len(language_list)}")
print(f"Total families: {len(unique_families)}")
print(f"Total unique phonemes: {len(all_phonemes)}")


# ============================================================
# RANDOM FOREST + SHAP: Predict MDL from Family Phoneme Frequencies
# ============================================================

print("\n" + "=" * 60)
print("Random Forest + SHAP Analysis: Predicting MDL from Phoneme Frequencies")
print("=" * 60)

# Create dataset: rows = families (with >= 5 languages), columns = phoneme frequencies + average MDL
print("\nPreparing dataset for Random Forest...")

# Step 1: Identify families with at least 5 languages
families_with_min_languages = {}
for family in unique_families:
    family_languages = [lang for i, lang in enumerate(language_list) if family_list[i] == family]
    if len(family_languages) >= 5:
        families_with_min_languages[family] = family_languages

print(f"Families with >= 5 languages: {len(families_with_min_languages)}")

# Step 2: Build family-level dataset with phoneme frequencies and average MDL
family_rf_data = []

for family, family_languages in families_with_min_languages.items():
    # Compute phoneme frequencies for this family
    family_phoneme_freq = {}
    for phoneme in all_phonemes:
        count = sum(1 for lang in family_languages if phoneme in phoneme_inventories.get(lang, set()))
        freq = count / len(family_languages)
        family_phoneme_freq[phoneme] = freq
    
    # Compute average MDL across all languages in this family (across all inventories)
    family_mdl_values = []
    inv = "JFH"
    if family in family_mdl_heatmap_data[inv]:
        family_mdl_values.extend(family_mdl_heatmap_data[inv][family])
    
    if family_mdl_values:
        median_mdl = np.median(family_mdl_values)
        
        # Create row for this family
        row = {'family': family, 'median_mdl': median_mdl}
        row.update(family_phoneme_freq)
        family_rf_data.append(row)

# Convert to DataFrame
df_rf = pd.DataFrame(family_rf_data)

# Ensure all columns are present (fill missing phonemes with 0)
for phoneme in all_phonemes:
    if phoneme not in df_rf.columns:
        df_rf[phoneme] = 0.0

print(f"\nDataset shape: {df_rf.shape}")
print(f"  Families (with >= 5 languages): {len(df_rf)}")
print(f"  Phoneme features: {len(all_phonemes)}")
print(f"\nDataset summary (first 10 families):")
print(df_rf[['family', 'median_mdl']].head(10))

# Step 3: Prepare X and y for Random Forest
X_rf = df_rf[[col for col in df_rf.columns if col not in ['family', 'median_mdl']]]
y_rf = df_rf['median_mdl']

print(f"\nX shape: {X_rf.shape}, y shape: {y_rf.shape}")
print(f"Median MDL across families: {y_rf.median():.4f} (min: {y_rf.min():.4f}, max: {y_rf.max():.4f})")

# Step 4: Train Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf_mdl_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

rf_mdl_model.fit(X_rf, y_rf)

# Evaluate on training data
train_score = rf_mdl_model.score(X_rf, y_rf)
print(f"Training R² score: {train_score:.4f}")

# Predictions and residuals
y_pred = rf_mdl_model.predict(X_rf)
residuals = y_rf - y_pred
print(f"Mean absolute error: {np.mean(np.abs(residuals)):.4f}")

# Create predicted vs actual plot
print("\nGenerating predicted vs actual plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot of actual vs predicted
ax.scatter(y_rf, y_pred, s=100, alpha=0.6, edgecolors='black', linewidth=0.5, color='steelblue')


# Perfect prediction line (y=x)
min_val = min(y_rf.min(), y_pred.min())
max_val = max(y_rf.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, alpha=0.5)

# Labels and title
ax.set_xlabel('Actual Median MDL', fontsize=13)
ax.set_ylabel('Predicted Median MDL', fontsize=13)
ax.set_title(f'Random Forest: Predicted vs Actual Median MDL\n(Training R² = {train_score:.4f})', fontsize=14)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

# Equal aspect ratio for better visualization
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('rf_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: rf_predicted_vs_actual.png")

# Step 5: Compute SHAP values
print("\nComputing SHAP values...")
explainer_mdl = shap.TreeExplainer(rf_mdl_model)
shap_values_mdl = explainer_mdl(X_rf)  # Call explainer on dataset to get Explanation object with feature values

# Get base value
base_value_mdl = explainer_mdl.expected_value
print(f"Base value (expected MDL): {base_value_mdl[0]:.4f}")

# Generate waterfall plots for all families
print("\nGenerating waterfall plots for all families...")
for sample_idx in range(len(df_rf)):
    family_name = df_rf.iloc[sample_idx]['family']
    pred_sample = rf_mdl_model.predict(X_rf.iloc[[sample_idx]])[0]
    reconstructed_sample = base_value_mdl + np.sum(shap_values_mdl.values[sample_idx])
    
    print(f"  {family_name}: prediction={pred_sample:.4f}, base+sum(SHAP)={reconstructed_sample[0]:.4f}")
    
    # Generate waterfall plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(shap_values_mdl[sample_idx], show=False)
    plt.tight_layout()
    
    # Create safe filename
    safe_family_name = family_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    plt.savefig(f"shap_waterfall_{safe_family_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"  Generated {len(df_rf)} waterfall plots")


# Step 6: SHAP visualizations
print("\nGenerating SHAP visualizations...")

# Summary bar plot (global importance)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_mdl, X_rf, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_summary_bar_mdl_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_summary_bar_mdl_prediction.png")

# Beeswarm summary plot
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values_mdl, X_rf, show=False)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm_mdl_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: shap_summary_beeswarm_mdl_prediction.png")

# Create scatter plots: SHAP values vs phoneme frequencies for each phoneme
print("\nGenerating SHAP vs phoneme frequency scatter plots...")

# Create output directory if it doesn't exist
import os
output_dir = "shap_per_phoneme"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Calculate mean absolute SHAP value for each feature
mean_abs_shap = np.mean(np.abs(shap_values_mdl.values), axis=0)

# Get indices of top 20 features by mean absolute SHAP value
top_20_indices = np.argsort(mean_abs_shap)[-20:][::-1]  # Sort descending
top_20_phonemes = [X_rf.columns[i] for i in top_20_indices]

print(f"  Top 20 phonemes by mean absolute SHAP value:")
for rank, (idx, phoneme) in enumerate(zip(top_20_indices, top_20_phonemes), 1):
    print(f"    {rank}. {phoneme}: {mean_abs_shap[idx]:.6f}")

# Generate scatter plots only for top 20 phonemes
for phoneme_idx in top_20_indices:
    phoneme = X_rf.columns[phoneme_idx]
    
    # Get feature values (frequencies) and SHAP values for this phoneme
    freq_values = X_rf[phoneme].values
    shap_values_phoneme = shap_values_mdl.values[:, phoneme_idx]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(freq_values, shap_values_phoneme, s=100, alpha=0.6, edgecolors='black', linewidth=0.5, color='steelblue')
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel(f'{phoneme} Frequency', fontsize=12)
    ax.set_ylabel(f'{phoneme} SHAP Value', fontsize=12)
    ax.set_title(f'SHAP Value vs Frequency for Phoneme "{phoneme}"', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_scatter_vs_freq_{phoneme}.png"), dpi=300, bbox_inches='tight')
    plt.close()

print(f"  Generated {len(top_20_indices)} scatter plots (top 20 phonemes by absolute SHAP value)")
print(f"  Saved in folder: {output_dir}")


# Summary of results
print("\n" + "=" * 60)
print("Random Forest + SHAP Analysis Complete")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  Training R²: {train_score:.4f}")
print(f"  Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
print(f"  Base value (expected MDL): {base_value_mdl[0]:.4f}")

print("\n" + "=" * 60)


# ============================================================
# CORRELATION ANALYSIS: Phoneme Frequencies Across Families
# ============================================================

print("\n" + "=" * 60)
print("Phoneme Frequency Correlation Analysis")
print("=" * 60)

# Use the X_rf dataframe (family × phoneme frequencies)
print(f"\nComputing correlation matrix for {X_rf.shape[1]} phonemes across {X_rf.shape[0]} families...")

# Compute Pearson correlation matrix
corr_matrix = X_rf.corr(method='pearson')

# Find top correlated phoneme pairs (excluding self-correlation)
# Flatten the upper triangle and sort by absolute correlation
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        phoneme1 = corr_matrix.columns[i]
        phoneme2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        corr_pairs.append({
            'phoneme1': phoneme1,
            'phoneme2': phoneme2,
            'correlation': corr_val,
            'abs_correlation': abs(corr_val)
        })

corr_pairs_df = pd.DataFrame(corr_pairs)
corr_pairs_df = corr_pairs_df.sort_values('abs_correlation', ascending=False)

# Print top 30 most correlated pairs
print("\nTop 30 most correlated phoneme pairs:")
print("(positive correlation = frequencies tend to increase together)")
print("(negative correlation = frequencies tend to move oppositely)")
print()
for idx, row in corr_pairs_df.head(30).iterrows():
    print(f"  {row['phoneme1']:8s} <-> {row['phoneme2']:8s}  r = {row['correlation']:7.4f}")

# Find phoneme pairs most correlated with the top impactful phoneme (dʑ if present)
top_phonemes_list = top_20_phonemes[:5]  # Use top 5 from SHAP analysis
print(f"\nCorrelations with top SHAP-important phonemes:")
for phoneme in top_phonemes_list:
    if phoneme in X_rf.columns:
        corr_with_phoneme = corr_matrix[phoneme].sort_values(ascending=False)
        print(f"\n  {phoneme} (top correlations):")
        for other_ph in corr_with_phoneme[1:6].index:  # Skip self, show top 5
            print(f"    {other_ph:8s}  r = {corr_with_phoneme[other_ph]:7.4f}")

# Create and save correlation heatmap (subset to most variable/important phonemes)
print("\nGenerating correlation heatmap (top 20 SHAP phonemes)...")

# Select subset of phonemes for heatmap (top 20 by SHAP)
top_20_cols = top_20_phonemes
subset_corr = corr_matrix.loc[top_20_cols, top_20_cols]

# Mask upper triangle (keep only lower triangle and diagonal)
mask = np.triu(np.ones_like(subset_corr, dtype=bool), k=1)
subset_corr_masked = subset_corr.copy()
subset_corr_masked[mask] = np.nan

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(subset_corr_masked, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(top_20_cols)))
ax.set_yticks(np.arange(len(top_20_cols)))
ax.set_xticklabels(top_20_cols, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(top_20_cols, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pearson Correlation', rotation=270, labelpad=20, fontsize=11)

# Add values in cells (only lower triangle)
for i in range(len(top_20_cols)):
    for j in range(i + 1):  # Only lower triangle and diagonal
        val = subset_corr.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

ax.set_title('Phoneme Frequency Correlations (Top 20 SHAP-Important Phonemes)', fontsize=13, pad=20)
ax.set_xlabel('Phoneme', fontsize=12)
ax.set_ylabel('Phoneme', fontsize=12)

plt.tight_layout()
plt.savefig('phoneme_frequency_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: phoneme_frequency_correlations.png")

# # Create scatter plots for all phoneme pairs (lower triangle only)
# print("\nGenerating scatter plots for phoneme frequency correlations...")

# corr_output_dir = "correlations_per_phonemes"
# if not os.path.exists(corr_output_dir):
#     os.makedirs(corr_output_dir)

# plot_count = 0
# for i in range(len(top_20_cols)):
#     for j in range(i):  # Only lower triangle (j < i)
#         phoneme1 = top_20_cols[j]
#         phoneme2 = top_20_cols[i]
        
#         freq1 = X_rf[phoneme1].values
#         freq2 = X_rf[phoneme2].values
#         corr_val = subset_corr.iloc[i, j]
        
#         # Create scatter plot
#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.scatter(freq1, freq2, s=100, alpha=0.6, edgecolors='black', linewidth=0.5, color='steelblue')
        
#         # Add trend line
#         z = np.polyfit(freq1, freq2, 1)
#         p = np.poly1d(z)
#         x_line = np.linspace(freq1.min(), freq1.max(), 100)
#         ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
        
#         # Labels and title
#         ax.set_xlabel(f'{phoneme1} Frequency', fontsize=12)
#         ax.set_ylabel(f'{phoneme2} Frequency', fontsize=12)
#         ax.set_title(f'Frequency Correlation: {phoneme1} vs {phoneme2}\n(r = {corr_val:.4f})', fontsize=13)
#         ax.grid(True, alpha=0.3)
#         ax.tick_params(labelsize=10)
#         ax.legend(fontsize=11)
        
#         plt.tight_layout()
        
#         # Save with safe filename
#         safe_name1 = phoneme1.replace('/', '_').replace(' ', '_')
#         safe_name2 = phoneme2.replace('/', '_').replace(' ', '_')
#         filename = f"correlation_{safe_name1}_vs_{safe_name2}.png"
#         plt.savefig(os.path.join(corr_output_dir, filename), dpi=300, bbox_inches='tight')
#         plt.close()
#         plot_count += 1

# print(f"  Generated {plot_count} scatter plots (one per unique phoneme pair)")
# print(f"  Saved in folder: {corr_output_dir}")

# Summary statistics
print(f"\nCorrelation matrix summary:")
print(f"  Mean absolute correlation (excluding self): {corr_pairs_df['abs_correlation'].mean():.4f}")
print(f"  Max positive correlation: {corr_pairs_df['correlation'].max():.4f}")
print(f"  Max negative correlation: {corr_pairs_df['correlation'].min():.4f}")
print(f"  Median absolute correlation: {corr_pairs_df['abs_correlation'].median():.4f}")

print("\n" + "=" * 60)
print("Correlation analysis complete!")
print("=" * 60)


# ============================================================
# FEATURE-MDL ANALYSIS: Top 10 Phonemes and Their JFH Features
# ============================================================

print("\n" + "=" * 60)
print("Feature-MDL Analysis: Top 10 Impactful Phonemes by JFH Features")
print("=" * 60)

# Get top 10 most impactful phonemes
top_10_indices = np.argsort(mean_abs_shap)[-10:][::-1]
top_10_phonemes = [X_rf.columns[i] for i in top_10_indices]

print(f"\nTop 10 phonemes by mean absolute SHAP value:")
for rank, (idx, phoneme) in enumerate(zip(top_10_indices, top_10_phonemes), 1):
    print(f"  {rank}. {phoneme}: {mean_abs_shap[idx]:.6f}")

# Create output directory for histograms
feature_mdl_dir = "feature_mdl_histograms"
if not os.path.exists(feature_mdl_dir):
    os.makedirs(feature_mdl_dir)

print(f"\nGenerating feature-MDL histograms for top 10 phonemes...")

# Load JFH feature set from file
df_jfh = pd.read_csv("feature_sets/JFH_features.txt", sep="\t", index_col=0)
df_jfh.index.name = 'segment'
df_jfh = df_jfh.reset_index()
jfh_fields = list(df_jfh.columns)

print(f"  Loaded {len(df_jfh)} phoneme feature records from JFH feature set")
print(f"  Features available: {jfh_fields[1:]}")

# For each top 10 phoneme, create a histogram of MDL by JFH features
for phoneme in top_10_phonemes:
    if phoneme not in df_jfh['segment'].values:
        print(f"  Warning: {phoneme} not found in JFH feature set, skipping...")
        continue
    
    # Get JFH features for this phoneme
    phoneme_jfh_row = df_jfh[df_jfh['segment'] == phoneme].iloc[0]
    
    # Get all families where this phoneme appears (frequency > 0)
    families_with_phoneme = df_rf[df_rf[phoneme] > 0]['family'].values
    
    # Collect MDL values by feature presence
    feature_mdl_data = {}
    
    for feature in jfh_fields[1:]:  # Skip 'segment'
        feature_val = phoneme_jfh_row[feature]
        
        # Convert feature to readable form
        if feature_val == '+':
            feature_label = f"{feature}\n(+)"
            feature_status = "present"
        elif feature_val == '-':
            feature_label = f"{feature}\n(-)"
            feature_status = "absent"
        else:
            feature_label = f"{feature}\n(n)"
            feature_status = "neutral"
        
        # Get MDL values for families with this phoneme
        mdl_values = []
        for family in families_with_phoneme:
            if family in df_rf['family'].values:
                mdl_val = df_rf[df_rf['family'] == family]['median_mdl'].values[0]
                mdl_values.append(mdl_val)
        
        if mdl_values:
            feature_mdl_data[feature_label] = {
                'values': mdl_values,
                'mean': np.mean(mdl_values),
                'median': np.median(mdl_values),
                'status': feature_status
            }
    
    # Create histogram showing mean MDL for each feature
    if feature_mdl_data:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        features_list = list(feature_mdl_data.keys())
        mean_mdl_list = [feature_mdl_data[f]['mean'] for f in features_list]
        median_mdl_list = [feature_mdl_data[f]['median'] for f in features_list]
        
        # Color bars based on feature status
        colors_list = []
        for f in features_list:
            status = feature_mdl_data[f]['status']
            if status == 'present':
                colors_list.append('#2ecc71')  # green for +
            elif status == 'absent':
                colors_list.append('#e74c3c')  # red for -
            else:
                colors_list.append('#95a5a6')  # gray for n
        
        # Create bar plot
        x_pos = np.arange(len(features_list))
        bars = ax.bar(x_pos, mean_mdl_list, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)

        
        # Set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features_list, fontsize=9, rotation=45, ha='right')
        ax.set_ylabel('Median MDL of Families with This Phoneme', fontsize=12)
        ax.set_xlabel('JFH Features', fontsize=12)
        ax.set_title(f'JFH Feature Characterization and MDL: {phoneme}\n(Green=+ | Red=- | Gray=n)', fontsize=13, pad=15)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=11)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, mean_mdl_list)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        safe_phoneme_name = phoneme.replace('/', '_').replace(' ', '_').replace('ʑ', 'dz')
        plt.savefig(os.path.join(feature_mdl_dir, f"feature_mdl_{safe_phoneme_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: feature_mdl_{safe_phoneme_name}.png")

print(f"\nFeature-MDL histograms saved in folder: {feature_mdl_dir}")

# Print summary: for each top 10 phoneme, show key features
print(f"\n{'Phoneme':<12} {'Key Features (JFH)':<60}")
print("-" * 72)

for phoneme in top_10_phonemes:
    if phoneme in df_jfh['segment'].values:
        phoneme_jfh_row = df_jfh[df_jfh['segment'] == phoneme].iloc[0]
        
        # Collect feature descriptions
        feature_desc = []
        for feature in jfh_fields[1:]:
            feature_val = phoneme_jfh_row[feature]
            if feature_val == '+':
                feature_desc.append(f"+{feature}")
            elif feature_val == '-':
                feature_desc.append(f"-{feature}")
        
        features_str = ", ".join(feature_desc[:8])  # Show first 8 features
        if len(feature_desc) > 8:
            features_str += "..."
        
        print(f"{phoneme:<12} {features_str:<60}")

print("\n" + "=" * 60)
print("Feature-MDL Analysis Complete")
print("=" * 60)






