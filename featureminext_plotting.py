import matplotlib.pyplot as plt
import json
import sys
import os
import pandas as pd
import numpy as np

def aux_plotting_function(values, title, xlabel, ylabel, path, xint):
    values = dict(sorted(values.items(), key=lambda item: item[1]))
    classes = list(values.keys())
    counts = list(values.values())

    # Plot the histogram
    # plt.figure(figsize=(15,5))
    plt.bar(classes, counts, width=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xint: plt.locator_params(axis="x", integer=True)
    else: plt.xticks(rotation=90)
    plt.savefig(path, bbox_inches='tight')
    plt.close() 

def aux_plotting_function_error_bars(xvalues, yvalues, std_devs, title, xlabel, ylabel, path, xint):
    plt.figure(figsize=(30,7))
    plt.bar(xvalues, yvalues, yerr=std_devs, width=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xint: plt.locator_params(axis="x", integer=True)
    else: plt.xticks(rotation=90)
    plt.savefig(path, bbox_inches='tight')
    plt.close() 

def group_info_by_family(loaded_dict, info, families):
    # Initialize dictionaries
    avg_per_family = {family: [] for family in families.unique()}

    # Compute the values
    for language in loaded_dict:
        selected_language = loaded_dict[language]
        language = language.replace(" or ", "/")
        family = df.loc[df['language'] == language, 'family'].values[0]

        sum_per_language = sum(selected_language[info].values())
        count_per_language = len(selected_language[info])
        avg_value = sum_per_language / count_per_language

        avg_per_family[family].append(avg_value)

    # Calculate means and standard deviations
    mean_per_family = {family: np.mean(values) for family, values in avg_per_family.items()}
    std_dev_per_family = {family: np.std(values) for family, values in avg_per_family.items()}

    # Plotting
    sorted_families = sorted(mean_per_family.items(), key=lambda x: x[1])
    sorted_family_names = [family for family, mean in sorted_families]
    sorted_means = [mean for family, mean in sorted_families]
    sorted_std_devs = [std_dev_per_family[family] for family in sorted_family_names]

    return sorted_family_names, sorted_means, sorted_std_devs

def plot_avg_measures_per_family(loaded_dict, info, df, output_dir):
    families = df['family'].unique()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collecting all features across languages
    all_features = list(next(iter(loaded_dict.values()))[info].keys())
    
    for feature in all_features:
        avg_per_family = {family: [] for family in families}
        
        # Compute the values per family
        for language in loaded_dict:
            selected_language = loaded_dict[language]
            language = language.replace(" or ", "/")
            family = df.loc[df['language'] == language, 'family'].values[0]
            
            if feature in selected_language[info]:
                measure = selected_language[info][feature]
                avg_per_family[family].append(measure)
        
        # Calculate means and standard deviations
        mean_per_family = {family: np.mean(values) for family, values in avg_per_family.items() if values}
        std_dev_per_family = {family: np.std(values) for family, values in avg_per_family.items() if values}
        
        # Plotting
        sorted_families = sorted(mean_per_family.items(), key=lambda x: x[1])
        sorted_family_names = [family for family, mean in sorted_families]
        sorted_means = [mean for family, mean in sorted_families]
        sorted_std_devs = [std_dev_per_family[family] for family in sorted_family_names]
        
        plt.figure(figsize=(20, 10))
        plt.errorbar(sorted_family_names, sorted_means, yerr=sorted_std_devs, ecolor='black', fmt='o', capsize=5)
        plt.xlabel('Language Families')
        plt.ylabel(f'Average minimal description length')
        plt.title(f'Feature: {feature} (Feature set: {inventoryfile})')
        plt.xticks(rotation=90)
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{feature}.png'))
        plt.close()
        
inventoryfile = sys.argv[1]
with open(f'data_all_languages_{inventoryfile}.json', 'r') as file:
    loaded_dict = json.load(file)
    
    # Plots all languages
    df = pd.read_csv('pb_languages_formatted.csv')
    # Extract language and family information
    languages = df['language'].apply(lambda x: x.replace(" or ", "/"))
    families = df['family']
    
    sorted_family_names, sorted_means, sorted_std_devs = group_info_by_family(loaded_dict, 'avg_lengths', families)
    title = f'Average length of feature descriptions per language family\n({inventoryfile} feature set)'
    path = f'avg_alldescrip_all_languages.jpg'
    aux_plotting_function_error_bars(sorted_family_names, sorted_means, sorted_std_devs, title, 'Language family', 'Average length', path, False)

    sorted_family_names, sorted_means, sorted_std_devs = group_info_by_family(loaded_dict, 'min_lengths', families)
    title = f'Average length of minimal feature descriptions per language family\n({inventoryfile} feature set)'
    path = f'avg_mindescrip_all_languages.jpg'
    aux_plotting_function_error_bars(sorted_family_names, sorted_means, sorted_std_devs, title, 'Language family', 'Average length', path, False)
    
    plot_avg_measures_per_family(loaded_dict, 'min_lengths', df, f'{inventoryfile}')
    # # Plots per language
    # for language in loaded_dict:
    #     loaded_dict_language = loaded_dict[language]
    #     min_lengths = loaded_dict_language['min_lengths']
    #     min_descriptions = loaded_dict_language['min_descriptions']
    #     count_phoneme = loaded_dict_language['count_phoneme']
    #     avg_lengths = loaded_dict_language['avg_lengths']
    #     count_lengths = loaded_dict_language['count_lengths']

    #     if not os.path.exists(f'count_minimal_description'):
    #         os.makedirs(f'count_minimal_description')
    #     if not os.path.exists(f'count_minimal_description/{inventoryfile}'):
    #         os.makedirs(f'count_minimal_description/{inventoryfile}')

    #     title = f'Count of minimal descriptions for various lengths \n({language} phonemic inventory with {inventoryfile} feature set)'
    #     path = f'count_minimal_description/{inventoryfile}/{language}.jpg'
    #     aux_plotting_function(count_lengths, title, 'Length', 'Count', path, True)

    #     total_sum = sum(count_lengths.values())
    #     normalized_lengths = {key: value / total_sum for key, value in count_lengths.items()}

    #     if not os.path.exists(f'normalized_count_minimal_description'):
    #         os.makedirs(f'normalized_count_minimal_description')
    #     if not os.path.exists(f'normalized_count_minimal_description/{inventoryfile}'):
    #         os.makedirs(f'normalized_count_minimal_description/{inventoryfile}')

    #     title = f'Normalized count of minimal descriptions for various lengths \n({language} phonemic inventory with {inventoryfile} feature set)'
    #     path = f'normalized_count_minimal_description/{inventoryfile}/{language}.jpg'
    #     aux_plotting_function(normalized_lengths, title, 'Length', 'Normalized count', path, True)

    #     if not os.path.exists(f'count_phonemes_per_feature'):
    #         os.makedirs(f'count_phonemes_per_feature')
    #     if not os.path.exists(f'count_phonemes_per_feature/{inventoryfile}'):
    #         os.makedirs(f'count_phonemes_per_feature/{inventoryfile}')

    #     title = f'The number of times the feature is included in the minimal description of a phoneme\n({language} phonemic inventory with {inventoryfile} feature set)'
    #     path = f'count_phonemes_per_feature/{inventoryfile}/{language}.jpg'
    #     aux_plotting_function(count_phoneme, title, 'Feature', 'Count', path, False)

    #     if not os.path.exists(f'min_feature_importance'):
    #         os.makedirs(f'min_feature_importance')
    #     if not os.path.exists(f'min_feature_importance/{inventoryfile}'):
    #         os.makedirs(f'min_feature_importance/{inventoryfile}')

    #     title = f'Length of minimal feature description\n({language} phonemic inventory with {inventoryfile} feature set)'
    #     path = f'min_feature_importance/{inventoryfile}/{language}.jpg'
    #     aux_plotting_function(min_lengths, title, 'Feature', 'Length', path, False)

    #     if not os.path.exists(f'avg_feature_importance'):
    #         os.makedirs(f'avg_feature_importance')
    #     if not os.path.exists(f'avg_feature_importance/{inventoryfile}'):
    #         os.makedirs(f'avg_feature_importance/{inventoryfile}')

    #     title = f'Average length of feature description\n({language} phonemic inventory with {inventoryfile} feature set)'
    #     path = f'avg_feature_importance/{inventoryfile}/{language}.jpg'
    #     aux_plotting_function(avg_lengths, title, 'Feature', 'Average length', path, False)