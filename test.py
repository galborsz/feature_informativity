import json

inv = "SPE"
# Load the JSON file
with open(f'data_all_languages_{inv}_features.json', 'r') as f:
    data = json.load(f)

# Extract and print the phonemes from min_descriptions keys
phonemes_SPE = []
for language, language_data in data.items():
    phonemes_SPE.extend(language_data.get('min_descriptions', {}).keys())
print(len(set(phonemes_SPE)))

inv = "JFH"
# Load the JSON file
with open(f'data_all_languages_{inv}_features.json', 'r') as f:
    data = json.load(f)

# Extract and print the phonemes from min_descriptions keys
phonemes_JFH = []
for language, language_data in data.items():
    phonemes_JFH.extend(language_data.get('min_descriptions', {}).keys())
print(len(set(phonemes_JFH)))


inv = "HC"
# Load the JSON file
with open(f'data_all_languages_{inv}_features.json', 'r') as f:
    data = json.load(f)

# Extract and print the phonemes from min_descriptions keys
phonemes_HC = []
for language, language_data in data.items():
    phonemes_HC.extend(language_data.get('min_descriptions', {}).keys())
print(len(set(phonemes_HC)))

# Find phonemes present in all three lists
common_phonemes = set(phonemes_SPE) & set(phonemes_JFH) & set(phonemes_HC)
print("Common phonemes:", sorted(list(common_phonemes)))
print("Count:", len(common_phonemes))
