# Quantification of Feature Informativity 

This code extends the work by [Chen and Hulden](https://github.com/mhulden/minphonfeat), , which verifies whether a set of phonemes forms a natural class. It employs two algorithms: Branch & Bound and Greedy Search. According to Chen and Hulden, the Greedy Search algorithm does not guarantee finding the correct minimal feature descriptions. Therefore, the results of the Branch & Bound algorithm are used to quantify feature informativity. This algorithm exhaustively explores all combinations of features that describe a given phoneme and verifies if that phoneme is a natural class (i.e., if it is the only phoneme described by that combination of features).

Each phoneme forming a natural class is stored in a dictionary along with all its possible feature descriptions. Additionally, some descriptive measurements (such as the length of the minimal feature descriptions, the average length of feature descriptions, etc.) are stored in this dictionary.

Two different scripts can be found in the repository: `featureinfo_alllanguages.py` and `featureinfo_selectlanguages.py`. The former generates a dictionary for a dataset containing the phonemic inventory of 629 languages and three feature systems (namely the binary-valued systems of Halle & Clements (HC), Sound Pattern of English (SPE), and Jakobson, Fant, and Halle (JFH)).

Usage:

```
python3 featureinfo_alllanguages.py inventory
```
For example:

```
python3 featureinfo_alllanguages.py HC_features
```

The latter script generates a dictionary for the given language and feature system.

Usage:

```
python3 featureinfo_selectlanguages.py inventory language
```
For example:

```
python3 featureinfo_selectlanguages.py riggle chinese
```
