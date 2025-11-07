# Quantification of Feature Informativity 

This code extends the work by [Chen and Hulden](https://github.com/mhulden/minphonfeat), which verifies whether a set of phonemes forms a natural class. It employs two algorithms: Branch & Bound and Greedy Search. According to Chen and Hulden, the Greedy Search algorithm does not guarantee finding the correct minimal feature descriptions. Therefore, the results of the Branch & Bound algorithm are used to quantify feature informativity. This algorithm exhaustively explores all combinations of features that describe a given phoneme and verifies if that phoneme is a natural class (i.e., if it is the only phoneme described by that combination of features). This project is based on the assumption that a feature exhibiting high informativity can isolate a phoneme within a natural class with minimal or no additional features. Hence, each phoneme forming a natural class is stored in a dictionary along with all its possible feature descriptions. Additionally, some descriptive measurements (such as the length of the minimal feature descriptions, the average length of feature descriptions, etc.) are stored in this dictionary.

Two different scripts can be found in the repository: `featureinfo_alllanguages.py` and `featureinfo_selectlanguages.py`. Despite their many similarities, each of them runs the code on a different dataset. The script operates on a fixed dataset that includes the phonemic inventories of 629 languages. While the dataset itself cannot be altered, users can select from three available feature systems: the binary-valued systems developed by Halle & Clements (HC), the Sound Pattern of English (SPE), and the system by Jakobson, Fant, and Halle (JFH).

Usage:

```
python3 featureinfo_alllanguages.py inventory
```
For example:

```
python3 featureinfo_alllanguages.py HC_features
```

The latter script allows users to create a dictionary for a chosen language and feature system. Users can select the phonemic inventory from Chinese, Dutch, French, or British English, and the feature system from either Riggle or Hayes.

Usage:

```
python3 featureinfo_selectlanguages.py inventory language
```
For example:

```
python3 featureinfo_selectlanguages.py riggle chinese
```
