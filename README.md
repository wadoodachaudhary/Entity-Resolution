# Entity-Resolution using Random Forest Classifier in Python
Entity Resolution, address matching for Four Square addresses with Locu using Random Forest Classifier in Python. 
In this lab, we take two datasets that describe the same entities, and identify which entity in one dataset is the same as an entity in the other dataset. Our datasets were provided by Foursquare and Locu, and contain descriptive information about various venues such as venue names and phone numbers.

![Alt text](/relative/path/to/leaderboardscore.jpg?raw=true "Preciosn Recall, F1-score")

This project uses several files. 
The data and matches for training in folder:
foursquare_train.json
locu_train.json
matches_train.csv
The data for online competition is in folder: online_competition
foursquare_test.json
locu_test.json

The `json` files contain a json-encoded list of venue attribute dictionaries. The `csv` file contains two columns, `locu_id` and `foursquare_id`, which reference the venue `id` fields that match in each dataset.

The program loads both datasets and identify matching venues in each dataset, measures the precision, recall, and F1-score against the ground truth in "matches_train.csv". 
The final program generates `matches_test.csv`, a mapping that looks like `matches_train.csv` but with mappings for the new test listings.

Here are a few notes:
The schemas for these datasets are aligned, but this was something that Locu and Foursquare engineers had to do ahead of time when we initially matched our datasets

The two datasets don't have the same exact formatting for some fields. Normaliziation of datasets was required. Some values in ground truth were not correct.That's fair: our data comes from matching algorithms and crowds, both of which can be imperfect. 
There are many different features that can suggest venue similarity. Field equality is a good one: if the names or phone numbers of venues are equal, the venues might be equal. But this is a relatively high-precision, low-recall feature (`Bob's Pizza` and `Bob's Pizzeria` aren't equal), and so we'll have to add other ones. We used Levenshtein distance between two strings offers finer granularity of how similar two strings are. At Locu we have many tens of features, ranging from field equality to more esoteric but useful ones (e.g., "Does the first numeric value in the address field match?").
Since there are many features, we needed some way to combine them. A simple weighted average of values, where more important values (similar names) are weighed more highly will get you quite far. In practice, we'd want to build a classifier that takes these features and learns the weights based on the training data. 
We saw good results with the decision tree ensemble/random forest techniques.
 
It's possible to be near 1 for precision/recall/F1 with enough training data and good enough machine learning models, but this took Locu engineers several months to get right. 

These datasets aren't too large, but in practice require matching several million venues across datasets. Performing an `O(N^2)` comparison on all venues would take too long in those cases so some heuristics are needed to narrow down the likely candidates.
