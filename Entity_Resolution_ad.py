import numpy as np
from sklearn.ensemble import RandomForestClassifier
import hungarian
import re
import argparse
import tldextract
import random, string
import math
import json
import editdistance
import csv
from unidecode import unidecode

path_train = "/Users/wadood/Library/Mobile Documents/com~apple~CloudDocs/Projects/DataSystems_COMS4121/hw3/train/"
path_test = "/Users/wadood/Library/Mobile Documents/com~apple~CloudDocs/Projects/DataSystems_COMS4121/hw3/online_competition/"


marker = "============================================================================================================="

# Generated using the True Negatives of my Algo below
blacklist = [
    #('705fe1489025117686ff','4f328ea619836c91c7e3714a'),
    ('c170270283ef870d546b','51eb7eed498e401ec51196b6'),
    ('825acefd3e298274a150','4f9ab1dbd4f2465542bc673f'),
    ('212dffb393f745df801a','51e869ac498e7e485cabcdeb'),
    #('9ea3254360d0fe59177e','4dc597c57d8b14fb462ed076'),
    ('e3f9d84c0c989f2e7928','51e25e57498e535de72f03e7'),
    ('edeba23f215dcc702220','51a11cbc498e4083823909f1'),
    #('496bd5b462f08383d880','3fd66200f964a5209eea1ee3'),
    ('66ef54d76ff989a91d52','51c9e1dd498e33ecd8670892'),
    ('5f3fd107090d0ddc658b','51ce011a498ed8dfb15381bb'),
    ('80afa95c01dae3ba5434','506fc5f6e4b0184b5ae01a83'),
    #('e863ac06811e469bde9b','4a7b1eaaf964a52012ea1fe3'),
    ('25485cb3241580f995ff','4dee9f25fa761efc37a0d1a4'),
    ('493f5e2798de851ec3b2','51f119e7498e9716f71f4413'),
]


def remove_non_ascii(text):
    return unidecode(unicode(text))


def get_alphanum(text):
    text_ascii = remove_non_ascii(text)
    text_alphanum = re.sub("[^0-9a-zA-Z ]", "", text_ascii)
    return text_alphanum


def get_alphabetic(text):
    text_ascii = remove_non_ascii(text)
    text_alpha = re.sub("[^a-zA-Z ]", "", text_ascii)
    return text_alpha


def get_numeric(text):
    text_ascii = remove_non_ascii(text)
    text_numeric = re.sub("[^0-9]", "", text_ascii)
    return text_numeric


def load_json(file_name, key_column):
    with file(file_name, 'r') as f:
        json_objects = json.load(f)

    data = {}
    for obj in json_objects:
        data[obj[key_column]] = obj

    return data


def load_csv(file_name, has_header=True):
    reader = csv.reader(open(file_name, 'r'))
    if has_header:
        next(reader, None)  # skip the headers
    data = {}
    for row in reader:
        k, v = row
        data[(k, v)] = 1
    return data


def dump_matching(matching, thold, locu, four):
    keys = [key for key in matching.keys() if (matching[key] > thold)]
    keys_um = list(set(matching.keys()) - set(keys))
    output_file = "matches_test_ad.csv"
    output = open(output_file, "w")
    output.write("locu_id,foursquare_id\n")
    for k in keys:
        (l, f) = k
        output.write("%s,%s\n" % k)

    locu_ids = [k[0] for k in keys_um]
    four_ids = [k[1] for k in keys_um]

    for l in locu_ids:
        for f in four_ids:
            if force_match(locu[l], four[f]):
                k = (l, f)
                output.write("%s,%s\n" % k)


# Features
def equal_phone_numbers(phone_l, phone_f):
    l = get_numeric(phone_l["phone"])
    f = get_numeric(phone_f["phone"])

    if not l or not f:
        return 0

    if l == f:
        return 1

    return -1


# Refer to https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
def get_distance(l, f):
    lat_l = l["latitude"]
    long_l = l["longitude"]
    lat_f = f["latitude"]
    long_f = f["longitude"]

    if lat_l is None or long_l is None or lat_f is None or long_f is None:
        return 6
    x_l = math.cos(lat_l) * math.cos(long_l)
    y_l = math.cos(lat_l) * math.sin(long_l)
    z_l = math.sin(lat_l)
    x_f = math.cos(lat_f) * math.cos(long_f)
    y_f = math.cos(lat_f) * math.sin(long_f)
    z_f = math.sin(lat_f)

    distance = (math.fabs(x_l - x_f) + math.fabs(y_l - y_f) + math.fabs(z_l - z_f))
    return distance


def equality_check(x, y):
    str_x = remove_non_ascii(x)
    str_y = remove_non_ascii(y)

    if not str_x or not str_y:
        return 0

    if str_x == str_y:
        return 1

    return -1


def existence_equality(x, y):
    str_x = remove_non_ascii(x)
    str_y = remove_non_ascii(y)
    if not str_x:
        if not str_y:
            return 1
        else:
            return 0
    else:
        if not str_y:
            return 0
        else:
            return 1


def name_containment(x, y):
    name_x = x["name"]
    name_y = y["name"]
    words_x = set(name_x.split(" "))
    words_y = set(name_y.split(" "))
    if words_x < words_y or words_x > words_y or words_x == words_y:
        return True
    return False


def edit_distance(x, y):
    l_str = get_alphanum(x)
    f_str = get_alphanum(y)
    return editdistance.eval(l_str, f_str)


def equal_websites(x, y):
    l = tldextract.extract(x)
    f = tldextract.extract(y)

    if not l.domain or not f.domain:
        return 0

    if l.domain == f.domain:
        return 1

    return 0

feature_names = [
    "existence_equality_phone",
    "existence_equality_postal_code",
    "existence_equality_street_address",
    "distance",
    "equal_phone_numbers",
    "equality_check_postal_code",
    "edit_distance_name",
    "edit_distance_street_address",
    "equal_websites",
    "name_containment"
]


def featurize(x, y):
    # print "Featurizing " + json.dumps(x) + " " + json.dumps(y)
    return [
        existence_equality(x["phone"], y["phone"]),
        existence_equality(x["postal_code"], y["postal_code"]),
        existence_equality(x["street_address"], y["street_address"]),
        get_distance(x, y),
        equal_phone_numbers(x, y),
        equality_check(x["postal_code"], y["postal_code"]),
        edit_distance(x["name"], y["name"]),
        edit_distance(x["street_address"], y["street_address"]),
        equal_websites(x["website"], y["website"]),
        name_containment(x, y)
    ]


def force_match(x, y):
    ph_l = get_numeric(x["phone"])
    ph_f = get_numeric(y["phone"])

    add_l = get_alphanum(x["street_address"])
    add_f = get_alphanum(y["street_address"])

    if ph_l and ph_f:
        if ph_l == ph_f:
            return True
        else:
            return False

    if add_l and add_f:
        if add_l == add_f:
            return True
        else:
            return False

    if name_containment(x, y):
        post_l = get_alphanum(x["postal_code"])
        post_f = get_alphanum(y["postal_code"])
        if post_l and post_f:
            if post_l != post_f:
                return False
            else:
                return True

        country_l = get_alphabetic(x["country"])
        country_f = get_alphabetic(y["country"])
        if country_l and country_f:
            if country_l != country_f:
                return False
            else:
                return True
        return True

    return False


def get_pairs_of_ids(locu, four):
    id_pairs = [(key_l, key_f) for (key_l, value_l) in locu.iteritems() for (key_f, value_f) in four.iteritems()]
    return id_pairs


def create_filtered_features(locu, four, options):
    X = []
    id_pairs = []
    for key_l in locu.iterkeys():
        filtered_keys = []
        # Find if exact phone match exists
        ph_l = get_numeric(locu[key_l]["phone"])
        exact_match = False
        if ph_l:
            for key_f in four.iterkeys():
                ph_f = get_numeric(four[key_f]["phone"])
                if ph_f:
                    if ph_l == ph_f:
                        exact_match = True
                        filtered_keys.append(key_f)

        if not exact_match:
            lat_l = locu[key_l]["latitude"]
            long_l = locu[key_l]["longitude"]
            if lat_l is None or long_l is None:
                filtered_keys = [key for key, value in four.iteritems()
                                 if value["postal_code"] == locu[key_l]["postal_code"]]
            else:
                dst = {}
                for key_f in four.iterkeys():
                    lat_f = four[key_f]["latitude"]
                    long_f = four[key_f]["longitude"]
                    if lat_f is None or long_f is None:
                        if locu[key_l]["postal_code"] == four[key_f]["postal_code"]:
                            filtered_keys.append(key_f)
                    else:
                        dst[key_f] = get_distance(locu[key_l], four[key_f])
                imp_keys = [key for key, value in sorted(dst.iteritems(), key=lambda (k, v): (v, k))]
                for imp_key in imp_keys[:options]:
                    if imp_key not in filtered_keys:
                        filtered_keys.append(imp_key)

        for k in filtered_keys:
            features = featurize(locu[key_l], four[k])
            X.append(features)
            id_pairs.append((key_l, k))
    return X, id_pairs


def create_class(pairs, matches):
    y = [((key_l, key_f) in matches) for (key_l, key_f) in pairs]
    return y


# Compute f1_score and point out the unmatchable pairs
def f1_score(prediction, truth, thold):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for key in truth.iterkeys():
        if key in prediction and prediction[key] > thold:
            tp += 1
        else:
            tn += 1
            if thold == 0:
                print "ALGO CAN NOT MATCH " + remove_non_ascii(key) # these go to the blacklist after manual inspection

    for key in prediction.iterkeys():
        if prediction[key] > thold:
            if key not in truth:
                fp += 1
        else:
            if key not in truth:
                fn += 1

    if tp == 0:
        prec = 0
        recall = 0
        f1 = 0
    else :
        prec = tp / float(tp + fp)
        recall = tp / float(tp + tn)
        f1 = 2.0 * prec * recall / (prec + recall)
    print "TP: " + remove_non_ascii(tp)  + " TN: " + remove_non_ascii(tn) + " FP: " + remove_non_ascii(fp) + \
          " FN: " + remove_non_ascii(fn) + " PREC: " + remove_non_ascii(prec) + " RECALL: " + remove_non_ascii(recall) + " F1: " + remove_non_ascii(f1)
    return f1


# Create an id to integer and vice-a-versa index map for both locu and foursquare entries
def build_indices(pairs_of_ids):
    locu_id_to_index = {}
    four_id_to_index = {}
    locu_index_to_id = []
    four_index_to_id = []
    for locu_id, four_id in pairs_of_ids:
        if locu_id not in locu_id_to_index:
            locu_id_to_index[locu_id] = len(locu_id_to_index)
            locu_index_to_id.append(locu_id)
        if four_id not in four_id_to_index:
            four_id_to_index[four_id] = len(four_id_to_index)
            four_index_to_id.append(four_id)
    return locu_id_to_index, locu_index_to_id, four_id_to_index, four_index_to_id


# given a set of matching probabilities, get the maximum probability sum bipartite matching
def fit_best_matching(prob, pairs_of_ids, filtered_pairs_of_ids, X):
    locu_id_to_index, locu_index_to_id, four_id_to_index, four_index_to_id = build_indices(pairs_of_ids)
    matching_weights = np.zeros((len(locu_id_to_index), len(four_id_to_index)))
    for i in range(len(filtered_pairs_of_ids)):
        locu_id, four_id = filtered_pairs_of_ids[i]
        matching_weights[locu_id_to_index[locu_id]][four_id_to_index[four_id]] = prob[i][1]

    max_matching = hungarian.lap(-matching_weights)[0]

    #store all matches
    best_matching = {}

    for i in range(len(max_matching)):
        locu_id = locu_index_to_id[i]
        four_id = four_index_to_id[max_matching[i]]
        w = matching_weights[i][max_matching[i]]
        best_matching[(locu_id, four_id)] = w

    return best_matching


# Make both locu and four of equal length for the bipartite matching in hungarian method to work
def fix_lengths(locu, four):
    len_x = len(locu)
    len_y = len(four)
    #print "len_x " + remove_non_ascii(len_x) + " len_y " + remove_non_ascii(len_y)

    if len_x == len_y:
        print "Both datasets have the same number of entries. Nothing to be done here!"
    else:
        print "The datasets have different number of entries. Introducing dummy entries..."
        if len_x < len_y:
            while len_x < len_y:
                rand_id = ''.join(random.choice(string.ascii_lowercase
                                                + string.digits) for _ in range(24))
                # Picking an unused id
                while rand_id in locu.keys():
                    rand_id = ''.join(random.choice(string.ascii_lowercase
                                                    + string.digits) for _ in range(24))
                dummy = dict()
                dummy["country"] = "N/A"
                dummy["id"] = rand_id
                dummy["latitude"] = 0
                dummy["locality"] = "N/A"
                dummy["longitude"] = 0
                dummy["name"] = "N/A"
                dummy["phone"] = "N/A"
                dummy["postal_code"] = "N/A"
                dummy["region"] = "N/A"
                dummy["street_address"] = "N/A"
                dummy["website"] = "N/A"
                locu[rand_id] = dummy
                len_x = len(locu)
        else:
            while len_y < len_x:
                rand_id = ''.join(random.choice(string.ascii_lowercase
                                                + string.digits) for _ in range(24))
                # Picking an unused id
                while rand_id in four.keys():
                    rand_id = ''.join(random.choice(string.ascii_lowercase
                                                    + string.digits) for _ in range(24))
                dummy = dict()
                dummy["country"] = "N/A"
                dummy["id"] = rand_id
                dummy["latitude"] = 0
                dummy["locality"] = "N/A"
                dummy["longitude"] = 0
                dummy["name"] = "N/A"
                dummy["phone"] = "N/A"
                dummy["postal_code"] = "N/A"
                dummy["region"] = "N/A"
                dummy["street_address"] = "N/A"
                dummy["website"] = "N/A"
                dummy["id"] = rand_id
                four[rand_id] = dummy
                len_y = len(four)
    """
    print "Four Dummies:"
    for k, v in four.iteritems():
        if v["country"] == "N/A":
            print json.dumps(v)
    print "Locu Dummies:"
    for k, v in locu.iteritems():
        if v["country"] == "N/A":
            print json.dumps(v)
    """


def train_model(locu_train_path, foursquare_train_path, matches_train_path, model):
    locu = load_json(locu_train_path, "id")
    four = load_json(foursquare_train_path, "id")

    fix_lengths(locu, four)
    #print "Lengths: " + remove_non_ascii(len(locu)) + " and " + remove_non_ascii(len(four))
    print "Done"
    print marker
    # Read the class for each pair. Matched = 1, Unmatched = 0
    matches = load_csv(matches_train_path)
    #print matches
    for bl in blacklist:
        if bl in matches:
            print "Deleting Blacklisted Training Matching " +  remove_non_ascii(bl)
            del matches[bl]
    print "Done"
    print marker

    id_pairs = get_pairs_of_ids(locu, four)

    # Set up the feature matrix for Classifier
    print "Creating Features for training data..."
    (X, filtered_id_pairs) = create_filtered_features(locu, four, 20)
    y = create_class(filtered_id_pairs, matches)

    # Fit a model
    print "Fitting the classifier on the training data..."
    classifier = model
    classifier.fit(X, np.array(y))

    # Fit a Matching
    print "Fitting a Matching on the training data..."
    p = classifier.predict_proba(X)
    best_matching = fit_best_matching(p, id_pairs, filtered_id_pairs, X)
    print "Done"
    print marker

    # Now we have to pick a probability threshold, which would help us control the F1 score of our matching.
    score_map = {}
    for t in np.linspace(0, 0.9999, 100):
        print "Computing F1 score for thold = " + remove_non_ascii(t)
        f1 = f1_score(best_matching, matches, t)
        score_map[t] = f1

    max_f1 = max(score_map.values())
    max_f1_tholds = [i for i in score_map.keys() if score_map[i] == max_f1]

    # Take the mean threshold of the best case scenarios
    # print max_f1_tholds
    best_thold = np.mean(max_f1_tholds)
    print "Selected Threshold = " + remove_non_ascii(best_thold) + " with Best F1-score = " + remove_non_ascii(max_f1)
    print "Done"
    print marker
    return best_thold, max_f1


def run_model(locu_test_path, foursquare_test_path, model):
    locu_test = load_json(locu_test_path, "id")
    four_test = load_json(foursquare_test_path, "id")

    fix_lengths(locu_test, four_test)
    print marker

    id_pairs_test = get_pairs_of_ids(locu_test, four_test)
    # Set up the feature matrix for Classifier
    print "Creating Features for test data..."
    (X_test, filtered_id_pairs_test) = create_filtered_features(locu_test, four_test, 20)
    classifier = model
    p_test = classifier.predict_proba(X_test)

    # Fit a Matching
    print "Fitting a Matching on the test data..."
    best_matching_test = fit_best_matching(p_test, id_pairs_test, filtered_id_pairs_test, X_test)

    return locu_test, four_test, best_matching_test


def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    """
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.

    """
    # Pick a classification model
    model = RandomForestClassifier(n_estimators=64, random_state=4121)


    # Train the model on training data and chose an appropriate theshold for matching
    print marker
    (thold, f1) = train_model(locu_train_path, foursquare_train_path, matches_train_path, model)

    # Importance of features
    feature_imp = model.feature_importances_
    order = sorted(range(len(feature_names)), key=lambda k: feature_imp[k], reverse=True)
    for i in order:
        print "Feature: " + feature_names[i] + " Importance: " + remove_non_ascii(feature_imp[i])
    print marker

    # Test on the test data
    locu_test, four_test, best_matching_test = run_model(locu_test_path, foursquare_test_path, model)
    # Write the test matches to "matches_test.csv"
    print "Writing matches to \"matches_test.csv\" in the same directory..."
    dump_matching(best_matching_test, thold, locu_test, four_test)
    print "Done"
    print marker


def run_old():
    np.random.seed(4121)
    parser = argparse.ArgumentParser()
    parser.add_argument("--locu_train_path", help="Locu Train path")
    parser.add_argument("--foursquare_train_path", help="Foursquare Train path")
    parser.add_argument("--matches_train_path", help="Matches Train path")
    parser.add_argument("--locu_test_path", help="Locu Test path")
    parser.add_argument("--foursquare_test_path", help="Foursquare Test path")

    args = parser.parse_args()
    get_matches(args.locu_train_path, args.foursquare_train_path, args.matches_train_path, args.locu_test_path,
                args.foursquare_test_path)

def run():
    locu_train_path = path_train + "locu_train.json"
    foursquare_train_path = path_train + "foursquare_train.json"
    matches_train_path = path_train + "matches_train.csv"
    locu_test_path = path_test + "locu_test.json"
    foursquare_test_path = path_test + "foursquare_test.json"
    get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path)


if __name__ == '__main__':
    run()