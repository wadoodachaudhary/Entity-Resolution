import json
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import nltk
from fuzzywuzzy import fuzz
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
feature_imp=[]
"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Name : Wadood Chaudhary
Uni  : wc2595

Member 2: name, uni

Member 3: name, uni
"""

#path = "/Users/wadood/Library/Mobile Documents/com~apple~CloudDocs/Projects/DataSystems_COMS4121/hw3/train_modified/"
path_train = "/Users/wadood/Library/Mobile Documents/com~apple~CloudDocs/Projects/DataSystems_COMS4121/hw3/train/"
path_test = "/Users/wadood/Library/Mobile Documents/com~apple~CloudDocs/Projects/DataSystems_COMS4121/hw3/online_competition/"


def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    """
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.

    """
    df_train, locu, four_sq = build_features(locu_train_path, foursquare_train_path)
    y = match(matches_train_path,df_train)
    features = [
        'name_ratio','website_ratio', 'longitude_ratio', 'phone_ratio',
        'postal_code_ratio', 'latitude_ratio', 'street_address_ratio', 'score'
    ]
    clf = RandomForestClassifier()
    clf.fit(df_train[features], y)
    # print important features
    print (clf.feature_importances_)
    #order_features = sorted(range(len(features)), key=lambda f: important_features[f], reverse=True)
    #for f in order_features:
    #    print (features[f] + " Importance: " + important_features[f])

    df_test, locu_test, four_sq_test = build_features(locu_test_path, foursquare_test_path)
    # p_test = clf.predict_proba(df_train[features])
    match_test = clf.predict(df_test[features])

    # print_result(df_test,p_test)
    save_result(df_test, match_test)
    # print(p_test)
    # print(pd.crosstab(clf.predict(df_train[features]), y))
    # clf.predict(df_test[features])
    # print(d)
    pass

def read_jason(fileName):
    with file(fileName, 'r') as f:
        data = f.read()
    return json.loads(data)

def print_unique(dic,col):
    s = set()
    for x in dic:
        s.add(x[col])
    print(s)

def normalize_phone(dic):
    for x in dic:
        phone = str(x["phone"])
        phone=phone.replace("(","")
        phone=phone.replace(")", "")
        phone=phone.replace(" ", "")
        phone=phone.replace("-","")
        x["phone"]=phone

def normalize_longlat(dic):
    for x in dic:
        lon = str(x["longitude"])
        lat = str(x["latitude"])
        lon= lon[0:10]
        lat = lat[0:10]
        x["longitude"]=lon
        x["latitude"] =lat

#blacklist="(c170270283ef870d546b,51eb7eed498e401ec51196b6),(825acefd3e298274a150-4f9ab1dbd4f2465542bc673f)212dffb393f745df801a-51e869ac498e7e485cabcdeb),(e3f9d84c0c989f2e7928-51e25e57498e535de72f03e7)"
blacklist="(c170270283ef870d546b,51eb7eed498e401ec51196b6),(825acefd3e298274a150-4f9ab1dbd4f2465542bc673f),(212dffb393f745df801a-51e869ac498e7e485cabcdeb),(e3f9d84c0c989f2e7928-51e25e57498e535de72f03e7),(edeba23f215dcc702220-51a11cbc498e4083823909f1),(66ef54d76ff989a91d52-51c9e1dd498e33ecd8670892)(5f3fd107090d0ddc658b-51ce011a498ed8dfb15381bb),(80afa95c01dae3ba5434-506fc5f6e4b0184b5ae01a83),(25485cb3241580f995ff-4dee9f25fa761efc37a0d1a4)(493f5e2798de851ec3b2-51f119e7498e9716f71f4413)"
def build_features(locu_file,four_sq_file):
    locu    = read_jason(locu_file)
    four_sq = read_jason(four_sq_file)
    COLUMN_NAMES = ['locu_name','four_sq_name','name_ratio', 'country_ratio', 'website_ratio', 'locality_ratio', 'region_ratio', 'longitude_ratio',
                    'phone_ratio', 'postal_code_ratio', 'latitude_ratio', 'street_address_ratio', "locu_id",
                    "four_sq_id", "score","match"]

    COLUMN_TYPES = [str, str, float, float, float, float,
                    float, float,
                    float, float, float, float,str,
                    str, float, bool]

    df = pd.DataFrame(columns=COLUMN_NAMES)
    df = df.astype(dtype={'locu_name':'str','four_sq_name':'str','name_ratio':'float', 'country_ratio':'float', 'website_ratio':'float',
                          'locality_ratio':'float', 'region_ratio':'float', 'longitude_ratio':'float',
                    'phone_ratio':'float', 'postal_code_ratio':'float', 'latitude_ratio':'float', 'street_address_ratio':'float',
                          "locu_id":'str',"four_sq_id":"str", "score":"float","match":np.int})

    #df['match'] = df['match'].astype(bool)
    normalize_phone(locu)
    normalize_phone(four_sq)
    normalize_longlat(locu)
    normalize_longlat(four_sq)
    w = 0
    r = 1
    for x in locu:

        # print(list(dfid.index),dfid.loc[4, "foursquare_id"])
        score=0
        for y in four_sq:
            if (x["id"]+"-"+y["id"]) in blacklist:
                continue
            # print(w,x['name'], y['name'],x['longitude'], y['longitude'])
            name_ratio = fuzz.ratio(x['name'], y['name']) * 1.5
            website_ratio = fuzz.ratio(x['website'], y['website']) * 1.5
            street_address_ratio = fuzz.ratio(x['street_address'], y['street_address']) * 1.5
            if (x['longitude'] == y['longitude']):
                longitude_ratio = 100
            else:
                longitude_ratio = 0
            if (x['latitude'] == y['latitude']):
                latitude_ratio = 100
            else:
                latitude_ratio = 0
            if (x['phone'] == y['phone']):
                phone_ratio = 100
            else:
                phone_ratio = 0
            if (x['postal_code'] == y['postal_code']):
                postal_code_ratio = 50
            else:
                postal_code_ratio = 0
            score = name_ratio+website_ratio+street_address_ratio+longitude_ratio+latitude_ratio+phone_ratio+postal_code_ratio

            if (score>200):
                # print(x["id"],y["id"],ratio)
                w = w + 1
                df.loc[w, 'locu_id'] = x["id"]
                df.loc[w, 'four_sq_id'] = y["id"]
                df.loc[w, 'locu_name'] = x["name"]
                df.loc[w, 'four_sq_name'] = y["name"]
                df.loc[w, 'name_ratio'] = name_ratio
                df.loc[w, 'website_ratio'] = website_ratio
                df.loc[w, 'longitude_ratio'] = longitude_ratio
                df.loc[w, 'latitude_ratio'] = latitude_ratio
                df.loc[w, 'postal_code_ratio'] = postal_code_ratio
                df.loc[w, 'street_address_ratio'] = street_address_ratio
                df.loc[w, 'phone_ratio'] = phone_ratio
                df.loc[w, 'score'] = score
                #df.loc[w, 'match'] = 0

        r = r + 1
        #if (r> 5): break
    #print(df)
    return df,locu,four_sq

def match(match_path,df_train):
    dfmatch = pd.read_csv(match_path)
    y=[]
    for index, row in df_train.iterrows():
        locu_id = row['locu_id']
        four_sq_id = row['four_sq_id']
        dfid = dfmatch.loc[dfmatch['locu_id'] == locu_id]
        # print(dfid.shape[0])
        if (dfid.shape[0] > 0):
           index = list(dfid.index)[0]
           four_sq_matched_id = dfid.loc[index,"foursquare_id"]
        else:
            four_sq_matched_id = -1
        if (four_sq_id==four_sq_matched_id):
            #print("Matched:",locu_id,four_sq_id,four_sq_matched_id,row['score'])
            row['match']=1.0
            y.append(True)
        else:
            row['match'] = 0.0
            y.append(False)

    matches=0
    for index, row in df_train.iterrows():
        #print("Match",row['match'])
        if y[index-1]:
            matches = matches+1
            #print(matches,index,row['match'], row['locu_name'],row['four_sq_name'], row['score'])
    #print (df_train.dtypes)
    return y

def print_result(df,match):
    #print(match)
    ctr =0
    for index in range(0, len(match)):
        if match[index]:
            ctr = ctr+1
            print("Matched:",ctr,index,df.loc[index+1,"locu_name"],df.loc[index+1,"four_sq_name"],df.loc[index+1,"score"])

def save_result(df,match):
    #print(match)
    output_file = "matches_test.csv"
    output = open(output_file, "w")
    output.write("locu_id,foursquare_id\n")
    for index in range(0, len(match)):
        if match[index]:
            k = (df.loc[index+1,"locu_id"],df.loc[index+1,"four_sq_id"])
            output.write("%s,%s\n" % k)
            #print("Matched:",index,df.loc[index+1,"locu_id"],df.loc[index+1,"four_sq_id"])


def main():
    df_train,locu,four_sq = build_features(path_train+"locu_train.json",path_train+"foursquare_train.json")
    y=match(path_train+"matches_train.csv",df_train)
    #df_test = build_features(path_test, "locu_test.json", "foursquare_test.json")
    features = [
            'name_ratio',
            'website_ratio','longitude_ratio','phone_ratio',
            'postal_code_ratio','latitude_ratio','street_address_ratio','score'
    ]
    clf = RandomForestClassifier()
    #print(df_train[match])
    clf.fit(df_train[features], y)
    feature_imp = clf.feature_importances_
    print(range(len(features)))
    order = sorted(range(len(features)), key=lambda k: feature_imp[k], reverse=True)

    #order = sorted(range(len(features)), key=f, reverse=True)
    for i in order:
        print(i,features[i],feature_imp[i])
        #print ("Feature: " + features[i] + " Importance: " + feature_imp[i])


    df_test, locu_test, four_sq_test = build_features(path_test+"locu_test.json", path_test+"foursquare_test.json")
    #p_test = clf.predict_proba(df_train[features])
    p_test = clf.predict(df_test[features])

    #print_result(df_test,p_test)
    save_result(df_test,p_test)
    #print(p_test)
    #print(pd.crosstab(clf.predict(df_train[features]), y))
    #clf.predict(df_test[features])
    #print(d)



def match_files():
    dfw = pd.read_csv("matches_test.csv")
    dfa = pd.read_csv("matches_test_ad.csv")
    y = []
    matched=0
    missed=0

    for index, row in dfw.iterrows():
        locu_id = row['locu_id']
        four_sq_id = row['foursquare_id']
        dfid = dfa.loc[dfa['locu_id'] == locu_id]
        if (dfid.shape[0] > 0):
            index = list(dfid.index)[0]
            four_sq_matched_id = dfid.loc[index, "foursquare_id"]
        else:
            four_sq_matched_id = -1
        if (four_sq_id == four_sq_matched_id):
            print("Matched:",locu_id,four_sq_id,four_sq_matched_id)
            matched=matched+1
            y.append(True)
        else:
            print("UnMatched:", locu_id, four_sq_id, four_sq_matched_id)
            missed=missed+1
            y.append(False)
    print(matched,missed)

if __name__ == '__main__':
    #main()
    locu_train_path=path_train+"locu_train.json"
    foursquare_train_path=path_train+"foursquare_train.json"
    matches_train_path=path_train+"matches_train.csv"
    locu_test_path=path_test+"locu_test.json"
    foursquare_test_path=path_test+"foursquare_test.json"
    #match_files()
    get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path)
    #main()

