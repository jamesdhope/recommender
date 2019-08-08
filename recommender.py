from lightfm import LightFM
from scipy.sparse import coo_matrix as sp
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import csv
import requests
import json
from itertools import islice
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split 

#################################
#								#
#  Fetching the training data 	#
#								#
#################################

def _download(url: str, dest_path: str):

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)

def get_data():

    ratings_url = ("http://www2.informatik.uni-freiburg.de/" "~cziegler/BX/BX-CSV-Dump.zip")

    if not os.path.exists("data"):
        os.makedirs("data")

        _download(ratings_url, "data/data.zip")

    with zipfile.ZipFile("data/data.zip") as archive:
        return (
            csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Book-Ratings.csv")),
                delimiter=";",
            ),
            csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Books.csv")), delimiter=";"
            ),
            csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Users.csv")), delimiter=";"
            ),
        )

def get_ratings():
    return get_data()[0]

def get_book_features():
    return get_data()[1]

def get_user_features():
    return get_data()[2]

#################################
#                               #
#       Building the Model      #
#                               #
#################################

dataset = Dataset()
dataset.fit((x['User-ID'] for x in get_ratings()),
            (x['ISBN'] for x in get_ratings()))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(users=(x['User-ID'] for x in get_user_features()),
                    items=(x['ISBN'] for x in get_book_features()),
                    item_features=(x['Book-Author'] for x in get_book_features()),
                    user_features=(x['Age'] for x in get_user_features()))

(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN'])
                                                      for x in get_ratings()))

#print(repr(interactions))

item_features = dataset.build_item_features(((x['ISBN'], [x['Book-Author']])
                                              for x in get_book_features()))
#print(repr(item_features))


user_features = dataset.build_user_features(((x['User-ID'], [x['Age']])
                                              for x in get_user_features()))


labels = np.array([x['ISBN'] for x in get_ratings()])

#################################
#								#
#  		Training the Model 		#
#								#
#################################

model = LightFM(loss='warp')

(train, test) = random_train_test_split(interactions=interactions, test_percentage=0.2)

model.fit(train, item_features=item_features, user_features=user_features, epochs=2)

### model performnce evaluation

#train_precision = precision_at_k(model, train,item_features=item_features, k=10).mean()
#test_precision = precision_at_k(model, test, item_features=item_features,k=10).mean()

#train_auc = auc_score(model, train,item_features=item_features).mean()
#test_auc = auc_score(model, test,item_features=item_features).mean()

#print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
#print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

#print("testing testing testing")
#print('printing labels', get_ratings()['ISBN'])

#########################################
#                                       #
# Computing Recommendations for Group   #
#                                       #
#########################################

labels = np.array([x['ISBN'] for x in get_ratings()])

def sample_recommendation(model, data, user_ids):

    n_users, n_items = data.shape

    #build a structure to store user scores for each item
    all_scores = np.empty(shape=(0,n_items))

    #iterate through the group and build the scores
    for user_id in user_ids:
        #known_positives = labels[data.tocsr()[user_id].indices]

        scores = model.predict(user_id,np.arange(n_items),item_features,user_features)
        
        top_items_for_user = labels[np.argsort(-scores)]
        print("Top Recommended ISBN For User: ", user_id)
        for x in top_items_for_user[:3]:
            print("     %s" % x)

        #vertically stack the user scores (items are columns)
        all_scores = np.vstack((all_scores, scores))
        #print(all_top_items)

    #compute the average rating for each item in the group
    item_averages = np.mean(all_scores.astype(np.float), axis=0)
    top_items_for_group = labels[np.argsort(-item_averages)]

    print("Top Recommended ISBN for Group:")

    for x in top_items_for_group[:5]:
        print("     %s" % x)

#################################
#                               #
#  Sampling Recommended Events  #
#                               #
#################################

#fetch user_ids of users in group
group = [3,26,451,23,24,25]

#sample recommendations for the group
sample_recommendation(model, interactions, group)

#############################################
#                                           #
#  Discounting Events based on Constraints  #
#                                           #
#############################################



