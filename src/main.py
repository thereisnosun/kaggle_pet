import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go


import json
import os

from src.data_exploration.age_exploration import AgeExplorer
from src.data_exploration.breed_exploration import BreedExplorer
from src.data_exploration.color_exploration import ColorExplorer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('../input/train/train.csv')
test_data = pd.read_csv('../input/test/test.csv')

train_sen = os.listdir('../input/train_sentiment/')
test_sen = os.listdir('../input/test_sentiment/')

train_metadata = os.listdir('../input/train_metadata')
test_metadata = os.listdir('../input/test_metadata')


train_data = train_data.drop('Description', 1)
train_data = train_data.drop('RescuerID', 1)

test_data = test_data.drop('Description', 1)
test_data = test_data.drop('RescuerID', 1)

print('Data HEAD: ')
print(train_data.head())
print('-----------------------------------------------------')

print('SHAPE: ')
print(train_data.shape)

print('LENGTH:')
print(len(train_data))

print("TEST START- ", len(test_data))


def get_sentiment(df, sen_source, test_train, key):
    sen = []
    for i in df['PetID']:
        a = i + '.json'
        if a in sen_source:
            x = '../input/%s_sentiment/%s' % (test_train, a)
            with open(x, 'r') as f:
                sentiment = json.load(f)

            y = sentiment['documentSentiment'][key]
        else:
            y = 0

        sen.append(y)
    return sen


def create_submission(ids, prediction):
    df = pd.DataFrame({'PetID': ids,'AdoptionSpeed': prediction }, columns=['PetID', 'AdoptionSpeed']) #TODO: sort columns correctly
    df.to_csv('../submission.csv',index=False)


# TODO: uncommented as soon as basic features are analyzed
# train_data['sent_score'] = get_sentiment(train_data, train_sen, 'train', 'score')
# train_data['sent_magnitude'] = get_sentiment(train_data, train_sen, 'train', 'magnitude')


# train_data['sent_score'].plot.hist()
# plt.show()
#
# train_data['sent_magnitude'].plot.hist()
# plt.show()

# train_data['PhotoAmt'].value_counts().plot.bar()
# plt.show()
#print(train_data.corr())
#
# train_data["AdoptionSpeed"].value_counts().sort_index().plot('barh')
# plt.show();
#
# train_data["Type"].value_counts().sort_index().plot('barh')
# plt.show();


#train_data['Type'] = train_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')


def assign_gender(type):
    if type == 1:
        return 'Male'
    elif  type == 2:
        return 'Female'
    else:
        return 'Mixed'


# print( train_data['Gender'].value_counts())
# train_data['Gender'] = train_data['Gender'].apply(assign_gender)
# print( train_data['Gender'].value_counts())

#make_count_plot(train_data, x='Type', tittle='by pet Type')
#make_count_plot(train_data, x='Age', tittle='by pet Age')

# sns.barplot(x='AdoptionSpeed', y='IsMixed', data=train_data)
# plt.show()


def final_preperations(data_features, sent, is_train):
    data_features['sent_score'] = get_sentiment(data_features, sent, 'train' if is_train else 'test', 'score')
    data_features['sent_magnitude'] = get_sentiment(data_features, sent, 'train' if is_train else 'test', 'magnitude')

    if is_train:
        train_predictions = data_features['AdoptionSpeed'].values
    else:
        train_predictions = None
    pet_ids = data_features['PetID']
    if is_train:
        data_features = data_features.drop('AdoptionSpeed', 1)
    data_features = data_features.drop('Name', 1) # for the very first test lets drop it
    data_features = data_features.drop('PetID', 1) # TODO: identify later by it

    return data_features, pet_ids, train_predictions


data_explorers = [AgeExplorer(), BreedExplorer(), ColorExplorer()]
train_data_features = train_data
for explorer in data_explorers:
    explorer.set_data_frame(train_data_features)
    explorer.basic_check()
    train_data_features = explorer.get_additional_features()
    explorer.plot_data()


#exit(0)

test_data_features = test_data
for explorer in data_explorers:
    explorer.set_data_frame(test_data_features)
    test_data_features = explorer.get_additional_features()




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import cohen_kappa_score


train_data_features, train_pet_ids, train_predictions = final_preperations(train_data_features, train_sen, True)
print(train_data_features.head())
random_forest = RandomForestClassifier(n_estimators=100, random_state=39)
#this one is better
# random_forest = AdaBoostClassifier(RandomForestClassifier(random_state=39, n_estimators=1000),
#                                  n_estimators=200,
#                                  algorithm="SAMME.R", learning_rate=0.5)
train_data_features = train_data_features.reset_index().values
random_forest.fit(train_data_features, train_predictions)
val_score = cross_val_score(random_forest, train_data_features, train_predictions, cv=3, scoring='accuracy', n_jobs=-1).mean()
train_predictions_result = random_forest.predict(train_data_features)
kappa_score = cohen_kappa_score(train_predictions, train_predictions_result).mean()
print('Cross val score {0}. Cohen kappa - {1}'.format(val_score, kappa_score))

# just to test output
print("Before test data is - ", len(test_data_features))
test_data_features, test_pet_ids, _ = final_preperations(test_data_features, test_sen, False)
print("test data is - ", len(test_data_features))
#print("Train len {0} test len {1}".format(len(train_data_features.columns, len(test_data_features.columns))))
test_data_features = test_data_features.reset_index().values
predictions = random_forest.predict(test_data_features)
create_submission(test_pet_ids, predictions)

