import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from wordcloud import WordCloud

import json
import os

from src.data_exploration.age_exploration import AgeExplorer
from src.data_exploration.breed_exploration import BreedExplorer
from src.data_exploration.color_exploration import ColorExplorer


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('../input/train/train.csv')
test_data = pd.read_csv('../input/test/test.csv')

all_data = pd.concat([train_data, test_data])

train_sen = os.listdir('../input/train_sentiment/')
test_sen = os.listdir('../input/test_sentiment/')

train_metadata = os.listdir('../input/train_metadata')
test_metadata = os.listdir('../input/test_metadata')

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


def check_name(name):
    if len(name) <= 2:
        return 1

    return 1 if any(char.isdigit() for char in name) else 0


def feature_engineer(data_features):
    data_features['HasName'] = data_features['Name'].apply(lambda x: 1 if x else 0)
    data_features['BadName'] = data_features['Name'].apply(lambda name: check_name(str(name))) # for some reason does not help

    data_features['IsFree'] = data_features['Fee'].apply(lambda price: 1 if price > 0 else 0)
    data_features = data_features.drop('Fee', 1)

    data_features['DescriptionLen'] = data_features['Description'].apply(lambda descr: len(str(descr)))
    data_features['DescriptionWords'] = data_features['Description'].apply(lambda descr: str(descr).count(' '))
    #TODO: classify manitude and score, to positive and negative
    data_features['WordsLen'] = data_features['DescriptionLen'] / data_features['DescriptionWords']
    data_features['WordsLen'] = data_features['WordsLen'].fillna(0)
    data_features['WordsLen'] = data_features['WordsLen'].apply(lambda x: 0 if x == float("inf") else 0)
    # calc description words, calc description length


    data_features = data_features.drop('Description', 1)
    data_features = data_features.drop('RescuerID', 1)




    return data_features


def plot_text(name):
    plt.subplot(1, 2, 1)
    text_cat = ' '.join(all_data.loc[all_data['Type'] == name, 'Name'].fillna('').values)
    wordcloud = WordCloud(max_font_size=None, background_color='white',
                          width=1200, height=1000).generate(text_cat)
    plt.imshow(wordcloud)
    plt.title('Top {0} names'.format(name))
    plt.axis("off")



def plot_data(data_features):
    #text
    #fig, ax = plt.subplot(figsize=(16, 12))
    #plot_text('1')
    pass


data_explorers = [AgeExplorer(), BreedExplorer(), ColorExplorer()]
train_data_features = train_data
for explorer in data_explorers:
    explorer.set_data_frame(train_data_features)
    #explorer.basic_check()
    train_data_features = explorer.get_additional_features()
    explorer.plot_data()

print("Does not have name - ", train_data['Name'].isna().sum())

plot_data(train_data_features)
train_data_features = feature_engineer(train_data_features)
#print('Bad named pets - ', train_data_features.where(train_data_features['BadName'] == 1).count())

#exit(0)

test_data_features = test_data
for explorer in data_explorers:
    explorer.set_data_frame(test_data_features)
    test_data_features = explorer.get_additional_features()

test_data_features = feature_engineer(test_data_features)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


train_data_features, train_pet_ids, train_predictions = final_preperations(train_data_features, train_sen, True)
print(train_data_features.head())

n_folds = 5
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=39)
total_kappa = 0
for n_folds, (train_index, valid_index) in enumerate(folds.split(train_data_features, train_predictions)):

    features_train, features_valid = train_data_features.iloc[train_index], train_data_features.iloc[valid_index]
    predictions_train, predictions_valid = train_predictions[train_index], train_predictions[valid_index]

    final_model = RandomForestClassifier(n_estimators=100, random_state=39)
    final_model.fit(features_train, predictions_train)

    predict_res_train = final_model.predict(features_train)
    predict_res_valid = final_model.predict(features_valid)
    kappa_score = cohen_kappa_score(predictions_valid, predict_res_valid, weights='quadratic')
    total_kappa += kappa_score

    print('{0} number cappa score is {1}'.format(n_folds, kappa_score))

print('Total kappa is ',  (total_kappa / 5) )





#this one is better
# final_model = AdaBoostClassifier(RandomForestClassifier(random_state=39, n_estimators=1000),
#                                  n_estimators=200,
#                                  algorithm="SAMME.R", learning_rate=0.5)
train_data_features = train_data_features.reset_index().values
final_model.fit(train_data_features, train_predictions)
val_score = cross_val_score(final_model, train_data_features, train_predictions, cv=3, scoring='accuracy', n_jobs=-1).mean()
train_predictions_result = final_model.predict(train_data_features)

kappa_score = cohen_kappa_score(train_predictions, train_predictions_result, weights='quadratic')
print(kappa_score)
print('Cross val score {0}. Cohen kappa - {1}'.format(val_score, kappa_score))

# just to test output
print("Before test data is - ", len(test_data_features))
test_data_features, test_pet_ids, _ = final_preperations(test_data_features, test_sen, False)
print("test data is - ", len(test_data_features))
#print("Train len {0} test len {1}".format(len(train_data_features.columns, len(test_data_features.columns))))
test_data_features = test_data_features.reset_index().values
predictions = final_model.predict(test_data_features)
create_submission(test_pet_ids, predictions)



