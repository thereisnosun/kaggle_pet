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

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('../data/all/train/train.csv')

train_sen = os.listdir('../data/all/train_sentiment/')
train_metadata = os.listdir('../data/all/train_metadata')


train_data = train_data.drop('Description', 1)
train_data = train_data.drop('RescuerID', 1)
print('Data HEAD: ')
print(train_data.head())
print('-----------------------------------------------------')

print('SHAPE: ')
print(train_data.shape)

print('LENGTH:')
print(len(train_data))


def get_sentiment(df, sen_source, test_train, key):
    sen = []
    for i in df['PetID']:
        a = i + '.json'
        if a in sen_source:
            x = '../data/all/%s_sentiment/%s' % (test_train, a)
            with open(x, 'r') as f:
                sentiment = json.load(f)

            y = sentiment['documentSentiment'][key]
        else:
            y = 0

        sen.append(y)
    return sen


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


train_data['Type'] = train_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')



#make_count_plot(train_data, x='Type', tittle='by pet Type')
#make_count_plot(train_data, x='Age', tittle='by pet Age')

# sns.barplot(x='AdoptionSpeed', y='IsMixed', data=train_data)
# plt.show()


data_explorers = [AgeExplorer(train_data), BreedExplorer(train_data)]
for explorer in data_explorers:
    explorer.basic_check()
    train_data_features = explorer.get_additional_features()
    explorer.plot_data()


