import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import os

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('../data/all/train/train.csv')

train_sen = os.listdir('../data/all/train_sentiment/')
train_metadata = os.listdir('../data/all/train_metadata')

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


train_data['sent_score'] = get_sentiment(train_data, train_sen, 'train', 'score')
train_data['sent_magnitude'] = get_sentiment(train_data, train_sen, 'train', 'magnitude')

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

# train_data["Age"].plot(kind='hist')
# plt.show()
#

train_data['Type'] = train_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')

# sns.barplot(x='AdoptionSpeed', y='Age', hue='Type', data=train_data)
# plt.show()


train_data['IsMixed'] = train_data['Breed2'].apply(lambda x: 0 if x == 0 else 1)
print(train_data['IsMixed'].unique())

#train_data['CatIsMixed'] = train_data

sns.barplot(x='AdoptionSpeed', y='IsMixed', data=train_data)
plt.show()