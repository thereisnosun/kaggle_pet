import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import os

from src.data_exploration.age_exploration import AgeExplorer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('../data/all/train/train.csv')

train_sen = os.listdir('../data/all/train_sentiment/')
train_metadata = os.listdir('../data/all/train_metadata')

#
# train_data = train_data.drop('Description', 1)
# train_data = train_data.drop('RescuerID', 1)
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


train_data['Type'] = train_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')


train_data['IsMixed'] = train_data['Breed2'].apply(lambda x: 0 if x == 0 else 1)
print(train_data['IsMixed'].unique())

#train_data['CatIsMixed'] = train_data

main_count = train_data["AdoptionSpeed"].value_counts(normalize=True).sort_index()
print(main_count)


def prepare_plot_dict(df, col, main_count):
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col]==1, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values()))/ main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0
    return plot_dict


def make_count_plot(df, x, hue='AdoptionSpeed', tittle='', main_count=main_count):
    g = sns.countplot(x=x, data=df, hue=hue)
#    plt.title(f'Adoption speed {title}')
    plt.title('Adoption speed {0}'.format(tittle))
    ax = g.axes

    plot_dict = prepare_plot_dict(df, x, main_count)

    plt.show()


#make_count_plot(train_data, x='Type', tittle='by pet Type')
#make_count_plot(train_data, x='Age', tittle='by pet Age')

# sns.barplot(x='AdoptionSpeed', y='IsMixed', data=train_data)
# plt.show()


age_explorer = AgeExplorer(train_data)
age_explorer.basic_check()
age_explorer.plot_data()
train_data_features = age_explorer.get_additional_features()
