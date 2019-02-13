

import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt


def basic_check(train_data):
    print('Age observation - \n');
    print('Max age - ', train_data['Age'].max())
    print(train_data['Age'].std())
    print(train_data['Age'].mean())
    print(train_data['Age'].median())
    print(train_data['Age'].isna().sum())


    train_data['LessThenYear'] = train_data['Age'].apply(lambda x: 'YES' if x < 12 else 'NO')
    print('Less then a year pets - ', train_data['LessThenYear'].value_counts())

    sns.catplot(x='LessThenYear', y='Age', data=train_data)
    plt.show()

    # plt.figure(figsize=(10,10))
    # sns.violinplot(x='AdoptionSpeed', y='Age', hue='Type',  data=train_data)
    # plt.show()

    # sns.lineplot(y='AdoptionSpeed', x='Age', data=train_data)
    # plt.show()

