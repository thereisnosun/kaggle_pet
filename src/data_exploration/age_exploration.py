import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt

from src.data_exploration.basic_explorer import BasicExplorer


class AgeExplorer(BasicExplorer):
    def __init__(self, data_frame):
        BasicExplorer.__init__(self, data_frame)

    def basic_check(self):
        BasicExplorer.basic_check(self, 'Age')

    def plot_data(self):
        sns.catplot(x='LessThenYear', y='Age', data=self.train_data)
        plt.show()

        # plt.figure(figsize=(10,10))
        # sns.violinplot(x='AdoptionSpeed', y='Age', hue='Type',  data=self.train_data)
        # plt.show()

        # sns.lineplot(y='AdoptionSpeed', x='Age', data=self.train_data)
        # plt.show()

        # self.train_data["Age"].plot(kind='hist')
        # plt.show()

    def get_additional_features(self):
        self.train_data['LessThenYear'] = self.train_data['Age'].apply(lambda x: 'YES' if x < 12 else 'NO')
        print('Less then a year pets - ', self.train_data['LessThenYear'].value_counts())
        return self.train_data



