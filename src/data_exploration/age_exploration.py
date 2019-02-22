import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import plotly.plotly as py
import plotly.graph_objs as go

from src.data_exploration.basic_explorer import BasicExplorer
import src.data_exploration.utils_plotting as plotting

class AgeExplorer(BasicExplorer):
    def __init__(self):
        BasicExplorer.__init__(self)

    def basic_check(self):
        BasicExplorer.basic_check(self, 'Age')

    def plot_data(self):
        # sns.violinplot(x='LessThenYear', y='AdoptionSpeed', data=self.data_frame)
        # plt.show()

        # plt.figure(figsize=(10,10))
        # sns.violinplot(x='AdoptionSpeed', y='Age', hue='Type',  data=self.data_frame)
        # plt.show()


        #plotting.adoption_trends_plot(self.data_frame,'Age')

        # sns.lineplot(y='AdoptionSpeed', x='Age', data=self.data_frame)
        # plt.show()
        #
        # self.data_frame["Age"].plot(kind='hist')
        # plt.show()
        pass

    def get_additional_features(self):
        self.data_frame['LessThenYear'] = self.data_frame['Age'].apply(lambda x: 1 if x < 12 else 0)
        print('Less then a year pets - ', self.data_frame['LessThenYear'].value_counts())
        return self.data_frame



