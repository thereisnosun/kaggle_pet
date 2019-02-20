import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import plotly.plotly as py
import plotly.graph_objs as go

from src.data_exploration.basic_explorer import BasicExplorer


class BreedExplorer(BasicExplorer):
    def __init__(self):
        BasicExplorer.__init__(self)

    def basic_check(self):
        BasicExplorer.basic_check(self, 'Breed1')
        BasicExplorer.basic_check(self, 'Breed2')

    def plot_data(self):
        pass

    def get_additional_features(self):
        self.data_frame['PureBreed'] = self.data_frame['Breed2'].apply(lambda x: 1 if x == 0 else 0)
        pure_breed = self.data_frame['PureBreed'].where(self.data_frame['PureBreed'] == 1).count()
        print('Pure bree number - ', pure_breed)
        #self.data_frame.where
        # TODO: calc mixed cats and mixed dogs, calc ratio mixed to pure breed


        return self.data_frame