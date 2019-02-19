import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


from src.data_exploration.basic_explorer import BasicExplorer


def get_color(row):
    if row['Color2'] == 0 and row['Color3'] == 0:
        return 1
    else:
        return 0

class ColorExplorer(BasicExplorer):
    def __init__(self, data_frame):
        BasicExplorer.__init__(self, data_frame)

    def basic_check(self):
        BasicExplorer.basic_check(self, 'Color1')
        BasicExplorer.basic_check(self, 'Color2')
        BasicExplorer.basic_check(self, 'Color3')

    def plot_data(self):
        pass



    def get_additional_features(self):
        # one_color = self.data_frame.where(np.logical_and(self.data_frame['Color2'] == 0,
        self.data_frame['OneColored'] = self.data_frame.apply(lambda row: get_color(row), axis=1)
        #self.data_frame['OneColored'] = one_color.dropna().iloc[0]
        print(self.data_frame['OneColored'].value_counts())
        #print('TYPE - ', one_color.dropna().iloc[0])
        # TODO: features:
        # 2-colored, 3-colored, one-colored
