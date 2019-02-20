import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


from src.data_exploration.basic_explorer import BasicExplorer


def get_one_color(row):# TODO: check if other permutations possible
    if row['Color2'] == 0 and row['Color3'] == 0:
        return 1
    else:
        return 0


def get_two_color(row):
    if (row['Color1'] != 0 and row['Color2'] != 0 and row['Color3'] == 0) \
            or (row['Color2'] != 0 and row['Color3'] != 0 and row['Color1'] == 0) \
            or (row['Color1'] != 0 and row['Color3'] != 0 and row['Color2'] == 0):
        return 1
    else:
        return 0


def get_three_color(row):
    if row['Color1'] != 0 and row['Color2'] != 0 and row['Color3'] != 0:
        return 1
    else:
        return 0

class ColorExplorer(BasicExplorer):
    def __init__(self):
        BasicExplorer.__init__(self)

    def basic_check(self):
        BasicExplorer.basic_check(self, 'Color1')
        BasicExplorer.basic_check(self, 'Color2')
        BasicExplorer.basic_check(self, 'Color3')

    def plot_data(self):
        pass

    def get_additional_features(self):
        # one_color = self.data_frame.where(np.logical_and(self.data_frame['Color2'] == 0,
        self.data_frame['OneColored'] = self.data_frame.apply(lambda row: get_one_color(row), axis=1)
        self.data_frame['TwoColored'] = self.data_frame.apply(lambda row: get_two_color(row), axis=1)
        self.data_frame['ThreeColored'] = self.data_frame.apply(lambda  row: get_three_color(row), axis=1)
        print('One colored- ', self.data_frame['OneColored'].value_counts())
        print('Two colored- ', self.data_frame['TwoColored'].value_counts())
        print('Three colored- ', self.data_frame['ThreeColored'].value_counts())
        return self.data_frame
        #print('TYPE - ', one_color.dropna().iloc[0])
        # TODO: features:
        # 2-colored, 3-colored, one-colored
