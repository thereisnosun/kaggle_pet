import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import plotly.plotly as py
import plotly.graph_objs as go

from src.data_exploration.basic_explorer import BasicExplorer

mixed_breeds_id = [307]

def check_if_pure_breed(row, breed_df):
    if int(row['Breed2']):
        return 0
    breed = row['Breed1']
    int_breed = int(breed)
    return int_breed in mixed_breeds_id

class BreedExplorer(BasicExplorer):
    def __init__(self):
        self.breed_df = pd.read_csv('../input/breed_labels.csv')
        self.breed_df['BreedName'] = self.breed_df['BreedName'].apply(lambda name: name.lower())
        BasicExplorer.__init__(self)

    def basic_check(self):
        BasicExplorer.basic_check(self, 'Breed1')
        BasicExplorer.basic_check(self, 'Breed2')


    def plot_data(self):
        pass

    def get_additional_features(self):
        self.data_frame['PureBreed'] = self.data_frame.apply(lambda row: check_if_pure_breed(row, self.breed_df), axis=1)
        pure_breed = self.data_frame['PureBreed'].where(self.data_frame['PureBreed'] == 1).count()
        print('Pure bree number - ', pure_breed)
        # TODO: calc mixed cats and mixed dogs, calc ratio mixed to pure breed


        return self.data_frame