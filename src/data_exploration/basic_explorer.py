class BasicExplorer:
    def __init__(self):
        self.data_frame = None

    def set_data_frame(self, data_frame):
        self.data_frame = data_frame

    def basic_check(self, column_name):
        print('{0} observation:'.format(column_name));
        print('Max value - ', self.data_frame[column_name].max())
        print('STD - ', self.data_frame[column_name].std())
        print('Mean - ', self.data_frame[column_name].mean())
        print('Median - ', self.data_frame[column_name].median())
        print('Num of NA - ', self.data_frame[column_name].isna().sum())
        print('--------------------------------------------')

    def get_additional_features(self):
        return self.data_frame

    def plot_data(self):
        pass

