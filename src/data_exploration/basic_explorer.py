class BasicExplorer:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def basic_check(self, column_name):
        print('{0} observation:'.format(column_name));
        print('Max value - ', self.data_frame[column_name].max())
        print(self.data_frame[column_name].std())
        print(self.data_frame[column_name].mean())
        print(self.data_frame[column_name].median())
        print(self.data_frame[column_name].isna().sum())

    def get_additional_features(self):
        return self.data_frame

    def plot_data(self):
        pass

