from utils.dataset import DataSet

import pandas as pd
import datetime as dt

class MyDataSet():
    def __init__(self, path="fnc-1"):
        self.path = path

        print("Reading dataset")
        self.filename = "{name}_dataframe.pkl"

        self.train = self.load(name='train')
        # self.test = self.load(name='test')  FIXME
        self.competition_test = self.load(name='competition_test')

    def load(self, name='test'):
        try:
            return pd.read_pickle(self.filename.format(name=name))
        except FileNotFoundError:
            return self.save(name)  # also returns df

    def save(self, name, suffix=''):
        data = DataSet(name=name, path='fnc-1')
        stances = pd.DataFrame(data=data.stances)
        stances.set_index('Body ID', drop=True, inplace=True)

        articles = pd.DataFrame.from_dict(data.articles, orient='index')
        articles.rename(columns={0: 'body'}, inplace=True)
        articles.index.names = ['Body ID']

        df = pd.merge(
            stances,
            articles,
            how='left',
            left_index=True,
            right_index=True
        )

        df.to_pickle(self.filename.format(name=name))
        return df
