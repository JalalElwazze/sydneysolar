import pandas as pd
from scipy.stats import spearmanr
import numpy as np


class Data:

    @staticmethod
    def exposure():
        df = pd.read_csv("data_files/exposure.csv")
        df['Time'] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df['Exposure kWh'] = df["Daily global solar exposure (MJ/m*m)"]*0.277778
        df.drop(df.columns[0:6], axis=1, inplace=True)

        return df

    @staticmethod
    def max_temp():
        df = pd.read_csv("data_files/max_temp.csv")
        df['Time'] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.drop(df.columns[0:5], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.columns = ["Max Temp (C)", 'Time']
        return df

    @staticmethod
    def min_temp():
        df = pd.read_csv("data_files/min_temp.csv")
        df['Time'] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.drop(df.columns[0:5], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.columns = ["Min Temp (C)", 'Time']
        return df

    @staticmethod
    def rain():
        df = pd.read_csv("data_files/rain.csv")
        df['Time'] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.drop(df.columns[0:5], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)
        df.columns = ["Rain (mm)", 'Time']
        return df

    @staticmethod
    def merged_units():
        df = Data.exposure().merge(Data.rain(), on='Time', how="left")
        df = df.merge(Data.min_temp(), on='Time', how="left")
        df = df.merge(Data.max_temp(), on='Time', how="left")
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)

        return df

    @staticmethod
    def merged_norm():
        df = Data.exposure().merge(Data.rain(), on='Time', how="left")
        df = df.merge(Data.min_temp(), on='Time', how="left")
        df = df.merge(Data.max_temp(), on='Time', how="left")
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        for col in df.columns[1:]:
            df[col] /= df[col].max()

        df.columns = ['Time', 'Exposure', 'Rain', 'Min Temp', 'Max Temp']

        return df

    @staticmethod
    def compute_rank(comparison):
        dataset = Data.merged_norm()
        rank = spearmanr(dataset["Exposure"], dataset[comparison])[0]
        return np.abs(round(rank, 2))

