import os
from glob import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

import tsfresh.utilities.dataframe_functions as df_utilities

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


class Preprocess(object):
    def __init__(
        self,
        directory,
        y_cols,
        target,
        add_crypto_id=False,
        pca_components=10,
        filter_value=None,
        test_ratio=0.2,
        val_ratio=0.1,
    ):
        self.directory = directory
        self.y_cols = y_cols
        self.y = target
        self.add_crypto_id = add_crypto_id
        self.pca_components = pca_components
        self.filter_value = filter_value
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        self._flagpreprocess = False
        self._flagtensors = False

    def _collect_files(self):
        files = glob(f"{self.directory}/*.csv")
        self.dfs = [pd.read_csv(file) for file in files]
        self.cryptos = [os.path.split(file)[1].split("_")[1] for file in files]

        for df in self.dfs:
            df.fillna(method="ffill", inplace=True)
            dates = df.date
            df.drop(columns=["date"], inplace=True)
            # This indirectly maps all BOOL values to 1's and 0's of type float64
            df *= 1.0
            # This function maps inf->max and -inf->min for each column
            df_utilities.impute(df)
            df.insert(0, column="date", value=dates)

    def _split_data(self, X_list, y_list):
        # This is an additional step to filter out any crypto df with less than a significant amount of data (atleast 1000 rows).
        # Completely optional
        if self.filter_value:
            X_above_1k = []
            y_above_1k = []

            for i in range(len(X_list)):
                if len(X_list[i]) >= self.filter_value:
                    X_above_1k.append(X_list[i])
                    y_above_1k.append(y_list[i])
            X_list = X_above_1k
            y_list = y_above_1k

        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        X_val_list = []
        y_val_list = []

        for i in range(len(X_list)):
            X_train, X_test, y_train, y_test = train_test_split(
                X_list[i], y_list[i], test_size=self.test_ratio, random_state=42
            )  # 20% for test
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.val_ratio / (1 - self.test_ratio),
                random_state=42,
            )  # 0.1/0.8 = 0.125 for val, since 0.8*0.125=10%
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
            X_val_list.append(X_val)
            y_val_list.append(y_val)

        return (
            X_train_list,
            y_train_list,
            X_test_list,
            y_test_list,
            X_val_list,
            y_val_list,
        )

    def _processtarget(self):
        self.targets = []  # This will contain all the output columns for each df

        for i, df in enumerate(self.dfs):
            if self.add_crypto_id:
                le = LabelEncoder()
                le.fit(self.cryptos)

                if "CryptoID" in df.columns:
                    del df["CryptoID"]

                df.insert(
                    1, "CryptoID", pd.Series(le.transform([self.cryptos[i]] * len(df)))
                )
                # adding unique id's for cryptos as an additional feature
                # to make sure our model learns the uniqueness of each crypto
            # This step is to take care of any change in the sequence of columns in different files
            if i == 0:
                cols = df.columns
                df = df.reindex(columns=cols)

            self.targets.append(df[self.y])
            self.dfs[i] = df[[col for col in df.columns if col not in self.y_cols]]
            self.columns = self.dfs[0].columns

    def _processinput(self, inputs):
        self.scalers = []  # list of scalers
        for i in range(len(inputs)):
            scaler = MinMaxScaler()
            if self.add_crypto_id:
                inputs[i].iloc[:, 2:] = scaler.fit_transform(
                    inputs[i].copy().iloc[:, 2:]
                )
            else:
                inputs[i].iloc[:, 1:] = scaler.fit_transform(
                    inputs[i].copy().iloc[:, 1:]
                )  # leaving out date column
            self.scalers.append(scaler)

        self.pcas = []
        for i, data in enumerate(inputs):
            if "date" in data.columns:
                data.index = data.date
                del data["date"]
            pca = PCA(n_components=self.pca_components)
            if self.add_crypto_id:
                non_reduced = np.array(data.iloc[:, :5])
                # saving columns cryptoID, open,high, low and close
                pca.fit(data.iloc[:, 5:])
                reduced = pca.transform(data.iloc[:, 5:])
                reduced = np.append(non_reduced, reduced, axis=1)
            else:
                non_reduced = np.array(data.iloc[:, :4])
                # saving columns open,high, low and close
                pca.fit(data.iloc[:, 4:])
                reduced = pca.transform(data.iloc[:, 4:])
                reduced = np.append(non_reduced, reduced, axis=1)
            self.pcas.append(pca)
            inputs[i] = reduced

        return inputs

    def _prepare_test_val_set(self, inputs_test, inputs_val):
        X_test_mm = []
        X_test_processed = []
        X_val_mm = []
        X_val_processed = []
        for i, df in enumerate(inputs_test):
            if "date" in df.columns:
                df.index = df.date
                del df["date"]
            if self.add_crypto_id:
                test_mm = self.scalers[i].transform(df.copy().iloc[:, 1:])
                # leaving out cryptoID column
                X_test_mm.append(df)
                X_test_mm[i].iloc[:, 1:] = test_mm
                non_reduced = np.array(X_test_mm[i].iloc[:, :5])
                # saving columns cryptoID, open,high, low and close
                reduced = self.pcas[i].transform(X_test_mm[i].iloc[:, 5:])
                reduced = np.append(non_reduced, reduced, axis=1)
            else:
                test_mm = self.scalers[i].transform(df.copy())
                X_test_mm.append(df)
                X_test_mm[i].iloc[:, 0:] = test_mm
                non_reduced = np.array(X_test_mm[i].iloc[:, :4])
                # saving columns open,high, low and close
                reduced = self.pcas[i].transform(X_test_mm[i].iloc[:, 4:])
                reduced = np.append(non_reduced, reduced, axis=1)

            X_test_processed.append(reduced)

        for i, df in enumerate(inputs_val):
            if "date" in df.columns:
                df.index = df.date
                del df["date"]
            if self.add_crypto_id:
                val_mm = self.scalers[i].transform(df.copy().iloc[:, 1:])
                # leaving out cryptoID column
                X_val_mm.append(df)
                X_val_mm[i].iloc[:, 1:] = val_mm
                non_reduced = np.array(X_val_mm[i].iloc[:, :5])
                # saving columns cryptoID, open,high, low and close
                reduced = self.pcas[i].transform(X_val_mm[i].iloc[:, 5:])
                reduced = np.append(non_reduced, reduced, axis=1)
            else:
                val_mm = self.scalers[i].transform(df.copy())
                X_val_mm.append(df)
                X_val_mm[i].iloc[:, 0:] = val_mm
                non_reduced = np.array(
                    X_val_mm[i].iloc[:, :4]
                )  # saving columns open,high, low and close
                reduced = self.pcas[i].transform(X_val_mm[i].iloc[:, 4:])
                reduced = np.append(non_reduced, reduced, axis=1)

            X_val_processed.append(reduced)

        return X_test_processed, X_val_processed

    def process(self):
        self._collect_files()
        self._processtarget()
        (
            X_train_list,
            y_train_list,
            X_test_list,
            y_test_list,
            X_val_list,
            y_val_list,
        ) = self._split_data(self.dfs, self.targets)

        procesed_X_train = self._processinput(X_train_list)
        procesed_X_test, procesed_X_val = self._prepare_test_val_set(
            X_test_list, X_val_list
        )
        X_train = procesed_X_train[0]
        y_train = np.array(y_train_list[0])
        X_test = procesed_X_test[0]
        y_test = np.array(y_test_list[0])
        X_val = procesed_X_val[0]
        y_val = np.array(y_val_list[0])
        # now we stack all the data
        for i in range(1, len(procesed_X_train)):
            X_train = np.row_stack((X_train, procesed_X_train[i]))
            y_train = np.row_stack((y_train, np.array(y_train_list[i])))
        for i in range(1, len(procesed_X_test)):
            X_test = np.row_stack((X_test, procesed_X_test[i]))
            y_test = np.row_stack((y_test, np.array(y_test_list[i])))
        for i in range(1, len(procesed_X_val)):
            X_val = np.row_stack((X_val, procesed_X_val[i]))
            y_val = np.row_stack((y_val, np.array(y_val_list[i])))

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        self._flagpreprocess = True
        return (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.X_val,
            self.y_val,
        )

    def prepare_tensors(self):
        if not self._flagpreprocess:
            self.process()
        self.X_train = torch.tensor(self.X_train, dtype=torch.double)
        self.X_test = torch.tensor(self.X_test, dtype=torch.double)
        self.X_val = torch.tensor(self.X_val, dtype=torch.double)

        self.y_train = torch.tensor(self.y_train, dtype=torch.int64)
        self.y_test = torch.tensor(self.y_test, dtype=torch.int64)
        self.y_val = torch.tensor(self.y_val, dtype=torch.int64)

        self.X_train = torch.reshape(
            self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1])
        )
        self.X_test = torch.reshape(
            self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1])
        )
        self.X_val_ = torch.reshape(
            self.X_val, (self.X_val.shape[0], 1, self.X_val.shape[1])
        )
        self.inputshape = self.X_train.shape[1]
        self._flagtensors = True
        return (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.X_val,
            self.y_val,
        )

    def prepare_data_loaders(self, batch_size=64):
        if not self._flagtensors:
            if not self._flagpreprocess:
                self.process()
            self.prepare_tensors()
        train_data = []
        val_data = []
        test_data = []
        for i in range(len(self.X_train)):
            train_data.append([self.X_train[i], self.y_train[i]])

        for i in range(len(self.X_val)):
            val_data.append([self.X_val[i], self.y_val[i]])

        for i in range(len(self.X_test)):
            test_data.append([self.X_test[i], self.y_test[i]])

        loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        loaders = {"train": loader_train, "valid": loader_val, "test": loader_test}

        return loaders

    def process_testdata(self, df, scaler=None, pca=None):
        if scaler == None:
            scaler = self.scalers[0]
        if pca == None:
            pca = self.pcas[0]
        if "date" in df.columns:
            df.index = df.date
            del df["date"]
        df = df[self.columns]
        if self.add_crypto_id:
            test = scaler.transform(df.copy().iloc[:, 1:])
            df.iloc[:, 1:] = test
            non_reduced = np.array(df.iloc[:, :5])
            # saving columns cryptoID, open,high, low and close
            reduced = pca[0].transform(df.iloc[:, 5:])
            reduced = np.append(non_reduced, reduced, axis=1)
        else:
            test = scaler.transform(df.copy())
            df.iloc[:, 0:] = test
            non_reduced = np.array(
                df.iloc[:, :4]
            )  # saving columns open,high, low and close
            reduced = pca.transform(df.iloc[:, 4:])
            reduced = np.append(non_reduced, reduced, axis=1)

        test = torch.tensor(reduced, dtype=torch.int64)
        test = torch.reshape(test, (test.shape[0], 1, test.shape[1]))

        return test