import os

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.utils.data import TensorDataset, DataLoader

from project.models import RNNPred

import warnings

warnings.simplefilter(action="ignore")


class TrainIndividual:
    def train(self, train_data, labels):
        batch_size = 16
        rnn_model = RNNPred(input_dim=10, hidden_size=64, no_of_hidden_layers=3, output_size=1)
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001, weight_decay=1e-6)
        loss_fn = MSELoss()

        x = torch.from_numpy(train_data.values).double()
        train_dataset = TensorDataset(x, torch.tensor(labels, dtype=torch.double))
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

        for epoch in range(10):
            rnn_model.zero_grad()
            epoch_loss = 0
            for x, y in train_loader:
                # x = x.reshape(10, 1)
                # y = torch.tensor([y], dtype=torch.double)
                x = x.view([16, -1, 10])
                output = rnn_model(x)
                # print(output[-1])
                # loss += loss_fn(y, output[-1])
                loss = loss_fn(y, output.reshape(16, 1, 1))
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 5)
            # optimizer.step()
            print(f"Done with one epoch, loss is :{np.mean(epoch_loss)}")


class DataFiles:
    """
    More info on what CSV contains what is provided in README.
    It can also be found here: https://www.kaggle.com/c/store-sales-time-series-forecasting/data?select=transactions.csv
    """
    OIL_PRICES = 'oil.csv'
    TEST = 'test.csv'
    TRAIN = 'train.csv'
    STORE_METADATA = 'stores.csv'
    HOLIDAYS = 'holidays_events.csv'
    TRANSACTIONS = 'transactions.csv'


class Cleaning:
    def clean2(self, base_path):
        oe_locale = OrdinalEncoder(dtype=np.int64)
        oe_city = OrdinalEncoder(dtype=np.int64)
        oe_state = OrdinalEncoder(dtype=np.int64)
        oe_type_y = OrdinalEncoder(dtype=np.int64)

        train_data = pd.read_csv(os.path.join(base_path, DataFiles.TRAIN))
        store_metadata = pd.read_csv(
            os.path.join(base_path, DataFiles.STORE_METADATA))  # not used in this implementation
        transactions = pd.read_csv(os.path.join(base_path, DataFiles.TRANSACTIONS))
        oil_prices = pd.read_csv(os.path.join(base_path, DataFiles.OIL_PRICES))
        holidays = pd.read_csv(os.path.join(base_path, DataFiles.HOLIDAYS))

        num_oil_prices = len(oil_prices)
        for index, item in enumerate(oil_prices.values):
            if not np.isnan(item[1]):
                continue
            if index == 0:
                value_to_replace_na = oil_prices.loc[index + 1]['dcoilwtico']
            elif index == num_oil_prices - 1:
                value_to_replace_na = oil_prices.loc[index - 1]['dcoilwtico']
            else:
                value_to_replace_na = (oil_prices.loc[index + 1]['dcoilwtico'] +
                                       oil_prices.loc[index + - 1]['dcoilwtico']) / 2
            oil_prices.at[index, 'dcoilwtico'] = value_to_replace_na

        enhanced_train_data = train_data.merge(oil_prices, left_on='date', right_on='date', how='left')
        enhanced_train_data['dcoilwtico'].fillna(method='ffill', inplace=True)
        transactions = transactions.groupby(['date']).agg({'transactions': 'mean'})
        enhanced_train_data = enhanced_train_data.merge(transactions, left_on=['date'], right_on=['date'], how='left')

        holiday_types = ['Holiday', 'Additional', 'Bridge', 'Event', 'Transfer']
        holidays = holidays.query("(type in @holiday_types) and (transferred == 0)")
        holidays.drop(columns=['locale_name', 'description', 'transferred'], inplace=True)
        holidays.drop_duplicates(inplace=True, keep="first", subset=['date'])
        enhanced_train_data = enhanced_train_data.merge(holidays, left_on=['date'], right_on=['date'], how='left')

        enhanced_train_data['locale'].fillna(value='WorkDay', inplace=True)
        enhanced_train_data['type'].fillna(value=0, inplace=True)
        enhanced_train_data['type'] = enhanced_train_data['type'].astype(bool).astype(int)
        enhanced_train_data["locale"] = oe_locale.fit_transform(enhanced_train_data[["locale"]])

        enhanced_train_data = enhanced_train_data.merge(store_metadata, left_on='store_nbr', right_on='store_nbr',
                                                        how='left')
        enhanced_train_data["city"] = oe_city.fit_transform(enhanced_train_data[["city"]])
        enhanced_train_data["state"] = oe_state.fit_transform(enhanced_train_data[["state"]])
        enhanced_train_data["type_y"] = oe_type_y.fit_transform(enhanced_train_data[["type_y"]])
        enhanced_train_data.set_index('date', inplace=True)

        grouped_data = enhanced_train_data.groupby(by=['store_nbr', 'family'])

        rows_to_train_on = grouped_data.get_group((1, "GROCERY I"))
        rows_to_train_on.drop(columns=['id', 'family', 'store_nbr'], inplace=True)
        y = rows_to_train_on.sales
        rows_to_train_on.drop(columns=['sales'])
        return rows_to_train_on, y

    def clean(self, base_path):
        oe_locale = OrdinalEncoder(dtype=np.int64)

        train_data = pd.read_csv(os.path.join(base_path, DataFiles.TRAIN))
        store_metadata = pd.read_csv(
            os.path.join(base_path, DataFiles.STORE_METADATA))  # not used in this implementation
        transactions = pd.read_csv(os.path.join(base_path, DataFiles.TRANSACTIONS))
        oil_prices = pd.read_csv(os.path.join(base_path, DataFiles.OIL_PRICES))
        holidays = pd.read_csv(os.path.join(base_path, DataFiles.HOLIDAYS))

        train_data_copy = train_data.copy()
        train_data = train_data_copy.groupby(['date']).agg({'sales': 'mean', 'onpromotion': 'mean'})
        # enhanced_train_data = train_data.merge(store_metadata, left_on='store_nbr', right_on='store_nbr', how='left')
        num_oil_prices = len(oil_prices)
        for index, item in enumerate(oil_prices.values):
            if not np.isnan(item[1]):
                continue
            if index == 0:
                value_to_replace_na = oil_prices.loc[index + 1]['dcoilwtico']
            elif index == num_oil_prices - 1:
                value_to_replace_na = oil_prices.loc[index - 1]['dcoilwtico']
            else:
                value_to_replace_na = (oil_prices.loc[index + 1]['dcoilwtico'] +
                                       oil_prices.loc[index + - 1]['dcoilwtico']) / 2
            oil_prices.at[index, 'dcoilwtico'] = value_to_replace_na

        enhanced_train_data = train_data.merge(oil_prices, left_on='date', right_on='date', how='left')
        enhanced_train_data['dcoilwtico'].fillna(method='ffill', inplace=True)
        transactions = transactions.groupby(['date']).agg({'transactions': 'mean'})
        enhanced_train_data = enhanced_train_data.merge(transactions, left_on=['date'], right_on=['date'], how='left')

        holiday_types = ['Holiday', 'Additional', 'Bridge', 'Event', 'Transfer']
        holidays = holidays.query("(type in @holiday_types) and (transferred == 0)")
        holidays.drop(columns=['locale_name', 'description', 'transferred'], inplace=True)
        holidays.drop_duplicates(inplace=True, keep="first", subset=['date'])
        enhanced_train_data = enhanced_train_data.merge(holidays, left_on=['date'], right_on=['date'], how='left')

        enhanced_train_data['locale'].fillna(value='WorkDay', inplace=True)
        enhanced_train_data['type'].fillna(value=0, inplace=True)
        enhanced_train_data['type'] = enhanced_train_data['type'].astype(bool).astype(int)
        enhanced_train_data["locale"] = oe_locale.fit_transform(enhanced_train_data[["locale"]])

        enhanced_train_data.set_index('date', inplace=True)

        scaler = MinMaxScaler()
        scaler.fit(enhanced_train_data)
        enhanced_train_data = scaler.transform(enhanced_train_data)

        train_data_copy.sort_values(by=['date', 'store_nbr', 'family'], inplace=True)
        labels = train_data_copy.groupby('date')['sales'].apply(list)
        labels = labels.to_frame()
        labels = pd.DataFrame(
            labels.sales.to_list(),
            labels.index, [f"dim_{n}" for n in range(1782)]
        )
        return enhanced_train_data, labels


class TrainRnn:
    def train(self, train_data, labels):
        # train_data: 1684 * 6 ; labels: 1684 * 1782
        rnn_model = RNNPred(input_dim=6, hidden_size=128, no_of_hidden_layers=3, output_size=1782)
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
        loss_fn = MSELoss(reduction="mean")

        hidden = torch.zeros(3, 1, 128, dtype=torch.double)

        for epoch in range(10):
            optimizer.zero_grad()
            loss_float = 0
            for x, y in zip(train_data, labels.to_numpy()):
                x = torch.tensor(x.reshape(6, 1).view(), dtype=torch.double)
                y = torch.tensor(y.reshape(1, 1782).view(), dtype=torch.double)
                output, hidden = rnn_model(x, hidden)

                loss = loss_fn(y, output[-1])
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            print("Done with one epoch")
            # loss_float += loss_fn(y, output[-1]).item()

            # epoch_loss = loss / len(train_data)


def run(base_path):
    train_data, labels = Cleaning().clean2(base_path)
    TrainIndividual().train(train_data, labels)
    # train_data, labels = Cleaning().clean(base_path)
    # TrainRnn().train(train_data, labels)


if __name__ == '__main__':
    run(base_path='store-sales-time-series-forecasting')
