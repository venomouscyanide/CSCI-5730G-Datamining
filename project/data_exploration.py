import os

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.utils.data import TensorDataset, DataLoader

from project.models import RNNPred, MV_LSTM

import warnings

warnings.simplefilter(action="ignore")


class TrainIndividual:
    def train(self, train_data, labels):
        batch_size = 8

        inp_dim = 9
        n_timesteps = 30
        hidden_size = 128
        n_layers = 3
        output_size = 1

        rnn_model = RNNPred(inp_dim, n_timesteps, hidden_size, n_layers, output_size)
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001, weight_decay=1e-6)
        loss_fn = MSELoss()

        # x = torch.from_numpy(train_data.values).double()
        # train_dataset = TensorDataset(x, torch.tensor(labels, dtype=torch.double))
        # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

        rnn_model.train()

        for epoch in range(50):
            counter = 0
            for b in range(0, len(train_data), batch_size):
                inpt = train_data[b:b + batch_size, :, :]
                target = labels[b:b + batch_size]

                x_batch = torch.tensor(inpt, dtype=torch.float32)
                y_batch = torch.tensor(target, dtype=torch.float32)

                rnn_model.init_hidden(x_batch.size(0))
                #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
                #    lstm_out.contiguous().view(x_batch.size(0),-1)
                output = rnn_model(x_batch)
                loss = loss_fn(output.view(-1), y_batch)

                loss.backward()
                counter += 1
                # print(loss.item(), counter)
                # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 5)
                # optimizer.step()
            print('step : ', epoch, 'loss : ', loss.item())


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


def split_sequences(sequences, n_steps):
    # https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch
    x, y = list(), list()
    # sequences = sequences.to_numpy()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


class Cleaning:
    def clean(self, base_path):
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

        rows_to_train_on = grouped_data.get_group((1, "GROCERY II"))
        rows_to_train_on.drop(columns=['id', 'family', 'store_nbr'], inplace=True)
        y = rows_to_train_on.sales
        rows_to_train_on.drop(columns=['sales'], inplace=True)
        rows_to_train_on.insert(loc=9, column='sales', value=y)

        scaler = MinMaxScaler(feature_range=(0, 100))
        rows_to_train_on = scaler.fit_transform(rows_to_train_on)
        rows_to_train_on = rows_to_train_on[:750]

        x, y = split_sequences(rows_to_train_on, 30)
        return x, y


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
    train_data, labels = Cleaning().clean(base_path)
    TrainIndividual().train(train_data, labels)


if __name__ == '__main__':
    run(base_path='store-sales-time-series-forecasting')
