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
        return rnn_model

    def inference(self, rnn_model, test_x, batch_size):
        rnn_model.eval()
        outputs = list()
        with torch.no_grad():
            for b in range(0, len(test_x), batch_size):
                inpt = test_x[b:b + batch_size, :, :]

                x_batch = torch.tensor(inpt, dtype=torch.float32)

                rnn_model.init_hidden(x_batch.size(0))
                #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
                #    lstm_out.contiguous().view(x_batch.size(0),-1)
                output = rnn_model(x_batch)
                outputs.append(output.detach().numpy())
        return np.vstack(outputs)


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


def split_sequences(sequences, n_steps, if_y=True):
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
        if if_y:
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
            x.append(seq_x)
            y.append(seq_y)
        else:
            seq_x = sequences[i:end_ix, :]
            x.append(seq_x)
    return np.array(x), np.array(y)


class CleaningAndTrain:
    def base_cleaning(self, base_path, data, oe_locale, oe_city, oe_state, oe_type_y):
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

        enhanced_train_data = data.merge(oil_prices, left_on='date', right_on='date', how='left')
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

        return enhanced_train_data

    def data_prep(self):
        pass

    def execute(self, base_path):
        oe_locale = OrdinalEncoder(dtype=np.int64)
        oe_city = OrdinalEncoder(dtype=np.int64)
        oe_state = OrdinalEncoder(dtype=np.int64)
        oe_type_y = OrdinalEncoder(dtype=np.int64)

        train_data = pd.read_csv(os.path.join(base_path, DataFiles.TRAIN))
        test_data = pd.read_csv(os.path.join(base_path, DataFiles.TEST))

        enhanced_train_data = self.base_cleaning(base_path, train_data, oe_locale, oe_city, oe_state, oe_type_y)
        enhanced_test_data = self.base_cleaning(base_path, test_data, oe_locale, oe_city, oe_state, oe_type_y)
        enhanced_test_data['transactions'].fillna(value=enhanced_train_data['transactions'].mean(),
                                                  inplace=True)  # TODO: be the mean of the respective groups

        grouped_train_data = enhanced_train_data.groupby(by=['store_nbr', 'family'])
        grouped_test_data = enhanced_test_data.groupby(by=['store_nbr', 'family'])

        all_groups = grouped_train_data.groups.keys()

        final_outputs = []
        all_groups_len = len(all_groups)
        for index, group in enumerate(all_groups):
            print(f"At index {index} of {all_groups_len}")
            rows_to_train_on = grouped_train_data.get_group(group)
            rows_to_test_on = grouped_test_data.get_group(group)

            rows_to_train_on.drop(columns=['id', 'family', 'store_nbr'], inplace=True)
            y = rows_to_train_on.sales
            rows_to_train_on.drop(columns=['sales'], inplace=True)
            rows_to_train_on.insert(loc=9, column='sales', value=y)

            scaler_train = MinMaxScaler(feature_range=(0, 1))
            rows_to_train_on = scaler_train.fit_transform(rows_to_train_on)
            org_rows_to_train_on = rows_to_train_on.copy()
            rows_to_train_on = rows_to_train_on[:800]

            train_test = TrainIndividual()
            x, y = split_sequences(rows_to_train_on, 30)
            rnn_model = train_test.train(x, y)

            rows_to_test_on.drop(columns=['id', 'family', 'store_nbr'], inplace=True)
            scaler_test = MinMaxScaler(feature_range=(0, 1))
            org_rows_to_test_on = rows_to_test_on.copy()
            rows_to_test_on = scaler_test.fit_transform(rows_to_test_on)
            rows_to_test_on = np.vstack((org_rows_to_train_on[-60:, :-1], rows_to_test_on))
            test_x, _ = split_sequences(rows_to_test_on, 30, if_y=False)

            test_outputs = train_test.inference(rnn_model, test_x, batch_size=8)
            test_outputs_for_csv = test_outputs[-16:, :]

            org_rows_to_train_on[:, -1] = np.pad(np.reshape(test_outputs_for_csv, newshape=(16,)), (0, 1668),
                                                 constant_values=0)
            final_output = scaler_train.inverse_transform(org_rows_to_train_on)
            final_output_for_group = final_output[:16, -1]

            csv_pd_data = grouped_test_data.get_group(group)
            csv_pd_data['store_nbr'] = final_output_for_group
            csv_pd_data.drop(
                columns=['family', 'onpromotion', 'dcoilwtico', 'transactions', 'type_x', 'locale', 'city', 'state',
                         'type_y', 'cluster'], inplace=True)

            csv_pd_data['store_nbr'] = csv_pd_data['store_nbr'].abs()
            final_outputs.append(csv_pd_data)

            print(group)
            print(csv_pd_data['store_nbr'])
        final_df = pd.concat(final_outputs)
        final_df.rename(columns={"store_nbr": "sales"}, inplace=True)
        final_df.to_csv('submission_rnn.csv', index=False)


def run(base_path):
    CleaningAndTrain().execute(base_path)


if __name__ == '__main__':
    run(base_path='store-sales-time-series-forecasting')
