import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.utils.data import TensorDataset, DataLoader

from project.models import RNNPred, RNNModel, LSTMModel

import warnings

warnings.simplefilter(action="ignore")


class TrainIndividual:
    def train(self, train_data, labels):
        batch_size = 64
        rnn_model = LSTMModel(input_dim=63, hidden_dim=64, layer_dim=3, output_dim=1, dropout_prob=0.5)
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001, weight_decay=1e-6)
        loss_fn = MSELoss()


        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.double), torch.tensor(labels, dtype=torch.double))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

        opt = Optimization(model=rnn_model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(train_loader, val_loader=None, batch_size=batch_size, n_epochs=5, n_features=63)

        for epoch in range(10):
            rnn_model.zero_grad()
            epoch_loss = 0
            for x, y in train_loader:
                # x = x.reshape(10, 1)
                # y = torch.tensor([y], dtype=torch.double)
                x = x.view([64, -1, 10])
                output = rnn_model(x)
                # print(output[-1])
                # loss += loss_fn(y, output[-1])
                loss = loss_fn(y, output.reshape(64, 1, 1))
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
        enhanced_train_data.index = pd.to_datetime(enhanced_train_data.index)
        enhanced_train_data = (
            enhanced_train_data
                .assign(day=enhanced_train_data.index.day)
                .assign(month=enhanced_train_data.index.month)
                .assign(day_of_week=enhanced_train_data.index.dayofweek)
                .assign(week_of_year=enhanced_train_data.index.week)
        )

        def onehot_encode_pd(df, cols):
            for col in cols:
                dummies = pd.get_dummies(df[col], prefix=col)

            return pd.concat([df, dummies], axis=1).drop(columns=cols)

        enhanced_train_data = onehot_encode_pd(enhanced_train_data, ['month', 'day', 'day_of_week', 'week_of_year'])

        grouped_data = enhanced_train_data.groupby(by=['store_nbr', 'family'])

        rows_to_train_on = grouped_data.get_group((1, "GROCERY I"))
        rows_to_train_on.drop(columns=['id', 'family', 'store_nbr'], inplace=True)
        y = rows_to_train_on.sales
        rows_to_train_on.drop(columns=['sales'])

        scaler = MinMaxScaler()
        rows_to_train_on = scaler.fit_transform(rows_to_train_on)
        y = scaler.fit_transform(y.values.reshape(1684, 1))
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

device = 'cpu'
class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """

    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        model_path = f'{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            # with torch.no_grad():
            #     batch_val_losses = []
            #     for x_val, y_val in val_loader:
            #         x_val = x_val.view([batch_size, -1, n_features]).to(device)
            #         y_val = y_val.to(device)
            #         self.model.eval()
            #         yhat = self.model(x_val)
            #         val_loss = self.loss_fn(y_val, yhat).item()
            #         batch_val_losses.append(val_loss)
            #     validation_loss = np.mean(batch_val_losses)
            #     self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: "
                )

        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        device = 'cpu'
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values




if __name__ == '__main__':
    run(base_path='store-sales-time-series-forecasting')
