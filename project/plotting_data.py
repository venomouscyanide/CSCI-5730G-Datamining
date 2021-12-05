import os

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

from project.prep_and_train import CleaningAndTrain, DataFiles
import matplotlib.pyplot as plt

import seaborn as sns

def earth_quake(enhanced_train_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
    copy_of_df = enhanced_train_data.copy()
    copy_of_df.drop(columns=['date', 'family'], inplace=True)

    scaled = min_max_scaler.fit_transform(copy_of_df)
    df = pd.DataFrame(scaled)
    enhanced_train_data['sales'] = df[2]

    # https://www.kaggle.com/luisblanche/pytorch-forecasting-temporalfusiontransformer
    fig = px.line(enhanced_train_data[(enhanced_train_data['date'] > pd.to_datetime("2016-03-16")) & (
            enhanced_train_data['date'] < pd.to_datetime("2016-06-16"))], x='date',
                  y=['earthquake_effect', 'sales'])

    fig.update_layout(title_text="Analyzing the effect of earthquake on sales")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Earthquake and Sale values")

    fig.show()
    fig.write_image("earthquake.png")


def oil_price_vs_sales(enhanced_train_data):
    fig = px.line(enhanced_train_data, x='date', y=['dcoilwtico', ])

    fig.update_layout(title_text="Variation of oil prices over dates")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Oil Prices")

    fig.show()
    fig.write_image("oil_prices.png")

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    copy_of_df = enhanced_train_data.copy()
    copy_of_df.drop(columns=['date', 'family'], inplace=True)

    scaled = min_max_scaler.fit_transform(copy_of_df)
    df = pd.DataFrame(scaled)
    enhanced_train_data['sales'] = df[2]
    enhanced_train_data['dcoilwtico'] = df[4]

    fig = px.line(enhanced_train_data[(enhanced_train_data['date'] > pd.to_datetime("2015-01-01"))], x='date',
                  y=['dcoilwtico', 'sales'])

    fig.update_layout(title_text="Tracking oil prices vs sales from 2015")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales and Oil prices")

    fig.show()
    fig.write_image("oil_prices_vs_sales.png")


def payday_analysis(enhanced_train_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    copy_of_df = enhanced_train_data.copy()
    copy_of_df.drop(columns=['date', 'family'], inplace=True)

    scaled = min_max_scaler.fit_transform(copy_of_df)
    df = pd.DataFrame(scaled)
    enhanced_train_data['sales'] = df[2]
    enhanced_train_data['days_from_payday'] = df[13]

    fig = px.line(enhanced_train_data[(enhanced_train_data['date'] > pd.to_datetime("2017-01-01")) & (
            enhanced_train_data['date'] < pd.to_datetime("2017-04-01"))], x='date',
                  y=['days_from_payday', 'sales'])

    fig.update_layout(title_text="Tracking the effect of payday on sales")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales and Days from Payday")

    fig.write_image("payday.png")


def transactions(enhanced_train_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    copy_of_df = enhanced_train_data.copy()
    copy_of_df.drop(columns=['date', 'family'], inplace=True)

    scaled = min_max_scaler.fit_transform(copy_of_df)
    df = pd.DataFrame(scaled)
    enhanced_train_data['sales'] = df[2]
    enhanced_train_data['transactions'] = df[5]

    fig = px.line(enhanced_train_data[(enhanced_train_data['date'] > pd.to_datetime("2017-04-01"))], x='date',
                  y=['transactions', 'sales'])

    fig.update_layout(title_text="Tracking transactions vs sales from April 2017")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales and Transactions")

    fig.show()
    fig.write_image("transactions.png")


def holidays(enhanced_train_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    copy_of_df = enhanced_train_data.copy()
    copy_of_df.drop(columns=['date', 'family'], inplace=True)

    scaled = min_max_scaler.fit_transform(copy_of_df)
    df = pd.DataFrame(scaled)
    enhanced_train_data['sales'] = df[2]
    enhanced_train_data['holidays'] = df[6]

    fig = px.line(enhanced_train_data[(enhanced_train_data['date'] > pd.to_datetime("2016-06-01")) & (
            enhanced_train_data['date'] < pd.to_datetime("2016-12-31"))], x='date',
                  y=['holidays', 'sales'])

    fig.update_layout(title_text="Tracking Holidays vs Sales from June to End of year 2016")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales and Transactions")

    fig.show()
    fig.write_image("holidays.png")


def heat_map(enhanced_train_data):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15, left=0.15)
    sns.heatmap(enhanced_train_data.corr())
    plt.show()
    plt.savefig(fname='heatmap.png')


if __name__ == '__main__':
    oe_locale = OrdinalEncoder(dtype=np.int64)
    oe_city = OrdinalEncoder(dtype=np.int64)
    oe_state = OrdinalEncoder(dtype=np.int64)
    oe_type_y = OrdinalEncoder(dtype=np.int64)

    base_path = 'store-sales-time-series-forecasting'
    train_data = pd.read_csv(os.path.join(base_path, DataFiles.TRAIN))
    test_data = pd.read_csv(os.path.join(base_path, DataFiles.TEST))

    enhanced_train_data = CleaningAndTrain().base_cleaning(base_path, train_data, oe_locale, oe_city, oe_state,
                                                           oe_type_y, False)
    # earth_quake(enhanced_train_data)
    # oil_price_vs_sales(enhanced_train_data)
    # payday_analysis(enhanced_train_data)
    # transactions(enhanced_train_data)
    # holidays(enhanced_train_data)
    heat_map(enhanced_train_data)
