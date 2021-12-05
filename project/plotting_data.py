import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder

from project.prep_and_train import CleaningAndTrain, DataFiles


def earth_quake(enhanced_train_data):
    fig = px.line(enhanced_train_data[(enhanced_train_data['date'] > pd.to_datetime("2016-03-16")) & (
            enhanced_train_data['date'] < pd.to_datetime("2016-06-16")) & (enhanced_train_data['store_nbr'] == 2) & (
                                        enhanced_train_data['family'] == 'AUTOMOTIVE')], x='date',
            y=['earthquake_effect', 'sales'])
    fig.show()


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
    earth_quake(enhanced_train_data)
