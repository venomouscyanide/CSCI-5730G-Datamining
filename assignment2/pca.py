"""
author: Paul Louis
email: paul.louis@ontariotechu.net
example usage: python3.9 pca.py --iris_dataset_location "iris.data" --k 4
"""
import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from sklearn.metrics import mean_squared_error
import plotly.express as px


def plot_pca(projected_data: np.ndarray, labels: List[str]):
    labels = np.array(labels)

    if projected_data.shape[-1] == 2:
        fig = px.scatter(projected_data,
                         x=0, y=1,
                         title="PCA with K=2",
                         labels={'0': 'PC 1', '1': 'PC 2'},
                         color=labels)
    else:
        fig = px.scatter_3d(
            projected_data,
            color=labels,
            title="PCA with K=3",
            x=0,
            y=1,
            z=2,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
    fig.show()


class CustomPCA:
    def __init__(self, dataframe: pd.DataFrame, k: int = 4):
        self.evals = None
        self.evecs = None
        self.mean_values: List[float] = []
        self.dataframe = dataframe
        self.k = k

    def transform(self) -> np.ndarray:
        # calculate mean values
        for col in self.dataframe.columns:
            self.mean_values.append(self.dataframe[col].mean())

        # center each column value using the calculated mean value
        for index, col in enumerate(self.dataframe.columns):
            self.dataframe[col] = self.dataframe[col] - self.mean_values[index]

        # calculate covariance matrix
        cov = self.dataframe.cov()

        # get top `k` eigen values and eigen vectors of the covariance matrix
        self.evals, self.evecs = eigh(cov, eigvals=(4 - self.k, 4 - 1))

        # create projected data
        projected_data = np.dot(self.dataframe, self.evecs)
        return projected_data

    def inverse_transform(self, projected_data: np.ndarray) -> np.ndarray:
        inverse_t = np.dot(projected_data, self.evecs.T)
        # add mean values
        for index in range(len(self.dataframe.columns)):
            inverse_t[:, index] = inverse_t[:, index] + self.mean_values[index]
        return inverse_t


def run_PCA_on_iris(k: int, iris_path: str):
    # read the iris dataset and parse data into list of lines
    with open(iris_path) as iris_file:
        data = iris_file.read().splitlines()
    # get label colors for plotting
    labels = [data.split(',')[-1].strip() for data in data][:-1]

    # cleaning
    # only consider data ignoring the last column which is the label
    data = [data.split(',')[:-1] for data in data]
    # drop empty data
    data = list(filter(lambda x: x, data))

    # type cast the str numbers into floating point numbers
    for idx, row in enumerate(data):
        updated_row = [float(row[idx]) for idx in range(len(row))]
        data[idx] = updated_row

    # create a df with the cleaned list of data
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4'])

    # call PCA with k value
    custom_pca = CustomPCA(df, k=k)
    projected_data = custom_pca.transform()

    # calculating mse after inverse projection
    inverse = custom_pca.inverse_transform(projected_data)
    error = mean_squared_error(df, inverse)
    print(f"Error calculation for k: {k} is {error}")

    # draw the projected data if k is 2 or 3
    if k in [2, 3]:
        plot_pca(projected_data, labels)

    # sanity check against scikit learn values
    pca = PCA(n_components=k)
    pca.fit(df)
    X = pca.transform(df)
    X = pca.inverse_transform(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Custom PCA")
    parser.add_argument('--iris_dataset_location', help='iris.data file path', required=True, type=str,
                        default="IRIS data/iris.data")
    parser.add_argument('--k', help='k value', required=True, type=int)
    args = parser.parse_args()
    run_PCA_on_iris(args.k, args.iris_dataset_location)
