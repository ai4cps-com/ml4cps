"""
    The module provides base classes to represent the dynamics of cyber-physical systems (CPS): CPS and
    CPSComponent.

    Author: Nemanja Hranisavljevic, hranisan@hsu-hh.de
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class TimeSeriesDiscretizer:
    """
        Abstract class that encapsulates methods used for the discretization of time series.
    """
    def train(self, data, *args):
        pass

    def discretize(self, data):
        pass


class EqualWidthDiscretizer (TimeSeriesDiscretizer):
    """
    A class that implements equal-width interval (EWI) discretization method. It calculates range of every variable
    (column) of the input data into a predefined number of equal-width intervals.
    """
    def __init__(self):
        self.intervals = None
        self.min_d = None
        self.max_d = None

    def train(self, data, number_of_intervals=10):
        """
        Estimate model parameters, thresholds that divide each variable into equal-width intervals.
        :param data: Data to calculate model parameters from.
        :param number_of_intervals: Number of equal-width intervals per variable.
        :return:
        """
        data = [pd.DataFrame(d) for d in data]
        min_d = min(d.min(axis=0) for d in data)
        max_d = max(d.max(axis=0) for d in data)
        self.min_d = min_d
        self.max_d = max_d
        self.intervals = pd.DataFrame(np.linspace(self.min_d, self.max_d, number_of_intervals), columns=data[0].columns)

    def discretize(self, df, return_str=False, append_discr=None):
        """
        Discretize data into equal width intervals.
        :param data: Data to discretize.
        :return: Discretized data.
        """

        df = pd.DataFrame(df)
        binned_df = df.copy()
        for col in self.intervals.columns:
            col_min=self.min_d[col]
            col_max=self.max_d[col]
            bins=self.intervals[col]

            # Check for out-of-range values and adjust
            if (df[col] < col_min).any() or (df[col] > col_max).any():
                print(f"Warning: Column '{col}' contains values outside the range [{col_min}, {col_max}].")
                binned_df[col] = np.clip(df[col], col_min, col_max)
            else:
                binned_df[col] = df[col]

            # Apply binning with predefined bin edges
            try:
                binned_df[col] = pd.cut(
                    binned_df[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True)
            except:
                print(1)

        if append_discr is not None:
            binned_df = np.hstack((binned_df, append_discr))

        binned_df = pd.DataFrame(binned_df)
        if return_str:
            binned_df = binned_df.round(0).astype("Int64").astype(str)
            binned_df = binned_df.agg(','.join, axis=1).to_numpy()
        return binned_df


class EqualFrequencyDiscretizer(TimeSeriesDiscretizer):
    """
    A class that implements the equal-frequency interval (EFI) discretization method. It divides each variable
    (column) into intervals such that each interval contains approximately the same number of data points.
    """
    def __init__(self):
        self.intervals = None

    def train(self, data, number_of_intervals=10):
        """
        Estimate model parameters, thresholds that divide each variable into equal-frequency intervals.
        :param data: Data to calculate model parameters from.
        :param number_of_intervals: Number of equal-frequency intervals per variable.
        :return:
        """
        data = [pd.DataFrame(d) for d in data]
        combined_data = pd.concat(data, axis=0)  # Combine all datasets for quantile calculation
        self.intervals = {}
        for col in combined_data.columns:
            intvl = pd.qcut(combined_data[col], q=number_of_intervals, duplicates='drop', retbins=True)[1]
            if len(np.unique(intvl)) != len(intvl):
                print(1)
            if len(intvl) < number_of_intervals + 1:
                expanded_arr = np.empty(number_of_intervals+1, dtype=combined_data[col].dtype)
                expanded_arr[:] = np.nan
                expanded_arr[:len(intvl)] = intvl
                intvl = expanded_arr
            self.intervals[col] = intvl

        self.intervals = pd.DataFrame(self.intervals)

    def discretize(self, df, return_str=False, append_discr=None):
        """
        Discretize data into equal-frequency intervals.
        :param df: Data to discretize.
        :param return_str: Whether to return discretized data as a concatenated string per row.
        :return: Discretized data.
        """
        df = pd.DataFrame(df)
        binned_df = df.copy()

        for col in self.intervals.columns:
            bins = self.intervals[col]
            if bins is None:
                continue

            col_min, col_max = bins.iloc[0], bins.iloc[-1]

            # Check for out-of-range values and adjust
            if (df[col] < col_min).any() or (df[col] > col_max).any():
                print(f"Warning: Column '{col}' contains values outside the range [{col_min}, {col_max}].")
                binned_df[col] = np.clip(df[col], col_min, col_max)
            else:
                binned_df[col] = df[col]

            # Apply binning with predefined bin edges
            binned_df[col] = pd.cut(
                binned_df[col],
                bins=bins.dropna(),
                labels=False,
                include_lowest=True
            )

        if append_discr is not None:
            binned_df = np.hstack((binned_df, append_discr))

        binned_df = pd.DataFrame(binned_df)
        if return_str:
            binned_df = binned_df.round(0).astype("Int64").astype(str)
            binned_df = binned_df.agg(','.join, axis=1).to_numpy()

        return binned_df

class KMeansDiscretizer(TimeSeriesDiscretizer):
    """
    A class that implements K-means discretization. It clusters the data values of each variable
    into a predefined number of clusters using the K-means algorithm.
    """
    def __init__(self):
        self.cluster_centers = None
        self.kmeans_models = None

    def train(self, data, number_of_clusters_per_var=10):
        """
        Train the K-means discretizer by fitting K-means models for each variable (column) in the data.
        :param data: List of DataFrames to calculate model parameters from.
        :param number_of_clusters_per_var: Number of clusters (intervals) per variable.
        """
        data = [pd.DataFrame(d) for d in data]
        combined_data = pd.concat(data, ignore_index=True)  # Combine all datasets for training
        self.kmeans_models = {}
        self.cluster_centers = {}

        # Train a K-means model for each column
        for col in combined_data.columns:
            kmeans = KMeans(n_clusters=number_of_clusters_per_var, random_state=0)
            kmeans.fit(combined_data[[col]])
            self.kmeans_models[col] = kmeans
            self.cluster_centers[col] = np.sort(kmeans.cluster_centers_.flatten())  # Sort centers for logical ordering

    def discretize(self, df, return_str=False, append_discr=None):
        """
        Discretize data into clusters determined by K-means.
        :param df: DataFrame to discretize.
        :param return_str: Whether to return discretized data as concatenated strings.
        :return: Discretized data.
        """
        df = pd.DataFrame(df)
        discretized_df = df.copy()

        for col in df.columns:
            kmeans = self.kmeans_models.get(col)
            if not kmeans:
                raise ValueError(f"No K-means model trained for column '{col}'.")
            # Assign each value to the closest cluster center
            discretized_df[col] = kmeans.predict(df[[col]])

        if append_discr is not None:
            discretized_df = np.hstack((discretized_df, append_discr))

        discretized_df = pd.DataFrame(discretized_df)
        if return_str:
            discretized_df = discretized_df.round(0).astype("Int64").astype(str)
            discretized_df = discretized_df.agg(','.join, axis=1).to_numpy()
        return discretized_df


class MultivariateKMeansDiscretizer(TimeSeriesDiscretizer):
    """
    A class that implements multivariate K-means discretization. It clusters the data
    based on all variables (columns) together into a predefined number of clusters.
    """

    def __init__(self):
        self.kmeans_model = None
        self.cluster_centers = None

    def train(self, data, number_of_clusters=10):
        """
        Train the K-means discretizer by fitting a single K-means model for all variables.
        :param data: List of DataFrames to calculate model parameters from.
        :param number_of_clusters: Number of clusters.
        """
        data = [pd.DataFrame(d) for d in data]
        combined_data = pd.concat(data, ignore_index=True)  # Combine all datasets for training
        self.kmeans_model = KMeans(n_clusters=number_of_clusters, random_state=0)
        self.kmeans_model.fit(combined_data)
        self.cluster_centers = self.kmeans_model.cluster_centers_

    def discretize(self, df, return_str=False, append_discr=None):
        """
        Discretize data into clusters determined by K-means.
        :param df: DataFrame to discretize.
        :param return_str: Whether to return discretized data as concatenated strings.
        :return: Discretized data (cluster labels).
        """
        df = pd.DataFrame(df)
        if self.kmeans_model is None:
            raise ValueError("The K-means model has not been trained yet.")

        # Predict cluster labels for the entire dataset
        cluster_labels = self.kmeans_model.predict(df)

        if append_discr is not None:
            if cluster_labels.ndim == 1:
                cluster_labels = cluster_labels.reshape(-1, 1)
            cluster_labels = np.hstack((cluster_labels, append_discr))

        cluster_labels = pd.DataFrame(cluster_labels)
        if return_str:
            cluster_labels = cluster_labels.round(0).astype("Int64").astype(str)
            cluster_labels = cluster_labels.agg(','.join, axis=1).to_numpy()
        return cluster_labels


class ThresholdDiscretizer (TimeSeriesDiscretizer):
    def discretize(self, states):
        pass


if __name__ == "__main__":
    from ml4cps import examples
    from ml4cps.discretization import catvae

    discrete_data, time_col, discrete_cols = examples.conveyor_system_sfowl(variable_type="discrete")
    cont_data, _, cont_cols = examples.conveyor_system_sfowl(variable_type="continuous")
    catvae.train_cat_vae(cont_data, )
