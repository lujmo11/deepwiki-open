from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from neuron.data_splitting.base import DataSplitter
from neuron.utils import set_seed


class FeatureBasedSplitter(DataSplitter):
    name = "feature_group_split"

    def __init__(
        self,
        grouping_features: List[str],
        test_size: float = 0.2,
    ):
        self.test_size = test_size
        self.grouping_features = grouping_features

    def train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_data = data[self.grouping_features].drop_duplicates()
        boundary_data, inner_data = self.compute_data_feature_boundary(data=feature_data)

        inner_data_ratio = len(inner_data) / len(
            feature_data
        )  # verify if there's enough data for testing

        if inner_data_ratio < self.test_size:
            raise ValueError(
                f"Got a data ratio of {inner_data_ratio} for testing \
                    and the minimum required is {self.test_size}"
            )
        else:
            adjusted_test_size = self.test_size * len(feature_data) / len(inner_data)
            subset_train_features, test_features = self._kmeans_sample(
                data=inner_data, test_size=adjusted_test_size
            )
            train_features = pd.concat([subset_train_features, boundary_data])

            train_df = data.merge(train_features, on=self.grouping_features)
            test_df = data.merge(test_features, on=self.grouping_features)

        return train_df, test_df

    def compute_data_feature_boundary(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(data) + 1 < len(self.grouping_features):
            raise ValueError(
                "The limited number of unique combinations does not permit \
                the definition of a convex shape."
            )

        points = np.array(data[self.grouping_features])
        hull = ConvexHull(points)

        if len(hull.vertices) == len(data):
            raise ValueError(
                "The selected features describe a convex shape, leaving no inner data for testing."
            )

        boundary_mask = [i in hull.vertices for i in range(len(data))]
        boundary_data = data[boundary_mask]
        inner_data = data[~np.array(boundary_mask)]

        return boundary_data, inner_data

    def _kmeans_sample(
        self, data: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n_test_size = int(test_size * len(data))
        """
        Kmeans clustering is used to achieve an even distribution of cluster centers. 
        (Previously a random sample was used which could result in an uneven distribution)
        For each cluster of datapoints, the datapoint closest to the center is selected 
        as a test datapoint.
        """

        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_test_size, random_state=set_seed())
        kmeans.fit(data_scaled)
        cluster_labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        test_data_indices = []

        for i, center in enumerate(centers):
            # cluster datapoints
            cluster_data_indices = np.where(cluster_labels == i)[0]
            cluster_data = data_scaled[cluster_data_indices]

            # euclidean distance between cluster datapoints and center
            distances = np.zeros(len(cluster_data))
            for j, datapoint in enumerate(cluster_data):
                distances[j] = np.linalg.norm(datapoint - center)

            # find index of datapoint with minimum distance to the center
            closest_point_index = cluster_data_indices[np.argmin(distances)]
            test_data_indices.append(closest_point_index)

        test_data = data.iloc[test_data_indices]
        train_data = data.drop(test_data.index)

        return train_data, test_data

    def validate_params(self) -> bool:
        valid_test_size = 0 < self.test_size < 1
        # There is no good way of testing if the names in the grouping_features are valid
        # before we have the data, so we just check if they are strings
        valid_grouping_features = all(
            [isinstance(feature, str) for feature in self.grouping_features]
        )
        return valid_test_size and valid_grouping_features
