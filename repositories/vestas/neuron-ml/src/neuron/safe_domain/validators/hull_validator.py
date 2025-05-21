import logging
from itertools import combinations
from typing import Dict, List, Self, Tuple

import alphashape
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from neuron.safe_domain.validators.base import Validator
from neuron.safe_domain.validators.exceptions import ValidatorFittingError
from neuron.schemas.domain import Feature

logger = logging.getLogger(__name__)


class HullValidator(Validator):
    """To ensure backwards compatibility, we should avoid refactoring this class.

    If the functionality of this validator is insufficient, a new validator should be created.
    """

    def __init__(self, alpha_parameter: float = 0.0):
        self.alpha_parameter = alpha_parameter
        self.features_hull_dict_interp: Dict[str, list] = {}
        self.features_hull_dict_extrap: Dict[str, list] = {}

    def fit(self, df: pd.DataFrame, features: List[Feature]) -> None:
        """Define the alpha shape for each 2d feature combinations.
        If an extrapolation offset exists for a feature combination, an extrapolation
        alpha shape is defined."""

        feature_combinations = combinations(features, 2)
        for combination in feature_combinations:
            feature_names = [f.name for f in combination]
            points = df[feature_names].to_numpy()  # training data points
            hull = alphashape.alphashape(points, self.alpha_parameter)

            if isinstance(hull, MultiPolygon):
                raise ValidatorFittingError(
                    "Alpha parameter assignment results in multiple interpolation validation "
                    f"domains for the feature combination {feature_names}"
                )

            hull_vertices = [(val[0], val[1]) for val in zip(*hull.exterior.xy, strict=False)]

            self.features_hull_dict_interp["_combined_with_".join(feature_names)] = hull_vertices

            # Extrapolation hull
            hull_vertices_expanded = []
            hull_vertices_expanded.extend(hull_vertices)

            for i, feature in enumerate(combination):
                if feature.extrapolation_domain_offset is not None:
                    offset = feature.extrapolation_domain_offset
                    hull_vertices_positive = self.get_offset_vertices(hull_vertices, i, offset)
                    hull_vertices_negative = self.get_offset_vertices(hull_vertices, i, -offset)

                    hull_vertices_expanded.extend(hull_vertices_positive)
                    hull_vertices_expanded.extend(hull_vertices_negative)

            hull_expanded = alphashape.alphashape(hull_vertices_expanded, self.alpha_parameter)

            hull_expanded_vertices = [
                (val[0], val[1]) for val in zip(*hull_expanded.exterior.xy, strict=False)
            ]

            self.features_hull_dict_extrap[
                "_combined_with_".join(feature_names)
            ] = hull_expanded_vertices

    def validate_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if values are within the alpha shape for each 2d feature combinations.

        1: In alpha shape, 0: Out of alpha shape
        """
        df_out_interp = pd.DataFrame()
        for combination, vertices in self.features_hull_dict_interp.items():
            polygon = Polygon(vertices)
            features = combination.split("_combined_with_")

            points = [Point(x, y) for x, y in df[features].to_numpy()]
            df_out_interp[combination] = [polygon.contains(p) or polygon.touches(p) for p in points]

        return df_out_interp

    def validate_extrapolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if values are within the extrapolation alpha shape for each 2d feature combos.

        1: In alpha shape, 0: Out of alpha shape
        """
        df_out_extrap = pd.DataFrame()
        for combination, vertices in self.features_hull_dict_extrap.items():
            polygon = Polygon(vertices)
            features = combination.split("_combined_with_")

            points = [Point(x, y) for x, y in df[features].to_numpy()]
            df_out_extrap[combination] = [polygon.contains(p) or polygon.touches(p) for p in points]

        return df_out_extrap

    def get_model_data(self) -> Dict:
        return {
            "features_hull_dict_interp": self.features_hull_dict_interp,
            "features_hull_dict_extrap": self.features_hull_dict_extrap,
        }

    def _get_ranges(self, feature_name: str, method: str = "interp") -> Tuple:
        maximum_value = []
        minimum_value = []

        if method == "interp":
            safe_domain_dict_items = self.features_hull_dict_interp.items()
        elif method == "extrap":
            safe_domain_dict_items = self.features_hull_dict_extrap.items()

        for combination, vertices in safe_domain_dict_items:
            features = combination.split("_combined_with_")
            if feature_name in features:
                feature_pos = features.index(feature_name)
                maximum_value += [max(vertice[feature_pos] for vertice in vertices)]
                minimum_value += [min(vertice[feature_pos] for vertice in vertices)]

        range_values = (min(minimum_value), max(maximum_value))

        return range_values

    @classmethod
    def load(cls, model_data: Dict) -> Self:
        validator = cls()
        validator.features_hull_dict_interp = model_data["features_hull_dict_interp"]
        validator.features_hull_dict_extrap = model_data["features_hull_dict_extrap"]

        return validator

    @staticmethod
    def get_offset_vertices(
        hull_vertices: List[Tuple[float]], feature_index: int, offset: float
    ) -> List[Tuple[float, float]]:
        """Create the new coordinates for the offset hull vertices based on the
        feature index and offset value."""
        offset_vertices = []
        for vertex in hull_vertices:
            new_vertex = list(vertex)
            new_vertex[feature_index] += offset
            offset_vertices.append(tuple(new_vertex))
        return offset_vertices
