import pandas as pd

from neuron.safe_domain.safe_domain_validator import SafeDomainValidator
from neuron.schemas.domain import Feature, FeatureType, FeatureValueType, SafeDomainGroup


def test_valid_data() -> None:
    # Create training data
    training_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [1.1, 2.2, 3.3]})

    # Create test data
    test_data = pd.DataFrame({"feature1": [1, 2], "feature2": [1.1, 2.2]})

    # Define features
    features = [
        Feature(
            name="feature1",
            feature_value_type=FeatureValueType.DISCRETE,
            safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="feature2",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
    ]

    # Create validator
    validator = SafeDomainValidator(features)
    validator.fit(training_data)

    # Validate test data
    results_interp = validator.validate_interpolation(test_data)
    results_extrap = validator.validate_extrapolation(test_data)

    # Check validation status of each feature
    assert (results_interp["feature1"] == 1).all()
    assert (results_interp["feature2"] == 1).all()
    assert (results_extrap["feature1"] == 1).all()
    assert (results_extrap["feature2"] == 1).all()


def test_invalid_data() -> None:
    # Create training data
    training_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [1.1, 2.2, 3.3]})

    # Create test data
    test_data = pd.DataFrame({"feature1": [-1, 5], "feature2": [0.2, 5.5]})

    # Define features
    features = [
        Feature(
            name="feature1",
            feature_value_type=FeatureValueType.DISCRETE,
            safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="feature2",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
            feature_type=FeatureType.RAW,
            extrapolation_domain_offset=0.1,
            is_model_input=True,
        ),
    ]

    # Create validator
    validator = SafeDomainValidator(features)
    validator.fit(training_data)

    # Validate test data
    results_interp = validator.validate_interpolation(test_data)
    results_extrap = validator.validate_extrapolation(test_data)

    # Check validation status of each feature
    assert (results_interp["feature1"] == 0).all()
    assert (results_interp["feature2"] == 0).all()
    assert (results_extrap["feature1"] == 0).all()
    assert (results_extrap["feature2"] == 0).all()


def test_mix_data() -> None:
    # Create training data
    training_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [1.1, 2.2, 3.3]})

    # Create test data
    test_data = pd.DataFrame({"feature1": [1, 5], "feature2": [-1000, 2.0]})

    # Define features
    features = [
        Feature(
            name="feature1",
            feature_value_type=FeatureValueType.DISCRETE,
            safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="feature2",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
            extrapolation_domain_offset=0.1,
        ),
    ]

    # Create validator
    validator = SafeDomainValidator(features)
    validator.fit(training_data)

    # Validate test data
    results_interp = validator.validate_interpolation(test_data)

    # Check validation status of each feature
    assert results_interp["feature1"][0] == 1
    assert results_interp["feature1"][1] == 0
    assert results_interp["feature2"][0] == 0
    assert results_interp["feature2"][1] == 1


def test_save_load(tmp_path_factory) -> None:  # noqa: ANN001
    # Create training data

    temp_dir = tmp_path_factory.mktemp("temp_safe_domain_dir")
    temp_file_path = temp_dir / "validator.json"

    training_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [1.1, 2.2, 3.3]})

    # Create test data
    test_data = pd.DataFrame({"feature1": [1, 5], "feature2": [-1000, 2.0]})

    # Define features
    features = [
        Feature(
            name="feature1",
            feature_value_type=FeatureValueType.DISCRETE,
            safe_domain_method=SafeDomainGroup.DISCRETE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
        ),
        Feature(
            name="feature2",
            feature_value_type=FeatureValueType.CONTINUOUS,
            safe_domain_method=SafeDomainGroup.RANGE_VALIDATION,
            feature_type=FeatureType.RAW,
            is_model_input=True,
            extrapolation_domain_offset=0.1,
        ),
    ]

    # Create validator
    validator = SafeDomainValidator(features)
    validator.fit(training_data)

    res_interp = validator.validate_interpolation(test_data)
    res_extrap = validator.validate_extrapolation(test_data)

    validator.save(str(temp_file_path))
    loaded_validator = SafeDomainValidator.load(str(temp_file_path))
    loaded_res_interp = loaded_validator.validate_interpolation(test_data)
    loaded_res_extrap = loaded_validator.validate_extrapolation(test_data)

    assert (res_interp == loaded_res_interp).all().all()
    assert (res_extrap == loaded_res_extrap).all().all()
