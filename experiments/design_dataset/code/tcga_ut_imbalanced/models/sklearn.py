import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

AVAILABLE_MODELS = {"knn": KNeighborsClassifier, "ncc": NearestCentroid}
REQUIRED_ARGUMENTS = {"knn": ["n_neighbors"], "ncc": []}
ARGUMENT_ALIASES = {"n_neighbors": "k"}


class SKLearnModel:
    def __init__(self, model_type: str, args: dict[str, int]) -> None:
        super().__init__()
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(f"Model type: {model_type} not implemented.")
        self.model_type = model_type
        self.args = args
        self.model = AVAILABLE_MODELS[self.model_type](**self.args)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit the wrapped scikit-learn model."""
        self.model.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels with the wrapped scikit-learn model."""
        return self.model.predict(features)

    @staticmethod
    def get_required_arguments_per_model(model_type: str) -> list[str]:
        """Return constructor argument names required by a model type."""
        return REQUIRED_ARGUMENTS[model_type]

    @staticmethod
    def get_argument_aliases() -> dict[str, str]:
        """Return CLI argument aliases for model constructor arguments."""
        return ARGUMENT_ALIASES
