from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

AVAILABLE_MODELS = {
    "knn": KNeighborsClassifier,
    "ncc": NearestCentroid
}

REQUIRED_ARGUMENTS = {
    "knn": ["n_neighbors"],
    "ncc": []
}

ARGUMENT_ALIASES = {
    "n_neighbors": "k"
}

class SKLearnModel():
    def __init__(self, model_type, args):
        super(SKLearnModel, self).__init__()
        
        if model_type not in AVAILABLE_MODELS.keys(): raise ValueError(f"Model type: {model_type} not implemented.")
        self.model_type = model_type
        self.args = args

        self.model = AVAILABLE_MODELS[self.model_type](**self.args)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    @staticmethod
    def get_required_arguments_per_model(model_type):
        return REQUIRED_ARGUMENTS[model_type]
    
    @staticmethod
    def get_argument_aliases():
        return ARGUMENT_ALIASES