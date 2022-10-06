from typing import Dict, List

import pathlib
import os

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model = None

    def load(self):
        import joblib
        # Load model here and assign to self._model.
        parent_path = pathlib.Path('model.joblib').parent.resolve() # <- Use this path if the Truss is running in a container
        model_path = os.path.join(parent_path , "ner-truss", "data", "model.joblib") # <- Use this path if you want to run locally.
        #smodel_path = os.path.join(parent_path , "data", "model.joblib")
        self._model = joblib.load(model_path)

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        inputs = request["inputs"]  # noqa

        # Invoke model and calculate predictions here.
        results = self._model.predict(inputs)
        response["predictions"] = results
        return response
