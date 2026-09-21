from typing import List, Union

import torch


class ModelOutput:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def list_of_row_format(self):
        """
        Function used to convert the predictions into a format that can be written to BigQuery.
        """
        raise NotImplementedError

    def _convert_to_list(self, prediction: Union[torch.Tensor, List]) -> List:
        """
        Convert the prediction to a list so it can be serialized.
        """
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().tolist()

        return prediction


class SharedKeyAcrossPredictionsOutput(ModelOutput):
    """
    A class to represent the output of a model with a single key for all predictions in a batch.

        Attributes:
        key: The single key associated with all the predictions.
        predictions: The predictions made by the model.
        key_name (str): The name of the key attribute. Default is "idx".
        prediction_name (str): The name of the prediction attribute. Default is "prediction".
    """

    def __init__(
        self,
        key,
        predictions,
        key_name: str = "idx",
        prediction_name: str = "prediction",
    ):
        self.key = key
        self.predictions = predictions
        self.key_name = key_name
        self.prediction_name = prediction_name

    @property
    def list_of_row_format(self):
        return [
            {self.key_name: self.key, self.prediction_name: pred}
            for pred in self._convert_to_list(self.predictions)
        ]


class OneKeyPerPredictionOutput(ModelOutput):
    """
    A class used to represent the output of a model where each prediction is associated with a unique key.
    Attributes
    ----------
    keys : Any
        The keys associated with each prediction.
    predictions : Any
        The predictions made by the model.
    key_name : str, optional
        The name to be used for the key in the output dictionary (default is "idx").
    prediction_name : str, optional
        The name to be used for the prediction in the output dictionary (default is "prediction").
    """

    def __init__(
        self,
        keys,
        predictions,
        key_name: str = "idx",
        prediction_name: str = "prediction",
    ):
        self.keys = keys
        self.predictions = predictions
        self.key_name = key_name
        self.prediction_name = prediction_name

    @property
    def list_of_row_format(self):
        return [
            {self.key_name: key, self.prediction_name: pred}
            for key, pred in zip(
                self._convert_to_list(self.keys),
                self._convert_to_list(self.predictions),
            )
        ]