import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from typing import Union

class Model(ABC):
    """
    Abstract class for model.
    """
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Abstract method for training model.
        Args:
            X_train: The training features.
            y_train: The training labels.
        Returns:
            None
        """
        pass
    
class LinearRegressionModel(Model):
    """
        Linear Regression model.
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model.
        Args:
            X_train: The training features.
            y_train: The training labels.
            **kwargs: Additional arguments.
        Returns:
            None
        """
        try:
            logging.info("Training Linear Regression model...")
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained.")
            return reg
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise e
        return None