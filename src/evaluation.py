import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class Evaluation(ABC):
    """
    Abstract class for evaluation.
    """
    
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Abstract method for evaluating model.
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        Returns:
            None
        """
        pass
    
class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error.
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates MSE.
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        Returns:
            None
        """
        try:
            logging.info("Calculating MSE...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e
        return None
    
class R2(Evaluation):
    """
    Evaluation Strategy that uses R2.
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates R2.
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        Returns:
            None
        """
        try:
            logging.info("Calculating R2...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2: {e}")
            raise e

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error.
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates RMSE.
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        Returns:
            None
        """
        try:
            logging.info("Calculating RMSE...")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e
        return None