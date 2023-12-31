import logging
from abc import ABC, abstractmethod

import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split
import pandas as pd

class DataStrategy(ABC):
    """
    Abstract class for data cleaning strategy.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method for handling data.
        """
        pass
    
class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for pre-processing data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-processes data.
        """
        try:
            logging.info("Pre-processing data...")
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace = True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace = True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace = True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace = True)
            data['review_comment_message'].fillna('No review', inplace = True)
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error pre-processing data: {e}")
            raise e
        return data

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test splits.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Divides data into train and test splits.
        """
        try:
            X = data.drop(['review_score'], axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error dividing data: {e}")
            raise e
        
class DataCleaning:
    """
    Class for cleaning data which pre-processes the data and divides it into train and test splits.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error handling data: {e}")
            raise e
        