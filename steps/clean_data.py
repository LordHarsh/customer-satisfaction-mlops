import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"], 
    Annotated[pd.DataFrame, "X_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans the data.
    
    Args:
        df: The input DataFrame.
    Returns:
        X_train: The training features.
        X_test: The test features.
        y_train: The training labels.
        y_test: The test labels.
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e