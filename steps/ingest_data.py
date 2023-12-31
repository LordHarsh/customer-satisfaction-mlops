import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingests data from a given path and returns a pandas DataFrame.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data to be ingested.
        """
        self.url = data_path
        
    def get_data(self):
        """
            Ingests data from a given path and returns a pandas DataFrame.
        """
        logging.info(f"Ingesting Data from ${self.url}")
        return pd.read_csv(self.url)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a given path and returns a pandas DataFrame.

    Args:
        data_path: path to the data to be ingested.
    Returns:
        pd.DataFrame: the ingested data.
    """
    
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in ingesting data: {e}")
        raise e