import os
from typing import Optional
import time
import json

from tqdm import tqdm
import pandas as pd
import turboml as tb
from turboml.common.api import api
from loguru import logger

from config import config

# Establish connection to TurboML platform
tb.init(
    backend_url=config.turboml_backend_url,
    api_key=config.turboml_api_key,
)

def simulate_live_stream(
    df: pd.DataFrame,
    batch_size: int,
    sleep_sec: int,
    from_timestamp_sec: int,
):
    """
    Yields the given `df` in batches of `batch_size` elements, with a delay of `sleep_sec` seconds
    between batches
    """
    total_length = len(df)
    for start_idx in range(0, total_length, batch_size):
        end_idx = min(start_idx + batch_size, total_length)
        batch = df.iloc[start_idx:end_idx]

        if 'timestamp' in batch.columns:
            # Shift the `timestamp` of the batch
            batch['timestamp'] += from_timestamp_sec
        
        yield batch

        time.sleep(sleep_sec)

def push_transactions_and_labels_to_online_datasets(
    transactions_dataset_name: str,
    labels_dataset_name: str,
    batch_size: int,
    every_n_seconds: int,
):
    """
    Pushes transactions and labels into TurboML online datasets, by sampling historical
    data (aka push-based ingestion)

    In a real-world setting, live data is ingested by TurboML from places like
    - AWS S3 bucket
    - PostgreSQL database
    - Google Cloud Pub/Sub
    using so-called connectors (aka pull-based ingestion)
    https://docs.turboml.com/intro/#pull-based-ingestion

    Args:
        transactions_dataset_name (str): Name of the transactions dataset in TurboML platform
        labels_dataset_name (str): Name of the labels dataset in TurboML platform
        batch_size (int): Number of transactions to push in each batch
        every_n_seconds (int): Number of seconds to wait between consecutive batches
    """    
    # Load the fraud detection datasets
    logger.info('Load fraud detection datasets into pandas')
    transactions_df = tb.datasets.FraudDetectionDatasetFeatures().df
    labels_df = tb.datasets.FraudDetectionDatasetLabels().df
    
    # Normalize the dataframes to replace nan's with 0-values for each type
    logger.info('Normalize the dataframes')
    transactions_df = tb.datasets.PandasHelpers.normalize_df(transactions_df)
    labels_df = tb.datasets.PandasHelpers.normalize_df(labels_df)

    # Simulate live stream of transactions and labels dataframes, starting from now
    from_timestamp_sec = int(time.time())
    transactions_stream = simulate_live_stream(transactions_df, batch_size, every_n_seconds, from_timestamp_sec)
    labels_stream = simulate_live_stream(labels_df, batch_size, every_n_seconds, from_timestamp_sec)

    with tqdm(
        total=len(transactions_df),
        desc="Pushing transactions to online datasets",
        unit="transactions",
        unit_scale=True
    ) as pbar:
        for transactions_batch, labels_batch in zip(
            transactions_stream, labels_stream, strict=True
        ):
            upload_df_to_online_dataset(transactions_dataset_name, transactions_batch)
            upload_df_to_online_dataset(labels_dataset_name, labels_batch)
            pbar.update(len(transactions_batch))

def upload_df_to_online_dataset(dataset_id: str, df: pd.DataFrame):
    """
    Used the TurboML REST API to push a pandas dataframe to an online dataset.
    This is a synchronous operation, meaning that the function will block until the
    data is pushed to the online dataset.
    """
    row_list = json.loads(df.to_json(orient="records"))
    api.post(f"dataset/{dataset_id}/upload", json=row_list)
    
if __name__ == '__main__':
    from config import config
    push_transactions_and_labels_to_online_datasets(
        transactions_dataset_name=config.transactions_dataset_name,
        labels_dataset_name=config.labels_dataset_name,
        batch_size=100,
        every_n_seconds=0.1,
    )