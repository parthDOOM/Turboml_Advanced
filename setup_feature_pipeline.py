from typing import Optional

import turboml as tb
from loguru import logger

from config import config

# Establish connection to TurboML platform
tb.init(
    backend_url=config.turboml_backend_url,
    api_key=config.turboml_api_key,
)

def create_datasets(
    transactions_dataset_name: str,
    labels_dataset_name: str,
    n_samples: Optional[int] = None,
) -> tuple[tb.OnlineDataset, tb.OnlineDataset]:
    """
    Pushes and initial set of (raw) data (transactions, labels) into the TurboML platform,
    as online datasets.

    Args:
        - transactions_dataset_name (str): Name of the transactions dataset in TurboML platform
        - labels_dataset_name (str): Name of the labels dataset in TurboML platform
    
    Returns:
        - transactions (tb.OnlineDataset): TurboML OnlineDataset object for transactions
        - labels (tb.OnlineDataset): TurboML OnlineDataset object for labels
    """
    # Load the fraud detection datasets
    logger.info('Load fraud detection datasets into pandas')
    transactions_df = tb.datasets.FraudDetectionDatasetFeatures().df
    labels_df = tb.datasets.FraudDetectionDatasetLabels().df

    if n_samples is not None:
        logger.info(f"Subsample datasets to have {n_samples} samples")
        transactions_df = transactions_df.head(n_samples)
        labels_df = labels_df.head(n_samples)

    logger.info(f'transactons_df has {len(transactions_df)} rows')
    logger.info(f'labels_df has {len(labels_df)} rows')

    # Push datasets to TurboML platform
    logger.info('Push datasets to TurboML platform.')
    transactions = tb.OnlineDataset.from_pd(
        id=transactions_dataset_name,
        df=transactions_df,
        key_field="transactionID",
        load_if_exists=True,
    )
    labels = tb.OnlineDataset.from_pd(
        id=labels_dataset_name,
        df=labels_df,
        key_field="transactionID",
        load_if_exists=True,
    )
    logger.info('Successfully pushed transactions and labels datasets to TurboML platform!')

    return transactions, labels

def get_feature_names(transactions: tb.OnlineDataset) -> list[str]:
    """
    Returns list of materialized feature names for the given `transactions` dataset.

    Args:
        - transactions (tb.OnlineDataset): TurboML OnlineDataset object for transactions

    Returns:
        - list[str]: List of materialized feature names
    """
    return transactions.feature_engineering.get_materialized_features().columns.tolist()

def define_feature_engineering(
    transactions: tb.OnlineDataset, 
):
    """
    Defines the feature engineering pipeline that will be used to map this raw data into
    ML model features, that will be used to compute fresh predictions in real-time.

    """
    # We need to tell TurboML how to measure time, before we can do time-windows aggreagations
    transactions.feature_engineering.register_timestamp(
        column_name="timestamp", format_type="epoch_seconds"
    )
    
    # Add feature definitions to the dataset, that will be used by the TurboML platform
    # to generate ML model features in real time from the datasets
    # In this case, a time-window aggregation feature is created, with total transaction volume
    if "my_sum_feat" in get_feature_names(transactions):
        logger.info('Feature "my_sum_feat" already existed in the dataset. Nothing to do')
        return
    
    logger.info('Add feature definitions to the dataset.')
    transactions.feature_engineering.create_aggregate_features(
        column_to_operate="transactionAmount",
        column_to_group="accountID",
        operation="SUM",
        new_feature_name="my_sum_feat",
        timestamp_column="timestamp",
        window_duration=24,
        window_unit="hours",
    )

    # Submit these feature definitions to the TurboML platform so that this can be
    # computed continously for the input data stream.
    logger.info("Submit materialization job to the platform")
    transactions.feature_engineering.materialize_features(["my_sum_feat"])
    

if __name__ == '__main__':
    
    transactions, labels = create_datasets(
        transactions_dataset_name=config.transactions_dataset_name,
        labels_dataset_name=config.labels_dataset_name,
        n_samples=100,
    )

    define_feature_engineering(transactions)