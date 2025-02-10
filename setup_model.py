import turboml as tb
from loguru import logger

from config import config

# Establish connection to TurboML platform
tb.init(
    backend_url=config.turboml_backend_url,
    api_key=config.turboml_api_key,
)


def setup_model(
    model_name: str,
    transactions_dataset_name: str,
    labels_dataset_name: str,
) -> str:
    """
    Deploys a streaming ML job. This means we define the model and push it to the TurboML platform.
    
    From that moment, the model is continuously

    - listening to new incoming transactions and genearating predictions, that are saved
    into another online dataset (aka Kafka topic)
    
    - listening to new incoming labels (fraud/not-fraud) for each recent transaction.
    These paris are used to incrementally adjust the model (aka incremental model training)
    Not all ML models support incremental training.

    Args:
        model_name (str): Name of the model in the TurboML platform
        transactions_dataset_name (str): Name of the online dataset where the model is listening to transactions.
        labels_dataset_name (str): Name of the online dataset where the model is listening to labels.

    Returns:
        str: The URL of the deployed model (for synchronous models)
    """
    # Get the transactions and labels datasets from TurboML platform
    logger.info(f"Connect to transactions dataset {transactions_dataset_name}")
    transactions = tb.OnlineDataset.load(dataset_id=transactions_dataset_name)
    logger.info(f"Connect to labels dataset {labels_dataset_name}")
    labels = tb.OnlineDataset.load(dataset_id=labels_dataset_name)
    
    # Define the model
    model = tb.HoeffdingTreeClassifier(n_classes=2)
    # Here you can use play with other models. To check all the available models for
    # Supervised ML, you can use the following command:
    #   tb.ml_algorithms(have_labels=True)
    #
    # Alternatively, you can define your own model in Python
    # https://docs.turboml.com/wyo_models/native_python_model/
    #
    # and even train it on a batch of data, before pushing it to the platform
    # https://docs.turboml.com/wyo_models/batch_python_model/

    numerical_fields = [
        "transactionAmount",
        "localHour",
        "my_sum_feat",
    ]
    categorical_fields = [
        "digitalItemCount",
        "physicalItemCount",
        "isProxyIP",
    ]
    features = transactions.get_model_inputs(
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields
    )
    label = labels.get_model_labels(label_field="is_fraud")

    logger.info(f"Try to deploy model {model_name} to platform")
    try:
        deployed_model = model.deploy(model_name, input=features, labels=label)
        logger.info(f"Deployment of {model_name} completed!")

        logger.info(f"Get the URL of the deployed model")
        model_endpoints = deployed_model.get_endpoints()
        logger.info(f"Model endpoints: {model_endpoints}")

    except Exception as e:
        logger.error(f"Failed to re-deploy model {model_name} to platform because the model with name {model_name} already exists.")
        logger.error(e)

if __name__ == '__main__':
    
    setup_model(
        model_name=config.model_name,
        transactions_dataset_name=config.transactions_dataset_name,
        labels_dataset_name=config.labels_dataset_name,
    )