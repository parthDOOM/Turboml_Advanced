from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

class Config(BaseSettings):
    # so environment variables can set secrets and override defaults for non-secrets
    model_config = SettingsConfigDict(env_file=".env")

    transactions_dataset_name: str = "qs_transactions"
    labels_dataset_name: str = "qs_transaction_labels"
    model_name: str = "fraud_detection_model_3"

    # secrets
    turboml_backend_url: str
    turboml_api_key: str
    
config = Config()