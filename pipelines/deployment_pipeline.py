import numpy as np
import pandas as pd
import json
# from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from .utils import get_data_for_test
docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0
    
@step(enable_cache=False)
def dynamic_importer() -> str:
    """Imports the custom materializer."""
    data = get_data_for_test()
    return data
@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Triggers the deployment if the accuracy is above the threshold."""
    return accuracy >= config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLFlow Deployment Loader Step Parameters
    Attributes:
        pipeline_name: Name of the pipeline.
        step_name: Name of the step.
        running: when the flag is set, it retuen the running services
        model_name: Name of the model.
    """
    pipeline_name: str
    step_name: str
    running: bool = True
    model_name: str
    
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
)-> MLFlowDeploymentService:
    """Loads the prediction service.

    Args:
        pipeline_name (str): _description_
        pipeline_step_name (str): _description_
        running (bool, optional): _description_. Defaults to True.
        model_name (str, optional): _description_. Defaults to "model".
    Returns:
        MLFlowDeploymentService: _description_
    """
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=running,
        model_name=model_name
    )
    if not existing_services:
        raise RuntimeError(
            f"No mlflow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}."
            f"Pipelines for model {model_name} is currently running"
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Predicts the data using the service.

    Args:
        service (MLFlowDeploymentService): _description_
        data (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    service.start(timeout=30)
    data = json.loads(data)
    data.pop('columns')
    data.pop('index')
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]
    df = pd.DataFrame(data['data'], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, mse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )
    
    
@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False
    )
    data = dynamic_importer()
    prediction = predictor(service=service, data=data)
    return prediction