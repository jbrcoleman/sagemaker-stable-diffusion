"""
This script deploys a text to image stable diffusion model
to AWS SageMaker. This model can be used for inference
after deployment.
"""

import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base

aws_role = get_execution_role()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()


model_id, model_version = " model-txt2img-stabilityai-stable-diffusion-v1-4", "*"

endpoint_name = name_from_base(f"jumpstart-example-infer-{model_id}")

# Use ml.g5.24xlarge instance type
inference_instance_type = "ml.g4dn.2xlarge"

# Get the inference docker container uri.
deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,  # automatically inferred from model_id
    image_scope="inference",
    model_id=model_id,
    model_version=model_version,
    instance_type=inference_instance_type,
)

# Get the inference script uri.
deploy_source_uri = script_uris.retrieve(
    model_id=model_id, model_version=model_version, script_scope="inference"
)


# Get the model uri.
model_uri = model_uris.retrieve(
    model_id=model_id, model_version=model_version, model_scope="inference"
)

# To increase the maximum response size from the endpoint.
env = {
    "MMS_MAX_RESPONSE_SIZE": "20000000",
}

# Create the SageMaker model instance
model = Model(
    image_uri=deploy_image_uri,
    source_dir=deploy_source_uri,
    model_data=model_uri,
    entry_point="inference.py",  # entry point file in source_dir and present in deploy_source_uri
    role=aws_role,
    predictor_cls=Predictor,
    name=endpoint_name,
    env=env,
)

# deploy the Model. Note that we need to pass Predictor class when we deploy model through,
# Model class, for being able to run inference through the sagemaker API.
model_predictor = model.deploy(
    initial_instance_count=1,
    instance_type=inference_instance_type,
    predictor_cls=Predictor,
    endpoint_name=endpoint_name,
)
