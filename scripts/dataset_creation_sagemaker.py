from sagemaker.session import Session
from sagemaker.processing import ProcessingOutput
from sagemaker.huggingface import HuggingFaceProcessor
import os
from dotenv import load_dotenv

load_dotenv()

session = Session()
role = "arn:aws:iam::063876841214:role/SagemakerRole"

hf_processor = HuggingFaceProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    transformers_version="4.56.2",
    pytorch_version="2.8.0",
    py_version="py312",
    sagemaker_session=session,
)

hf_processor.run(
    code="scripts/dataset_creation/1_dataset_creation.py",
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{os.environ['bucket']}/Data/alphonse_mucha_encoded_dataset",
        )
    ],
    arguments=[
        "--hf-token",
        os.environ["HF_TOKEN"],
        "--model-id",
        "black-forest-labs/FLUX.2-klein-9B",
        "--source-dataset",
        "derekl35/alphonse-mucha-style",
        "--cache-dir",
        "/opt/ml/processing/cache",
    ],
)
