from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput,ProcessingOutput
from sagemaker.session import Session
from dotenv import load_dotenv
import os

load_dotenv()


role = "arn:aws:iam::063876841214:role/SagemakerRole"
session = Session()

hf_processor= HuggingFaceProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    transformers_version="4.56.2",
    pytorch_version="2.8.0",
    py_version="py312",
    sagemaker_session=session,
)

hf_processor.run(
    source_dir="scripts/evaluation/generating_predictions",
    code="generating_model_predictions.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{os.environ['bucket']}/Data/alphonse_mucha_test_set",
            destination="/opt/ml/processing/test_set",
        ),
        ProcessingInput(
            source=f"s3://pankaj-flux2klein/checkpoints/pytorch-training-2026-07-26-06-57-36-923/output/model.tar.gz",
            destination="/opt/ml/processing/model",
        )

    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{os.environ['bucket']}/Data/predictions",
        )
    ],
    arguments=[
        "--hf-token",
        os.environ["HF_TOKEN"]
    ],
)
