from sagemaker.session import Session
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
import os
from dotenv import load_dotenv

load_dotenv()

session = Session()
role = "arn:aws:iam::063876841214:role/SagemakerRole"


# using sklearn processor because i just need load_dataset lib which will be downloaded from requirements.txt
hf_processor = SKLearnProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    framework_version="1.2-1",
    sagemaker_session=session,
)

hf_processor.run(
    code="scripts/evaluation/test_set_generation/creating_test_set.py",
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{os.environ['bucket']}/Data/alphonse_mucha_test_set",
        )
    ],
    arguments=[
        "--HF_TOKEN",
        os.environ["HF_TOKEN"],
    ],
)
