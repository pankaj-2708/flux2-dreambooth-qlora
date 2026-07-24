from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv
import os

load_dotenv()

role = "arn:aws:iam::063876841214:role/SagemakerRole"

pytorch_trainer=PyTorch(
    entry_point="dreambooth_qlora_train.py",
    source_dir="scripts/dreambooth_qlora/",
    role=role,
    framework_version="2.8",
    py_version="py312",
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    hyperparameters={
        "hf_token": os.environ["HF_TOKEN"],
        "cache_dir": "/opt/ml/processing/cache",
        "epochs":20
    },
    output_path="s3://pankaj-flux2klein/checkpoints/"
)

pytorch_trainer.fit({
    "train": "s3://pankaj-flux2klein/Data/alphonse_mucha_encoded_dataset"
})