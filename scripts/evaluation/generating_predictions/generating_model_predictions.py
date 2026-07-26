from diffusers import Flux2KleinPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import load_from_disk, load_dataset
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
import torch
import tarfile
import os
import argparse
import subprocess


pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
        "text_encoder": TransformersBitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
    }
)

def load_pipeline():
    pipeline = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
    return pipeline

def print_gpu_info() -> None:
    """Print GPU information if nvidia-smi is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  nvidia-smi not found or failed – running on CPU.")


def generate_predictions(pipeline,prompts,dir_path,n_step=4):
    os.makedirs(dir_path,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    for i,prompt in enumerate(prompts):
        with torch.autocast(device,dtype=torch.bfloat16):
            image = pipeline(
            prompt=prompt,
            height=768,
            width=512,
            guidance_scale=1.0,
            num_inference_steps=n_step,
            generator=torch.Generator(device=device).manual_seed(0)
        ).images[0]
        image.save(os.path.join(dir_path,f"{i}.png"))

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--hf-token",type=str,required=True,help="Huggingface token")
    return parser.parse_args()

def setup_env_from_args(args):

    os.environ['HF_TOKEN']=args.hf_token

def reset_model_lora_alpha(model, lora_alpha: float, adapter: str):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            reset_lora_alpha(module, lora_alpha, adapter)


def reset_lora_alpha(lora_layer, lora_alpha: float, adapter: str):
    if adapter not in lora_layer.active_adapters:
        return

    lora_layer.lora_alpha[adapter] = lora_alpha
    if lora_layer.r[adapter] > 0:
        lora_layer.scaling[adapter] = (lora_layer.lora_alpha[adapter]
                                       / lora_layer.r[adapter])

def main():
    parser = parse_args()
    setup_env_from_args(parser)

    test_set_path="/opt/ml/processing/test_set"
    base_model_path="/opt/ml/processing/output/base_model_predictions"
    lora_model_path="/opt/ml/processing/output/lora_model_predictions"
    model_tar_path="/opt/ml/processing/model/model.tar.gz"

    # unzipping the model
    print("Unzipping the model...")
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path="/opt/ml/processing/model")
        
    lora_weights_path="/opt/ml/processing/model/model/50/transformer"

    
    # loading base pipeline
    print("Loading base model...")
    pipeline = load_pipeline()
    

    # confirm the path of test set
    print("Loading test set...")
    if test_set_path:
        dataset_path=test_set_path
        dataset = load_from_disk(dataset_path)
        prompts=[item['prompt'] for item in dataset]
        print("Loaded dataset from disk")
    else:
        dataset = load_dataset("Pankaj121212/alphonse_mucha_test_set")
        prompts=[item['prompt'] for item in dataset['train']]
        print("Loaded dataset from hf cloud")

    print("Generating Base model predictions...")
    generate_predictions(pipeline,prompts,base_model_path)
    
    print("Loading LoRA weights...")
    pipeline.load_lora_weights(
        lora_weights_path, 
        weight_name="adapter_model.safetensors",
        adapter_name="custom_style"
    )
    lora_config=LoraConfig.from_pretrained(lora_weights_path)
    reset_model_lora_alpha(pipeline.transformer,lora_config.lora_alpha,"custom_style")

    print_gpu_info()
    print("Generating LoRA predictions...")
    generate_predictions(pipeline,prompts,lora_model_path,n_step=10)
    

if __name__=="__main__":
    main()