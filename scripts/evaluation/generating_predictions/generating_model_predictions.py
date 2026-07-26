from diffusers import Flux2KleinPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import load_from_disk, load_dataset
import torch
import tarfile
import os
import argparse


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
    torch_dtype=torch.bfloat16
)
    return pipeline


def generate_predictions(pipeline,prompts,dir_path):
    os.makedirs(dir_path,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    for i,prompt in enumerate(prompts[:1]):
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            image = pipeline(
            prompt=prompt,
            height=768,
            width=512,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(0)
        ).images[0]
        image.save(os.path.join(dir_path,f"{i}.png"))

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--hf-token",type=str,required=True,help="Huggingface token")
    return parser.parse_args()

def setup_env_from_args(args):

    os.environ['HF_TOKEN']=args.hf_token

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
        
    lora_weights_path="/opt/ml/processing/model/model/transformer"

    
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

    # adding lora adater
    print("Loading LoRA weights...")
    pipeline.load_lora_weights(
        lora_weights_path, 
        weight_name="adapter_model.safetensors",
        adapter_name="custom_style"
    )
    pipeline.enable_lora()


    # pipeline.set_adapters("custom_style")
    # pipeline.enable_model_cpu_offload()  

    print("Generating LoRA predictions...")
    generate_predictions(pipeline,prompts,lora_model_path)

    pipeline.disable_lora()

    #loading base model  predictions
    print("Generating Base model predictions...")
    # generate_predictions(pipeline,prompts,base_model_path)

if __name__=="__main__":
    main()