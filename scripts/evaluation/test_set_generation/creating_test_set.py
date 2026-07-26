import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "numpy","-y"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0.0", "datasets"])


from datasets import Dataset
import argparse
import os

mucha_prompts = [
    # training prompts
    {'prompt': 'Red-haired woman amid grapevines with bowl, alphonse mucha style'},
    {'prompt': 'Two figures surrounded by white lilies, alphonse mucha style'},
    {'prompt': 'Princess with blossom crown and star halo, alphonse mucha style'},
    {'prompt': 'Hooded woman beside snow-covered branches, alphonse mucha style'},
    {'prompt': 'Daydreaming nude framed by gothic arch, alphonse mucha style'},
    {'prompt': 'Reader with floral backdrop and book, alphonse mucha style'},
    {'prompt': 'Regal figure in gold robes with palm, alphonse mucha style'},
    {'prompt': 'Smoky-haired woman against purple lettering, alphonse mucha style'},
    {'prompt': 'Redhead in white gown before starry sky, alphonse mucha style'},
    {'prompt': 'Woman toasting beneath ornate champagne emblem, alphonse mucha style'},
    {'prompt': 'Slavic maiden with wreath, orb, and eagle, alphonse mucha style'},
    {'prompt': 'Blonde figure weaving blossoms on flowering branches, alphonse mucha style'},
    {'prompt': 'Redhead with poppy crown reclining among sunset vines, alphonse mucha style'},


    # new prompts
    {"prompt": "Blonde woman weaving blossoms on flowering branches, alphonse mucha style"},
    {"prompt": "Serene raven-haired woman, moonlit lilies, swirling botanicals, alphonse mucha style"},
    {"prompt": "a puppy in a pond, alphonse mucha style"},
    {"prompt": "Ornate fox with a collar of autumn leaves and berries, amidst a tapestry of forest foliage, alphonse mucha style"},
    {"prompt": "Slavic maiden holding a wreath and golden orb, alphonse mucha style"},
    {"prompt": "Ornate fox with a collar of autumn leaves, alphonse mucha style"},
    {"prompt": "Majestic lion with flowing stylized mane and mosaic background, alphonse mucha style"},
    {"prompt": "Golden retriever puppy sitting next to a lily pond, alphonse mucha style"},
    {"prompt": "Detailed owl perched on an arched orchid branch, alphonse mucha style"},
    {"prompt": "Futuristic sports car surrounded by botanical vines, alphonse mucha style"},
    {"prompt": "Cyberpunk city skyline inside circular floral frame, alphonse mucha style"},
    {"prompt": "Robotic woman with glowing wires and copper halo, alphonse mucha style"},
    {"prompt": "Astronaut floating among stylized stars and lilies, alphonse mucha style"},
    {"prompt": "Modern electric guitar with ornate whiplash curves, alphonse mucha style"},
    {"prompt": "Male scholar in flowing robes holding an open book, alphonse mucha style"},
    {"prompt": "Knight in ornate armor surrounded by red roses, alphonse mucha style"},
    {"prompt": "Ornamental border pattern of intertwined ivy and daisies, alphonse mucha style"},
    {"prompt": "Moravian countryside landscape at sunset with tall lilies, alphonse mucha style"}
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Pankaj121212/alphonse_mucha_test_set", help="Dataset name")
    parser.add_argument("--HF_TOKEN", type=str, required=True, help="Huggingface token")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset=Dataset.from_list(mucha_prompts)
    os.environ["HF_TOKEN"]=args.HF_TOKEN
    dataset.push_to_hub(args.dataset_name)
    dataset.save_to_disk("/opt/ml/processing/output/")

if __name__ == "__main__":
    main()