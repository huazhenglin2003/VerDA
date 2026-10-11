import datetime
import os
import json
import random
import argparse
import torch
import yaml
from math import ceil
from tqdm import tqdm
from eval.datasets import *
from scripts.internvl_series.internvl_utils import get_internvl_model, load_image, replace_intern_forward
from scripts.internvl_series.internvl3_14b_utils import replace_intern3_14b_forward


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# TODO: Set MODEL_CACHE_ROOT or pass --model_path when checkpoints are not in ./models.
MODEL_CACHE_ROOT = os.path.expanduser(
    os.environ.get("MODEL_CACHE_ROOT", os.path.join(PROJECT_ROOT, "models"))
)
DEFAULT_MODEL_PATH = os.path.join(
    MODEL_CACHE_ROOT, "models--OpenGVLab--InternVL2-8B"
)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "verda_configs.yaml")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pope", help="Dataset name (options: chair, pope)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the pretrained model")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the model configuration file")
    parser.add_argument("--output_file", type=str, default="test/test.json", help="Path to save the output results (default: test/test.json)")
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--subset", type=str, default="adversarial", choices=["popular", "adversarial", "random"], help="Subset of dataset to use (default: popular)")
    parser.add_argument("--use_verda", action="store_true", default=False, help="Whether to apply attention verda during inference")
    
    return parser.parse_args()

                
def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    random.seed(args.seed)

    # Load model
    device = torch.device(args.device)
    tokenizer, model, image_start, num_img_tokens = get_internvl_model(args.model_path, device, "eager")

    # Load dataset
    if args.dataset == "chair":
        dataset = CHAIRBench(500)
        print(f"seed: {args.seed}")
    elif args.dataset == "pope":
        dataset = POPE("pope_" + args.subset)
        print(f"dataset: pope_{args.subset}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    chunk_size = ceil(len(dataset.all) / args.num_chunks)

    output_file = open(args.output_file, "w")

    generate_kwargs = {"num_beams": 1, 
                       "max_new_tokens": args.max_new_tokens,
                       "do_sample": False}
    
    if args.use_verda:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_name = args.model_path.rstrip("/").split("/")[-1].split("--", 2)[-1]
        cfg = config[model_name]

        if "InternVL3-14B" in model_name:
            replace_intern3_14b_forward(model, image_start, image_start + num_img_tokens, rho=cfg["rho"], gamma=cfg["gamma"], target_layers=cfg["layers"])
        else:
            replace_intern_forward(model, image_start, image_start + num_img_tokens, rho=cfg["rho"], gamma=cfg["gamma"], target_layers=cfg["layers"])
       
    start = datetime.datetime.now()
    for qid in tqdm(dataset.all[args.chunk_idx*chunk_size:(args.chunk_idx+1)*chunk_size], ncols=100):

        # prepare for inputs
        img, qu, gt = dataset[qid]
        pixel_values = load_image(img, max_num=12).to(torch.bfloat16).to(device)        
        question = f'<image>\n{qu}.'
        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, question, generate_kwargs)
        gen_answer = response.strip()
        ans_conf = "N/A"

        output_file.write(json.dumps({"question_id": qid, "text": gen_answer, "gt": gt, "ans_conf": ans_conf, "question": qu})+"\n")
        output_file.flush()

    output_file.close()
    end = datetime.datetime.now()
    elapsed = (end - start).seconds
    print(f"inference_time_s: {elapsed}")

if __name__ == "__main__":
    args = argument_parser()
    main(args)
