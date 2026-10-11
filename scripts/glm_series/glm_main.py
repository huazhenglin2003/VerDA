import datetime
import os
import json
import random
import argparse
import sys
import torch
import yaml
from math import ceil
from tqdm import tqdm
from eval.datasets import *
from scripts.glm_series.glm_utils import *


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# TODO: Set MODEL_CACHE_ROOT or pass --model_path when checkpoints are not in ./models.
MODEL_CACHE_ROOT = os.path.expanduser(
    os.environ.get("MODEL_CACHE_ROOT", os.path.join(PROJECT_ROOT, "models"))
)
DEFAULT_MODEL_PATH = os.path.join(
    MODEL_CACHE_ROOT, "models--zai-org--glm-4v-9b"
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
    tokenizer, model, num_img_tokens = get_glm_model(args.model_path, device, "eager")

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

    generate_kwargs = {"num_beams": 1, 
                       "max_new_tokens": args.max_new_tokens,
                       "do_sample": False}
    
    start = datetime.datetime.now()
    skipped_qids = []
    successful_samples = 0
    chunk_qids = dataset.all[args.chunk_idx*chunk_size:(args.chunk_idx+1)*chunk_size]

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for qid in tqdm(chunk_qids, ncols=100):
            img = None
            try:
                # prepare for inputs
                img, qu, gt = dataset[qid]
                if img.mode != "RGB":
                    raise ValueError(
                        f"GLM-4V requires an RGB image, but received mode={img.mode}"
                    )
                inputs = tokenizer.apply_chat_template([{"role": "user", "image": img, "content": qu}],
                                               add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)
                image_start = (inputs['input_ids'] == model.config.boi_token_id).nonzero(as_tuple=True)[1].item()
                
                if args.use_verda:
                    with open(args.config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)

                    model_name = args.model_path.rstrip("/").split("/")[-1].split("--", 2)[-1]
                    cfg = config[model_name]

                    replace_glm_forward(model, image_start, image_start + num_img_tokens, rho=cfg["rho"], gamma=cfg["gamma"], target_layers=cfg["layers"])
                
                inputs = {key: value.to(device) for key, value in inputs.items()}
                    
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generate_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]

                gen_answer = tokenizer.decode(outputs[0])

                output_file.write(json.dumps({"question_id": qid, "text": gen_answer, "gt": gt, "question": qu})+"\n")
                output_file.flush()
                successful_samples += 1
            except Exception as exc:
                skipped_qids.append(qid)
                tqdm.write(
                    f"[WARNING] Skipping sample question_id={qid}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            finally:
                close_image = getattr(img, "close", None)
                if callable(close_image):
                    close_image()

    if skipped_qids:
        print(
            f"[WARNING] Chunk {args.chunk_idx} skipped {len(skipped_qids)} sample(s): "
            f"{skipped_qids}",
            file=sys.stderr,
        )
    if chunk_qids and successful_samples == 0:
        raise RuntimeError(
            f"Chunk {args.chunk_idx} failed to process every sample; "
            "see the warnings above for the underlying errors."
        )
    end = datetime.datetime.now()
    elapsed = (end - start).seconds
    print(f"inference_time_s: {elapsed}")

if __name__ == "__main__":
    args = argument_parser()
    main(args)
