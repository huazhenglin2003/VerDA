# Attention Heads are Complementary Visual Units: Mitigating Hallucinations in LVLMs via Adaptive Visual Cues Focusing

This repository contains the official implementation of our paper **"Attention Heads are Complementary Visual Units: Mitigating Hallucinations in LVLMs via Adaptive Visual Cues Focusing"**, accepted at **NeurIPS 2026**.

---

## 🛠️ Installation

Create and activate a Conda environment, then install the required dependencies:

```bash
conda create -n verda python=3.10.14 -y
conda activate verda
pip install -r requirements.txt
```

> ⚠️ **Note:** For models in the **Qwen-VL series**, additional dependencies may be required. Install them using the following command:

```bash
pip install -r requirements-qwen.txt
```

---

## 📂 Dataset and Model Preparation

1. **📊 Datasets.** Download the [MSCOCO](https://cocodataset.org/#download) dataset and the POPE evaluation annotations, and organize them under `./data` as follows:

   ```text
   data/
   ├── coco/
   │   ├── val2014/
   │   └── annotations/
   │       ├── captions_val2014.json
   │       └── instances_val2014.json
   └── pope/
       └── coco/
           ├── coco_pope_popular.json
           ├── coco_pope_random.json
           └── coco_pope_adversarial.json
   ```

   The default dataset root is `./data`. To keep datasets elsewhere, set
   `VERDA_DATA_ROOT` to a directory with the same internal structure.

2. **🧠 Large Vision-Language Models (LVLMs).** Download the pretrained checkpoints for the following LVLMs. By default, the inference scripts look for them under `./models`; set `MODEL_CACHE_ROOT` to use another directory. The checkpoint directory names used by the scripts are listed in the `MODEL_DIRS` arrays in `infer.sh` and `infer-qwen.sh` and can be adjusted to match your local names.

   - [LLaVA-NeXT-8B](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/main)
   - [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)
   - [InstructBLIP-13B](https://huggingface.co/Salesforce/instructblip-vicuna-13b/tree/main)
   - [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)
   - [InternVL3-9B](https://huggingface.co/OpenGVLab/InternVL3-9B/tree/main)
   - [InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B/tree/main)
   - [GLM-4V-9B](https://huggingface.co/zai-org/glm-4v-9b/tree/main)
   - [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)
   - [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main)

---

## 🚀 Inference and Evaluation

Run inference using the provided shell scripts:

```bash
# General inference
bash infer.sh

# Inference with Qwen-VL models
bash infer-qwen.sh
```

The scripts use repository-relative defaults and can also be configured without
editing the source:

```bash
VERDA_DATA_ROOT=/path/to/data \
MODEL_CACHE_ROOT=/path/to/models \
OUTPUT_ROOT=/path/to/outputs \
bash infer.sh
```

If these variables are omitted, the defaults are `./data`, `./models`, and
`./outputs`, respectively. Direct Python entry points also honor
`VERDA_DATA_ROOT` and `MODEL_CACHE_ROOT`.

👉 **Note:**

- `num_chunks` specifies the number of inference chunks for parallel execution across GPUs (default: 8).
- Update the `GPUS` array and, if necessary, the `MODEL_DIRS`, `METHODS`,
  `dataset`, and `POPE_SUBSETS` values in the inference script before running.
- You can select the target LVLM in the corresponding `.sh` script and configure the **`gamma`** and **`rho`** hyperparameters in `./configs/verda_configs.yaml`.

## 📚 Citations

If you find this work useful, please cite us. 😉