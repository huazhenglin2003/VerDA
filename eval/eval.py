import argparse
import json
import os
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from eval.chair import CHAIR, load_generated_captions
from eval.cocoeval import COCOEvalCap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# TODO: Set VERDA_DATA_ROOT when datasets are not stored under ./data.
DATA_ROOT = Path(os.environ.get("VERDA_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
# TODO: Set OUTPUT_ROOT or pass the output arguments explicitly when evaluation
# artifacts should be written outside ./outputs.
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser()


def compute_distinct_1(texts):
    """Return corpus-level Distinct-1."""
    if not texts:
        return {"distinct_1": 0.0}

    all_tokens = []
    strip_punctuation = str.maketrans("", "", string.punctuation)
    for text in texts:
        all_tokens.extend(text.lower().translate(strip_punctuation).split())

    return {
        "distinct_1": len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def eval_chair(answer_file, anno_file, anno_dir):
    answers = load_jsonl(answer_file)
    if not answers:
        raise ValueError(f"No predictions found in {answer_file}")

    for answer in answers:
        answer["caption"] = answer["text"]
        answer["image_id"] = answer["question_id"]

    result_path = Path(answer_file).resolve()
    captions = [answer["caption"] for answer in answers]

    # Compute Distinct-1
    metrics = compute_distinct_1(captions)
    d1_file = result_path.with_name("d1_" + result_path.name)
    save_json(d1_file, metrics)
    print(f"Distinct-1 saved to {d1_file}")

    # Compute COCO caption metrics
    coco = COCO(anno_file)
    overall_scores = defaultdict(list)
    img_to_eval = {}

    for start in tqdm(range(0, len(answers), 100)):
        coco_res = coco.loadRes(answers[start:start + 100])
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params["image_id"] = coco_res.getImgIds()
        coco_eval.evaluate()

        for metric, score in coco_eval.eval.items():
            overall_scores[metric].append(score)

        for i, image_id in enumerate(coco_res.getImgIds()):
            eval_item = coco_eval.evalImgs[i]
            eval_item["caption"] = coco_res.imgToAnns[image_id][0]["caption"]
            img_to_eval[image_id] = eval_item

    chair_file = result_path.with_name("chair_" + result_path.name)
    save_json(chair_file, {
        "overall": {
            name: float(np.mean(scores))
            for name, scores in overall_scores.items()
        },
        "imgToEval": img_to_eval,
    })

    # Compute CHAIR metrics
    _, image_ids, _ = load_generated_captions(str(chair_file))
    evaluator = CHAIR(image_ids, anno_dir)
    evaluator.get_annotations()
    chair_metrics = evaluator.compute_chair(str(chair_file))

    metrics_file = result_path.with_name("metrics_" + result_path.name)
    save_json(metrics_file, chair_metrics.get("overall_metrics", {}))
    print(f"CHAIR metrics saved to {metrics_file}")


def eval_pope(answers, labels, question_ids):
    predictions, targets = [], []

    for question_id in question_ids:
        if question_id not in answers:
            continue

        text = answers[question_id].split(".", 1)[0].replace(",", "")
        predictions.append(
            0 if any(w in text.split(" ") for w in ("No", "not", "no")) else 1
        )
        targets.append(
            0 if labels[question_id] and "no" in labels[question_id].lower() else 1
        )

    if not predictions:
        raise ValueError("No matching POPE prediction IDs found.")

    tp = sum(p == 1 and t == 1 for p, t in zip(predictions, targets))
    tn = sum(p == 0 and t == 0 for p, t in zip(predictions, targets))
    fp = sum(p == 1 and t == 0 for p, t in zip(predictions, targets))
    fn = sum(p == 0 and t == 1 for p, t in zip(predictions, targets))

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    accuracy = (tp + tn) / len(predictions)
    yes_ratio = predictions.count(1) / len(predictions)

    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}, Total={len(predictions)}")
    print(
        f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
        f"Recall={recall:.4f}, F1={f1:.4f}, Yes ratio={yes_ratio:.4f}"
    )

    return [accuracy, precision, recall, f1, yes_ratio]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question-file",
        type=str,
        default=str(DATA_ROOT / "pope" / "coco" / "coco_pope_adversarial.json")
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=str(OUTPUT_ROOT / "predictions.json")
    )
    parser.add_argument("--eval_pope", action="store_true")
    parser.add_argument("--eval_chair", action="store_true")
    parser.add_argument(
        "--pope_savepath",
        type=str,
        default=str(OUTPUT_ROOT / "pope-metrics.json")
    )
    parser.add_argument(
        "--anno-file",
        type=str,
        default=str(DATA_ROOT / "coco" / "annotations" / "captions_val2014.json")
    )
    parser.add_argument(
        "--anno-dir",
        type=str,
        default=str(DATA_ROOT / "coco" / "annotations")
    )

    args = parser.parse_args()

    if args.eval_chair:
        eval_chair(args.result_file, args.anno_file, args.anno_dir)

    elif args.eval_pope:
        questions = load_jsonl(args.question_file)
        answers = load_jsonl(args.result_file)

        question_ids = list(dict.fromkeys(q["question_id"] for q in questions))
        labels = {q["question_id"]: q["label"] for q in questions}
        predictions = {a["question_id"]: a["text"] for a in answers}

        scores = eval_pope(predictions, labels, question_ids)
        scaled = np.round(np.asarray(scores) * 100, decimals=2)

        names = ["accuracy", "precision", "recall", "f1_score", "yes_ratio"]

        pope_metrics = {
            "raw_metrics": dict(zip(names, map(float, scores))),
            "Average_metrics": dict(zip(names, map(float, scaled))),
        }

        save_path = Path(args.pope_savepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(save_path, pope_metrics)

        print("Average:", scaled)

    print("====================================")


if __name__ == "__main__":
    main()
