"""
Batch Inference for EGTR Scene Graph Generation with Hyperparameter Sweeping

This script performs inference on a folder of images using EGTR, sweeping over
multiple hyperparameters (object threshold, relation threshold, top-k). The results
are saved in a structured JSON format for each image and parameter combination.

Example usage:
    python infer_egtr_batch.py \
        --image_folder /path/to/image/folder \
        --artifact_path /path/to/egtr/artifact \
        --data_path ./dataset/visual_genome \
        --output_dir ./result_sg
"""

import argparse
import json
import os
from glob import glob
from itertools import product

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from data.visual_genome import VGDataset

def argsort_desc(a: np.ndarray) -> np.ndarray:
    """Sort indices of a 2D array in descending order of values."""
    flat_indices = np.argsort(a, axis=None)[::-1]
    coords = np.array(np.unravel_index(flat_indices, a.shape)).T
    return coords

@torch.no_grad()
def inference_single_image(model, feature_extractor, image, obj_threshold, rel_threshold,
                           top_k, id2label, id2relation, device):
    """Run inference on a single image and extract scene graph triplets.

    Args:
        model: Trained EGTR model.
        feature_extractor: Deformable DETR feature extractor.
        image: PIL image input.
        obj_threshold (float): Object detection threshold.
        rel_threshold (float): Relationship score threshold.
        top_k (int): Number of top triplets to retain.
        id2label (dict): Mapping from object ID to label.
        id2relation (dict): Mapping from relation ID to label.
        device (str): Device to run inference on.

    Returns:
        dict: Contains triplets, unique subjects, predicates, and objects.
    """
    image_input = feature_extractor(image, return_tensors="pt")
    image_input = {k: v.to(device) for k, v in image_input.items()}

    outputs = model(
        pixel_values=image_input["pixel_values"],
        pixel_mask=image_input["pixel_mask"],
        output_attention_states=True
    )

    pred_logits = outputs["logits"][0]
    obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), dim=-1)

    valid_obj_mask = obj_scores >= obj_threshold
    if valid_obj_mask.sum() == 0:
        return {
            "triplets": [],
            "subjects": [],
            "predicates": [],
            "objects": []
        }

    valid_obj_indices = valid_obj_mask.nonzero(as_tuple=False).squeeze(1)
    valid_obj_scores = obj_scores[valid_obj_indices]
    valid_pred_classes = pred_classes[valid_obj_indices]

    pred_rel = outputs["pred_rel"][0]
    pred_connectivity = outputs["pred_connectivity"][0]
    pred_rel = pred_rel * pred_connectivity
    valid_pred_rel = pred_rel[valid_obj_indices][:, valid_obj_indices]

    best_rel_score, best_rel_class = torch.max(valid_pred_rel, dim=-1)

    valid_obj_scores = valid_obj_scores.unsqueeze(1)
    sub_ob_scores = valid_obj_scores * valid_obj_scores.t()
    diag_idx = torch.arange(sub_ob_scores.size(0), device=device)
    sub_ob_scores[diag_idx, diag_idx] = 0.0

    triplet_scores = best_rel_score * sub_ob_scores
    triplet_scores_np = triplet_scores.cpu().numpy()

    valid_pairs = np.where(triplet_scores_np >= rel_threshold)
    if len(valid_pairs[0]) == 0:
        return {
            "triplets": [],
            "subjects": [],
            "predicates": [],
            "objects": []
        }

    pairs = np.stack(valid_pairs, axis=-1)
    pair_scores = triplet_scores_np[valid_pairs]

    sorted_indices = np.argsort(pair_scores)[::-1]
    top_pairs = pairs[sorted_indices][:top_k]
    top_scores = pair_scores[sorted_indices][:top_k]

    triplets = []
    subjects = set()
    predicates = set()
    objects = set()

    for idx, (i, j) in enumerate(top_pairs):
        subject_label = id2label.get(valid_pred_classes[i].item(), f"obj_{valid_pred_classes[i].item()}")
        object_label = id2label.get(valid_pred_classes[j].item(), f"obj_{valid_pred_classes[j].item()}")
        predicate_label = id2relation.get(best_rel_class[i, j].item(), f"rel_{best_rel_class[i, j].item()}")
        triplets.append({
            "subject": subject_label,
            "predicate": predicate_label,
            "object": object_label,
            "score": float(top_scores[idx])
        })
        subjects.add(subject_label)
        predicates.add(predicate_label)
        objects.add(object_label)

    return {
        "triplets": triplets,
        "subjects": sorted(list(subjects)),
        "predicates": sorted(list(predicates)),
        "objects": sorted(list(objects))
    }

def load_model(artifact_path, architecture, device):
    """Load EGTR model from checkpoint."""
    config = DeformableDetrConfig.from_pretrained(artifact_path)
    model = DetrForSceneGraphGeneration.from_pretrained(
        architecture, config=config, ignore_mismatched_sizes=True
    )
    ckpt_paths = sorted(
        glob(os.path.join(artifact_path, "checkpoints", "epoch=*.ckpt")),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0])
    )
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoint files found in the artifact path.")
    ckpt_path = ckpt_paths[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Batch EGTR Inference")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder of input images")
    parser.add_argument("--artifact_path", type=str, required=True, help="EGTR artifact folder")
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument("--output_dir", type=str, default="result_sg")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )

    dataset = VGDataset(
        data_folder=args.data_path,
        feature_extractor=feature_extractor,
        split="train", # TODO: change to "test" or "train"
        num_object_queries=200 # NOTE: maybe can be hyperparameter
    )
    id2label = {k - 1: v["name"] for k, v in dataset.coco.cats.items()}
    id2relation = {i: name for i, name in enumerate(dataset.rel_categories)}

    model = load_model(args.artifact_path, args.architecture, args.device)

    # TODO: Change hyperparameters in lists based on your needs
    obj_thresholds = [0.1]
    rel_thresholds = [0.00001]
    top_ks = [20]

    images = sorted(glob(os.path.join(args.image_folder, "*.jpg")))

    for image_path in tqdm(images, desc="Processing images"):
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        image_results = {}

        for obj_thres, rel_thres, top_k in product(obj_thresholds, rel_thresholds, top_ks):
            key = f"obj_{obj_thres}_rel_{rel_thres}_topk_{top_k}"
            inference_result = inference_single_image(
                model, feature_extractor, image,
                obj_thres, rel_thres, top_k, id2label, id2relation,
                args.device
            )
            
            structured_result = {
                "parameters": {
                    "obj_threshold": obj_thres,
                    "rel_threshold": rel_thres,
                    "top_k": top_k
                },
                "image_name": image_name,
                "triplets": inference_result["triplets"],
                "subjects": inference_result["subjects"],
                "predicates": inference_result["predicates"],
                "objects": inference_result["objects"]
            }

            image_results[key] = structured_result

        output_path = os.path.join(args.output_dir, f"{image_name}.json")
        with open(output_path, "w") as f:
            json.dump(image_results, f, indent=4)

if __name__ == "__main__":
    main()
