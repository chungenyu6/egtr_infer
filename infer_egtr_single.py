import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from data.visual_genome import VGDataset


def argsort_desc(a: np.ndarray) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="EGTR Single Image Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--artifact_path", type=str, required=True, help="EGTR artifact folder")
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_queries", type=int, default=200, help="How many potential objects the model “asks” about in each image. (typically 200 for Visual Genome, matching the config)")
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--obj_threshold", type=float, default=0.3)
    parser.add_argument("--rel_threshold", type=float, default=0.0001)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_json", type=str, default="single_output.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.min_size, max_size=args.max_size
    )

    dataset = VGDataset(
        data_folder=args.data_path,
        feature_extractor=feature_extractor,
        split=args.split,
        num_object_queries=args.num_queries
    )
    id2label = {k - 1: v["name"] for k, v in dataset.coco.cats.items()}
    id2relation = {i: name for i, name in enumerate(dataset.rel_categories)}

    model = load_model(args.artifact_path, args.architecture, args.device)
    image = Image.open(args.image_path).convert("RGB")

    inference_result = inference_single_image(
        model, feature_extractor, image,
        args.obj_threshold, args.rel_threshold,
        args.top_k, id2label, id2relation,
        args.device
    )

    structured_result = {
        "parameters": {
            "obj_threshold": args.obj_threshold,
            "rel_threshold": args.rel_threshold,
            "top_k": args.top_k
        },
        "image_name": args.image_path,
        "triplets": inference_result["triplets"],
        "subjects": inference_result["subjects"],
        "predicates": inference_result["predicates"],
        "objects": inference_result["objects"]
    }

    with open(args.output_json, "w") as f:
        json.dump(structured_result, f, indent=4)

    print(f"Saved top {args.top_k} triplets to {args.output_json}")


if __name__ == "__main__":
    main()