from typing import Tuple, Set

import numpy as np


def calculate_recall(gt: Set[Tuple[int, int, int]], pred: Set[Tuple[int, int, int]]):
    return len(gt & pred) / len(gt)


def calculate_mean_recall(gt: Set[Tuple[int, int, int]], pred: Set[Tuple[int, int, int]]):
    gt_predicates = {triplet[1] for triplet in gt}
    recall_list = []
    for predicate in gt_predicates:
        p_gt = {triplet for triplet in gt if triplet[1] == predicate}
        recall_list.append(len(p_gt & pred) / len(p_gt))
    return np.mean(recall_list)


def evaluate_recall_at_k(data: dict, prediction_dict: dict):
    recall_list = [0] * 5
    count = 0
    for image_id in prediction_dict:
        predictions = [item[:3] for item in prediction_dict[image_id]]
        targets = set(data[image_id]['vrd_list'])
        if len(targets) == 0:
            continue
        for i, k in enumerate([5, 10, 20, 50, 100]):
            recall_list[i] += len(set(predictions[:k]) & targets) / len(targets)
        count += 1
    for i in range(len(recall_list)):
        recall_list[i] /= count
    return recall_list
