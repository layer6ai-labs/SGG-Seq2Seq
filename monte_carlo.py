import argparse
import copy
from typing import List, Tuple, Set

import numpy as np
import torch

from evaluation_utils import calculate_recall, calculate_mean_recall
from transformer import Transformer


class MonteCarlo(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def rollout(
            self,
            model: Transformer,
            box_features: torch.FloatTensor,
            pairwise_features: torch.FloatTensor,
            box_padding_mask: torch.BoolTensor,
            batch_indices: np.ndarray,
            subject_indices: np.ndarray,
            object_indices: np.ndarray,
            candidates_list: List[List[int]],
            contextual_box_features: torch.FloatTensor,
            decoded_subject_features: torch.FloatTensor,
            decoded_relation_ids: torch.LongTensor,
            decoded_object_features: torch.FloatTensor,
            decoded_pairwise_features: torch.FloatTensor,
            greedy_decoding_results: List[List[Tuple[int, int, int, float]]],
            gt_list: List[Set[Tuple[int, int, int]]]
    ):
        greedy_pred_list = [set(item[:3] for item in sublist) for sublist in greedy_decoding_results]
        baseline_rewards = np.array([
            [
                self.args.alpha * calculate_recall(gt=gt, pred=pred) + (1.0 - self.args.alpha) * calculate_mean_recall(gt=gt, pred=pred)
            ] for gt, pred in zip(gt_list, greedy_pred_list)
        ])
        rewards = np.zeros(shape=(len(gt_list), self.args.max_k), dtype=np.float32)
        for t in range(1, self.args.max_k + 1):
            for playout in range(self.args.num_playouts):
                decoding_results = copy.deepcopy(greedy_decoding_results)
                decoding_results = [item[: t - 1] for item in decoding_results]
                decoding_results, _ = model.decode(
                    box_features=box_features,
                    pairwise_features=pairwise_features,
                    box_padding_mask=box_padding_mask,
                    batch_indices=batch_indices,
                    subject_indices=subject_indices,
                    object_indices=object_indices,
                    candidates_list=candidates_list,
                    contextual_box_features=contextual_box_features,
                    decoded_subject_features=decoded_subject_features,
                    decoded_relation_ids=decoded_relation_ids,
                    decoded_object_features=decoded_object_features,
                    decoded_pairwise_features=decoded_pairwise_features,
                    decoding_results=decoding_results,
                    t_start=t,
                    sample=True
                )
                sample_pred_list = [set(item[:3] for item in sublist) for sublist in decoding_results]
                sample_rewards = np.array([
                    self.args.alpha * calculate_recall(gt=gt, pred=pred) + (1.0 - self.args.alpha) * calculate_mean_recall(gt=gt, pred=pred)
                    for gt, pred in zip(gt_list, sample_pred_list)
                ])
                rewards[:, t - 1] += sample_rewards
        rewards /= self.args.num_playouts
        rewards -= baseline_rewards
        return rewards
