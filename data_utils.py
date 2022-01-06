import argparse
import copy

import numpy as np
import random
import torch
from torch.utils.data import Dataset

from feature_utils import extract_pairwise_features
from file_utils import load_object_from_file


class VRDDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = load_object_from_file('output/{}/train_data.dat'.format(args.dataset))
        self.image_ids = list(self.data.keys())

    def __getitem__(self, index):
        data = self.data[self.image_ids[index]]
        bbox_list = data['bbox_list']
        box_features = data['box_features']
        vrd_list = copy.deepcopy(data['vrd_list'])
        negative_set = data['negative_set']
        random.shuffle(vrd_list)
        pairwise_features = np.array([extract_pairwise_features(bbox_list[item[0]], bbox_list[item[2]]) for item in vrd_list], dtype=np.float32)
        num_negatives = len(vrd_list) * self.args.num_negatives
        sampled_negative_list = random.choices(list(negative_set), k=num_negatives) if len(negative_set) > 0 else [(0, 0)] * num_negatives
        pairwise_features = np.concatenate([
            pairwise_features[:, None, :],
            np.array([extract_pairwise_features(bbox_list[item[0]], bbox_list[item[1]]) for item in sampled_negative_list], dtype=np.float32).reshape((-1, self.args.num_negatives, self.args.num_pairwise_features))
        ], axis=1)
        subject_indices, relation_ids, object_indices = zip(*vrd_list)
        subject_features = np.array([box_features[index, :] for index in subject_indices], dtype=np.float32)
        object_features = np.array([box_features[index, :] for index in object_indices], dtype=np.float32)
        sampled_negative_list = np.array(sampled_negative_list, dtype=np.int64)
        subject_indices = np.concatenate([
            np.array(subject_indices, dtype=np.int64)[:, None],
            sampled_negative_list[:, 0].reshape((-1, self.args.num_negatives))
        ], axis=-1)
        object_indices = np.concatenate([
            np.array(object_indices, dtype=np.int64)[:, None],
            sampled_negative_list[:, 1].reshape((-1, self.args.num_negatives))
        ], axis=-1)
        return box_features, pairwise_features, subject_features, relation_ids, object_features, subject_indices, object_indices

    def __len__(self):
        return len(self.image_ids)


class VRDInferenceDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str = 'valid'):
        self.args = args
        self.data = load_object_from_file(f'output/{args.dataset}/{split}_data.dat')
        self.pair_predicate_dict = load_object_from_file(f'output/{args.dataset}/pair_predicate_dict.dat')
        self.image_ids = list(self.data.keys())

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.data[image_id]
        bbox_list = data['bbox_list']
        box_features = data['box_features']
        vrd_list = data['vrd_list']
        positive_list = list(set((item[0], item[2]) for item in vrd_list))
        negative_list = list(data['negative_set'])
        pairwise_features = np.array(
            [
                extract_pairwise_features(
                    subject_bbox=bbox_list[subject_index],
                    object_bbox=bbox_list[object_index])
                for subject_index, object_index in positive_list
            ] +
            [
                extract_pairwise_features(
                    subject_bbox=bbox_list[subject_index],
                    object_bbox=bbox_list[object_index]
                ) for subject_index, object_index in negative_list
            ],
            dtype=np.float32
        )
        subject_indices, object_indices = zip(*positive_list)
        if len(negative_list) > 0:
            negative_subject_indices, negative_object_indices = zip(*negative_list)
            subject_indices = list(subject_indices) + list(negative_subject_indices)
            object_indices = list(object_indices) + list(negative_object_indices)
        candidates = [
            [
                i * self.args.num_relations + k
                for k in self.pair_predicate_dict[(bbox_list[subject_index].category, bbox_list[object_index].category)]
            ]
            for i, (subject_index, object_index) in enumerate(positive_list)
        ]
        candidates = [item for sublist in candidates for item in sublist]
        return image_id, box_features, pairwise_features, subject_indices, object_indices, candidates, set(vrd_list)

    def __len__(self):
        return len(self.image_ids)


def collate_fn(args: argparse.Namespace, batch: list):
    box_features_list, pairwise_features_list, subject_features_list, relation_ids_list, object_features_list, subject_indices_list, object_indices_list = zip(*batch)
    box_features = np.zeros(shape=(len(box_features_list), max(item.shape[0] for item in box_features_list), box_features_list[0].shape[1]), dtype=np.float32)
    box_padding_mask = np.zeros(shape=(len(box_features_list), max(item.shape[0] for item in box_features_list)), dtype=np.uint8)
    for i, features in enumerate(box_features_list):
        box_features[i, :features.shape[0], :] = features
        box_padding_mask[i, features.shape[0]:] = 1
    pairwise_features = np.zeros(shape=(len(pairwise_features_list), max(item.shape[0] for item in pairwise_features_list), args.num_negatives + 1, args.num_pairwise_features), dtype=np.float32)
    triplet_padding_mask = np.zeros(shape=(len(pairwise_features_list), max(item.shape[0] for item in pairwise_features_list)), dtype=np.uint8)
    for i, features in enumerate(pairwise_features_list):
        pairwise_features[i, :features.shape[0], :, :] = features
        triplet_padding_mask[i, features.shape[0]:] = 1
    subject_features = np.zeros(shape=(len(subject_features_list), max(item.shape[0] for item in subject_features_list), subject_features_list[0].shape[1]), dtype=np.float32)
    for i, features in enumerate(subject_features_list):
        subject_features[i, 1: features.shape[0], :] = features[:-1, :]
    relation_ids = np.zeros(shape=(len(relation_ids_list), max(len(item) for item in relation_ids_list)), dtype=np.int64)
    for i, ids in enumerate(relation_ids_list):
        relation_ids[i, 1: len(ids)] = ids[:-1]
    object_features = np.zeros(shape=(len(object_features_list), max(item.shape[0] for item in object_features_list), object_features_list[0].shape[1]), dtype=np.float32)
    for i, features in enumerate(object_features_list):
        object_features[i, 1: features.shape[0], :] = features[:-1, :]
    batch_indices = np.zeros(shape=(pairwise_features.shape[0], pairwise_features.shape[1], args.num_negatives + 1), dtype=np.int64)
    for i in range(batch_indices.shape[0]):
        batch_indices[i, :, :] = i
    batch_indices = batch_indices.flatten()
    subject_indices = np.zeros(shape=(len(subject_indices_list), max(item.shape[0] for item in subject_indices_list), (args.num_negatives + 1)), dtype=np.int64)
    for i, indices in enumerate(subject_indices_list):
        subject_indices[i, :indices.shape[0], :] = indices
    subject_indices = subject_indices.flatten()
    object_indices = np.zeros(shape=(len(object_indices_list), max(item.shape[0] for item in object_indices_list), (args.num_negatives + 1)), dtype=np.int64)
    for i, indices in enumerate(object_indices_list):
        object_indices[i, :indices.shape[0], :] = indices
    object_indices = object_indices.flatten()
    targets = np.zeros(shape=(len(relation_ids_list), max(len(item) for item in relation_ids_list), args.num_negatives + 1), dtype=np.int64)
    for i, ids in enumerate(relation_ids_list):
        targets[i, :len(ids), 0] = ids
        targets[i, len(ids):, :] = -100
    box_features = torch.FloatTensor(box_features)
    pairwise_features = torch.FloatTensor(pairwise_features)
    subject_features = torch.FloatTensor(subject_features)
    relation_ids = torch.LongTensor(relation_ids)
    object_features = torch.FloatTensor(object_features)
    box_padding_mask = torch.BoolTensor(box_padding_mask)
    triplet_padding_mask = torch.BoolTensor(triplet_padding_mask)
    targets = torch.LongTensor(targets).flatten()
    return box_features, pairwise_features, subject_features, relation_ids, object_features, box_padding_mask, triplet_padding_mask, batch_indices, subject_indices, object_indices, targets


def inference_collate_fn(args: argparse.Namespace, batch: list):
    image_id_list, box_features_list, pairwise_features_list, subject_indices_list, object_indices_list, candidates_list, gt_list = zip(*batch)
    image_id_list = list(image_id_list)
    candidates_list = list(candidates_list)
    gt_list = list(gt_list)
    batch_size = len(image_id_list)
    max_num_boxes = max(item.shape[0] for item in box_features_list)
    max_num_pairs = max(item.shape[0] for item in pairwise_features_list)
    box_features = np.zeros(shape=(batch_size, max_num_boxes, args.num_box_features), dtype=np.float32)
    box_padding_mask = np.zeros(shape=(batch_size, max_num_boxes), dtype=np.uint8)
    for i, features in enumerate(box_features_list):
        box_features[i, :features.shape[0], :] = features
        box_padding_mask[i, features.shape[0]:] = 1
    pairwise_features = np.zeros(shape=(batch_size, max_num_pairs, args.num_pairwise_features), dtype=np.float32)
    for i, features in enumerate(pairwise_features_list):
        pairwise_features[i, :features.shape[0], :] = features
    batch_indices = np.array([[i] * max_num_pairs for i in range(batch_size)], dtype=np.int64).flatten()
    subject_indices = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.int64)
    object_indices = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.int64)
    for i, indices in enumerate(subject_indices_list):
        subject_indices[i, : len(indices)] = indices
    for i, indices in enumerate(object_indices_list):
        object_indices[i, : len(indices)] = indices
    subject_indices = subject_indices.flatten()
    object_indices = object_indices.flatten()
    box_features = torch.FloatTensor(box_features)
    pairwise_features = torch.FloatTensor(pairwise_features)
    box_padding_mask = torch.BoolTensor(box_padding_mask)
    return image_id_list, box_features, pairwise_features, box_padding_mask, batch_indices, subject_indices, object_indices, candidates_list, gt_list
