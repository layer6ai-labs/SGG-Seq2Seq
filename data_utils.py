import numpy as np
import random
import torch
from torch.utils.data import Dataset

from config import NUM_BOX_FEATURES, NUM_PAIRWISE_FEATURES, NUM_NEGATIVES
from feature_utils import extract_pairwise_features
from file_utils import load_object_from_file


class VRDDataset(Dataset):
    def __init__(self):
        self.data = load_object_from_file('output/train_data.dat')
        self.image_ids = list(self.data.keys())

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.data[image_id]
        bbox_list = data['bbox_list']
        box_features = data['box_features']
        vrd_list = data['vrd_list']
        negative_set = data['negative_set']
        random.shuffle(vrd_list)
        pairwise_features = np.array([extract_pairwise_features(bbox_list[item[0]], bbox_list[item[2]]) for item in vrd_list], dtype=np.float32)
        num_negatives = pairwise_features.shape[0] * NUM_NEGATIVES
        sampled_negative_list = random.choices(list(negative_set), k=num_negatives) if len(negative_set) > 0 else [(0, 0)] * num_negatives
        pairwise_features = np.concatenate([
            pairwise_features[:, None, :],
            np.array([extract_pairwise_features(bbox_list[item[0]], bbox_list[item[1]]) for item in sampled_negative_list], dtype=np.float32).reshape(-1, NUM_NEGATIVES, NUM_PAIRWISE_FEATURES)
        ], axis=1)
        subject_indices, relation_ids, object_indices = zip(*vrd_list)
        subject_features = np.array([box_features[index, :] for index in subject_indices], dtype=np.float32)
        object_features = np.array([box_features[index, :] for index in object_indices], dtype=np.float32)
        sampled_negative_list = np.array(sampled_negative_list, dtype=np.long)
        subject_indices = np.concatenate([
            np.array(subject_indices, dtype=np.long)[:, None],
            sampled_negative_list[:, 0].reshape((-1, NUM_NEGATIVES))
        ], axis=-1)
        object_indices = np.concatenate([
            np.array(object_indices, dtype=np.long)[:, None],
            sampled_negative_list[:, 1].reshape((-1, NUM_NEGATIVES))
        ], axis=-1)
        return box_features, pairwise_features, subject_features, relation_ids, object_features, subject_indices, object_indices

    def __len__(self):
        return len(self.image_ids)


class VRDInferenceDataset(Dataset):
    def __init__(self):
        self.data = load_object_from_file('output/valid_data.dat')
        self.pair_predicate_dict = load_object_from_file('output/pair_predicate_dict.dat')
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
            [extract_pairwise_features(bbox_list[item[0]], bbox_list[item[1]]) for item in positive_list] + [extract_pairwise_features(bbox_list[item[0]], bbox_list[item[1]]) for item in negative_list],
            dtype=np.float32
        )
        subject_indices, object_indices = zip(*positive_list)
        if len(negative_list) > 0:
            negative_subject_indices, negative_object_indices = zip(*negative_list)
            subject_indices = list(subject_indices) + list(negative_subject_indices)
            object_indices = list(object_indices) + list(negative_object_indices)
        candidates = [[i * 71 + k for k in self.pair_predicate_dict[(bbox_list[item[0]].category, bbox_list[item[1]].category)]] for i, item in enumerate(zip(subject_indices, object_indices))]
        candidates = [item for sublist in candidates for item in sublist]
        return image_id, box_features, pairwise_features, subject_indices, object_indices, candidates

    def __len__(self):
        return len(self.image_ids)


def collate_fn(batch):
    box_features_list, pairwise_features_list, subject_features_list, relation_ids_list, object_features_list, subject_indices_list, object_indices_list = zip(*batch)
    box_features = np.zeros(shape=(len(box_features_list), max(item.shape[0] for item in box_features_list), NUM_BOX_FEATURES), dtype=np.float32)
    src_key_padding_mask = np.zeros(shape=(len(box_features_list), max(item.shape[0] for item in box_features_list)), dtype=np.int8)
    for i, features in enumerate(box_features_list):
        box_features[i, :features.shape[0], :] = features
        src_key_padding_mask[i, features.shape[0]:] = 1
    pairwise_features = np.zeros(shape=(len(pairwise_features_list), max(item.shape[0] for item in pairwise_features_list), NUM_NEGATIVES + 1, NUM_PAIRWISE_FEATURES), dtype=np.float32)
    tgt_key_padding_mask = np.zeros(shape=(len(pairwise_features_list), max(item.shape[0] for item in pairwise_features_list)), dtype=np.int8)
    for i, features in enumerate(pairwise_features_list):
        pairwise_features[i, :features.shape[0], :, :] = features
        tgt_key_padding_mask[i, features.shape[0]:] = 1
    subject_features = np.zeros(shape=(len(subject_features_list), max(item.shape[0] for item in subject_features_list), NUM_BOX_FEATURES), dtype=np.float32)
    for i, features in enumerate(subject_features_list):
        subject_features[i, 1: features.shape[0], :] = features[:-1, :]
    relation_ids = np.zeros(shape=(len(relation_ids_list), max(len(item) for item in relation_ids_list)), dtype=np.long)
    for i, ids in enumerate(relation_ids_list):
        relation_ids[i, 1: len(ids)] = ids[:-1]
    object_features = np.zeros(shape=(len(object_features_list), max(item.shape[0] for item in object_features_list), NUM_BOX_FEATURES), dtype=np.float32)
    for i, features in enumerate(object_features_list):
        object_features[i, 1: features.shape[0], :] = features[:-1, :]
    batch_indices = np.zeros(shape=(pairwise_features.shape[0], pairwise_features.shape[1], NUM_NEGATIVES + 1), dtype=np.long)
    for i in range(batch_indices.shape[0]):
        batch_indices[i, :, :] = i
    batch_indices = batch_indices.reshape((-1,))
    subject_indices = np.zeros(shape=(len(subject_indices_list), max(item.shape[0] for item in subject_indices_list), (NUM_NEGATIVES + 1)), dtype=np.long)
    for i, indices in enumerate(subject_indices_list):
        subject_indices[i, :indices.shape[0], :] = indices
    subject_indices = subject_indices.reshape((-1,))
    object_indices = np.zeros(shape=(len(object_indices_list), max(item.shape[0] for item in object_indices_list), (NUM_NEGATIVES + 1)), dtype=np.long)
    for i, indices in enumerate(object_indices_list):
        object_indices[i, :indices.shape[0], :] = indices
    object_indices = object_indices.reshape((-1,))
    targets = np.zeros(shape=(len(relation_ids_list), max(len(item) for item in relation_ids_list), NUM_NEGATIVES + 1), dtype=np.long)
    for i, ids in enumerate(relation_ids_list):
        targets[i, :len(ids), 0] = ids
        targets[i, len(ids):, :] = -100
    box_features = torch.FloatTensor(box_features)
    pairwise_features = torch.FloatTensor(pairwise_features)
    subject_features = torch.FloatTensor(subject_features)
    relation_ids = torch.LongTensor(relation_ids)
    object_features = torch.FloatTensor(object_features)
    src_key_padding_mask = torch.BoolTensor(src_key_padding_mask)
    tgt_key_padding_mask = torch.BoolTensor(tgt_key_padding_mask)
    targets = torch.LongTensor(targets).view(-1)
    return box_features, pairwise_features, subject_features, relation_ids, object_features, src_key_padding_mask, tgt_key_padding_mask, batch_indices, subject_indices, object_indices, targets


def inference_collate_fn(batch):
    image_id_list, box_features_list, pairwise_features_list, subject_indices_list, object_indices_list, candidates_list = zip(*batch)
    box_features = np.zeros(shape=(len(box_features_list), max(item.shape[0] for item in box_features_list), NUM_BOX_FEATURES), dtype=np.float32)
    src_key_padding_mask = np.zeros(shape=(len(box_features_list), max(item.shape[0] for item in box_features_list)), dtype=np.int8)
    for i, features in enumerate(box_features_list):
        box_features[i, :features.shape[0], :] = features
        src_key_padding_mask[i, features.shape[0]:] = 1
    pairwise_features = np.zeros(shape=(len(pairwise_features_list), max(item.shape[0] for item in pairwise_features_list), NUM_PAIRWISE_FEATURES), dtype=np.float32)
    for i, features in enumerate(pairwise_features_list):
        pairwise_features[i, :features.shape[0], :] = features
    batch_indices = np.array([[i] * pairwise_features.shape[1] for i in range(pairwise_features.shape[0])], dtype=np.long).reshape((-1,))
    subject_indices = np.zeros(shape=(pairwise_features.shape[0], pairwise_features.shape[1]), dtype=np.long)
    for i, indices in enumerate(subject_indices_list):
        subject_indices[i, :len(indices)] = indices
    subject_indices = subject_indices.reshape((-1,))
    object_indices = np.zeros(shape=(pairwise_features.shape[0], pairwise_features.shape[1]), dtype=np.long)
    for i, indices in enumerate(object_indices_list):
        object_indices[i, :len(indices)] = indices
    object_indices = object_indices.reshape((-1,))
    box_features = torch.FloatTensor(box_features)
    pairwise_features = torch.FloatTensor(pairwise_features)
    src_key_padding_mask = torch.BoolTensor(src_key_padding_mask)
    return image_id_list, box_features, pairwise_features, src_key_padding_mask, batch_indices, subject_indices, object_indices, candidates_list
