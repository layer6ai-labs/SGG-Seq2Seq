import argparse
import gc
import math
import os
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from data_utils import VRDDataset, VRDInferenceDataset, collate_fn, inference_collate_fn
from evaluation_utils import evaluate_recall_at_k
from feature_utils import extract_pairwise_features
from file_utils import load_object_from_file
from monte_carlo import MonteCarlo
from transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vrd')
    parser.add_argument('--max_k', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--relation_embedding_dim', type=int, default=32)
    parser.add_argument('--dnn_dropout', type=float, default=0.2)
    parser.add_argument('--num_negatives', type=int, default=4)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--rl_interval', type=int, default=2)
    parser.add_argument('--num_playouts', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=4000)
    parser.add_argument('--num_warmup_steps', type=int, default=1000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=10.0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--normalization', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def get_cosine_schedule_with_warmup(
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_train_steps: int,
        num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_train_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def no_decay(n: str):
    return n.endswith('.bias') or n.endswith('.norm1.weight') or n.endswith('.norm2.weight') or n.endswith('.norm3.weight')


def main():
    args = parse_args()

    assert args.batch_size % args.gradient_accumulation_steps == 0
    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    if not os.path.exists(f'model/{args.dataset}'):
        pathlib.Path(f'model/{args.dataset}').mkdir(parents=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.dataset == 'vrd':
        args.num_classes = 100
        args.num_relations = 71
        args.num_box_features = 109
    elif args.dataset == 'vg':
        args.num_classes = 150
        args.num_relations = 51
        args.num_box_features = 159
    else:
        raise ValueError()

    args.num_pairwise_features = 15

    train_dataset = VRDDataset(args=args)
    rl_dataset = VRDInferenceDataset(args=args, split='train')
    valid_dataset = VRDInferenceDataset(args=args, split='valid')
    valid_data = load_object_from_file(f'output/{args.dataset}/valid_data.dat')

    if args.normalization:
        train_box_features = np.concatenate([train_dataset.data[image_id]['box_features'] for image_id in train_dataset.data], axis=0)
        box_features_scaler = StandardScaler()
        box_features_scaler.fit(train_box_features)
        for image_id in train_dataset.data:
            train_dataset.data[image_id]['box_features'] = box_features_scaler.transform(train_dataset.data[image_id]['box_features'])
        for image_id in rl_dataset.data:
            rl_dataset.data[image_id]['box_features'] = box_features_scaler.transform(rl_dataset.data[image_id]['box_features'])
        for image_id in valid_dataset.data:
            valid_dataset.data[image_id]['box_features'] = box_features_scaler.transform(valid_dataset.data[image_id]['box_features'])
        for image_id in valid_data:
            valid_data[image_id]['box_features'] = box_features_scaler.transform(valid_data[image_id]['box_features'])
        train_pairwise_features = []
        for image_id in train_dataset.data:
            bbox_list = train_dataset.data[image_id]['bbox_list']
            train_pairwise_features.append(np.array([extract_pairwise_features(bbox_list[item[0]], bbox_list[item[2]]) for item in train_dataset.data[image_id]['vrd_list']], dtype=np.float32))
            if len(train_dataset.data[image_id]['negative_set']) > 0:
                train_pairwise_features.append(np.array([extract_pairwise_features(bbox_list[item[0]], bbox_list[item[1]]) for item in train_dataset.data[image_id]['negative_set']], dtype=np.float32))
        train_pairwise_features = np.concatenate(train_pairwise_features, axis=0)
        pairwise_features_scaler = StandardScaler()
        pairwise_features_scaler.fit(train_pairwise_features)
        pairwise_features_mean = torch.from_numpy(pairwise_features_scaler.mean_).float().cuda()
        pairwise_features_std = torch.from_numpy(pairwise_features_scaler.scale_).float().cuda()
        del train_box_features, train_pairwise_features
        gc.collect()
    else:
        box_features_scaler = None
        pairwise_features_scaler = None
        pairwise_features_mean = 0.0
        pairwise_features_std = 1.0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(args=args, batch=batch)
    )
    rl_loader = DataLoader(
        dataset=rl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: inference_collate_fn(args=args, batch=batch)
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: inference_collate_fn(args=args, batch=batch)
    )

    model = Transformer(args=args)

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not no_decay(n)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if no_decay(n)],
            'weight_decay': 0.0
        }
    ]
    tf_optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr)
    rl_optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args.gamma * args.lr)

    num_tf_steps = args.num_epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    tf_scheduler = get_cosine_schedule_with_warmup(
        optimizer=tf_optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_train_steps=num_tf_steps
    )
    num_rl_steps = args.num_epochs * math.ceil(len(rl_loader) / args.gradient_accumulation_steps / args.rl_interval)
    rl_scheduler = get_cosine_schedule_with_warmup(
        optimizer=rl_optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_train_steps=num_rl_steps
    )

    model.cuda()
    model.train()

    tf_scaler = GradScaler() if args.fp16 else None
    rl_scaler = GradScaler() if args.fp16 else None

    monte_carlo = MonteCarlo(args=args)

    tf_global_step = 0
    rl_global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        for (
                box_features,
                pairwise_features,
                subject_features,
                relation_ids,
                object_features,
                box_padding_mask,
                triplet_padding_mask,
                batch_indices,
                subject_indices,
                object_indices,
                targets
        ) in train_loader:
            box_features = box_features.cuda()
            pairwise_features = pairwise_features.cuda()
            subject_features = subject_features.cuda()
            relation_ids = relation_ids.cuda()
            object_features = object_features.cuda()
            box_padding_mask = box_padding_mask.cuda()
            triplet_padding_mask = triplet_padding_mask.cuda()
            targets = targets.cuda()
            if args.normalization:
                pairwise_features = (pairwise_features - pairwise_features_mean) / pairwise_features_std
            with autocast(enabled=args.fp16):
                predictions = model.forward(
                    box_features=box_features,
                    pairwise_features=pairwise_features,
                    subject_features=subject_features,
                    relation_ids=relation_ids,
                    object_features=object_features,
                    box_padding_mask=box_padding_mask,
                    triplet_padding_mask=triplet_padding_mask,
                    batch_indices=batch_indices,
                    subject_indices=subject_indices,
                    object_indices=object_indices
                )
            loss = F.cross_entropy(input=predictions.flatten(0, 2), target=targets, ignore_index=-100, reduction='mean')
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                tf_scaler.scale(loss).backward()
            else:
                loss.backward()
            tf_global_step += 1
            if tf_global_step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm is not None:
                    if args.fp16:
                        tf_scaler.unscale_(tf_optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=False)
                if args.fp16:
                    tf_scaler.step(tf_optimizer)
                    tf_scaler.update()
                else:
                    tf_optimizer.step()
                tf_optimizer.zero_grad()
                tf_scheduler.step()

        if epoch % args.rl_interval == 0:
            for (
                    image_id_list,
                    box_features,
                    pairwise_features,
                    box_padding_mask,
                    batch_indices,
                    subject_indices,
                    object_indices,
                    candidates_list,
                    gt_list
            ) in rl_loader:
                box_features = box_features.cuda()
                pairwise_features = pairwise_features.cuda()
                box_padding_mask = box_padding_mask.cuda()
                if args.normalization:
                    pairwise_features = (pairwise_features - pairwise_features_mean) / pairwise_features_std
                greedy_decoding_results, greedy_decoding_cache = model.decode(
                    box_features=box_features,
                    pairwise_features=pairwise_features,
                    box_padding_mask=box_padding_mask,
                    batch_indices=batch_indices,
                    subject_indices=subject_indices,
                    object_indices=object_indices,
                    candidates_list=candidates_list
                )
                contextual_box_features, decoded_subject_features, decoded_relation_ids, decoded_object_features, decoded_pairwise_features = greedy_decoding_cache
                rewards = monte_carlo.rollout(
                    model=model,
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
                    greedy_decoding_results=greedy_decoding_results,
                    gt_list=gt_list
                )
                batch_size = len(greedy_decoding_results)
                max_num_pairs = max(len(item) for item in greedy_decoding_results)
                decoded_batch_indices = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.int64)
                decoded_subject_indices = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.int64)
                decoded_object_indices = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.int64)
                triplet_padding_mask = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.uint8)
                decoded_targets = np.zeros(shape=(batch_size, max_num_pairs), dtype=np.int64)
                for i, results in enumerate(greedy_decoding_results):
                    decoded_batch_indices[i, :] = i
                    decoded_subject_indices[i, : len(results)] = [item[0] for item in results]
                    decoded_object_indices[i, : len(results)] = [item[2] for item in results]
                    triplet_padding_mask[i, len(results):] = 1
                    decoded_targets[i, :len(results)] = [item[1] for item in results]
                    decoded_targets[i, len(results):] = -100
                decoded_batch_indices = torch.from_numpy(decoded_batch_indices).cuda()
                decoded_subject_indices = torch.from_numpy(decoded_subject_indices).cuda()
                decoded_object_indices = torch.from_numpy(decoded_object_indices).cuda()
                triplet_padding_mask = torch.from_numpy(triplet_padding_mask).cuda().bool()
                decoded_targets = torch.from_numpy(decoded_targets).cuda()
                with autocast(enabled=args.fp16):
                    predictions = model(
                        box_features=box_features,
                        pairwise_features=decoded_pairwise_features[:, 1:, :],
                        box_padding_mask=box_padding_mask,
                        triplet_padding_mask=triplet_padding_mask,
                        subject_features=decoded_subject_features[:, : -1, :],
                        relation_ids=decoded_relation_ids[:, : -1],
                        object_features=decoded_object_features[:, : -1, :],
                        batch_indices=decoded_batch_indices,
                        subject_indices=decoded_subject_indices,
                        object_indices=decoded_object_indices,
                        contextual_box_features=contextual_box_features
                    )
                loss = F.cross_entropy(input=predictions.flatten(0, 1), target=decoded_targets.flatten(), ignore_index=-100, reduction='none')
                loss = loss.view(batch_size, max_num_pairs) * torch.from_numpy(rewards).cuda()
                triplet_padding_mask = 1.0 - triplet_padding_mask.float()
                loss = torch.sum(loss * triplet_padding_mask) / torch.sum(triplet_padding_mask)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    rl_scaler.scale(loss).backward()
                else:
                    loss.backward()
                rl_global_step += 1
                if rl_global_step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm is not None:
                        if args.fp16:
                            rl_scaler.unscale_(rl_optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=False)
                    if args.fp16:
                        rl_scaler.step(rl_optimizer)
                        rl_scaler.update()
                    else:
                        rl_optimizer.step()
                    rl_optimizer.zero_grad()
                    rl_scheduler.step()

        if epoch % args.eval_interval == 0:
            prediction_dict = dict()
            for (
                    image_id_list,
                    box_features,
                    pairwise_features,
                    box_padding_mask,
                    batch_indices,
                    subject_indices,
                    object_indices,
                    candidates_list,
                    gt_list
            ) in valid_loader:
                box_features = box_features.cuda()
                pairwise_features = pairwise_features.cuda()
                box_padding_mask = box_padding_mask.cuda()
                if args.normalization:
                    pairwise_features = (pairwise_features - pairwise_features_mean) / pairwise_features_std
                decoding_results, _ = model.decode(
                    box_features=box_features,
                    pairwise_features=pairwise_features,
                    box_padding_mask=box_padding_mask,
                    batch_indices=batch_indices,
                    subject_indices=subject_indices,
                    object_indices=object_indices,
                    candidates_list=candidates_list
                )
                for image_id, predictions in zip(image_id_list, decoding_results):
                    prediction_dict[image_id] = predictions
            recall_at_k = evaluate_recall_at_k(data=valid_data, prediction_dict=prediction_dict)
            print(f'Epoch {epoch} Recall@5 {recall_at_k[0]} Recall@10 {recall_at_k[1]} Recall@20 {recall_at_k[2]} Recall@50 {recall_at_k[3]} Recall@100 {recall_at_k[4]}...')
            torch.save({
                'model': model.state_dict(),
                'tf_optimizer': tf_optimizer.state_dict(),
                'rl_optimizer': rl_optimizer.state_dict(),
                'epoch': epoch,
                'Recall@5': recall_at_k[0],
                'Recall@10': recall_at_k[1],
                'Recall@20': recall_at_k[2],
                'Recall@50': recall_at_k[3],
                'Recall@100': recall_at_k[4],
                'box_features_scaler': box_features_scaler,
                'pairwise_features_scaler': pairwise_features_scaler
            }, f'model/{args.dataset}/model_epoch_{epoch}.bin')


if __name__ == '__main__':
    main()
