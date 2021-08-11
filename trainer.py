import argparse
from collections import defaultdict
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import NUM_BOX_FEATURES, MAX_LEN
from data_utils import VRDDataset, VRDInferenceDataset, collate_fn, inference_collate_fn
from evaluation_utils import evaluate_per_image_recall
from file_utils import load_object_from_file, dump_object_to_file
from transformer import Transformer, generate_square_subsequent_mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-relations', type=int, default=71)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], default='relu')
    parser.add_argument('--num-encoder-layers', type=int, default=2)
    parser.add_argument('--num-decoder-layers', type=int, default=2)
    parser.add_argument('--train-batch-size', type=int, default=512)
    parser.add_argument('--valid-batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    args.d_model = NUM_BOX_FEATURES
    args.fp16 = args.fp16 and torch.cuda.is_available()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Transformer(args=args)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    train_dataset = VRDDataset()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_data = load_object_from_file('output/valid_data.dat')
    valid_dataset = VRDInferenceDataset()
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=inference_collate_fn, drop_last=False)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    inference_tgt_mask = generate_square_subsequent_mask(seq_len=MAX_LEN + 1).to(device)
    if args.fp16:
        scaler = GradScaler()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        progress_bar = tqdm(train_dataloader)
        epoch_loss = 0.0
        epoch_count = 0
        for batch in progress_bar:
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            box_features, pairwise_features, subject_features, relation_ids, object_features, src_key_padding_mask, tgt_key_padding_mask, batch_indices, subject_indices, object_indices, targets = batch
            optimizer.zero_grad()
            if args.fp16:
                with autocast():
                    predictions = model.forward(
                        src=box_features,
                        tgt_subject=subject_features,
                        tgt_relation=relation_ids,
                        tgt_object=object_features,
                        tgt_features=pairwise_features,
                        tgt_mask=None,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        batch_indices=batch_indices,
                        subject_indices=subject_indices,
                        object_indices=object_indices
                    )
                    loss = criterion(predictions.view(-1, args.num_relations), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model.forward(
                    src=box_features,
                    tgt_subject=subject_features,
                    tgt_relation=relation_ids,
                    tgt_object=object_features,
                    tgt_features=pairwise_features,
                    tgt_mask=None,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    batch_indices=batch_indices,
                    subject_indices=subject_indices,
                    object_indices=object_indices
                )
                loss = criterion(predictions.view(-1, args.num_relations), targets)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_count += 1
            progress_bar.set_description('Epoch {}: loss {:.4f}'.format(epoch, epoch_loss / epoch_count))
        if epoch % 5 == 0:
            model.eval()
            prediction_dict = defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(valid_dataloader):
                    batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
                    image_id_list, box_features, pairwise_features, src_key_padding_mask, batch_indices, subject_indices, object_indices, candidates_list = batch
                    batch_size = len(candidates_list)
                    max_num_pairs = pairwise_features.shape[1]
                    subject_features = torch.zeros(size=(batch_size, MAX_LEN + 1, NUM_BOX_FEATURES), dtype=torch.float, device=device)
                    relation_ids = torch.zeros(size=(batch_size, MAX_LEN + 1), dtype=torch.long, device=device)
                    object_features = torch.zeros(size=(batch_size, MAX_LEN + 1, NUM_BOX_FEATURES), dtype=torch.float, device=device)
                    tgt_key_padding_mask = torch.ones(size=(batch_size, MAX_LEN + 1), dtype=torch.bool, device=device)
                    tgt_key_padding_mask[:, 0] = False
                    for t in range(1, MAX_LEN + 1):
                        predictions = model.forward(
                            src=box_features,
                            tgt_subject=subject_features,
                            tgt_relation=relation_ids,
                            tgt_object=object_features,
                            tgt_features=pairwise_features,
                            tgt_mask=inference_tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            batch_indices=batch_indices,
                            subject_indices=subject_indices,
                            object_indices=object_indices,
                            cursor=t
                        )
                        predictions = torch.softmax(predictions, dim=-1)
                        predictions = predictions.cpu().numpy().reshape((len(candidates_list), -1))
                        for i, candidates in enumerate(candidates_list):
                            if len(candidates_list[i]) == 0:
                                continue
                            prediction = np.argmax(predictions[i, candidates_list[i]])
                            score = predictions[i, candidates_list[i][prediction]]
                            prediction = candidates_list[i].pop(prediction)
                            subject_index = subject_indices[max_num_pairs * i + prediction // 71]
                            relation_id = prediction % 71
                            object_index = object_indices[max_num_pairs * i + prediction // 71]
                            subject_features[i, t, :] = box_features[i, subject_index, :]
                            relation_ids[i, t] = relation_id
                            object_features[i, t, :] = box_features[i, object_index, :]
                            prediction_dict[image_id_list[i]].append((subject_index, relation_id, object_index, score))
                        tgt_key_padding_mask[:, t] = False
            print(evaluate_per_image_recall(data=valid_data, prediction_dict=prediction_dict))
            # dump_object_to_file(prediction_dict, 'cache/prediction_dict_epoch_{}.dat'.format(epoch))


if __name__ == '__main__':
    main()
