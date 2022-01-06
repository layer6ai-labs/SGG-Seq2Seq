import argparse
import copy
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def generate_square_subsequent_mask(seq_len: int) -> torch.FloatTensor:
    mask = (torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    return mask


class Transformer(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Transformer, self).__init__()
        self.args = args

        self.box_features_fc = nn.Linear(in_features=args.num_box_features, out_features=args.d_model)
        self.triplet_encoder = TripletEncoder(args=args)
        self.triplet_decoder = TripletDecoder(args=args)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=args.activation
            ),
            num_layers=args.num_encoder_layers,
            norm=nn.LayerNorm(normalized_shape=args.d_model)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=args.activation
            ),
            num_layers=args.num_decoder_layers,
            norm=nn.LayerNorm(normalized_shape=args.d_model)
        )

        self.triplet_mask = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            box_features: torch.FloatTensor,
            pairwise_features: torch.FloatTensor,
            subject_features: torch.FloatTensor,
            relation_ids: torch.LongTensor,
            object_features: torch.FloatTensor,
            box_padding_mask: torch.BoolTensor,
            triplet_padding_mask: torch.BoolTensor,
            batch_indices: np.ndarray,
            subject_indices: np.ndarray,
            object_indices: np.ndarray,
            contextual_box_features: Optional[torch.FloatTensor] = None
    ):
        # if num_negatives > 0:
        #     batch_size, max_num_pairs, _, _ = pairwise_features.size()
        # else:
        #     batch_size, max_num_pairs, _ = pairwise_features.size()

        batch_size, max_num_pairs = pairwise_features.size()[:2]

        if contextual_box_features is None:
            contextual_box_features = self.box_features_fc(box_features)
            contextual_box_features = self.encoder(
                src=contextual_box_features.transpose(0, 1),
                mask=None,
                src_key_padding_mask=box_padding_mask
            ).transpose(0, 1)

        triplet_encoded = self.triplet_encoder(
            subject_features=subject_features,
            object_features=object_features,
            relation_ids=relation_ids
        )
        decoder_states = self.decoder(
            tgt=triplet_encoded.transpose(0, 1),
            memory=contextual_box_features.transpose(0, 1),
            tgt_mask=generate_square_subsequent_mask(seq_len=max_num_pairs).to(box_features),
            memory_mask=None,
            tgt_key_padding_mask=triplet_padding_mask,
            memory_key_padding_mask=box_padding_mask
        ).transpose(0, 1)

        if pairwise_features.ndim == 3:
            decoder_outputs = self.triplet_decoder(
                subject_contextual_features=contextual_box_features[batch_indices, subject_indices, :].view(batch_size, self.args.max_k, self.args.d_model),
                object_contextual_features=contextual_box_features[batch_indices, object_indices, :].view(batch_size, self.args.max_k, self.args.d_model),
                decoder_states=decoder_states,
                pairwise_features=pairwise_features
            )
        else:
            decoder_outputs = self.triplet_decoder(
                subject_contextual_features=contextual_box_features[batch_indices, subject_indices, :].view(batch_size, max_num_pairs, self.args.num_negatives + 1, self.args.d_model),
                object_contextual_features=contextual_box_features[batch_indices, object_indices, :].view(batch_size, max_num_pairs, self.args.num_negatives + 1, self.args.d_model),
                decoder_states=decoder_states[:, :, None, :].repeat(1, 1, self.args.num_negatives + 1, 1),
                pairwise_features=pairwise_features
            )
        return decoder_outputs

    def decode(
            self,
            box_features: torch.FloatTensor,
            pairwise_features: torch.FloatTensor,
            box_padding_mask: torch.BoolTensor,
            batch_indices: np.ndarray,
            subject_indices: np.ndarray,
            object_indices: np.ndarray,
            candidates_list: List[List[int]],
            contextual_box_features: Optional[torch.FloatTensor] = None,
            decoded_subject_features: Optional[torch.FloatTensor] = None,
            decoded_relation_ids: Optional[torch.LongTensor] = None,
            decoded_object_features: Optional[torch.FloatTensor] = None,
            decoded_pairwise_features: Optional[torch.FloatTensor] = None,
            decoding_results: Optional[list] = None,
            t_start: Optional[int] = None,
            sample: Optional[bool] = False
    ):
        batch_size, max_num_pairs, _ = pairwise_features.size()
        candidates_list = copy.deepcopy(candidates_list)
        decoding_results = [[] for _ in range(batch_size)] if decoding_results is None else copy.deepcopy(decoding_results)

        with torch.no_grad():
            if decoded_subject_features is not None and decoded_relation_ids is not None and decoded_object_features is not None and decoded_pairwise_features is not None:
                decoded_subject_features = decoded_subject_features.clone()
                decoded_relation_ids = decoded_relation_ids.clone()
                decoded_object_features = decoded_object_features.clone()
                decoded_pairwise_features = decoded_pairwise_features.clone()
            else:
                decoded_subject_features = torch.zeros(size=(batch_size, self.args.max_k + 1, self.args.num_box_features), dtype=torch.float).to(box_features.device)
                decoded_relation_ids = torch.zeros(size=(batch_size, self.args.max_k + 1), dtype=torch.long).to(box_features.device)
                decoded_object_features = torch.zeros(size=(batch_size, self.args.max_k + 1, self.args.num_box_features), dtype=torch.float).to(box_features.device)
                decoded_pairwise_features = torch.zeros(size=(batch_size, self.args.max_k + 1, self.args.num_pairwise_features), dtype=torch.float).to(box_features.device)
            triplet_padding_mask = torch.ones(size=(batch_size, self.args.max_k + 1), dtype=torch.bool).to(box_features.device)
            if t_start is not None:
                triplet_padding_mask[:, : t_start - 1] = False
            if contextual_box_features is None:
                contextual_box_features = self.box_features_fc(box_features)
                contextual_box_features = self.encoder(
                    src=contextual_box_features.transpose(0, 1),
                    mask=None,
                    src_key_padding_mask=box_padding_mask
                ).transpose(0, 1)

            if self.triplet_mask is None:
                self.triplet_mask = generate_square_subsequent_mask(seq_len=self.args.max_k + 1).to(box_features.device)

            for t in range(1, self.args.max_k + 1) if t_start is None else range(t_start, self.args.max_k + 1):
                triplet_padding_mask[:, t - 1] = False
                triplet_encoded = self.triplet_encoder(
                    subject_features=decoded_subject_features,
                    object_features=decoded_object_features,
                    relation_ids=decoded_relation_ids
                )
                decoder_states = self.decoder(
                    tgt=triplet_encoded.transpose(0, 1),
                    memory=contextual_box_features.transpose(0, 1),
                    tgt_mask=self.triplet_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=triplet_padding_mask,
                    memory_key_padding_mask=box_padding_mask
                )[t - 1: t, :, :].transpose(0, 1)
                decoder_outputs = self.triplet_decoder(
                    subject_contextual_features=contextual_box_features[batch_indices, subject_indices, :].view(batch_size, max_num_pairs, self.args.d_model),
                    object_contextual_features=contextual_box_features[batch_indices, object_indices, :].view(batch_size, max_num_pairs, self.args.d_model),
                    decoder_states=decoder_states.repeat(1, max_num_pairs, 1),
                    pairwise_features=pairwise_features
                )
                predictions = torch.softmax(decoder_outputs, dim=-1).view(batch_size, max_num_pairs * self.args.num_relations)
                predictions = predictions.cpu().numpy()
                for i in range(len(candidates_list)):
                    if len(candidates_list[i]) == 0:
                        continue
                    if sample:
                        probabilities = predictions[i, candidates_list[i]]
                        probabilities /= np.sum(probabilities)
                        prediction = np.random.choice(np.arange(len(candidates_list[i])), p=probabilities)
                    else:
                        prediction = np.argmax(predictions[i, candidates_list[i]])
                    score = predictions[i, candidates_list[i][prediction]]
                    prediction = candidates_list[i].pop(prediction)
                    subject_index = subject_indices[max_num_pairs * i + prediction // self.args.num_relations]
                    relation_id = prediction % self.args.num_relations
                    object_index = object_indices[max_num_pairs * i + prediction // self.args.num_relations]
                    decoded_subject_features[i, t, :] = box_features[i, subject_index, :]
                    decoded_relation_ids[i, t] = relation_id
                    decoded_object_features[i, t, :] = box_features[i, object_index, :]
                    decoded_pairwise_features[i, t, :] = pairwise_features[i, prediction // self.args.num_relations, :]
                    decoding_results[i].append((subject_index, relation_id, object_index, score))
            decoding_cache = contextual_box_features, decoded_subject_features, decoded_relation_ids, decoded_object_features, decoded_pairwise_features

            return decoding_results, decoding_cache


class TripletEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(TripletEncoder, self).__init__()
        in_features = args.num_box_features * 2 + args.relation_embedding_dim
        hidden_features = 2 ** math.floor(math.log2((in_features * args.d_model) ** 0.5))
        self.relation_embeddings = nn.Embedding(
            num_embeddings=args.num_relations,
            embedding_dim=args.relation_embedding_dim
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=args.dnn_dropout),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dnn_dropout),
            nn.Linear(in_features=hidden_features, out_features=args.d_model)
        )

    def forward(
            self,
            subject_features: torch.FloatTensor,
            object_features: torch.FloatTensor,
            relation_ids: torch.LongTensor
    ):
        return self.fc(
            torch.cat([
                subject_features,
                object_features,
                self.relation_embeddings(relation_ids)
            ], dim=-1)
        )


class TripletDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(TripletDecoder, self).__init__()
        in_features = args.d_model * 3 + args.num_pairwise_features
        hidden_features = 2 ** math.floor(math.log2((in_features * args.num_relations) ** 0.5))
        self.fc = nn.Sequential(
            nn.Dropout(p=args.dnn_dropout),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dnn_dropout),
            nn.Linear(in_features=hidden_features, out_features=args.num_relations)
        )

    def forward(
            self,
            subject_contextual_features: torch.Tensor,
            object_contextual_features: torch.Tensor,
            decoder_states: torch.Tensor,
            pairwise_features: torch.Tensor
    ):
        return self.fc(
            torch.cat([
                subject_contextual_features,
                object_contextual_features,
                decoder_states,
                pairwise_features
            ], dim=-1)
        )
