import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from config import NUM_BOX_FEATURES, NUM_PAIRWISE_FEATURES, NUM_NEGATIVES


def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.tgt_relation_embeddings = nn.Embedding(
            num_embeddings=args.num_relations,
            embedding_dim=args.d_model
        )

        self.tgt_fc = nn.Linear(
            in_features=args.d_model * 3,
            out_features=args.d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation
        )
        encoder_norm = nn.LayerNorm(normalized_shape=args.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args.num_encoder_layers,
            norm=encoder_norm
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation
        )
        decoder_norm = nn.LayerNorm(normalized_shape=args.d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=args.num_decoder_layers,
            norm=decoder_norm
        )

        num_features = NUM_BOX_FEATURES * 3 + NUM_PAIRWISE_FEATURES

        self.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=num_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=num_features // 2, out_features=args.num_relations)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            src: torch.Tensor,
            tgt_subject: torch.Tensor,
            tgt_relation: torch.Tensor,
            tgt_object: torch.Tensor,
            tgt_features: torch.Tensor,
            tgt_mask: Optional[torch.Tensor],
            src_key_padding_mask: torch.Tensor,
            tgt_key_padding_mask: torch.Tensor,
            batch_indices: np.ndarray,
            subject_indices: np.ndarray,
            object_indices: np.ndarray,
            cursor: int = None
    ):
        tgt = self.tgt_fc(torch.cat([
            tgt_subject,
            self.tgt_relation_embeddings(tgt_relation),
            tgt_object
        ], dim=-1))
        src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
        memory = self.encoder(
            src=src,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask if tgt_mask is not None else generate_square_subsequent_mask(seq_len=tgt.size(0)).to(tgt.device),
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        memory, output = memory.transpose(0, 1), output.transpose(0, 1)
        if cursor is not None:
            features = torch.cat([
                memory[batch_indices, subject_indices, :].view(memory.size(0), -1, memory.size(-1)),
                output[:, cursor - 1, None, :].repeat(1, tgt_features.size(1), 1),
                memory[batch_indices, object_indices, :].view(memory.size(0), -1, memory.size(-1)),
                tgt_features
            ], dim=-1)
        else:
            features = torch.cat([
                memory[batch_indices, subject_indices, :].view(memory.size(0), -1, NUM_NEGATIVES + 1, memory.size(-1)),
                output[:, :, None, :].repeat(1, 1, NUM_NEGATIVES + 1, 1),
                memory[batch_indices, object_indices, :].view(memory.size(0), -1, NUM_NEGATIVES + 1, memory.size(-1)),
                tgt_features
            ], dim=-1)
        predictions = self.fc(features)
        return predictions
