import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class CodePredictionLoss(nn.Module):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, latent_size=256):
        super(CodePredictionLoss, self).__init__()

        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False)
        )
        
    def forward(self, y_hat):
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, y_hat.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(y_hat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n
        return logits
    

class CodeLevelLoss(nn.Module):

    def __init__(self, codebook_size, dim_embd=512, 
                 n_head=8, n_layers=9, latent_size=256):
        super(CodeLevelLoss, self).__init__()
        self.codebook_size = codebook_size
        self.transformer = CodePredictionLoss(
            dim_embd=dim_embd, 
            n_head=n_head,
            n_layers=n_layers,
            codebook_size=codebook_size,
            latent_size=latent_size,
        )
        self.mse = nn.MSELoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, y_hat, y_gt, y_label, use_transformer=True):
        B, C, H, W = y_hat.shape
        N = H * W

        y_label = y_label.reshape(B * N)

        # part 1: mse 
        feat_mse_loss = self.mse(y_hat, y_gt)

        # part 2: code idx prediction loss
        if use_transformer:
            y_hat = y_hat.reshape(B, C, N)
            logits = self.transformer(y_hat)
            logits = logits.reshape(B * C, -1)
            cross_entropy_loss = self.cross_entropy(logits, y_label)
        else:
            cross_entropy_loss = 0.0
        
        return {
            "mse_loss": feat_mse_loss,
            "cross_entropy_loss": cross_entropy_loss
        }