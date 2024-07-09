'''
Vit as backbone
'''
from Vision_Transformer_with_mask import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type
from torch import Tensor, nn

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class MLP_att_out(nn.Module):

    def __init__(self, input_dim, inter_dim=None, output_dim=None, activation="relu", drop=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim
        if inter_dim is None: self.inter_dim=input_dim
        if output_dim is None: self.output_dim=input_dim

        self.linear1 = nn.Linear(self.input_dim, self.inter_dim)
        self.activation = self._get_activation_fn(activation)
        self.dropout3 = nn.Dropout(drop)
        self.linear2 = nn.Linear(self.inter_dim, self.output_dim)
        self.dropout4 = nn.Dropout(drop)
        self.norm3 = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        x = x + self.dropout4(x)
        x = self.norm3(x)
        return x

    def _get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class FusionAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
        """
        super().__init__()
        self.self_attn = Attention_ori(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_mask_to_image = CrossAttention(dim=embedding_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_mask = CrossAttention(dim=embedding_dim, num_heads=num_heads)


    def forward(self, img_emb: Tensor, mask_emb: Tensor, atten_mask: Tensor) -> Tuple[ Tensor]:
        # Self attention block #最开始的时候 queries=query_pe
        #queries: Tensor, keys: Tensor
        queries = mask_emb
        attn_out = self.self_attn(queries)  #小图
        queries = attn_out
        #queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, mask attending to image embedding
        q = queries #1,5,256
        k = img_emb  # v是值，因此用keys？
        input_x = torch.cat((q, k), dim=1)  # 2 50 768
        attn_out = self.cross_attn_mask_to_image(input_x) #TODO 要不要mask呢 交叉的时候 先不用试试
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = img_emb
        k = queries
        input_x = torch.cat((q, k), dim=1)
        attn_out = self.cross_attn_image_to_mask(input_x)
        img_emb = img_emb + attn_out
        img_emb = self.norm4(img_emb)

        return img_emb



class my_model2(nn.Module):
    '''不用mask的版本'''
    def __init__(self, pretrained=False,num_classes=3,in_chans=1,img_size=224, **kwargs):
        super().__init__()
        self.backboon1 = vit_base_patch16_224(pretrained=False,in_chans=in_chans, as_backbone=True,img_size=img_size)

        self.self_atten_img = Attention_ori(dim= self.backboon1.embed_dim, num_heads=self.backboon1.num_heads)
        self.self_atten_mask = Attention_ori(dim=self.backboon1.embed_dim, num_heads=self.backboon1.num_heads)
        self.cross_atten = FusionAttentionBlock(embedding_dim=self.backboon1.embed_dim, num_heads=self.backboon1.num_heads)

        self.mlp = MLP_att_out(input_dim=self.backboon1.embed_dim * 3, output_dim=self.backboon1.embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.backboon1.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        ### 新加的
        self.norm1 = nn.LayerNorm(self.backboon1.embed_dim)
        self.norm2 = nn.LayerNorm(self.backboon1.embed_dim)
        self.norm3 = nn.LayerNorm(self.backboon1.embed_dim)
    def forward(self, img, mask):
        x1 = self.backboon1(torch.cat((img, torch.zeros_like(img)), dim=1))  # TODO 是否用同一模型 还是不同 中间是否融合多尺度
        x2 = self.backboon1(torch.cat((img * mask, torch.zeros_like(img)), dim=1))  # 输出经过了归一化层 #小图
        #自注意力+残差
        x2_atten_mask = self.backboon1.atten_mask
        x1_atten = self.self_atten_img(x1)
        x2_atten = self.self_atten_mask(x2)
        x1_out = self.norm1(x1 + x1_atten)
        x2_out = self.norm2(x2 + x2_atten)
        #交叉注意力
        corss_out = self.norm3(self.cross_atten(x1, x2, x2_atten_mask))
        #得到输出特征
        out = torch.concat((x1_out, corss_out, x2_out), dim=1)
        out= self.avgpool(out.transpose(1, 2))  # B C 1
        out = torch.flatten(out, 1)
        out = self.head(out)
        return out




