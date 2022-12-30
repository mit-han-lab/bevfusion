"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout



def nan_to_num(tensor, nan=0.0):
    mask = ~torch.isfinite(tensor)
    new_tensor = tensor.masked_fill(mask, 0)
    return new_tensor

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)

        # Remove all nans from softmax
        A = nan_to_num(A)

        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()





import torch
import torch.nn as nn


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    feature = feature.permute(0, 3, 1, 2).contiguous()
  
    return feature

def get_graph_xyz(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    feature = feature.permute(0, 3, 1, 2).contiguous()
  
    return feature

class corss_attention(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(corss_attention, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, search_feat, search_xyz, template_feat, template_xyz, mask=None):
        """
        search_feat:          [B, C32, Ns1024]
        search_xyz:           [B, Ns1024, C3]
        template_feat:        [B, C32, Nt512]
        template_xyz:         [B, Nt512, C3]
        """
        bs = search_feat.size(0)
        search_feat = search_feat.permute(0, 2, 1)
        template_feat = template_feat.permute(0, 2, 1)

        # search_feat_pos = search_feat + self.pos_mlp(search_xyz)
        template_feat_pos = template_feat + self.pos_mlp(template_xyz)
        
        # multi-head attention
        query = self.q_proj(search_feat).view(bs, -1, self.nhead, self.dim)          
        key = self.k_proj(template_feat).view(bs, -1, self.nhead, self.dim)           
        value = self.v_proj(template_feat_pos).view(bs, -1, self.nhead, self.dim)        
        
        message = self.attention(query, key, value, q_mask=None, kv_mask=None)    
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))            
        message = self.norm1(message)                                             
        
        # feed-forward network
        message = self.mlp(torch.cat([search_feat, message], dim=2))                   
        message = self.norm2(message)
        
        return (search_feat + message).permute(0, 2, 1)

class local_self_attention(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 knum=32):
        super(local_self_attention, self).__init__()
        
        self.d_model = d_model
        self.dim = d_model // nhead
        self.nhead = nhead
        self.knum = knum

        # position encoding
        self.pos_mlp_knn = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(True),
            nn.Linear(32, 32)
        )

        # multi-head attention
        self.q_proj_knn = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_knn = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_knn = nn.Linear(d_model, d_model, bias=False)
        self.attention_knn = LinearAttention()
        self.merge_knn = nn.Linear(d_model, d_model, bias=False)
        
        # feed-forward network
        self.mlp_knn = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1_knn = nn.LayerNorm(d_model)
        self.norm2_knn = nn.LayerNorm(d_model)

    def forward(self, search_feat, search_xyz, mask=None):
        """
        search_feat:          [B, C32, Ns1024]
        search_xyz:           [B, Ns1024, C3]
        """
        bs, ch, ns = search_feat.size()
        
        # ------------------------ find knn ---------------------
        kidx = knn(search_feat, k=self.knum)    # feature space knn
        fea_knn = get_graph_feature(search_feat, k=self.knum, idx=kidx)             # [B, C, N, K]
        fea_knn = fea_knn.permute(0, 2, 3, 1).contiguous()                          # [B, N, K, C]
        fea_knn = fea_knn.view(bs*ns, self.knum, -1)                                # [BN, K, C]
        
        xyz_knn = get_graph_xyz(search_xyz.permute(0, 2, 1), k=self.knum, idx=kidx) # [B, 3, N, K]
        xyz_knn = xyz_knn.permute(0, 2, 3, 1).contiguous()                          # [B, N, K, 3]
        xyz_knn = xyz_knn.view(bs*ns, self.knum, 3)                                 # [BN, K, 3]
        
        search_feat = search_feat.permute(0, 2, 1).contiguous()     # [B, N, C]
        search_feat = search_feat.view(bs*ns, 1, ch)                # [BN, 1, C]

        # position embedding
        search_feat_pos = search_feat + self.pos_mlp_knn(search_xyz.view(bs*ns, 1, 3))  # [BN, 1, C]
        fea_knn_pos = fea_knn + self.pos_mlp_knn(xyz_knn)                               # [BN, K, C]

        query = self.q_proj_knn(search_feat_pos).view(bs*ns, -1, self.nhead, self.dim)  # [BN, X, (H, D)]
        key = self.k_proj_knn(fea_knn_pos).view(bs*ns, -1, self.nhead, self.dim)        # [BN, X, (H, D)]
        value = self.v_proj_knn(fea_knn_pos).view(bs*ns, -1, self.nhead, self.dim)      # [BN, X, (H, D)]
        
        message = self.attention_knn(query, key, value, q_mask=None, kv_mask=None)      # [BN, X, (H, D)]
        message = self.merge_knn(message.view(bs*ns, -1, self.nhead*self.dim))          # [BN, X, C=H*D]
        message = self.norm1_knn(message)                                               # [BN, X, C=H*D]
        
        message = self.mlp_knn(torch.cat([search_feat, message], dim=2))                # [BN, X, C=H*D]
        message = self.norm2_knn(message)                                               # [BN, X, C=H*D]

        return (search_feat+message).view(bs, ns, self.d_model).permute(0, 2, 1)        # [B, C, N]