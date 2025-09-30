# coding=utf-8
import os
from turtle import forward
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.functional import norm

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    # norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    # X = torch.div(X, norm)
    X = X / (torch.norm(X,p=1,dim=dim,keepdim=True)+eps)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    # norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    # X = torch.div(X, norm)
    X = X / (torch.norm(X,p=2,dim=dim,keepdim=True)+eps)
    return X

def cosine_similarity(x1, x2, dim=-1):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2)).squeeze(dim=-1)

# calculate the mask according to the given lens
def get_mask(lens):
    """
    :param lens: length of the sequence
    :return: 
    """
    batch = lens.shape[0]
    max_l = int(lens.max())
    mask = torch.arange(max_l).expand(batch, max_l).to(lens.device)
    mask = (mask<lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(lens.device)
    return mask

def get_padding_mask(lens):
    """
    :param lens: length of the sequence
    :return: 
    """
    batch = lens.shape[0]
    max_l = int(lens.max())
    mask = torch.arange(max_l).expand(batch, max_l).to(lens.device)
    mask = (mask>=lens.long().unsqueeze(dim=1)).to(lens.device)
    return mask

# calculate the fine-grained similarity according to the given images and captions
def get_fgsims(imgs, caps):
    bi, n_r, embi = imgs.shape
    bc, n_w, embc = caps.shape
    imgs = imgs.reshape(bi*n_r, embi)
    caps = caps.reshape(bc*n_w, embc).t()
    sims = torch.matmul(imgs,caps)
    sims = sims.reshape(bi, n_r, bc, n_w).permute(0,2,1,3)
    return sims

# calculate the mask of fine-grained similarity according to the given images length and captions length
def get_fgmask(img_lens, cap_lens):
    bi = img_lens.shape[0]
    bc = cap_lens.shape[0]
    max_r = int(img_lens.max())
    max_w = int(cap_lens.max())

    mask_i = torch.arange(max_r).expand(bi, max_r).to(img_lens.device)
    mask_i = (mask_i<img_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(img_lens.device)
    mask_i = mask_i.reshape(bi*max_r,1)

    mask_c = torch.arange(max_w).expand(bc,max_w).to(cap_lens.device)
    mask_c = (mask_c<cap_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(cap_lens.device)
    mask_c = mask_c.reshape(bc*max_w,1).t()

    mask = torch.matmul(mask_i,mask_c).reshape(bi, max_r, bc, max_w).permute(0,2,1,3)
    return mask

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8):
        super(SelfAttention, self).__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def attention(self, x: torch.Tensor, lens: torch.Tensor=None):
        mask = get_padding_mask(lens).squeeze() if lens is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=mask)[0]

    def forward(self, x: torch.Tensor, lens: torch.Tensor=None):
        return x + self.attention(self.ln(x),lens)

class Transformer(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def SCAN_attention(query, context, smooth=9):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext, attn


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        # 初始化两个线性变换层，用于生成注意力映射
        # mk: 将输入特征从d_model维映射到S维，即降维到共享内存空间的大小
        self.mk = nn.Linear(d_model, S, bias=False)
        # mv: 将降维后的特征从S维映射回原始的d_model维
        self.mv = nn.Linear(S, d_model, bias=False)
        # 使用Softmax函数进行归一化处理
        self.softmax = nn.Softmax(dim=1)
        # 调用权重初始化函数
        self.init_weights()
        # 对输入特征进行归一化
        self.ln = nn.LayerNorm(d_model)

    def init_weights(self):
        # 自定义权重初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层的权重进行Kaiming正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    # 如果有偏置项，则将其初始化为0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 对批归一化层的权重和偏置进行常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对线性层的权重进行正态分布初始化，偏置项（如果存在）初始化为0
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        # 归一化处理
        queries1 = self.ln(queries)
        # 前向传播函数
        attn = self.mk(queries1)  # 使用mk层将输入特征降维到S维
        attn = self.softmax(attn)  # 对降维后的特征进行Softmax归一化处理
        # 对归一化后的注意力分数进行标准化，使其和为1
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)  # 使用mv层将注意力特征映射回原始维度
        return out + queries

# 定义XNorm函数，对输入x进行规范化  ------ 协助定义UFO
def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor

# UFOAttention类继承自nn.Module
class UFOAttention(nn.Module):
    '''
    实现一个改进的自注意力机制，具有线性复杂度。
    '''

    # 初始化函数
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: 模型的维度
        :param d_k: 查询和键的维度
        :param d_v: 值的维度
        :param h: 注意力头数
        '''
        super(UFOAttention, self).__init__()
        # 初始化四个线性层：为查询、键、值和输出转换使用
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        # gamma参数用于规范化
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 前向传播
    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # 通过线性层将查询、键、值映射到新的空间
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        # 计算键和值的乘积，然后对结果进行规范化
        kv = torch.matmul(k, v)  # bs,h,c,c
        kv_norm = XNorm(kv, self.gamma)  # bs,h,c,c
        q_norm = XNorm(q, self.gamma)  # bs,h,n,c
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

# 1、sim3D注意力
class SimAM3D(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM3D, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
        
    def forward(self, x):
        # 假设输入形状为 [batch_size, height, width]
        b, h, w = x.size()
        
        # 添加一个通道维度
        x_4d = x.unsqueeze(1)  # [batch_size, 1, height, width]
        
        # 计算特征图的元素数量减一
        n = h * w - 1
        
        # 计算与均值的差的平方
        x_minus_mu_square = (x_4d - x_4d.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # 计算注意力权重
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 应用注意力并移除通道维度
        out = (x_4d * self.act(y)).squeeze(1)
        return out

# 2、提出的ESAttention
class ESAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.attention1 = ExternalAttention(d_model, S)
        self.attention2 = SimAM3D()

    def forward(self, x):
        value = self.attention1(x)
        out = self.attention2(value)
        return out

# 3、SE Net模块
class SENet3D(nn.Module):
    # 初始化SE模块，适用于三维输入，feature_dim为特征维度，reduction为降维比率
    def __init__(self, feature_dim=1024, reduction=16):
        super().__init__()
        # 不再使用2D池化，而是直接在特征维度上进行操作
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(feature_dim // reduction, feature_dim, bias=False),  # 升维
            nn.Sigmoid()  # Sigmoid激活函数，输出特征维度的重要性系数
        )
        self.ln = nn.LayerNorm(feature_dim)

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 前向传播方法
    def forward(self, x):
        # x的形状为 [batch_size, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = x.size()
        
        # 计算每个特征维度的平均值，得到全局特征表示
        x_temp = x
        x = self.ln(x)
        # 在序列维度上进行平均池化
        y = torch.mean(x, dim=1)  # [batch_size, feature_dim]
        
        # 通过全连接层计算特征重要性
        y = self.fc(y)  # [batch_size, feature_dim]
        
        # 将特征重要性扩展为与输入相同的形状
        y = y.unsqueeze(1).expand_as(x)  # [batch_size, seq_len, feature_dim]
        
        # 将特征重要性应用到原始输入上
        return x * y + x_temp

# 4、提出的SEAttention
class SEAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.attention1 = SelfAttention(d_model)
        self.attention2 = SENet3D(feature_dim=d_model, reduction=S)

    def forward(self, x):
        value = self.attention1(x)
        out = self.attention2(value)
        return out

# 5、ASPP-PPM
class ASPPConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.bn.weight is not None:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ASPPPolling1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPolling1D, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.bn.weight is not None:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        size = x.shape[2:] # (N,)
        pooled_x = self.gap(x)
        out = self.conv(pooled_x)
        out = self.bn(out)
        out = self.relu(out)
        # Upsample to original sequence length N
        return F.interpolate(out, size=size, mode='linear', align_corners=False)

class ASPPModule1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super(ASPPModule1D, self).__init__()
        self.dilations = dilations
        
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        ) # 1x1 conv branch

        for dilation in dilations:
            self.convs.append(ASPPConv1D(in_channels, out_channels, dilation))
        
        self.image_pooling = ASPPPolling1D(in_channels, out_channels)
        
        # Total channels after concatenation
        total_out_channels = (len(dilations) + 2) * out_channels 
        self.project = nn.Sequential(
            nn.Conv1d(total_out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1) # Optional dropout
        )
        
        # Initialize 1x1 projection
        if isinstance(self.project[0], nn.Conv1d):
            nn.init.kaiming_normal_(self.project[0].weight, mode='fan_out', nonlinearity='relu')
        if isinstance(self.project[1], nn.BatchNorm1d) and self.project[1].weight is not None:
             nn.init.constant_(self.project[1].weight, 1)
             nn.init.constant_(self.project[1].bias, 0)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, C_in, N)
        Returns:
            torch.Tensor: (B, C_out, N)
        """
        res = []
        for conv_module in self.convs:
            res.append(conv_module(x))
        res.append(self.image_pooling(x))
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class PPMPoolingBranch1D(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch, pool_scale):
        super(PPMPoolingBranch1D, self).__init__()
        self.pool_scale = pool_scale # Number of segments to pool into
        self.pool = nn.AdaptiveAvgPool1d(pool_scale)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU()
        )
        # Initialize weights
        if isinstance(self.conv[0], nn.Conv1d):
            nn.init.kaiming_normal_(self.conv[0].weight, mode='fan_out', nonlinearity='relu')
        if isinstance(self.conv[1], nn.BatchNorm1d) and self.conv[1].weight is not None:
             nn.init.constant_(self.conv[1].weight, 1)
             nn.init.constant_(self.conv[1].bias, 0)

    def forward(self, x, original_size_n):
        """
        Args:
            x (torch.Tensor): (B, C_in, N)
            original_size_n (int): Original sequence length for upsampling
        Returns:
            torch.Tensor: (B, C_out_branch, N_original)
        """
        pooled = self.pool(x) # (B, C_in, pool_scale)
        conved = self.conv(pooled) # (B, C_out_branch, pool_scale)
        # Upsample to original sequence length N
        upsampled = F.interpolate(conved, size=original_size_n, mode='linear', align_corners=False)
        return upsampled

class PPMModule1D(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch, pool_scales=[1, 2, 3, 6]):
        super(PPMModule1D, self).__init__()
        self.pool_scales = pool_scales
        self.branches = nn.ModuleList()
        for scale in pool_scales:
            self.branches.append(PPMPoolingBranch1D(in_channels, out_channels_per_branch, scale))
        
        # Total channels after concatenation (input + all branches)
        total_out_channels = in_channels + (len(pool_scales) * out_channels_per_branch)
        self.project = nn.Sequential(
            nn.Conv1d(total_out_channels, in_channels, kernel_size=1, bias=False), # Project back to in_channels or a new dim
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.1) # Optional dropout
        )
         # Initialize 1x1 projection
        if isinstance(self.project[0], nn.Conv1d):
            nn.init.kaiming_normal_(self.project[0].weight, mode='fan_out', nonlinearity='relu')
        if isinstance(self.project[1], nn.BatchNorm1d) and self.project[1].weight is not None:
             nn.init.constant_(self.project[1].weight, 1)
             nn.init.constant_(self.project[1].bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, C_in, N)
        Returns:
            torch.Tensor: (B, C_projected_out, N)
        """
        B, C, N_original = x.shape
        
        ppm_outputs = [x] # Include the original feature
        for branch in self.branches:
            ppm_outputs.append(branch(x, N_original))
            
        combined = torch.cat(ppm_outputs, dim=1)
        return self.project(combined)

class ASPP_PPM(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 aspp_out_channels_per_branch=256, # ASPP branches output channels
                 aspp_dilations=[1, 3, 6],         # Reduced for 1D context
                 ppm_out_channels_per_branch=256,  # PPM branches output channels
                 ppm_pool_scales=[1, 2, 4],        # Pyramid levels for PPM
                 final_output_dim=512):
        super(ASPP_PPM, self).__init__()

        self.input_dim = input_dim
        
        # ASPP Module
        # The ASPPModule1D's 'out_channels' parameter is the target for its internal projection
        # It will internally calculate total channels and project to this value.
        aspp_internal_projection_dim = aspp_out_channels_per_branch * (len(aspp_dilations) + 2) // 2 # Example intermediate dim
        aspp_internal_projection_dim = max(aspp_internal_projection_dim, aspp_out_channels_per_branch) # ensure it's at least aspp_out_channels_per_branch

        self.aspp_module = ASPPModule1D(in_channels=input_dim,
                                        out_channels=aspp_internal_projection_dim, # This is the output of ASPP's final projection
                                        dilations=aspp_dilations)
        
        # PPM Module
        # PPM will take output from ASPP.
        # PPM's internal projection will output to ppm_internal_projection_dim (same as its input from aspp)
        ppm_input_dim = aspp_internal_projection_dim
        self.ppm_module = PPMModule1D(in_channels=ppm_input_dim,
                                      out_channels_per_branch=ppm_out_channels_per_branch,
                                      pool_scales=ppm_pool_scales)
                                      # PPM's project layer outputs ppm_input_dim channels

        # Final projection to desired output dimension
        self.final_projection = nn.Linear(ppm_input_dim, final_output_dim)
        self.final_activation = nn.ReLU() # Or other activation
        self.final_norm = nn.LayerNorm(final_output_dim) # Optional LayerNorm

        # Initialize final projection
        nn.init.xavier_uniform_(self.final_projection.weight)
        if self.final_projection.bias is not None:
            nn.init.zeros_(self.final_projection.bias)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image features, shape (B, n, input_dim)
        Returns:
            torch.Tensor: Processed features, shape (B, n, final_output_dim)
        """
        B, N, D_in = x.shape
        
        # Transpose for Conv1D: (B, N, D_in) -> (B, D_in, N)
        x_transposed = x.transpose(1, 2)
        
        # ASPP Module
        aspp_out = self.aspp_module(x_transposed) # (B, C_aspp_out, N)
        
        # PPM Module
        ppm_out = self.ppm_module(aspp_out) # (B, C_ppm_out, N), where C_ppm_out is same as C_aspp_out
        
        # Transpose back for Linear layer: (B, C_ppm_out, N) -> (B, N, C_ppm_out)
        ppm_out_transposed = ppm_out.transpose(1, 2)
        
        # Final Projection
        final_features = self.final_projection(ppm_out_transposed) # (B, N, final_output_dim)
        final_features = self.final_norm(self.final_activation(final_features))
        
        return final_features

# 6、ASPP
# 定义一个包含空洞卷积、批量归一化和ReLU激活函数的子模块
# 针对 [B, C, H, 1] 的输入进行优化
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            # 空洞卷积，kernel_size=(3,1) 对应 H 维度
            # padding=(dilation,0) 保持 H 维度大小不变
            # dilation=(dilation,1) 只在 H 维度应用膨胀
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), 
                      padding=(dilation, 0), dilation=(dilation, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 定义一个全局平均池化后接卷积、批量归一化和ReLU的子模块
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出 (B, C, 1, 1)
            nn.Conv2d(in_channels, out_channels, 1, bias=False), # 1x1卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [B, C_in, H, W_in=1]
        size = x.shape[-2:]  # 保存输入特征图的空间维度, e.g., (H, 1)
        # super.forward(x) output shape: [B, C_out, 1, 1]
        pooled_x = super(ASPPPooling, self).forward(x)
        # 通过双线性插值将特征图大小调整回原始输入空间大小 (H, 1)
        return F.interpolate(pooled_x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256): # 允许指定 out_channels
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), # 1x1卷积，kernel_size=(1,1)
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 根据不同的膨胀率添加空洞卷积模块
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 添加全局平均池化模块
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 投影层：融合所有分支的输出
        # 分支数量 = 1 (1x1 conv) + len(atrous_rates) (ASPPConv) + 1 (Pooling)
        num_branches = 1 + len(atrous_rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, 1, bias=False), # 1x1卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        # 输入 x 的形状: [B, N, C_in]
        # 1. 调整形状以适应 Conv2d: [B, C_in, N, 1]
        #    C_in 是输入通道数，N 是可变长度（我们视其为H），W=1
        tmp = x
        x = x.permute(0, 2, 1)  # [B, C_in, N]
        x = x.unsqueeze(-1)     # [B, C_in, N, 1]

        res = []
        for conv_module in self.convs: # Renamed 'conv' to 'conv_module' to avoid conflict
            res.append(conv_module(x))
        
        res = torch.cat(res, dim=1) # 在通道维度上拼接, dim=1 for [B, C, H, W]
        # res shape: [B, num_branches * C_out, N, 1]
        
        x_projected = self.project(res)
        # x_projected shape: [B, C_out, N, 1]

        # 2. 调整回原始类似的形状: [B, N, C_out]
        x_projected = x_projected.squeeze(-1)    # [B, C_out, N]
        x_projected = x_projected.permute(0, 2, 1) # [B, N, C_out]
        
        return x_projected + tmp


class PPM(nn.Module):
    """
    金字塔池化模块 (Pyramid Pooling Module)
    适应 [B, N, C] 形状的输入, 其中 N 是可变序列长度。
    """
    def __init__(self, in_channels, reduction_channels, pool_sizes):
        """
        初始化PPM模块。

        参数:
        in_channels (int): 输入特征的通道数 (对应你数据中的 1024)。
        reduction_channels (int): 每个金字塔层级池化后，通过1x1卷积降维到的通道数。
                                  这有助于减少后续拼接时的参数量。
        pool_sizes (tuple of int): 一个包含多个整数的元组，定义了自适应平均池化的输出尺寸。
                                   例如 (1, 2, 3, 6) 表示分别池化成 1个、2个、3个、6个特征点。
        """
        super(PPM, self).__init__()
        self.in_channels = in_channels
        self.reduction_channels = reduction_channels
        self.pool_sizes = pool_sizes

        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            self.stages.append(nn.Sequential(
                # 1D 自适应平均池化，作用于序列长度 N 这个维度
                nn.AdaptiveAvgPool1d(output_size=pool_size),
                # 1x1 卷积，用于降低通道数
                nn.Conv1d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(reduction_channels),
                nn.ReLU(inplace=True)
            ))

        # 融合层：将原始特征和所有金字塔池化后的特征拼接起来，再通过一个1x1卷积
        # 拼接后的总通道数 = 原始通道数 + 金字塔层数 * 每个金字塔层的降维后通道数
        total_out_channels = in_channels + (len(pool_sizes) * reduction_channels)
        
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(total_out_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量，形状为 [B, N, C_in]，其中 C_in == self.in_channels。
                          B是批次大小, N是序列长度, C_in是特征维度。

        返回:
        torch.Tensor: 输出张量，形状与输入x相同 [B, N, C_in]。
        """
        tmp = x
        batch_size, sequence_length, num_channels = x.shape

        # PyTorch的Conv1d和AdaptiveAvgPool1d期望的输入是 [B, C, N]
        # 所以我们需要先转换维度
        x_transposed = x.permute(0, 2, 1)  # 形状变为 [B, C_in, N]

        # 存储所有需要拼接的特征，首先加入原始特征（转置后的）
        features_to_concat = [x_transposed]

        for stage in self.stages:
            # 对 x_transposed 进行池化和卷积
            pooled_features = stage(x_transposed)  # 形状 [B, reduction_channels, pool_size]
            
            # 将池化后的特征上采样到原始序列长度 N
            # 使用 'linear' 模式进行1D插值
            upsampled_features = F.interpolate(
                pooled_features, 
                size=sequence_length, 
                mode='linear', 
                align_corners=False # 对于linear模式，通常设为False
            ) # 形状 [B, reduction_channels, N]
            features_to_concat.append(upsampled_features)

        # 沿着通道维度 (dim=1) 拼接所有特征
        concatenated_features = torch.cat(features_to_concat, dim=1) 
        # 形状 [B, in_channels + len(pool_sizes)*reduction_channels, N]

        # 通过融合卷积层，将通道数恢复到原始的 in_channels
        fused_features = self.conv_fusion(concatenated_features) # 形状 [B, in_channels, N]

        # 将维度转置回原始的 [B, N, C] 格式
        output = fused_features.permute(0, 2, 1) # 形状 [B, N, in_channels]

        return output + tmp

# 8、