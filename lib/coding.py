# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules import get_mask, get_fgsims, get_fgmask, l2norm, cosine_similarity, SCAN_attention

EPS = 1e-8 # epsilon 
MASK = -1 # padding value

# Visual Hard Assignment Coding 静态的VHA
class VHACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims
# class VHACoding(nn.Module):
#     def __init__(self, k=1):  # 新增参数k，默认保持原论文的Top-1行为
#         super().__init__()
#         self.k = k            # 可设为可学习参数或通过外部传入
    
#     def forward(self, imgs, caps, img_lens, cap_lens):
#         max_r = int(img_lens.max())
#         max_w = int(cap_lens.max())
#         sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
#         mask = get_fgmask(img_lens, cap_lens)
#         sims = sims.masked_fill(mask == 0, MASK)
        
#         # 动态选择Top-K区域并聚合（例如取平均）
#         topk_sims = sims.topk(k=self.k, dim=-2)[0]  # (Bi, Bt, K, L)
#         sims = topk_sims.mean(dim=-2)               # 沿区域维度聚合
        
#         return sims

#Texual Hard Assignment Coding
class THACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-1)[0]
        return sims
# class THACoding(nn.Module):
#     def __init__(self, dim=512, init_threshold=0.5):  # 新增动态阈值参数
#         super().__init__()
#         self.threshold_generator = nn.Sequential(
#             nn.Linear(dim, 1),
#             nn.Sigmoid()  # 输出范围 [0, 1]
#         )
#         self.init_threshold = init_threshold
    
#     def forward(self, imgs, caps, img_lens, cap_lens):
#         max_r = int(img_lens.max())
#         max_w = int(cap_lens.max())
#         sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
#         mask = get_fgmask(img_lens, cap_lens)
#         sims = sims.masked_fill(mask == 0, MASK)
        
#         # 动态生成阈值（基于文本全局特征）
#         global_feat = caps.mean(dim=1)  # 文本全局特征 (Bt, d)
#         thresholds = self.threshold_generator(global_feat) * self.init_threshold  # (Bt, 1)
        
#         # 应用阈值过滤低相似度对齐
#         dynamic_mask = (sims >= thresholds.unsqueeze(-1)).float()  # 阈值广播对齐维度
#         dynamic_sims = sims * dynamic_mask
        
#         return dynamic_sims.max(dim=-1)[0]  # 保留最大相似度

class VSACoding(nn.Module):
    def __init__(self,temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)

        # calculate attention
        sims = sims / self.temperature

        sims = torch.softmax(sims.masked_fill(mask==0, -torch.inf),dim=-1) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,caps) # Bi x Bt x K x D
        sims = torch.mul(sims.permute(1,0,2,3),imgs).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bi x Bt x K

        mask = get_mask(img_lens).permute(0,2,1).repeat(1,cap_lens.size(0),1)
        sims = sims.masked_fill(mask==0, -1)
        return sims

class T2ICrossAttentionPool(nn.Module):
    def __init__(self,smooth=9):
        super().__init__()
        self.labmda = smooth

    def forward(self, imgs, caps, img_lens, cap_lens):
        return self.xattn_score_t2i(imgs,caps,cap_lens)

    def xattn_score_t2i(self, images, captions, cap_lens, return_attn=False):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        attentions = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = int(cap_lens[i].item())
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d)
                attn: (n_image, n_region, n_word)
            """
            if return_attn:
                weiContext,attn = SCAN_attention(cap_i_expand, images,self.labmda)
                attentions.append(attn)
            else:
                weiContext,_ = SCAN_attention(cap_i_expand, images,self.labmda)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            col_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
            col_sim = col_sim.mean(dim=1, keepdim=True)
            similarities.append(col_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        if return_attn:return torch.cat(attentions, 0)
        else:return similarities

# max pooling
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims = sims.max(dim=-1)[0]
        return sims

# mean pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        lens = (sims!=MASK).sum(dim=-1)
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)/lens
        return sims

# sum pooling
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)
        return sims

# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        sims = torch.logsumexp(sims/self.temperature,dim=-1)
        return sims

# softmax pooling
class SoftmaxPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        weight = torch.softmax(sims/self.temperature,dim=-1)
        sims = (weight*sims).sum(dim=-1)
        return sims

# def get_coding(coding_type, **args):
#     alpha = args["opt"].alpha
#     if coding_type=="VHACoding":
#         return VHACoding()
#     elif coding_type=="THACoding":
#         return THACoding()
#     elif coding_type=="VSACoding":
#         return VSACoding(alpha)
#     else:
#         raise ValueError("Unknown coding type: {}".format(coding_type))

def get_pooling(pooling_type, **args):
    belta = args["opt"].belta
    if pooling_type=="MaxPooling":
        return MaxPooling()
    elif pooling_type=="MeanPooling":
        return MeanPooling()
    elif pooling_type=="SumPooling":
        return SumPooling()
    elif pooling_type=="SoftmaxPooling":
        return SoftmaxPooling(belta)
    elif pooling_type=="LSEPooling":
        return LSEPooling(belta)
    else:
        raise ValueError("Unknown pooling type: {}".format(pooling_type))


# 新的动态对齐的代码部分
# 开始

# 新增: 自适应混合对齐编码
class AdaptiveMixedCoding(nn.Module):
    def __init__(self, temperature=0.1, init_alpha=0.5):
        super().__init__()
        self.temperature = temperature
        # 初始化混合比例参数，可学习
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        imgs = imgs[:, :max_r, :]
        caps = caps[:, :max_w, :]
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]
        mask = get_fgmask(img_lens, cap_lens)
        
        # 计算硬对齐 (VHA)
        hard_sims = sims.clone()
        hard_sims = hard_sims.masked_fill(mask == 0, MASK)
        hard_attn = torch.zeros_like(sims).to(sims.device)
        
        # 对每个区域找到最匹配的文本词
        max_indices = hard_sims.max(dim=-1, keepdim=True)[1]
        hard_attn.scatter_(-1, max_indices, 1.0)
        hard_attn = hard_attn.masked_fill(mask == 0, 0)
        
        # 计算软对齐 (VSA)
        soft_sims = sims.clone() / self.temperature
        soft_attn = torch.softmax(soft_sims.masked_fill(mask==0, -torch.inf), dim=-1)
        soft_attn = soft_attn.masked_fill(mask == 0, 0)
        
        # 自适应混合软硬对齐
        alpha = self.sigmoid(self.alpha)  # 将alpha限制在0-1之间
        mixed_attn = alpha * soft_attn + (1 - alpha) * hard_attn
        
        # 使用混合注意力进行特征融合
        weighted_caps = torch.matmul(mixed_attn, caps)  # Bi x Bt x K x D
        sims = torch.mul(weighted_caps.permute(1, 0, 2, 3), imgs).permute(1, 0, 2, 3).sum(dim=-1) / (torch.norm(weighted_caps, p=2, dim=-1, keepdim=False) + EPS)  # Bi x Bt x K
        
        mask = get_mask(img_lens).permute(0, 2, 1).repeat(1, cap_lens.size(0), 1)
        sims = sims.masked_fill(mask == 0, -1)
        
        if return_attn:
            return sims, mixed_attn, alpha
        return sims

# 新增: 基于不确定性的混合对齐编码
class UncertaintyMixedCoding(nn.Module):
    def __init__(self, temperature=0.1, hidden_dim=512):
        super().__init__()
        self.temperature = temperature
        # 不确定性估计器
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        imgs = imgs[:, :max_r, :]
        caps = caps[:, :max_w, :]
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]
        mask = get_fgmask(img_lens, cap_lens)
        
        # 估计每个图像区域的匹配不确定性
        uncertainties = self.uncertainty_estimator(imgs)  # Bi x K x 1
        
        # 计算硬对齐
        hard_sims = sims.clone()
        hard_sims = hard_sims.masked_fill(mask == 0, MASK)
        hard_attn = torch.zeros_like(sims).to(sims.device)
        
        # 对每个区域找到最匹配的文本词
        max_indices = hard_sims.max(dim=-1, keepdim=True)[1]
        hard_attn.scatter_(-1, max_indices, 1.0)
        hard_attn = hard_attn.masked_fill(mask == 0, 0)
        
        # 计算软对齐
        soft_sims = sims.clone() / self.temperature
        soft_attn = torch.softmax(soft_sims.masked_fill(mask==0, -torch.inf), dim=-1)
        soft_attn = soft_attn.masked_fill(mask == 0, 0)
        
        # 基于不确定性的混合
        # 不确定性高 -> 更多软对齐; 不确定性低 -> 更多硬对齐
        alpha = uncertainties.expand(-1, -1, max_w)  # Bi x K x L
        alpha = alpha.unsqueeze(1).expand(-1, caps.size(0), -1, -1)  # Bi x Bt x K x L
        
        mixed_attn = alpha * soft_attn + (1 - alpha) * hard_attn
        
        # 使用混合注意力进行特征融合
        weighted_caps = torch.matmul(mixed_attn, caps)  # Bi x Bt x K x D
        
        # 重写相似度计算部分，避免复杂的permute操作
        batch_size_img = imgs.size(0)
        batch_size_cap = caps.size(0)
        
        # 归一化特征
        weighted_caps_norm = F.normalize(weighted_caps, p=2, dim=-1)  # Bi x Bt x K x D
        imgs_norm = F.normalize(imgs, p=2, dim=-1)  # Bi x K x D
        
        # 计算相似度
        result = []
        for i in range(batch_size_img):
            img_feats = imgs_norm[i]  # K x D
            img_result = []
            for j in range(batch_size_cap):
                weighted_feats = weighted_caps_norm[i, j]  # K x D
                # 计算每个区域的相似度
                sim = torch.sum(img_feats * weighted_feats, dim=-1)  # K
                img_result.append(sim)
            img_result = torch.stack(img_result, dim=0)  # Bt x K
            result.append(img_result)
        
        sims = torch.stack(result, dim=0)  # Bi x Bt x K
        
        # 应用掩码
        mask = get_mask(img_lens).permute(0, 2, 1).repeat(1, cap_lens.size(0), 1)
        sims = sims.masked_fill(mask == 0, -1)
        
        if return_attn:
            return sims, mixed_attn, uncertainties
        return sims

# 新增: 门控混合对齐编码
class GatedMixedCoding(nn.Module):
    def __init__(self, temperature=0.1, hidden_dim=512):
        super().__init__()
        self.temperature = temperature
        # 软对齐门控
        self.soft_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        # 硬对齐门控
        self.hard_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        imgs = imgs[:, :max_r, :]
        caps = caps[:, :max_w, :]
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]
        mask = get_fgmask(img_lens, cap_lens)
        
        # 计算硬对齐
        hard_sims = sims.clone()
        hard_sims = hard_sims.masked_fill(mask == 0, MASK)
        hard_attn = torch.zeros_like(sims).to(sims.device)
        
        # 对每个区域找到最匹配的文本词
        max_indices = hard_sims.max(dim=-1, keepdim=True)[1]
        hard_attn.scatter_(-1, max_indices, 1.0)
        hard_attn = hard_attn.masked_fill(mask == 0, 0)
        
        # 计算软对齐
        soft_sims = sims.clone() / self.temperature
        soft_attn = torch.softmax(soft_sims.masked_fill(mask==0, -torch.inf), dim=-1)
        soft_attn = soft_attn.masked_fill(mask == 0, 0)
        
        # 计算硬对齐和软对齐的特征
        hard_weighted_caps = torch.matmul(hard_attn, caps)  # Bi x Bt x K x D
        soft_weighted_caps = torch.matmul(soft_attn, caps)  # Bi x Bt x K x D
        
        # 计算门控权重
        soft_gates = self.soft_gate(imgs)  # Bi x K x 1
        hard_gates = self.hard_gate(imgs)  # Bi x K x 1
        
        # 归一化门控权重
        gates_sum = soft_gates + hard_gates
        soft_gates = soft_gates / (gates_sum + EPS)
        hard_gates = hard_gates / (gates_sum + EPS)
        
        # 扩展门控权重维度
        soft_gates = soft_gates.unsqueeze(1).expand(-1, caps.size(0), -1, -1)  # Bi x Bt x K x 1
        hard_gates = hard_gates.unsqueeze(1).expand(-1, caps.size(0), -1, -1)  # Bi x Bt x K x 1
        
        # 混合特征
        mixed_weighted_caps = soft_gates * soft_weighted_caps + hard_gates * hard_weighted_caps
        
        # 完全重写相似度计算部分，避免复杂的permute操作
        # 计算每个区域的特征相似度
        batch_size_img = imgs.size(0)
        batch_size_cap = caps.size(0)
        region_size = max_r
        
        # 归一化特征
        mixed_caps_norm = F.normalize(mixed_weighted_caps, p=2, dim=-1)  # Bi x Bt x K x D
        imgs_norm = F.normalize(imgs, p=2, dim=-1)  # Bi x K x D
        
        # 计算相似度
        result = []
        for i in range(batch_size_img):
            img_feats = imgs_norm[i]  # K x D
            img_result = []
            for j in range(batch_size_cap):
                mixed_feats = mixed_caps_norm[i, j]  # K x D
                # 计算每个区域的相似度
                sim = torch.sum(img_feats * mixed_feats, dim=-1)  # K
                img_result.append(sim)
            img_result = torch.stack(img_result, dim=0)  # Bt x K
            result.append(img_result)
        
        sims = torch.stack(result, dim=0)  # Bi x Bt x K
        
        # 应用掩码
        mask = get_mask(img_lens).permute(0, 2, 1).repeat(1, cap_lens.size(0), 1)
        sims = sims.masked_fill(mask == 0, -1)
        
        if return_attn:
            mixed_attn = soft_gates * soft_attn + hard_gates * hard_attn
            return sims, mixed_attn, (soft_gates, hard_gates)
        return sims


def get_coding(coding_type, **args):
    alpha = args["opt"].alpha
    hidden_dim = args["opt"].embed_size if hasattr(args["opt"], "embed_size") else 512
    
    if coding_type=="VHACoding":
        return VHACoding()
    elif coding_type=="THACoding":
        return THACoding()
    elif coding_type=="VSACoding":
        return VSACoding(alpha)
    elif coding_type=="AdaptiveMixedCoding":
        return AdaptiveMixedCoding(alpha, init_alpha=0.5)
    elif coding_type=="UncertaintyMixedCoding":
        return UncertaintyMixedCoding(alpha, hidden_dim)
    elif coding_type=="GatedMixedCoding":
        return GatedMixedCoding(alpha, hidden_dim)
    else:
        raise ValueError("Unknown coding type: {}".format(coding_type))
# 结束