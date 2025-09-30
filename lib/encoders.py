"""VSE modules"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import DistilBertTokenizer, DistilBertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sru import SRU

from transformers import BertModel

from lib.coding import get_coding, get_pooling
from lib.modules import l2norm, SelfAttention, Transformer, ExternalAttention, SimAM3D, ESAttention, SEAttention, ASPP, PPM

import logging

logger = logging.getLogger(__name__)


def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, text_enc_type="bigru", 
                    use_bi_gru=True, no_txtnorm=False, **args):
    """A wrapper to text encoders."""
    if text_enc_type == "bigru":
        txt_enc = EncoderTextBigru(vocab_size, embed_size, word_dim, num_layers, use_bi_gru=use_bi_gru, no_txtnorm=no_txtnorm, **args)
    elif text_enc_type == "bert":
        txt_enc = EncoderTextBert(embed_size, no_txtnorm=no_txtnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(text_enc_type))
    return txt_enc


def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, **args):
    """A wrapper to image encoders."""
    img_enc = EncoderImagePrecomp(img_dim, embed_size, precomp_enc_type, no_imgnorm, **args)
    return img_enc


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, **args):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.fc_down = nn.Linear(1024, 128)
        self.fc_up = nn.Linear(128, 1024)

        if precomp_enc_type=="basic":
            self.feedforward = nn.Identity()
        elif precomp_enc_type=="selfattention":   # --------------------------------------- 调取函数 
            self.feedforward = SelfAttention(embed_size)
            # self.feedforward = ExternalAttention(d_model=embed_size, S=8)
            # self.feedforward = ESAttention(d_model=embed_size, S=8)
            # self.feedforward = SimAM3D()
            # self.feedforward = SEAttention(d_model=embed_size, S=8)
        elif precomp_enc_type=="transformer":
            self.feedforward = Transformer(embed_size)
        else:
            raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
        self.init_weights()

        self.ASPP = ASPP(in_channels=128, atrous_rates=[6, 18], out_channels=128) # 6,12,18； AP01：6,18； 02:完整版超参
        self.PPM = PPM(in_channels=128, reduction_channels=32, pool_sizes=(1, 2, 4))


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""

        features = self.fc(images) # 1024

        # # 保存原始特征用于最后的残差连接
        # residual_1024 = features

        # features = self.fc_down(features) # 1024->256
        # features = self.ASPP(features) # 256
        # features = self.PPM(features)  # 256
        # features = self.fc_up(features)# 256->1024

        # # 将升维后的特征与进入瓶颈前的原始特征相加
        # features = residual_1024 + features
       
        features = self.feedforward(features) # 原来的函数

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


# Language Model with 轻量级卷积
class HardAssignmentTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, 
                 num_layers=1, no_txtnorm=False, **args):
        super().__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # 词嵌入层（完整实现）
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed.requires_grad = False  # 冻结嵌入层

        # 轻量级动态特征提取
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(word_dim, embed_size, kernel_size=3, padding=1, groups=word_dim//2),
            nn.GELU(),
            nn.Conv1d(embed_size, embed_size, kernel_size=1)
        )
        
        # 硬分配特征选择
        self.selection_gate = nn.Sequential(
            nn.Linear(embed_size, embed_size//4),
            nn.Sigmoid(),
            nn.Linear(embed_size//4, 1)
        )
        
        # 初始化权重（完整实现）
        self.init_weights(wemb_type=args["wemb_type"],
                         word2idx=args["word2idx"],
                         word_dim=word_dim,
                         cache_dir=args.get("cache_dir", "~/.cache/torch/hub/"))

    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/.cache/torch/hub/"):
        """完整预训练词向量加载逻辑"""
        if not wemb_type or not word2idx:
            nn.init.xavier_uniform_(self.embed.weight)
            return

        # 处理缓存路径
        cache_dir = os.path.expanduser(os.path.join(cache_dir, wemb_type))
        os.makedirs(cache_dir, exist_ok=True)

        # 加载预训练词向量
        try:
            if wemb_type.lower() == 'fasttext':
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif wemb_type.lower() == 'glove':
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise ValueError(f"不支持的词向量类型: {wemb_type}")
        except Exception as e:
            print(f"加载词向量失败: {e}")
            nn.init.xavier_uniform_(self.embed.weight)
            return

        # 检查维度匹配
        assert wemb.vectors.shape[1] == word_dim, \
            f"词向量维度不匹配: {wemb.vectors.shape[1]} vs {word_dim}"

        # 处理词汇表
        missing_words = []
        for word, idx in word2idx.items():
            # 清洗单词（处理特殊字符）
            cleaned_word = word.lower().replace('-', '').replace('.', '')
            if '/' in cleaned_word:
                cleaned_word = cleaned_word.split('/')[0]

            # 查找词向量
            if cleaned_word in wemb.stoi:
                vec = wemb.vectors[wemb.stoi[cleaned_word]]
                self.embed.weight.data[idx] = vec
            else:
                missing_words.append(word)

        # 统计信息
        print(f"词向量加载完成: {len(word2idx)-len(missing_words)}/{len(word2idx)} 找到，"
              f"{len(missing_words)} 未找到（如：{missing_words[:5]}...）")

    def forward(self, x, lengths):
        # 词嵌入
        x_emb = self.embed(x)  # (batch, seq_len, word_dim)
        
        # 轻量级特征提取
        x_conv = self.temporal_conv(x_emb.transpose(1,2))  # (batch, embed_size, seq_len)
        x_conv = x_conv.transpose(1,2)  # (batch, seq_len, embed_size)
        
        # 硬分配选择关键特征
        weights = self.selection_gate(x_conv)  # (batch, seq_len, 1)
        weights = torch.softmax(weights.squeeze(-1), dim=1)  # 归一化
        
        # 保留top-50%关键特征（动态选择）
        k = max(1, int(weights.size(1) * 0.5))
        topk_weights, topk_indices = torch.topk(weights, k=k, dim=1)
        selected_features = x_conv.gather(
            1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, x_conv.size(-1))
        )
        
        # 特征聚合
        cap_emb = torch.mean(selected_features * topk_weights.unsqueeze(-1), dim=1)
        
        # 归一化
        if not self.no_txtnorm:
            cap_emb = nn.functional.normalize(cap_emb, p=2, dim=-1)
        
        return cap_emb.unsqueeze(1)  # 保持与原始输出维度一致

# # 原BIGRU代码开始
# # Language Model with BiGRU
class EncoderTextBigru(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False, **args):
        super(EncoderTextBigru, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        hidden_size = embed_size
        self.rnn = nn.GRU(word_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.fc = nn.Linear(hidden_size, embed_size)
        self.init_weights(wemb_type=args["wemb_type"],word2idx=args["word2idx"],word_dim=word_dim)
        # self.attention1 = ExternalAttention(d_model=embed_size, S=8) # ------------新加的注意力机制
        # self.attention2 = SimAM3D()


    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/.cache/torch/hub/"):
        if wemb_type is None or word2idx is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            cache_dir = os.path.expanduser(cache_dir+wemb_type)
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            ##
            self.embed.requires_grad = False
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)

        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # cap_emb_tmp = self.attention1(cap_emb) # ------------------ 新加，经过注意力机制处理
        # cap_emb_tmp = self.attention2(cap_emb_tmp)
        # cap_emb = cap_emb + cap_emb_tmp # -----------------残差连接

        # cap_emb = self.attention1(cap_emb) # ---------不残差

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        # print("检查文本特征的形状：")
        # print(cap_emb.shape) torch.Size([384, 56, 1024]) torch.Size([384, 47, 1024]) torch.Size([384, 32, 1024])
        return cap_emb
# # 原BIGRU代码结束
# 新的BIGRU代码开始

# 新的BIGRU代码结束

# Language Model with SRU
class EncoderTextSRU(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_sru=True, no_txtnorm=False, **args):
        super().__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # 词嵌入层（保持原结构）
        self.embed = nn.Embedding(vocab_size, word_dim)

        # SRU 配置（替换 GRU）
        self.rnn = SRU(
            input_size=word_dim,
            hidden_size=embed_size,
            num_layers=num_layers,
            bidirectional=use_bi_sru,
            dropout=0.1 if num_layers > 1 else 0,
            use_tanh=True,  # 使用 tanh 增强稳定性
            highway_bias=0.5,  # 增强梯度流
            rescale=True
        )

        # 双向输出处理
        self.bidirectional = use_bi_sru
        self._init_weights(args)

    def _init_weights(self, args):
        """初始化词嵌入和 SRU 参数"""
        wemb_type = args.get("wemb_type")
        word2idx = args.get("word2idx")
        word_dim = args.get("word_dim", 300)
        cache_dir = args.get("cache_dir", "~/.cache/torch/hub/")

        # 预训练词向量初始化
        if wemb_type and word2idx:
            cache_dir = os.path.expanduser(os.path.join(cache_dir, wemb_type))
            if wemb_type.lower() == 'fasttext':
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif wemb_type.lower() == 'glove':
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise ValueError(f"Unsupported embedding type: {wemb_type}")
            
            assert wemb.vectors.shape[1] == word_dim, "词向量维度不匹配"
            
            missing_words = []
            for word, idx in word2idx.items():
                processed_word = word.replace('-', '').replace('.', '').replace("'", '')
                if '/' in processed_word:
                    processed_word = processed_word.split('/')[0]
                if processed_word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[processed_word]]
            self.embed.requires_grad = False
            print(f"Loaded pretrained embeddings. Coverage: {len(word2idx)-len(missing_words)}/{len(word2idx)}")
        else:
            nn.init.xavier_uniform_(self.embed.weight)

        # SRU 参数初始化
        for name, param in self.rnn.named_parameters():
                if 'weight' in name:
                    if param.dim() >= 2:  # 仅处理二维及以上权重
                        nn.init.xavier_normal_(param)
                    else:  # 一维权重（如SRU的缩放因子）
                        nn.init.uniform_(param, -0.1, 0.1)  # 使用均匀分布初始化
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x, lengths):
        """
        输入: 
            x: (batch_size, seq_len)
            lengths: (batch_size,) 
        输出: 
            (batch_size, seq_len, embed_size)
        """
        # 词嵌入
        x_emb = self.embed(x)  # (batch_size, seq_len, word_dim)

        # 调整维度为 SRU 需要的格式 (seq_len, batch_size, input_size)
        x_emb = x_emb.transpose(0, 1)

        # SRU 前向传播
        output, _ = self.rnn(x_emb)  # (seq_len, batch_size, hidden_size * num_directions)

        # 调整回原始维度 (batch_size, seq_len, hidden_size * num_directions)
        output = output.transpose(0, 1)

        # 处理双向输出（与原 BiGRU 逻辑兼容）
        if self.bidirectional:
            forward_out = output[..., :self.embed_size]
            backward_out = output[..., self.embed_size:]
            output = (forward_out + backward_out) / 2

        # 归一化
        if not self.no_txtnorm:
            output = l2norm(output, dim=-1)

        return output

# 替换的HAN模型
class EncoderTextHAN(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False, **args):
        super(EncoderTextHAN, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, word_dim)

        # HAN 核心组件 -------------------------------------------------
        # 词级双向GRU（保持与原始结构兼容）
        self.word_rnn = nn.GRU(
            input_size=word_dim,
            hidden_size=embed_size,
            num_layers=num_layers,
            bidirectional=use_bi_gru,
            batch_first=True
        )

        # 词级注意力机制
        self.word_attn = nn.Sequential(
            nn.Linear(2 * embed_size if use_bi_gru else embed_size, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, 1, bias=False)
        )
        # --------------------------------------------------------------

        self.init_weights(wemb_type=args["wemb_type"], word2idx=args["word2idx"], word_dim=word_dim)

    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/.cache/torch/hub/"):
        # 保持原始词嵌入初始化逻辑不变
        if wemb_type is None or word2idx is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            cache_dir = os.path.expanduser(cache_dir + wemb_type)
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception(f'Unknown word embedding type: {wemb_type}')
            assert wemb.vectors.shape[1] == word_dim

            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            self.embed.requires_grad = False
            print(
                f'Words: {len(word2idx) - len(missing_words)}/{len(word2idx)} found in vocabulary; {len(missing_words)} words missing')

    def forward(self, x, lengths):
        """输入输出形状保持与原始 BiGRU 版本一致
        输入:
            x:      (batch_size, seq_len)
            lengths:(batch_size,)
        输出:
            cap_emb:(batch_size, seq_len, embed_size)
        """
        # 词嵌入
        x_emb = self.embed(x)  # (batch_size, seq_len, word_dim)

        # 处理变长序列
        self.word_rnn.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )

        # 词级双向GRU
        rnn_out, _ = self.word_rnn(packed)  # rnn_out: PackedSequence
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out,
                                                         batch_first=True)  # (batch_size, seq_len, 2*embed_size)

        # ----------------- HAN 核心修改部分 -----------------
        # 计算词级注意力权重
        attn_weights = self.word_attn(padded_out)  # (batch_size, seq_len, 1)

        # 对填充位置进行mask 生成 mask 并确保设备一致
        max_len = x.size(1)
        mask = (torch.arange(max_len, device=x.device).expand(len(lengths), max_len) < lengths.unsqueeze(1))

        # 使用 FP16 兼容的极小值（-1e4）并确保数据类型
        attn_weights = attn_weights.squeeze(-1).masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)

        # 注意力加权（保持序列长度不变）
        # 使用注意力权重对原始GRU输出进行加权（类似残差连接）
        weighted_out = padded_out * attn_weights + padded_out  # (batch_size, seq_len, 2*embed_size)
        # ----------------------------------------------------

        # 合并双向输出（与原始代码一致）
        if self.word_rnn.bidirectional:
            weighted_out = (weighted_out[:, :, :self.embed_size] + weighted_out[:, :, self.embed_size:]) / 2

        # 归一化处理
        if not self.no_txtnorm:
            weighted_out = l2norm(weighted_out, dim=-1)

        return weighted_out  # (batch_size, seq_len, embed_size)


# 修改后的 Language Model with BiLSTM 文本编码器类
class EncoderTextBilstm(nn.Module):  # 类名修改为更准确的名称
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_lstm=True, no_txtnorm=False, **args):
        super(EncoderTextBilstm, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # Word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # Caption embedding
        hidden_size = embed_size  # 保持隐藏层维度与嵌入维度一致
        self.rnn = nn.LSTM(  # 将 GRU 替换为 LSTM
            input_size=word_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=use_bi_lstm  # 参数名改为 use_bi_lstm
        )
        self.fc = nn.Linear(hidden_size, embed_size)
        self.init_weights(wemb_type=args["wemb_type"], word2idx=args["word2idx"], word_dim=word_dim)

    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/.cache/torch/hub/"):
        # 此部分无需修改，保持原有词嵌入初始化逻辑
        if wemb_type is None or word2idx is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            cache_dir = os.path.expanduser(cache_dir + wemb_type)
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            self.embed.requires_grad = False
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions"""
        # 词嵌入层
        x_emb = self.embed(x)

        # LSTM 前向传播
        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)
        
        # LSTM 返回 (output, (h_n, c_n))，这里只需 output
        out, _ = self.rnn(packed)  # 自动处理隐藏状态和细胞状态
        
        # 解包并处理双向输出
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        
        # 合并双向输出（取前后向平均值）
        if self.rnn.bidirectional:
            cap_emb = (cap_emb[:, :, :self.embed_size] + cap_emb[:, :, self.embed_size:]) / 2

        # 在联合嵌入空间做归一化
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # print("检查返回结果的形状：")
        # print(cap_emb.shape) torch.Size([384, 37, 1024])

        return cap_emb


# Language Model with BERT
class EncoderTextBert(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderTextBert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # root = os.path.expanduser("~/.cache/torch/hub/transformers")
        # self.bert = BertModel.from_pretrained(config=root,pretrained_model_name_or_path=root)

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')  # -------------------- distilbert 改变这个0

        # self.bert = model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        # 添加一个线性层将 312维 映射到 768 维 tiny                                # -------------------- distilbert 改变这个1 tiny
        # self.mapping_layer = nn.Linear(312,768)

        self.linear = nn.Linear(768, embed_size)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        # bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D  
        bert_emb = self.bert(x, bert_attention_mask)[0]  # -------------------- distilbert 改变这个2
        
        # bert_emb = self.mapping_layer(bert_emb) # 添加 tiny

        cap_len = lengths

        cap_emb = self.linear(bert_emb)
        cap_emb_tmp = cap_emb

        # 新添加GNN模块进行处理 start
        batch_size, seq_len, feature_dim = cap_emb.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=cap_emb.device)
        mask = torch.arange(seq_len, device=cap_emb.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)

        gnn_model = TextGNNWithSemanticGraph(
            input_dim=feature_dim,  # BERT输出的特征维度，这里是1024
            hidden_dim=512,         # GNN隐藏层维度，可以根据需要调整
            output_dim=feature_dim, # 输出维度，通常保持与输入相同
            num_layers=2,           # GNN层数
            dropout=0.1,            # Dropout率
            similarity_threshold=0.5  # 相似度阈值，用于构建语义图
        )
        gnn_model = gnn_model.to(cap_emb.device)
        gnn_output = gnn_model(cap_emb, mask)
        # ----end
        
        # normalization in the joint embedding space 
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb + cap_emb_tmp # 残差

class SimsEncoder(nn.Module):
    def __init__(self, coding_type, pooling_type, **args):
        super(SimsEncoder, self).__init__()
        self.opt = args["opt"]
        self.coding = get_coding(coding_type, opt=self.opt)
        self.pooling = get_pooling(pooling_type, opt=self.opt)

    def forward(self, img_emb, cap_emb, img_lens, cap_lens, return_attention=False): # 可视化修改了这里，但是对训练没有影响
        raw_sims = self.coding(img_emb, cap_emb, img_lens, cap_lens)
        pooled_sims = self.pooling(raw_sims)
        
        if return_attention:
            return pooled_sims, raw_sims
        else:
            return pooled_sims
   
# 图神经网络代码开始
class GraphAttentionLayer(nn.Module):# 图注意力网络
    """
    图注意力层 (GAT)
    用于处理文本中词与词之间的关系
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # 定义可学习的参数
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # 优化，改了forward
    def forward(self, input, adj):
        """
        input: 节点特征矩阵 [batch_size, seq_len, in_features]
        adj: 邻接矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = input.size()
        
        # 线性变换
        h = self.W(input)  # [batch_size, seq_len, out_features]
        
        # 优化注意力计算，减少内存使用
        attention = torch.zeros(batch_size, seq_len, seq_len, device=input.device)
        
        # 分批计算注意力系数，避免大内存分配
        chunk_size = 32  # 可以根据GPU内存调整这个值
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                
                # 计算当前块的注意力系数
                h_i = h[:, i:end_i, :]
                h_j = h[:, j:end_j, :]
                
                # 为当前块准备注意力输入
                h_i_expanded = h_i.unsqueeze(2).expand(-1, -1, end_j-j, -1)
                h_j_expanded = h_j.unsqueeze(1).expand(-1, end_i-i, -1, -1)
                
                # 拼接特征
                a_input = torch.cat([h_i_expanded, h_j_expanded], dim=-1)
                
                # 计算注意力系数
                e = self.leakyrelu(self.a(a_input).squeeze(-1))
                
                # 将计算结果存储到完整的注意力矩阵中
                attention[:, i:end_i, j:end_j] = e
        
        # 掩码机制，将不相连的边置为负无穷
        zero_vec = -9e15 * torch.ones_like(attention)
        attention = torch.where(adj > 0, attention, zero_vec)
        
        # 对注意力系数进行softmax归一化
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 聚合邻居信息
        h_prime = torch.bmm(attention, h)  # [batch_size, seq_len, out_features]
        
        return h_prime

# class GraphConvLayer(nn.Module):
#     """
#     图卷积网络层 (GCN)，比GAT更节省内存
#     """
#     def __init__(self, in_features, out_features, dropout=0.1):
#         super(GraphConvLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
        
#         # 定义可学习的参数
#         self.W = nn.Linear(in_features, out_features, bias=False)
        
#     def forward(self, input, adj):
#         """
#         input: 节点特征矩阵 [batch_size, seq_len, in_features]
#         adj: 邻接矩阵 [batch_size, seq_len, seq_len]
#         """
#         # 对邻接矩阵进行归一化
#         rowsum = adj.sum(dim=2, keepdim=True) + 1e-8
#         norm_adj = adj / rowsum
        
#         # 线性变换
#         h = self.W(input)  # [batch_size, seq_len, out_features]
        
#         # 聚合邻居信息
#         h_prime = torch.bmm(norm_adj, h)  # [batch_size, seq_len, out_features]
        
#         return h_prime

class TextGNN(nn.Module):
    """
    用于处理文本特征的图神经网络
    """

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_layers=2, dropout=0.1):
        super(TextGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # GNN层
        self.gnn_layers = nn.ModuleList()

        # 第一层
        self.gnn_layers.append(GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout)) # GraphConvLayer(input_dim, hidden_dim, dropout=dropout)

        # 中间层
        for i in range(num_layers - 2):
            self.gnn_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)) # GraphConvLayer(hidden_dim, hidden_dim, dropout=dropout)

        # 最后一层
        if num_layers > 1:
            self.gnn_layers.append(GraphAttentionLayer(hidden_dim, output_dim, dropout=dropout)) # GraphConvLayer(hidden_dim, output_dim, dropout=dropout)

        # 残差连接的投影层（如果输入维度和输出维度不同）
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

    def build_graph(self, x, mask=None):
        """
        构建文本的图结构
        x: 输入特征 [batch_size, seq_len, feature_dim]
        mask: 掩码 [batch_size, seq_len]

        返回邻接矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()

        # 初始化邻接矩阵 - 默认为全连接图
        adj = torch.ones(batch_size, seq_len, seq_len, device=x.device)

        # 可以根据需要实现更复杂的图构建逻辑
        # 例如：基于词共现、语义相似度等构建边

        # 应用掩码（如果提供）
        if mask is not None:
            mask_matrix = mask.unsqueeze(-1) * mask.unsqueeze(1)
            adj = adj * mask_matrix

        return adj

    def forward(self, x, mask=None):
        """
        x: 输入特征 [batch_size, seq_len, input_dim]
        mask: 掩码 [batch_size, seq_len]

        返回处理后的特征 [batch_size, seq_len, output_dim]
        """
        # 构建图
        adj = self.build_graph(x, mask)

        # 保存原始输入用于残差连接
        residual = x

        # 应用GNN层
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, adj)
            # 除最后一层外，应用ReLU和Dropout
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        x = x + residual

        # 层归一化
        x = self.layer_norm(x)

        return x

class TextGNNWithSemanticGraph(TextGNN):
    """
    使用语义相似度构建图的文本GNN
    """

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_layers=2, dropout=0.1,
                 similarity_threshold=0.5):
        super(TextGNNWithSemanticGraph, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.similarity_threshold = similarity_threshold

    def build_graph(self, x, mask=None):
        """
        基于语义相似度构建图
        """
        batch_size, seq_len, _ = x.size()

        # 计算特征之间的余弦相似度
        x_norm = F.normalize(x, p=2, dim=2)
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [batch_size, seq_len, seq_len]

        # 应用阈值，构建邻接矩阵
        adj = (similarity > self.similarity_threshold).float()

        # 确保自环
        eye = torch.eye(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        adj = (adj > 0).float()  # 确保值为0或1

        # 应用掩码（如果提供）
        if mask is not None:
            mask_matrix = mask.unsqueeze(-1) * mask.unsqueeze(1)
            adj = adj * mask_matrix

        return adj
# 图神经网络代码结束

