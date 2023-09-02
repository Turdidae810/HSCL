import torch
import torch.nn.functional as F
import time
import math
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig


class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        
        bertconfig = BertConfig.from_pretrained(hp.bert_path)
        bertconfig.update({'output_hidden_states':True})
        self.bertmodel = BertModel.from_pretrained(hp.bert_path, config=bertconfig)
            
    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask,
                                token_type_ids=bert_sent_type)
        bert_output = bert_output[0]
        return bert_output   # return head (sequence representation)
    
class MLP(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(MLP, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_3, fusion


class FC(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(FC, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.relu(self.linear_1(dropped))
        return y_1
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        if d_model % 2==1:
            dim = d_model+1
        else:
            dim = d_model

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe = self.pe[:x.size(0), :]
        # print(pe.shape)
        x = x + self.pe[:x.size(0), :,:self.d_model]
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self, ninp=300, nhead=4, nhid=128, nlayers=3, dropout=0.5):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        

    def generate_square_subsequent_mask(self, src, lenths):
        '''
        padding_mask
        src:max_lenth,batch_size,dim
        lenths:[lenth1,lenth2...]
        '''

        # mask num_of_sens x max_lenth
        mask = torch.ones(src.size(1), src.size(0)) == 1  # 全部初始化为 True
        # batch_size, seq_length
        for i in range(len(lenths)):# batch_size
            lenth = lenths[i]
            for j in range(lenth): 
                mask[i][j] = False  # 设置前面的部分为False

        return mask

    def forward(self, src, mask):
        '''
        src:num_of_all_sens,max_lenth,300
        '''
        if mask==None:
            src = src * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
            output = output
        
        else:
            self.src_mask = mask

            src = src * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_key_padding_mask=self.src_mask)
            output = output
        return output
    
    def generate_padding_mask(self,seq):
        """
        seq: tensor of shape (max_length, batch_size, embedding_dim)
        pad_token: int, the padding token id
        """
        mask = torch.ones(seq.size(1), seq.size(0)) == 1
        seq = seq.permute(1,0,2)  # batch, len, dim
        mask[seq[:, :, 0] != 0.0]=False  # batch_size, len
        seq = seq.permute(1,0,2)
        return mask.to(torch.bool)


class BimodalFusionLayer(nn.Module):
    def __init__(self, embed_dim=768, cross_heads=12, self_heads=12, kdim=20, vdim=20, 
    attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_heads = cross_heads
        self.self_heads = self_heads
        self.kdim = kdim
        self.vdim = vdim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.cross_heads,
            kdim=self.kdim,
            vdim=self.vdim,
            dropout=attn_dropout
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.self_heads,
            dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        self.fc1 = Linear(self.embed_dim, self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])
        # self.layer_norm_q = nn.LayerNorm(self.embed_dim)
        # self.layer_norm_k = nn.LayerNorm(self.kdim)
        # self.layer_norm_v = nn.LayerNorm(self.vdim)

    def forward(self, x, x_k, x_v, key_padding_mask, attn_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x

        # 1. attention
        x, _ = self.self_attn(query=x, key=x, value=x,key_padding_mask=attn_mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x)  # 有层归一化
 

        # 2. attention
        
        x, _ = self.cross_attn(query=x, key=x_k, value=x_v, key_padding_mask=key_padding_mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x)  # 层归一化


        # 3. feed forward
        residual = x
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x)  # 层归一化
        return x

    def maybe_layer_norm(self, i, x):
        return self.layer_norms[i](x)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m
