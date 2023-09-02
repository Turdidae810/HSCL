import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import *

from transformers import BertModel, BertConfig

class HSCL(nn.Module):  # Hierarchical Supervised Contrastive Learning

    def __init__(self, hp):
        super().__init__() 
        self.hp = hp

         # 1. unimodal feature extraction
        self.text_enc = LanguageEmbeddingLayer(hp)

        if hp.dataset=='mosi':
            self.v_dim=20
            self.a_dim=5
            self.visual_enc = TransformerEncoder(ninp=20, nhead=5, nhid=64, nlayers=3, dropout=0.1)
            self.acoustic_enc = TransformerEncoder(ninp=5, nhead=1, nhid=32, nlayers=3, dropout=0.1)

        elif hp.dataset=='mosei':
            self.v_dim=35
            self.a_dim=74
            self.visual_enc = TransformerEncoder(ninp=35, nhead=5, nhid=32, nlayers=3, dropout=0.1)
            self.acoustic_enc = TransformerEncoder(ninp=74, nhead=2, nhid=32, nlayers=3, dropout=0.1)

        #  2. bimodal fusion
        self.tv_encoder = BimodalFusionLayer(embed_dim=768, cross_heads=12, self_heads=12,kdim=20, vdim=20)
        self.ta_encoder = BimodalFusionLayer(embed_dim=768, cross_heads=12, self_heads=12,kdim=5, vdim=5)

        # 3. last fusion

        self.fusion_prj = MLP(
            in_size = 768,
            hidden_size = hp.d_prjh,  # 128
            n_class = 1,
            dropout = hp.dropout_prj
        )

        # 4. FC for contrast learning
        self.FC_at = FC(
            in_size = 768,
            hidden_size = hp.d_prjh,
            dropout = hp.dropout_prj
        )
        self.FC_vt = FC(
            in_size = 768,
            hidden_size = hp.d_prjh,
            dropout = hp.dropout_prj
        )
        self.FC_t = FC(
            in_size = 768,
            hidden_size = hp.d_prjh,
            dropout = hp.dropout_prj
        )
        self.FC_a = FC(
            in_size = self.a_dim,
            hidden_size = hp.d_prjh,
            dropout = hp.dropout_prj
        )
        self.FC_v = FC(
            in_size = self.v_dim,
            hidden_size = hp.d_prjh,
            dropout = hp.dropout_prj
        )
       
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask):
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        r_text = enc_word[:,0,:] # (batch_size, emb_size)
        text = enc_word.permute(1,0,2) # (seq_len, batch_size, emb_size)

        cls_a = nn.Parameter(torch.randn(1, 1, acoustic.shape[-1]))
        cls_v = nn.Parameter(0.2*torch.randn(1, 1, visual.shape[-1]))

        cls_a_ = cls_a.repeat(1, acoustic.shape[1], 1)
        cls_v_ = cls_v.repeat(1, visual.shape[1], 1)
        acoustic = torch.cat([cls_a_, acoustic], dim=0)
        visual = torch.cat([cls_v_, visual], dim=0)
    
        mask_t = ~bert_sent_mask.bool() 
        mask_a = self.acoustic_enc.generate_square_subsequent_mask(acoustic,a_len+1)     
        mask_v = self.visual_enc.generate_square_subsequent_mask(visual,v_len+1)

        acoustic = self.acoustic_enc(acoustic, mask_a)  # length, batch_size, dim
        visual = self.visual_enc(visual, mask_v)  # length, batch_size, dim
        r_a = acoustic[0] 
        r_v = visual[0]

        visual_based_text = self.tv_encoder(x=text, x_k=visual, x_v=visual,key_padding_mask=mask_v,attn_mask=None)
        acoustic_based_text = self.ta_encoder(x=text, x_k=acoustic,x_v=acoustic,key_padding_mask=mask_a,attn_mask=None)

        repre_v_t = visual_based_text[0,:,:]
        repre_a_t = acoustic_based_text[0,:,:]

        representation = 0.5*(repre_v_t+repre_a_t)
        z_at = self.FC_at(repre_a_t)
        z_vt = self.FC_vt(repre_v_t)
        z_at = z_at.unsqueeze(1)
        z_vt = z_vt.unsqueeze(1)
        feature = torch.cat([z_at,z_vt],dim=1)

        z_t = self.FC_t(r_text) 
        z_a = self.FC_a(r_a)
        z_v = self.FC_v(r_v)
        z_t = z_t.unsqueeze(1)
        z_a = z_a.unsqueeze(1)
        z_v = z_v.unsqueeze(1)
        feature_at = torch.cat([z_t,z_a],dim=1)
        feature_vt = torch.cat([z_t,z_v],dim=1)
        preds,fusion = self.fusion_prj(representation)

        return preds, feature, fusion, feature_at, feature_vt
