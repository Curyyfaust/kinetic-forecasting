# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:52:23 2025

@author: lenovo
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, series_decomp
import math
import numpy as np

class Autoformer(nn.Module):

    def __init__(self, parameter):
        super(Autoformer, self).__init__()
        self.seq_len = parameter["seq_len"] #输入长度
        self.label_len = parameter["label_len"]
        self.pred_len = parameter["pred_len"]
        #self.output_attention = parameter["output_attention"]
        
        #Decomp
        kernel_size = parameter["moving_avg"]
        self.decomp = series_decomp(kernel_size)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(AutoCorrelation(),parameter["d_model"], parameter["n_heads"]),
                    parameter["d_model"]
                ) for l in range(parameter["e_layers"])
            ]
        )
        
        self.decoder = Decoder(
                [
                    DecoderLayer(
                            AutoCorrelationLayer(AutoCorrelation(),parameter["d_model"], parameter["n_heads"]),
                            AutoCorrelationLayer(AutoCorrelation(),parameter["d_model"], parameter["n_heads"]),
                        parameter["d_model"],
                        parameter["c_out"]
                    )for l in range(parameter["d_layers"])
                ]
            )
        
        self.proj1 = nn.Linear(parameter["d_model"],parameter["c_out"])
        self.proj2 = nn.Linear(parameter["seq_len"],parameter["pred_len"])
    
    def forward(self, x_enc,x_dec):
        
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # mean--> (batch_size,pred_len,d_model)
        # zeros--> (batch_size,pred_len,d_model)
        # sea,trend --> (batch_size,seq_len,d_model)
        
        # enc 
        enc_out, attns = self.encoder(x_enc, attn_mask=None)
        
        #enc_out --> (B,S,D)
        # dec
        seasonal_part, trend_part = self.decoder(enc_out, enc_out, x_mask=None, cross_mask=None,trend=trend_init)
        dec_out = trend_part + seasonal_part
        #print(dec_out.shape)
        #dec_out = self.proj1(dec_out)
        dec_out = self.proj2(dec_out.permute(0,2,1))
        dec_out = dec_out.permute(0,2,1)
        
        #MLP
        
        
        return dec_out # [B, L, D]