import numpy as np
import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from math import pi, log
from functools import wraps
import einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce

# from .backbone import resnet18
from MRImodels import new_AttentionMultiScaleCNN
from Plasmamodels import NeuralNetwork
from healnet import *

class AVClassifier(nn.Module):
    def __init__(
        self,
        args,
        input_dim
    ):
        super(AVClassifier, self).__init__()

        self.dataset = args.dataset

        self.use_healnet=args.use_healnet
        # self.use_coatt =args.use_coatt

        if self.dataset=='ANM' or self.dataset=='ANM_early' or self.dataset=='ADNI' or self.dataset=='ADNI_early':
            n_classes = 3
        elif self.dataset=='ANM_MMSE' or self.dataset=='ANM_MMSE_early':
            n_classes = 5
        elif self.dataset=='ADNI_MMSE':
            n_classes = 4
            

        if args.ckpt_path == 'UMT':
            print('use_UMT')
            self.audio_net=new_AttentionMultiScaleCNN(num_classes=n_classes)
            if args.dataset == 'ANM':
                self.audio_net.load_state_dict(torch.load('../models/ANM/MRI/best_4.pth'))
            elif args.dataset == 'ANM_early':
                print('use ANM_early')
                self.audio_net.load_state_dict(torch.load('../models/ANM_early/MRI/best_3.pth'))
            elif args.dataset == 'ANM_MMSE':
                print('use ANM_MMSE')
                self.audio_net.load_state_dict(torch.load('../models/ANM_MMSE/MRI/best_0.pth'))
            elif args.dataset == 'ADNI':
                self.audio_net.load_state_dict(torch.load('../models/ADNI/MRI/best_3.pth'))
            self.audio_net.fc = nn.Linear(50 * 3 * 9 * 9, 64)

            self.visual_net=NeuralNetwork(input_dim,num_classes=n_classes)
            if args.dataset == 'ANM':
                self.visual_net.load_state_dict(torch.load('../models/ANM/plasma/best_4.pth'))
            elif args.dataset == 'ANM_early':
                print('use ANM_early')
                self.visual_net.load_state_dict(torch.load('../models/ANM_early/plasma/best_2.pth'))
            elif args.dataset == 'ANM_MMSE':
                print('use ANM_MMSE')
                self.visual_net.load_state_dict(torch.load('../models/ANM_MMSE/plasma/best_0.pth'))
            elif args.dataset == 'ADNI':
                self.visual_net.load_state_dict(torch.load('../models/ADNI/plasma/best_1.pth'))
            self.visual_net.fc4 = nn.Linear(64, 64)
        elif args.ckpt_path == 'No_UMT':
            print('NOuse_UMT')
            self.audio_net = new_AttentionMultiScaleCNN(num_classes=64)

            self.visual_net = NeuralNetwork(input_dim,num_classes=64)


        self.head = nn.Linear(128, n_classes)
        # self.headca = nn.Linear(256, n_classes)
        self.head_audio = nn.Linear(64, n_classes)
        self.head_video = nn.Linear(64, n_classes)

        self.heal_transformer = HealNet(
            n_modalities=2,
            channel_dims=[64, 64], # (2000, 3, 3) number of channels/tokens per modality
            num_spatial_axes=[1, 1], # (1, 2, 3) number of spatial axes (will be positionally encoded to preserve spatial information)
            out_dims = n_classes,
            depth=3,
            self_per_cross_attn=0
        )
        # self.get_local =  MLP(
        #     input_dim=100,
        #     output_dim=50,
        #     hidden_dims = [128,64]
        # )

        # self.cross_attention = CrossAttention(feature_dim=64, num_heads=8)

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        out_audio=self.head_audio(a)
        out_video=self.head_video(v)

        if self.use_healnet:
            # print('use_healnet')

            a_heal = einops.repeat(a, 'b d -> b c d', c=1) # spatial axis: None (pass as 1)
            v_heal = einops.repeat(v, 'b d -> b c d', c=1) # spatial axis: None (pass as 1)

            out = self.heal_transformer([a_heal,v_heal])
            # local_f = self.get_local(torch.cat((a,v),1))
            # out = self.head(torch.cat((global_f,local_f),1))
        # elif self.use_coatt:
        #     CA_output1 = self.cross_attention(a , v)
        #     CA_output2 = self.cross_attention(v , a)
        #     out = torch.cat((a,v,CA_output1,CA_output2),1)
        #     out = self.headca(out)
        else:
            #concate
            out = torch.cat((a,v),1)
            out = self.head(out)

        return out,out_audio,out_video,a,v

# class CrossAttention(nn.Module):
#     def __init__(self, feature_dim, num_heads):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.num_heads = num_heads
#         self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

#     def forward(self, rna, graph):
#         """
#         rna: (batch_size, 64)  --> Query
#         graph: (batch_size, 64)  --> Key,Value
#         """
#         # 变换维度以适配 MultiheadAttention: (batch_size, seq_len=1, embed_dim=64)
#         rna = rna.unsqueeze(1)    # (batch_size, 1, 64) 作为 Query
#         graph = graph.unsqueeze(1)  # (batch_size, 1, 64) 作为 Key/Value
        
#         # 计算交叉注意力
#         attn_output, _ = self.cross_attn(query=rna, key=graph, value=graph)  # (batch_size, 1, 64)

#         return attn_output.squeeze(1)  # 变回 (batch_size, 64)