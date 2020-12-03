import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VAE(nn.Module):

    def __init__(self, embed_size):
        super(VAE, self).__init__()

        Z_dim = X_dim = h_dim = embed_size
        self.Z_dim = Z_dim
        self.X_dim= X_dim
        self.h_dim = h_dim
        self.embed_size= embed_size

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

        # =============================== Q(z|X) ======================================
        self.dense_xh = nn.Linear(X_dim, h_dim)
        init_weights(self.dense_xh)

        self.dense_hz_mu = nn.Linear(h_dim, Z_dim)
        init_weights(self.dense_hz_mu)

        self.dense_hz_var = nn.Linear(h_dim, Z_dim)
        init_weights(self.dense_hz_var)

        # =============================== P(X|z) ======================================
        self.dense_zh = nn.Linear(Z_dim, h_dim)
        init_weights(self.dense_zh)

        self.dense_hx = nn.Linear(h_dim, X_dim)
        init_weights(self.dense_hx)

    def Q(self, X):
        h = nn.ReLU()(self.dense_xh(X))
        z_mu = self.dense_hz_mu(h)
        z_var = self.dense_hz_var(h)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        mb_size = mu.shape[0]
        eps = Variable(torch.randn(mb_size, self.Z_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def P(self, z):
        h = nn.ReLU()(self.dense_zh(z))
        X = self.dense_hx(h)
        return X


class AGNN(torch.nn.Module):
    def __init__(self, user_size, item_size, gender_size, age_size, occupation_size, genre_size, director_size, writer_size, star_size, country_size, embed_size, attention_size, dropout):
        super(AGNN, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.gender_size = gender_size
        self.age_size = age_size
        self.occupation_size = occupation_size
        self.genre_size = genre_size
        self.director_size = director_size
        self.writer_size = writer_size
        self.star_size = star_size
        self.country_size = country_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.attention_size = attention_size

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

        self.user_embed = torch.nn.Embedding(self.user_size, self.embed_size)
        self.item_embed = torch.nn.Embedding(self.item_size, self.embed_size)
        nn.init.xavier_uniform(self.user_embed.weight)
        nn.init.xavier_uniform(self.item_embed.weight)

        self.user_bias = torch.nn.Embedding(self.user_size, 1)
        self.item_bias = torch.nn.Embedding(self.item_size, 1)
        nn.init.constant(self.user_bias.weight, 0)
        nn.init.constant(self.item_bias.weight, 0)

        self.miu = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.gender_embed = torch.nn.Embedding(self.gender_size, self.embed_size)
        self.gender_embed.weight.data.normal_(0, 0.05)
        self.age_embed = torch.nn.Embedding(self.age_size, self.embed_size)
        self.age_embed.weight.data.normal_(0, 0.05)
        self.occupation_embed = torch.nn.Embedding(self.occupation_size, self.embed_size)
        self.occupation_embed.weight.data.normal_(0, 0.05)

        self.genre_embed = torch.nn.Embedding(self.genre_size, self.embed_size)
        self.genre_embed.weight.data.normal_(0, 0.05)
        self.director_embed = torch.nn.Embedding(self.director_size, self.embed_size)
        self.director_embed.weight.data.normal_(0, 0.05)
        self.writer_embed = torch.nn.Embedding(self.writer_size, self.embed_size)
        self.writer_embed.weight.data.normal_(0, 0.05)
        self.star_embed = torch.nn.Embedding(self.star_size, self.embed_size)
        self.star_embed.weight.data.normal_(0, 0.05)
        self.country_embed = torch.nn.Embedding(self.country_size, self.embed_size)
        self.country_embed.weight.data.normal_(0, 0.05)


        #--------------------------------------------------
        self.dense_item_self_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_item_self_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_item_onehop_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_item_onehop_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_self_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_self_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_onehop_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_onehop_siinter = nn.Linear(self.embed_size, self.embed_size)
        init_weights(self.dense_item_self_biinter)
        init_weights(self.dense_item_self_siinter)
        init_weights(self.dense_item_onehop_biinter)
        init_weights(self.dense_item_onehop_siinter)
        init_weights(self.dense_user_self_biinter)
        init_weights(self.dense_user_self_siinter)
        init_weights(self.dense_user_onehop_biinter)
        init_weights(self.dense_user_onehop_siinter)

        self.dense_item_cate_self = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_item_cate_hop1 = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_user_cate_self = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_user_cate_hop1 = nn.Linear(2 * self.embed_size, self.embed_size)
        init_weights(self.dense_item_cate_self)
        init_weights(self.dense_item_cate_hop1)
        init_weights(self.dense_user_cate_self)
        init_weights(self.dense_user_cate_hop1)

        self.dense_item_addgate = nn.Linear(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_item_addgate)
        self.dense_item_erasegate = nn.Linear(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_item_erasegate)
        self.dense_user_addgate = nn.Linear(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_user_addgate)
        self.dense_user_erasegate = nn.Linear(self.embed_size * 2, self.embed_size)

        self.user_vae = VAE(embed_size)
        self.item_vae = VAE(embed_size)

        #----------------------------------------------------
        #concat, mlp融合
        self.FC_pre = nn.Linear(2 * embed_size, 1)
        init_weights(self.FC_pre)

        """# dot
        self.user_bias = nn.Embedding(self.user_size, 1)
        self.item_bias = nn.Embedding(self.item_size, 1)
        self.user_bias.weight.data.normal_(0, 0.01)
        self.item_bias.weight.data.normal_(0, 0.01)
        self.bias = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.bias.data.uniform_(0, 0.1)"""

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

    def feat_interaction(self, feature_embedding, fun_bi, fun_si, dimension):
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)).pow(2)
        squared_sum_features_emb = torch.sum(feature_embedding.pow(2), dim=dimension)
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        deep_fm = self.leakyrelu(fun_bi(deep_fm))
        bias_fm = self.leakyrelu(fun_si(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm

    def forward(self, user, item, user_self_cate, user_onehop_id, user_onehop_cate, item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country, item_onehop_id, item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country, mode='train'):

        uids_list = user.cuda()
        sids_list = item.cuda()
        if mode == 'train' or mode == 'warm':
            user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
            item_embedding = self.item_embed(torch.autograd.Variable(sids_list))
        if mode == 'ics':
            user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
        if mode == 'ucs':
            item_embedding = self.item_embed(torch.autograd.Variable(sids_list))

        batch_size = item_self_cate.shape[0]
        cate_size = item_self_cate.shape[1]
        director_size = item_self_director.shape[1]
        writer_size = item_self_writer.shape[1]
        star_size = item_self_star.shape[1]
        country_size = item_self_country.shape[1]
        user_onehop_size = user_onehop_id.shape[1]
        item_onehop_size = item_onehop_id.shape[1]

        #------------------------------------------------------GCN-item
        # K=2
        item_onehop_id = self.item_embed(Variable(item_onehop_id))

        item_onehop_cate = self.genre_embed(Variable(item_onehop_cate).view(-1, cate_size)).view(batch_size,item_onehop_size,cate_size, -1)
        item_onehop_director = self.director_embed(Variable(item_onehop_director).view(-1, director_size)).view(batch_size, item_onehop_size, director_size, -1)
        item_onehop_writer = self.writer_embed(Variable(item_onehop_writer).view(-1, writer_size)).view(batch_size, item_onehop_size, writer_size, -1)
        item_onehop_star = self.star_embed(Variable(item_onehop_star).view(-1, star_size)).view(batch_size, item_onehop_size, star_size, -1)
        item_onehop_country = self.country_embed(Variable(item_onehop_country).view(-1, country_size)).view(batch_size, item_onehop_size, country_size, -1)

        item_onehop_feature = torch.cat([item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country], dim=2)
        item_onehop_embed = self.dense_item_cate_hop1(torch.cat([self.feat_interaction(item_onehop_feature, self.dense_item_onehop_biinter,  self.dense_item_onehop_siinter, dimension=2), item_onehop_id], dim=-1))

        # K=1
        item_self_cate = self.genre_embed(Variable(item_self_cate))
        item_self_director = self.director_embed(Variable(item_self_director))
        item_self_writer = self.writer_embed(Variable(item_self_writer))
        item_self_star = self.star_embed(Variable(item_self_star))
        item_self_country = self.country_embed(Variable(item_self_country))

        item_self_feature = torch.cat([item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country], dim=1)
        item_self_feature = self.feat_interaction(item_self_feature, self.dense_item_self_biinter, self.dense_item_self_siinter, dimension=1)

        if mode == 'ics':
            item_mu, item_var = self.item_vae.Q(item_self_feature)
            item_z = self.item_vae.sample_z(item_mu, item_var)
            item_embedding = self.item_vae.P(item_z)
        item_self_embed = self.dense_item_cate_self(torch.cat([item_self_feature, item_embedding], dim=-1))

        item_addgate = self.sigmoid(self.dense_item_addgate(torch.cat([item_self_embed.unsqueeze(1).repeat(1, item_onehop_size, 1), item_onehop_embed], dim=-1)))  # 商品的邻居门，控制邻居信息多少作为输入
        item_erasegate = self.sigmoid(self.dense_item_erasegate(torch.cat([item_self_embed, item_onehop_embed.mean(dim=1)], dim=-1)))
        item_onehop_embed_final = (item_onehop_embed * item_addgate).mean(1)
        item_self_embed = (1 - item_erasegate) * item_self_embed

        item_gcn_embed = self.leakyrelu(item_self_embed + item_onehop_embed_final)  # [batch, embed]

        #----------------------------------------------------------GCN-user
        # K=2
        user_onehop_id = self.user_embed(Variable(user_onehop_id))

        user_onehop_gender_emb = self.gender_embed(Variable(user_onehop_cate[:, :, 0]))
        user_onehop_age_emb = self.age_embed(Variable(user_onehop_cate[:, :, 1]))
        user_onehop_occupation_emb = self.occupation_embed(Variable(user_onehop_cate[:, :, 2]))

        user_onehop_feat = torch.cat([user_onehop_gender_emb.unsqueeze(2), user_onehop_age_emb.unsqueeze(2), user_onehop_occupation_emb.unsqueeze(2)], dim=2)
        user_onehop_embed = self.dense_user_cate_hop1(torch.cat([self.feat_interaction(user_onehop_feat, self.dense_user_onehop_biinter, self.dense_user_onehop_siinter, dimension=2), user_onehop_id], dim=-1))

        # K=1
        user_gender_emb = self.gender_embed(Variable(user_self_cate[:, 0]))
        user_age_emb = self.age_embed(Variable(user_self_cate[:, 1]))
        user_occupation_emb = self.occupation_embed(Variable(user_self_cate[:, 2]))

        user_self_feature = torch.cat([user_gender_emb.unsqueeze(1), user_age_emb.unsqueeze(1), user_occupation_emb.unsqueeze(1)], dim=1)
        user_self_feature = self.feat_interaction(user_self_feature, self.dense_user_self_biinter,  self.dense_user_onehop_siinter, dimension=1)

        if mode == 'ucs':
            user_mu, user_var = self.user_vae.Q(user_self_feature)
            user_z = self.user_vae.sample_z(user_mu, user_var)
            user_embedding = self.user_vae.P(user_z)
        user_self_embed = self.dense_user_cate_self(torch.cat([user_self_feature, user_embedding], dim=-1))

        user_addgate = self.sigmoid(self.dense_user_addgate(torch.cat([user_self_embed.unsqueeze(1).repeat(1, user_onehop_size, 1), user_onehop_embed],dim=-1)))
        user_erasegate = self.sigmoid(self.dense_user_erasegate(torch.cat([user_self_embed, user_onehop_embed.mean(dim=1)], dim=-1)))
        user_onehop_embed_final = (user_onehop_embed * user_addgate).mean(dim=1)
        user_self_embed = (1 - user_erasegate) * user_self_embed

        user_gcn_embed = self.leakyrelu(user_self_embed + user_onehop_embed_final)

        #--------------------------------------------------norm
        item_mu, item_var = self.item_vae.Q(item_self_feature)
        item_z = self.item_vae.sample_z(item_mu, item_var)
        item_preference_sample = self.item_vae.P(item_z)

        user_mu, user_var = self.user_vae.Q(user_self_feature)
        user_z = self.user_vae.sample_z(user_mu, user_var)
        user_preference_sample = self.user_vae.P(user_z)

        recon_loss = torch.norm(item_preference_sample - item_embedding) + torch.norm(user_preference_sample - user_embedding)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(item_z) + item_mu ** 2 - 1. - item_var, 1)) + \
                  torch.mean(0.5 * torch.sum(torch.exp(user_z) + user_mu ** 2 - 1. - user_var, 1))

        ####################################prediction#####################################################

        #concat -> mlp
        bu = self.user_bias(Variable(uids_list))
        bi = self.item_bias(Variable(sids_list))
        #pred = (user_gcn_embed * item_gcn_embed).sum(1, keepdim=True) + bu + bi + (self.miu).repeat(batch_size, 1)
        tmp = torch.cat([user_gcn_embed, item_gcn_embed], dim=1)
        pred = self.FC_pre(tmp) + (user_gcn_embed * item_gcn_embed).sum(1, keepdim=True) + bu + bi + (self.miu).repeat(batch_size, 1)

        return pred.squeeze(), recon_loss, kl_loss
