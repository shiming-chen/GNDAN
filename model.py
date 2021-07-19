import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

import torch_geometric as pyg
import torch_geometric.nn as pygnn
import torch_geometric.utils as pyg_utils


class RAN(nn.Module):
    def __init__(self, parts, map_threshold, att_dim,
                 drop_out=0., map_size=14, cov_channel=2048, is_hidden=False):
        super(RAN, self).__init__()
        self.map_threshold = map_threshold
        self.parts = parts
        self.map_size = map_size
        self.cov_channel = cov_channel

        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1)
        self.p_linear = nn.Linear(self.cov_channel*self.parts, att_dim, False)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, features):
        # assert features
        w = features.size()
        weights = torch.sigmoid(self.cov(features))

        # threshold the weights
        batch, parts, width, height = weights.size()
        weights_layout = weights.view(batch, -1)
        threshold_value, _ = weights_layout.max(dim=1)
        local_max, _ = weights.view(batch, parts, -1).max(dim=2)
        threshold_value = self.map_threshold*threshold_value.view(batch, 1) \
            .expand(batch, parts)
        weights = weights*local_max.ge(threshold_value).view(batch, parts, 1, 1). \
            float().expand(batch, parts, width, height)

        blocks = []
        for k in range(self.parts):
            Y = features*weights[:, k, :, :].unsqueeze(dim=1).expand(
                w[0], self.cov_channel, w[2], w[3])
            blocks.append(self.pool(Y).squeeze().view(-1, self.cov_channel))

        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1)))
        parts = torch.cat([b.unsqueeze(dim=1) for b in blocks], dim=1)

        return p_output, parts.permute(0, 2, 1)


class GAT(nn.Module):
    def __init__(self, num_node, input_dim, embed_dim, res=False,
                 dropout=0., concat_embed=True, layer_num=3, heads=1):
        super(GAT, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.res = res

        assert embed_dim % heads == 0

        self.conv_list = nn.ModuleList()
        self.conv_list.append(pygnn.GATConv(input_dim, embed_dim // heads,
                                            heads=heads, add_self_loops=False))
        for _ in range(layer_num - 1):
            conv = pygnn.GATConv(embed_dim, embed_dim // heads,
                                 heads=heads, add_self_loops=False)
            self.conv_list.append(conv)

        self.concat_embed = concat_embed
        conv_concat_dim = self.input_dim + len(self.conv_list) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(conv_concat_dim, input_dim),
            nn.ReLU())

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_concat = [x]
        for conv in self.conv_list:
            x = conv(x, edge_index=edge_index)
            x = x + F.relu(x) if self.res else F.relu(x)
            x_concat.append(x)
        return self.mlp(torch.cat(x_concat, dim=1))


class GNDAN(nn.Module):
    def __init__(self, config):
        super(GNDAN, self).__init__()

        self.config = config
        self.dim_f = config.dim_f
        self.dim_v = config.dim_v
        self.nclass = config.num_class

        self.att = nn.Parameter(torch.empty(
            self.nclass, config.num_attribute), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.mask_bias = nn.Parameter(torch.empty(
            1, self.nclass), requires_grad=False)
        self.V = nn.Parameter(torch.empty(
            config.num_attribute, self.dim_v), requires_grad=True)
        self.W_1 = nn.Parameter(torch.empty(
            self.dim_v, self.dim_f), requires_grad=True)
        self.W_2 = nn.Parameter(torch.empty(
            self.dim_v, self.dim_f), requires_grad=True)

        # graph model
        num_node = config.num_attribute
        self.graph_model = GAT(num_node=num_node,
                               input_dim=self.dim_f,
                               embed_dim=config.GAT_embed_dim,
                               res=config.GAT_res,
                               heads=config.GAT_heads,
                               concat_embed=config.GAT_concat_embed,
                               layer_num=config.GAT_layer_num)

        self.aren = RAN(parts=config.AREN_parts,
                        map_threshold=config.AREN_map_threshold,
                        att_dim=config.num_attribute,
                        map_size=int(np.sqrt(config.resnet_region)),
                        drop_out=config.AREN_dropout,
                        is_hidden=False)

        # bakcbone
        resnet101 = models.resnet101(pretrained=True)
        self.resnet101 = nn.Sequential(*list(resnet101.children())[:-2])

    def forward(self, imgs):
        Fs = self.resnet101(imgs)
        ran_embed, _ = self.aren(Fs)
        rgat_embed = self.forward_feature_graph(Fs)

        package = {}
        package['rgat_embed'] = self.forward_attribute(rgat_embed)
        package['ran_embed'] = self.forward_attribute(ran_embed)

        coef_ran = self.config.coef_ran
        coef_rgat = self.config.coef_rgat
        package['embed'] = (coef_rgat * package['rgat_embed'] +
                           coef_ran * package['ran_embed']) / (coef_rgat + coef_ran)
        return package

    def forward_attribute(self, embed):
        embed = torch.einsum('ki,bi->bk', self.att, embed)
        self.vec_bias = self.mask_bias*self.bias
        embed = embed + self.vec_bias
        return embed

    def forward_feature_graph(self, Fs):
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2]
                            * shape[3])  # batch x 2048 x 49

        V_n = F.normalize(self.V) if self.config.normalize_V else self.V
        Fs = F.normalize(Fs, dim=1)

        A = torch.einsum('iv,vf,bfr->bir', V_n, self.W_2, Fs)
        A = F.softmax(A, dim=-1)
        F_p = torch.einsum('bir,bfr->bif', A, Fs)

        F_p = F_p.permute(0, 2, 1)
        F_pn = F.normalize(F_p, dim=1)
        dense_adj_batch = F_pn.permute(0, 2, 1) @ F_pn
        feature_batch = F_p.permute(0, 2, 1)
        graph_list = []
        for node_feature, dense_adj in zip(feature_batch, dense_adj_batch):
            edge_index, edge_attr = pyg_utils.dense_to_sparse(dense_adj)
            graph_list.append(pyg.data.Data(
                node_feature, edge_index, edge_attr))
        graph = pyg.data.Batch.from_data_list(graph_list)
        outputs = self.graph_model(graph)
        F_p = pyg_utils.to_dense_batch(x=outputs, batch=graph.batch)[0]

        embed = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, F_p)

        return embed


if __name__ == '__main__':
    pass
