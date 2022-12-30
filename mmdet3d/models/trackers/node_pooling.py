#Taken from https://github.com/mods333/energy-based-scene-graph all credit to the authors.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
from torch_scatter import scatter

import torch


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


    
class EdgeGatedPooling(nn.Module):
    '''
    Modified Version of Global Pooling Layer from the “Gated Graph Sequence Neural Networks” paper
    Parameters:
    ----------
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
    '''

    def __init__(self, node_dim, edge_dim, pooling_dim):
        super(EdgeGatedPooling, self).__init__()

        ###############################################################
        # Gates to compute attention scores
        self.hgate_node = nn.Sequential(
            nn.Linear(node_dim, 1)
        )
        self.hgate_edge = nn.Sequential(
            nn.Linear(edge_dim, 1)
        )

        ##############################################################
        #Layers to tranfrom features before combinig
        self.htheta_node = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU()
        )
        self.htheta_edge = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU()
        )

        ################################################################
        #Final pooling layer
        self.poolingLayer = nn.Sequential(
            nn.Linear(node_dim + edge_dim, pooling_dim)
        )
    def forward(self, node_features, edge_features, node_batch_list, edge_batch_list):

        
        node_alpha = self.hgate_node(node_features)
        edge_alpha = self.hgate_edge(edge_features)
        
        node_pool = scatter(node_alpha*node_features, node_batch_list, dim=0)
        edge_pool = scatter(edge_alpha*edge_features, edge_batch_list, dim=0, dim_size = node_pool.shape[0])

        return self.poolingLayer(cat((node_pool, edge_pool), -1))

class GatedPooling(nn.Module):
    '''
    Modified Version of Global Pooling Layer from the “Gated Graph Sequence Neural Networks” paper
    Parameters:
    ----------
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
    '''

    def __init__(self, node_dim, pooling_dim):
        super(GatedPooling, self).__init__()

        ###############################################################
        # Gates to compute attention scores
        self.hgate_node = nn.Sequential(
            nn.Linear(node_dim, 1)
        )

        ##############################################################
        #Layers to tranfrom features before combinig
        self.htheta_node = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU()
        )

        ################################################################
        #Final pooling layer
        self.poolingLayer = nn.Sequential(
            nn.Linear(node_dim, pooling_dim)
        )
    def forward(self, node_features, batch_list):

        node_alpha = self.hgate_node(node_features)
        node_pool = scatter(node_alpha*node_features, batch_list, dim=0, reduce="sum")

        return self.poolingLayer(node_pool)