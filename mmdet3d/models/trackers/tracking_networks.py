import torch.nn as nn
from mmdet3d.models import TRACKERS
from .tracking_helpers import interpolate_bev_2d
from torch.nn.modules.transformer import (TransformerEncoder, TransformerDecoder, 
                                          TransformerDecoderLayer, TransformerEncoderLayer,)
from .lanegcn_nets import PostRes,LinearRes
from .loftr import LocalFeatureTransformer
# from .node_pooling import GatedPooling
# from torch_geometric.nn.conv import GATv2Conv,GatedGraphConv
# from torch_geometric.data import Data
# from torch_sparse import SparseTensor

import torch
import copy

module_obj = {
    'Linear':nn.Linear,
    'ReLU':nn.ReLU,
    'LSTM':nn.LSTM,
    'GroupNorm':nn.GroupNorm,
    'Embedding':nn.Embedding,
    'MultiheadAttention':nn.MultiheadAttention,
    'TransformerEncoder':TransformerEncoder,
    'TransformerEncoderLayer':TransformerEncoderLayer,
    'TransformerDecoder':TransformerDecoder,
    'TransformerDecoderLayer':TransformerDecoderLayer,
    'LayerNorm':nn.LayerNorm,
    'PostRes':PostRes,
    'LinearRes':LinearRes,
    'LocalFeatureTransformer':LocalFeatureTransformer,
    # 'GatedPooling':GatedPooling,
    # 'GATv2Conv':GATv2Conv,
    # 'GatedGraphConv':GatedGraphConv,
}

def build_module(cfg):
    if cfg == None or cfg == {}:
        return None

    if isinstance(cfg, list):
        return build_sequential(cfg)
    elif cfg['type'] == 'ModuleDict':
        del cfg['type']
        print(cfg)
        return nn.ModuleDict({k:build_module(v) for k,v in cfg.items()}) 

    cls_ = module_obj[cfg['type']]
    del cfg['type']
    return cls_(**cfg)


def build_sequential(module_list):
    if module_list == None or module_list == {}:
        return None
        
    modules = []
    for cfg in module_list:
        modules.append(build_module(cfg))
    return nn.Sequential(*modules)

def build_decisions(decisions):
    if decisions == None or decisions == {}:
        return None

    emb = decisions['embedding']
    return nn.ModuleDict({k:build_module(copy.deepcopy(emb)) for k in decisions if k != 'embedding'})





@TRACKERS.register_module()
class TrackingModules(nn.Module):
    def __init__(self,
                 #feature encoders
                 bev_attn,
                 MLP_encode_BEV1,
                 MLP_encode_BEV2,
                 MLP_encode_motion,
                 #transformer
                 EncoderNorm,
                 DecoderNorm,
                 TransformerEncoderLayer,
                 TransformerDecoderLayer,
                 TransformerEncoder,
                 TransformerDecoder,
                 pad,
                 #bev
                 bev_pos_enc,
                 bev_EncoderNorm,
                 bev_TransformerEncoderLayer,
                 bev_TransformerEncoder,
                 bev_forecast_proj,
                 bev_confidence_proj,
                 #decisions
                 MLPMerge,
                 MLPMatch,
                 MLPDetNewborn,
                 MLPDetFalsePositive,
                 MLPTrackFalsePositive,
                 MLPTrackFalseNegative,
                 #pretext
                 MLPRefine,
                 MLPPredict,
                 MLPProjPredict,
                 #lstm
                 lstm,
                 h0,
                 c0,
                 #pooling,
                 decision_pooling,
                 self_loop_pooling,
                 #embeddings
                 class_embeddings,
                 false_negative_emb,
                 decisions,
                 #EdgeGNN
                 edge_gnn,
                 decision_edge_linear):
        super().__init__()
        
        #Feature Ecoders
        self.MLPMerge = build_sequential(MLPMerge)
        self.MLP_encode_motion = build_sequential(MLP_encode_motion)
        self.bev_attn = build_module(bev_attn) 
        self.MLP_encode_BEV1 = build_sequential(MLP_encode_BEV1)
        self.MLP_encode_BEV2 = build_sequential(MLP_encode_BEV2)

        # Transformer 
        self.EncoderNorm = build_module(EncoderNorm)
        self.DecoderNorm = build_module(DecoderNorm)
        self.TransformerEncoderLayer = build_module(TransformerEncoderLayer)
        self.TransformerDecoderLayer = build_module(TransformerDecoderLayer)
        if TransformerEncoder != {} and  TransformerEncoder['type'] == 'TransformerEncoder':
            TransformerEncoder.update({'encoder_layer':self.TransformerEncoderLayer,
                                        'norm':self.EncoderNorm})
        if TransformerDecoder != {} and TransformerDecoder['type'] == 'TransformerDecoder':
            TransformerDecoder.update({'decoder_layer':self.TransformerDecoderLayer,
                                        'norm':self.DecoderNorm})
        self.TransformerEncoder = build_module(TransformerEncoder)
        self.TransformerDecoder = build_module(TransformerDecoder)
        self.pad = build_module(pad)#for padding the sequence

        #decisions
        self.MLPMatch = build_sequential(MLPMatch)
        self.MLPDetNewborn = build_sequential(MLPDetNewborn)
        self.MLPDetFalsePositive = build_sequential(MLPDetFalsePositive)
        self.MLPTrackFalsePositive = build_sequential(MLPTrackFalsePositive)
        self.MLPTrackFalseNegative = build_sequential(MLPTrackFalseNegative)
        self.decisions = build_decisions(decisions)

        #pretext MLPs
        self.MLPRefine = build_sequential(MLPRefine)
        self.MLPPredict = build_sequential(MLPPredict)
        self.MLPProjPredict = build_sequential(MLPProjPredict)

        #LSTM
        self.lstm = build_module(lstm)
        self.h0 = build_module(h0)
        self.c0 = build_module(c0)
        self.false_negative_emb = build_module(false_negative_emb) 


        #bev
        self.bev_forecast_proj = build_module(bev_forecast_proj)
        self.bev_confidence_proj = build_module(bev_confidence_proj)
        self.bev_maxpool = torch.nn.MaxPool1d(5)
        self.bev_pos_enc = build_module(bev_pos_enc)
        self.bev_EncoderNorm = build_module(bev_EncoderNorm)
        self.bev_TransformerEncoderLayer = build_module(bev_TransformerEncoderLayer)
        if bev_TransformerEncoder != {}:
            bev_TransformerEncoder.update({'encoder_layer':self.bev_TransformerEncoderLayer,
                                        'norm':self.bev_EncoderNorm})
        self.bev_TransformerEncoder = build_module(bev_TransformerEncoder)


        #pooling
        self.decision_pooling =  nn.ModuleDict({k:build_module(v) for k,v in decision_pooling.items()})
        self.self_loop_pooling =  nn.ModuleDict({k:build_module(v) for k,v in self_loop_pooling.items()})


        #embeddings
        self.class_embeddings = build_module(class_embeddings)


        #Edge GNN
        self.edge_gnn = build_module(edge_gnn)
        self.decision_edge_linear = build_module(decision_edge_linear)



def getEdgeGraph(nodes,decision_edges,num_tracks,num_dets,
                 tracking_decisions,detection_decisions,device,method='combination'):
    """Nodes have detection decisions and tracking decisions stacked on top of 
        nodes representing edges between tracks and detections"""
    
    #     num_tracks += len(detection_decisions)
    #     num_dets += len(tracking_decisions)
    assert nodes.size(0) == num_tracks * num_dets
    edges = []
    for x in range(num_tracks):
        r = torch.arange(x * num_dets, (x+1) * num_dets,device=device)
        if method == 'combination':
            e = torch.combinations(r)
        elif method == 'permutation':
            e = torch.cartesian_prod(r,r)
        edges.append(e)

    for x in range(num_dets):
        r = ( torch.arange(0,num_tracks,device=device) * num_dets ) + x
        if method == 'combination':
            e = torch.combinations(r)
        elif method == 'permutation':
            e = torch.cartesian_prod(r,r)
        edges.append(e)
    
    nodes_cat = []
    offset = nodes.size(0)
    #Each tracking decision represents a column in the matrix.
    #They are only connected column-wise
    for i,td in enumerate(tracking_decisions):
        nodes_cat.append(decision_edges[td])
        for x in range(num_tracks):
            edges.append(
                torch.cartesian_prod(torch.tensor([offset],device=device),
                                     torch.arange(x * num_dets, (x+1) * num_dets,device=device))
            )
            offset += 1
        
    #Each tracking decision represents a column in the matrix.
    #They are only connected row-wise
    for i,dd in enumerate(detection_decisions):
        nodes_cat.append(decision_edges[dd])
        for x in range(num_dets):
            edges.append(
                torch.cartesian_prod(torch.tensor([offset],device=device),
                                     torch.arange(0,num_tracks,device=device) * num_dets + x
            ))
            offset += 1
        
    nodes = torch.cat([nodes,torch.cat(nodes_cat,dim=0)],dim=0)
    edge_index = torch.cat(edges,dim=0).t().contiguous()
    return Data(x=nodes,edge_index=edge_index) #nodes,edge_index #


@TRACKERS.register_module()
class DecisionTracker(TrackingModules):

    def __init__(self,merge_forward,message_passing_forward,decisions_forward,get_edges_forward='simple',*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.merge_forward = merge_forward
        self.message_passing_forward = message_passing_forward
        self.decisions_forward = decisions_forward
        self.get_edges_forward = get_edges_forward


    def edge_graph_message_passing(self,object_edges,decision_edges,num_tracks,num_dets,tracking_decisions,detection_decisions,device,*args,**kwargs):
        if self.edge_gnn == {} or self.edge_gnn == None:
            return object_edges,decision_edges

        decision_edges = {k:self.decision_edge_linear[k](v) for k,v in decision_edges.items()}
        object_edges = self.decision_edge_linear['match'](object_edges)

        oe_offset = object_edges.size(0)
        data = getEdgeGraph(nodes=object_edges,
                            decision_edges=decision_edges,
                            num_tracks=num_tracks,
                            num_dets=num_dets,
                            tracking_decisions=tracking_decisions,
                            detection_decisions=detection_decisions,
                            device=device,
                            method='permutation')
        
        # print(x.shape,x)
        # print(edge_index.shape,edge_index)

        # adj = SparseTensor(row=data['edge_index'][0], col=data['edge_index'][1],
        #            sparse_sizes=(data['x'].size(0), data['x'].size(0)))

        out = data['x']
        
        for x in range(3):
            out = self.edge_gnn(out,adj)#data['edge_index'])

        # print('x',data['x'])
        # print('edge_index',data['edge_index'])
        object_edges = out[:oe_offset,...]

        for i,td in enumerate(tracking_decisions):
            decision_edges[td] = out[oe_offset:oe_offset+num_tracks,...]
            oe_offset += num_tracks
        
        for i,dd in enumerate(detection_decisions):
            decision_edges[dd] = out[oe_offset:oe_offset+num_dets,...]
            oe_offset += num_dets

        del data
        return  object_edges, decision_edges


    def forward_association(self,trk_feats,det_feats,tracking_decisions,detection_decisions,device,*args,**kwargs):
        num_track, num_det = trk_feats.shape[0], det_feats.shape[0]

        trk_feats, det_feats, decision_embs = self.message_passing(trk_feats=trk_feats,
                                                                    det_feats=det_feats,
                                                                    decision_feats=None,
                                                                    tracking_decisions=tracking_decisions,
                                                                    detection_decisions=detection_decisions,
                                                                    device=device,)

        object_edges, decision_edges, track_idx, det_idx = self.get_edges(trk_feats=trk_feats,
                                                                            det_feats=det_feats,
                                                                            decision_feats=None,
                                                                            tracking_decisions=tracking_decisions,
                                                                            detection_decisions=detection_decisions,
                                                                            num_det=num_det,
                                                                            num_track=num_track,
                                                                            device=device,)

        # before_shape = object_edges.shape

        # print('object_edges',object_edges.shape)
        # print('decision_edges',decision_edges)
        # print([x.shape for x in decision_edges.values()])
        # print('track_idx',track_idx.shape)
        # print('det_idx',det_idx.shape)
        # exit(0)

        object_edges, decision_edges = self.edge_graph_message_passing(num_dets=num_det,
                                                                        num_tracks=num_track,
                                                                        object_edges=object_edges,
                                                                        decision_edges=decision_edges,
                                                                        track_idx=track_idx,
                                                                        det_idx=det_idx,
                                                                        tracking_decisions=tracking_decisions,
                                                                        detection_decisions=detection_decisions,
                                                                        device=device,)
        
        # assert before_shape == object_edges.shape

                                
        supervise = {}
        supervise['cost_mat'] = self.decision_forward(decision='match',
                                                      track_idx=track_idx,
                                                      det_idx=det_idx,
                                                      edge_feats=object_edges,
                                                      num_det=num_det,
                                                      num_track=num_track,
                                                      device=device)

        supervise.update({k:self.decision_forward(decision=k,
                                                 edge_feats=decision_edges[k],
                                                 decision_embs=decision_embs,
                                                 device=device)
            for k in detection_decisions + tracking_decisions})


        return supervise, trk_feats, det_feats



    def forecast_bev(self,x):
        x = self.MLPProjPredict(x)
        x = self.MLPPredict(x)
        return x


    ################################ Decisions ################################
    def decision_forward(self,decision,*args,**kwargs):
        return getattr(self,f'{decision}_{self.decisions_forward[decision]}')(*args,**kwargs)

    #Matches
    def match_MLP(self,num_track, num_det,edge_feats,track_idx,det_idx,device,*args,**kwargs):
        cost_mat = torch.zeros((num_track, num_det),dtype=torch.float32,device=device)
        cost_mat[track_idx,det_idx] = self.MLPMatch(edge_feats).squeeze(1)
        return cost_mat

    def match_Dot(self,trk_feats,det_feats,*args,**kwargs):
        return trk_feats @ det_feats.T

    def match_dec_MLP(self,*args,**kwargs):
        return self.match_MLP(*args,**kwargs)


    #det false positives
    def det_false_positive_MLP(self,edge_feats,*args,**kwargs):
        return self.MLPDetFalsePositive(edge_feats).squeeze(1)

    def det_false_positive_Dot(self,det_feats,decision_embs,*args,**kwargs):
        return (decision_embs['det_false_positive'] * det_feats).sum(1)
    
    def det_false_positive_dec_MLP(self,det_feats,decision_embs,*args,**kwargs):
        # print(decision_embs['det_false_positive'].shape)
        # print(det_feats.shape)
        # print(decision_embs['det_false_positive'].unsqueeze(0).repeat(det_feats.size(0),1).shape)
        return self.MLPDetFalsePositive(torch.cat([decision_embs['det_false_positive'].unsqueeze(0).repeat(det_feats.size(0),1),det_feats],dim=1)).squeeze(1)


    #det newborn positives
    def det_newborn_MLP(self,edge_feats,*args,**kwargs):
        return self.MLPDetNewborn(edge_feats).squeeze(1)

    def det_newborn_Dot(self,det_feats,decision_embs,*args,**kwargs):
        return (decision_embs['det_newborn'] * det_feats).sum(1)

    def det_newborn_dec_MLP(self,det_feats,decision_embs,*args,**kwargs):
        return self.MLPDetFalsePositive(torch.cat([decision_embs['det_newborn'].unsqueeze(0).repeat(det_feats.size(0),1),det_feats],dim=1)).squeeze(1)


    #track false positives
    def track_false_positive_MLP(self,edge_feats,*args,**kwargs):
        return self.MLPTrackFalsePositive(edge_feats).squeeze(1)

    def track_false_positive_Dot(self,trk_feats,decision_embs,*args,**kwargs):
        return (decision_embs['track_false_positive'] * trk_feats).sum(1)

    def rack_false_positive_dec_MLP(self,trk_feats,decision_embs,*args,**kwargs):
        return self.MLPDetFalsePositive(torch.cat([trk_feats,decision_embs['track_false_positive'].unsqueeze(0).repeat(trk_feats.size(0),1)],dim=1)).squeeze(1)


    #track false negatives
    def track_false_negative_MLP(self,edge_feats,*args,**kwargs):
        return self.MLPTrackFalseNegative(edge_feats).squeeze(1)

    def track_false_negative_Dot(self,trk_feats,decision_embs,*args,**kwargs):
        return (decision_embs['track_false_negative'] * trk_feats).sum(1)

    def track_false_negative_dec_MLP(self,trk_feats,decision_embs,*args,**kwargs):
        return self.MLPDetFalsePositive(torch.cat([trk_feats,decision_embs['track_false_negative'].unsqueeze(0).repeat(trk_feats.size(0),1)],dim=1)).squeeze(1)



    ################################ Message Passing ################################
    def message_passing(self,*args,**kwargs):
        return getattr(self,f'message_passing_{self.message_passing_forward}')(*args,**kwargs)

    def message_passing_simple(self,trk_feats,det_feats,device,*args,**kwargs):
        """Enables information to be shared between tracks and detections"""
        trk_shape = trk_feats.shape
        det_shape = det_feats.shape

        det_size = det_feats.size(0)
        trk_size = trk_feats.size(0)

        if det_size == 0 and trk_size == 0:
            return trk_feats, det_feats, {}

        # print("Before Message Passing",trk_feats.shape, det_feats.shape)
        trk_feats = trk_feats.unsqueeze(1)
        det_feats = det_feats.unsqueeze(1)

        assert trk_feats.shape[1:] == det_feats.shape[1:]

        #get the features of matched dets
        #pad tracks
        if trk_size < det_size:
            diff = det_size - trk_size
            trk_feats = torch.cat([trk_feats,self.pad.weight.unsqueeze(0).repeat(diff,1,1)],dim=0)
        elif trk_size > det_size:
            diff = trk_size - det_size
            det_feats = torch.cat([det_feats,self.pad.weight.unsqueeze(0).repeat(diff,1,1)],dim=0)

        if self.TransformerEncoder == None:
            encoded = torch.cat([trk_feats,det_feats],dim=1)
        else:
            encoded = self.TransformerEncoder(torch.cat([trk_feats,det_feats],dim=1))


        if type(self.TransformerDecoder) == LocalFeatureTransformer:
            trk_feats, det_feats = self.TransformerDecoder(trk_feats,det_feats)
            # print(trk_shape,det_shape)
            # print(trk_feats.shape,det_feats.shape)
            trk_feats, det_feats = trk_feats[:trk_size,0,:], det_feats[:det_size,0,:]
            # print(trk_feats.shape,det_feats.shape)

        else:
            out = self.TransformerDecoder(encoded, torch.cat([encoded[:,1:2,:],encoded[:,0:1,:]],dim=1))
            trk_feats, det_feats = out[:trk_size,0,:], out[:det_size,1,:]

        assert trk_shape == trk_feats.shape
        assert det_shape == det_feats.shape

        return_decisions = {}
        return trk_feats, det_feats, return_decisions

    def message_passing_decisions(self,trk_feats,det_feats,decision_feats,tracking_decisions,detection_decisions,device,*args,**kwargs):
        """Enables information to be shared between tracks and detections"""
        if len(tracking_decisions) > 0:
            track_dec_feats = torch.cat([self.decisions[k].weight.to(device) for k in tracking_decisions])
        else:
            track_dec_feats = torch.empty([0,trk_feats.size(1)],device=device)


        if len(detection_decisions) > 0:
            det_dec_feats = torch.cat([self.decisions[k].weight.to(device) for k in detection_decisions])
        else:
            det_dec_feats = torch.empty([0,trk_feats.size(1)],device=device)


        # if type(list(self.decisions.values())[0]) == nn.Embedding:
            
            
        # else:
        #     track_dec_feats = torch.cat([self.decisions[k](decision_feats) for k in tracking_decisions])
        #     det_dec_feats = torch.cat([self.decisions[k](decision_feats) for k in detection_decisions])

        
        trk_shape = trk_feats.shape
        det_shape = det_feats.shape

        trk_size = trk_feats.size(0)
        det_size = det_feats.size(0)

        track_dec_size = track_dec_feats.size(0)
        det_dec_size = det_dec_feats.size(0)

        track_total_size = trk_size + track_dec_size
        det_total_size = det_size + det_dec_size

        trk_feats = trk_feats.unsqueeze(1)
        det_feats = det_feats.unsqueeze(1)
        track_dec_feats = track_dec_feats.unsqueeze(1)
        det_dec_feats = det_dec_feats.unsqueeze(1)

        assert trk_feats.shape[1:] == det_feats.shape[1:]


        #get the features of matched dets
        #pad tracks
        if track_total_size < det_total_size:
            diff = det_total_size - track_total_size
            trk_feats = torch.cat([track_dec_feats,trk_feats,self.pad.weight.unsqueeze(0).repeat(diff,1,1)],dim=0)
            det_feats = torch.cat([det_dec_feats,det_feats,],dim=0)
        elif det_total_size < track_total_size:
            diff = track_total_size - det_total_size
            det_feats = torch.cat([det_dec_feats,det_feats,self.pad.weight.unsqueeze(0).repeat(diff,1,1)],dim=0)
            trk_feats = torch.cat([track_dec_feats,trk_feats,],dim=0)
        else:
            trk_feats = torch.cat([track_dec_feats,trk_feats],dim=0)
            det_feats = torch.cat([det_dec_feats,det_feats],dim=0)
        
        if self.TransformerEncoder == None:
            encoded = torch.cat([trk_feats,det_feats],dim=1)
        else:
            encoded = self.TransformerEncoder(torch.cat([trk_feats,det_feats],dim=1))

        
        out = self.TransformerDecoder(encoded, torch.cat([encoded[:,1:2,:],encoded[:,0:1,:]],dim=1))
        track_dec_feats, det_dec_feats = out[:track_dec_size,0,:], out[:det_dec_size,1,:]

        trk_feats, det_feats = out[track_dec_size:track_total_size,0,:], out[det_dec_size:det_total_size,1,:]

        # try:
        assert trk_shape == trk_feats.shape
        assert det_shape == det_feats.shape

        return_decisions = {}
        for i,k in enumerate(tracking_decisions):
            return_decisions[k] = track_dec_feats[i,...]
        
        for i,k in enumerate(detection_decisions):
            return_decisions[k] = det_dec_feats[i,...]


        # print({k:v.shape for k,v in return_decisions.items()})

        return trk_feats, det_feats, return_decisions


    ################################ Merging BEV Feats ################################
    def getMergedFeats(self,*args,**kwargs):
        return getattr(self,f'getMergedFeats_{self.merge_forward}')(*args,**kwargs)

    def getMergedFeats_queries(self,ego,pred_cls,bbox_feats,bev_feats,queries,confidence_scores,point_cloud_range,device,*args,**kwargs):
        """TODO make this torch processed only"""

        if bbox_feats.nelement() == 0:
            return torch.tensor([],device=device)
        
        #process motion features
        ego_feat = ego.get_current_feat().repeat(bbox_feats.size(0),1)

        motion_feats = torch.cat([confidence_scores.unsqueeze(1), bbox_feats, ego_feat],dim=1)
        motion_feats = self.MLP_encode_motion(motion_feats)
        
        all_feats = torch.cat([queries, motion_feats],dim=1)

        merged = self.MLPMerge(all_feats)
        if self.class_embeddings != dict() and self.class_embeddings != None:
            merged = merged + self.class_embeddings.weight[pred_cls.long(),:]

        return merged, bev_feats, motion_feats


    def getMergedFeats_interpolate(self,ego,pred_cls,bbox_feats,bev_feats,confidence_scores,point_cloud_range,device,*args,**kwargs):
        """TODO make this torch processed only"""

        if bbox_feats.nelement() == 0:
            return torch.tensor([],device=device)
        
        #process motion features
        ego_feat = ego.get_current_feat().repeat(bbox_feats.size(0),1)

        motion_feats = torch.cat([confidence_scores.unsqueeze(1), bbox_feats, ego_feat],dim=1)
        motion_feats = self.MLP_encode_motion(motion_feats)
        
        bev_feats = interpolate_bev_2d(bev_feats=bev_feats,
                                       xy=bbox_feats[:,:2].unsqueeze(1),
                                       point_cloud_range=point_cloud_range,
                                       device=device)

        bev_feats = bev_feats.squeeze(1)
        bev_feats = self.MLP_encode_BEV1(bev_feats)
        
        all_feats = torch.cat([bev_feats, motion_feats],dim=1)

        merged = self.MLPMerge(all_feats)
        if self.class_embeddings != dict() and self.class_embeddings != None:
            merged = merged + self.class_embeddings.weight[pred_cls.long(),:]

        return merged, bev_feats, motion_feats



    def getMergedFeats_cpoint_interpolate_transformer(self,ego,pred_cls,bbox_side_and_center,bbox_feats,bev_feats,confidence_scores,point_cloud_range,device,*args,**kwargs):
        """TODO make this torch processed only"""

        if bbox_feats.nelement() == 0:
            return torch.tensor([],device=device)
        
        #process motion features
        ego_feat = ego.get_current_feat().repeat(bbox_feats.size(0),1)

        motion_feats = torch.cat([confidence_scores.unsqueeze(1), bbox_feats, ego_feat],dim=1)
        motion_feats = self.MLP_encode_motion(motion_feats)
        
        bev_feats = interpolate_bev_2d(bev_feats=bev_feats,
                                       xy=bbox_side_and_center,
                                       point_cloud_range=point_cloud_range,
                                       device=device)

        bev_feats = torch.cat([self.bev_maxpool(bev_feats.permute(0,2,1)).permute(0,2,1),bev_feats],dim=1)
        bev_feats = bev_feats + self.bev_pos_enc.weight.repeat(bev_feats.size(0),1,1)
        bev_feats = self.bev_TransformerEncoder(bev_feats)[:,0,:]


        bev_feats = self.MLP_encode_BEV1(bev_feats)

        all_feats = torch.cat([bev_feats, motion_feats],dim=1)

        merged = self.MLPMerge(all_feats)
        if self.class_embeddings != dict() and self.class_embeddings != None:
            merged = merged + self.class_embeddings.weight[pred_cls.long(),:]

        return merged, bev_feats, motion_feats

    

    def getMergedFeats_cpoint_interpolate_simple(self,ego,pred_cls,bbox_side_and_center,bbox_feats,bev_feats,confidence_scores,point_cloud_range,device,*args,**kwargs):
        """TODO make this torch processed only"""

        if bbox_feats.nelement() == 0:
            return torch.tensor([],device=device)
        
        #process motion features
        ego_feat = ego.get_current_feat().repeat(bbox_feats.size(0),1)

        motion_feats = torch.cat([confidence_scores.unsqueeze(1), bbox_feats, ego_feat],dim=1)
        motion_feats = self.MLP_encode_motion(motion_feats)
        
        bev_feats = interpolate_bev_2d(bev_feats=bev_feats,
                                       xy=bbox_side_and_center,
                                       point_cloud_range=point_cloud_range,
                                       device=device)

        bev_feats = bev_feats.reshape(bev_feats.size(0),-1)
        # print(bev_feats.shape)
        bev_feats = self.MLP_encode_BEV1(bev_feats)

        all_feats = torch.cat([bev_feats, motion_feats],dim=1)

        merged = self.MLPMerge(all_feats)
        if self.class_embeddings != dict() and self.class_embeddings != None:
            merged = merged + self.class_embeddings.weight[pred_cls.long(),:]

        return merged, bev_feats, motion_feats



    ################################ Retrieving decision features ################################


    def get_edges(self,*args,**kwargs):
        return getattr(self,f'get_edges_{self.get_edges_forward}')(*args,**kwargs)

    def get_edges_simple(self,num_track,num_det,trk_feats,det_feats,tracking_decisions,detection_decisions,device,*args,**kwargs):

        cp = torch.cartesian_prod(torch.arange(0,num_track,device=device),
                                    torch.arange(0,num_det,device=device))
        object_edges = torch.cat([trk_feats[cp[:,0],:],det_feats[cp[:,1],:]],dim=1)
        decision_edges={}
        for k in tracking_decisions:
            decision_edges[k] = trk_feats

        for k in detection_decisions:
            decision_edges[k] = det_feats

            
        return object_edges, decision_edges, cp[:,0], cp[:,1]

    def get_edges_gnn(self,num_track,num_det,trk_feats,det_feats,tracking_decisions,detection_decisions,device,*args,**kwargs):
        """Computes edge features when using a GNN for node pooling"""
        nodes = torch.cat([trk_feats,det_feats],dim=0)
        edges = torch.cartesian_prod(torch.arange(0,num_track),num_track+torch.arange(0,num_det))
        data = Data(x=nodes, edge_index=edges.t().contiguous())
        decision_edges = {}
        for k in tracking_decisions: 
            out = self.decision_pooling[k](data['x'],data['edge_index'])
            decision_edges[k] = torch.cat([nodes[:num_track,:],
                                        out[:num_track,:]],dim=1)
            
            
        for k in detection_decisions: 
            out = self.decision_pooling[k](data['x'],data['edge_index'])
            decision_edges[k] = torch.cat([nodes[num_track:,:],
                                        out[num_track:,:]],dim=1)
            
            
        cp = torch.cartesian_prod(torch.arange(0,num_track,device=device),
                                    torch.arange(0,num_det,device=device))
        object_edges = torch.cat([trk_feats[cp[:,0],:],det_feats[cp[:,1],:]],dim=1)
        
        
        return object_edges, decision_edges, cp[:,0], cp[:,1]

    def get_edges_pooling(self,num_track,num_det,trk_feats,det_feats,tracking_decisions,detection_decisions,device,*args,**kwargs):
        """Computes edge features when using a simple node pooler."""
        cp = torch.cartesian_prod(torch.arange(0,num_track,device=device),
                                    torch.arange(0,num_det,device=device))
        object_edges = torch.cat([trk_feats[cp[:,0],:],det_feats[cp[:,1],:]],dim=1)
        
        
        decision_edges = {}
        for k in tracking_decisions:
            if num_det > 0 and num_track > 0:
                decision_edges[k] = self.decision_pooling[k](object_edges,cp[:,0]) 
                # print(decision_edges[k].shape, self.self_loop_pooling[k](trk_feats).shape)
                decision_edges[k] = decision_edges[k] + self.self_loop_pooling[k](trk_feats)
            else:
                decision_edges[k] = self.self_loop_pooling[k](trk_feats)
            
        for k in detection_decisions: 
            if num_det > 0 and num_track > 0:
                decision_edges[k] = self.decision_pooling[k](object_edges,cp[:,1]) 
                # print(decision_edges[k].shape, self.self_loop_pooling[k](det_feats).shape)
                decision_edges[k] = decision_edges[k] +  self.self_loop_pooling[k](det_feats)
            else:
                decision_edges[k] = self.self_loop_pooling[k](det_feats)
        
        
        return object_edges, decision_edges, cp[:,0], cp[:,1]




