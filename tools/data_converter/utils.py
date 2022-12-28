
import numpy as np
from nuscenes.utils.geometry_utils import transform_matrix

def getFuture(nusc,annot_token,max_count):
    annot_token = nusc.get('sample_annotation',annot_token)
    if annot_token['next'] == '':
        return []
    future_anns = [annot_token['next']]
    count = 1
    while(True):
        annot_token = nusc.get('sample_annotation',annot_token['next'])
        
        if annot_token['next'] == '' or count >= max_count:
            break
        else: 
            future_anns.append(annot_token['next'])
        count += 1
    return future_anns
    
def getPast(nusc,annot_token,max_count):
    annot_token = nusc.get('sample_annotation',annot_token)
    if annot_token['prev'] == '':
        return []
    past_anns = [annot_token['prev']]
    count = 1
    while(True):
        annot_token = nusc.get('sample_annotation',annot_token['prev'])

        if annot_token['prev'] == '' or count >= max_count:
            break
        else: 
            past_anns.append(annot_token['prev'])
        count += 1
    return past_anns
    

def get_loc_offset(nusc,lidar_token,annot_token,future_count=6,past_count=4,padding_value=-5000.0):
    prev_tokens = getPast(nusc,annot_token,max_count=past_count)
    future_tokens = getFuture(nusc,annot_token,max_count=future_count)
    
    all_anns = prev_tokens + [annot_token] + future_tokens
    boxes = list(map(nusc.get_box, all_anns))
    locs = np.array([b.center for b in boxes])
    locs = np.concatenate([locs,np.ones((locs.shape[0],1))],axis=1)    
    TM = transform_matrix(boxes[len(prev_tokens)].center, boxes[len(prev_tokens)].orientation, inverse=True)
    new_locs = TM @ locs.T
    assert abs(np.sum(new_locs[:,len(prev_tokens)]) - 1.0) < 0.0000001
    
    transformed = new_locs.T[:,:2] # only get x and y
    pasts = transformed[:len(prev_tokens),:]
    current = transformed[len(prev_tokens),:]
    futures = transformed[1+len(prev_tokens):,:]

    past_pad = past_count - pasts.shape[0]
    pasts = np.pad(pasts,((past_pad,0),(0,0)),'constant',constant_values=(padding_value,padding_value))

    futures_pad = future_count - futures.shape[0]
    futures = np.pad(futures,((0,futures_pad),(0,0)),'constant',constant_values=(padding_value,padding_value))

    return pasts, futures



