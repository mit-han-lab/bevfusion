import random
import json
import copy

import numpy as np
import os.path as osp

from .builder import DATASETS, build_dataset
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

from mmcv.runner import get_dist_info


def load_nusc_scene_map(version,data_root):
    """Loads a cached version of the scene map or creates it outright"""
    filepath = osp.join(data_root,"scene_map_{}.json".format(version))
    try:
        sceneMap = json.load(open(filepath,'r'))
    except Exception:
        from nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        sceneMap = {}
        for x in nusc.scene:
            sceneMap[x['token']] = [x['first_sample_token']]
            nextToken = x['first_sample_token']
            while True:
                my_sample = nusc.get('sample', nextToken)
                if my_sample['next'] == '':
                    break
                else:
                    nextToken = my_sample['next']
                    sceneMap[x['token']].append(nextToken)
                    
            assert len(sceneMap[x['token']]) == x['nbr_samples']

        json.dump(sceneMap, open(filepath,'w'))
    return sceneMap


class SequenceSampler(object):
    """Helper class for sampling from sequences.
    
    Samples sequences of indices from an index list. Each sequence is of 
    specified length unless a number larger than the maximum length is provided.

    Class members:
        infos_idx_list (list[int]): Infos indices of the sequence.
        max_starting_idx (list[int]): Largest possible starting index
            for a sampled sequence.
        sample_size (int): The length of sequence to sample.
    """

    def __init__(self, infos_idx_list, sample_size, sampling_mode, max_seq_len, pad=-1):
        self.pad = pad
        self.sampling_mode = sampling_mode
        self.infos_idx_list = self.make_infos_list(infos_idx_list,max_seq_len)
        self.mod_sample_size(sample_size)


    def __str__(self):
        return "len sequence: {}, \nsample size: {},\nmax_starting_idx: {}".format(
            len(self.infos_idx_list),self.sample_size,self.max_starting_idx)


    def make_infos_list(self,infos_idx_list,max_seq_len):
        if self.sampling_mode == 'train':
            return infos_idx_list
        else:
            return infos_idx_list + [self.pad for x in range(max_seq_len - len(infos_idx_list))]

    def mod_sample_size(self,sample_size):
        self.sample_size = sample_size
        self.max_starting_idx = len(self.infos_idx_list) - sample_size

    def __call__(self):
        """samples one sequence form the indices using"""
        if self.sampling_mode == 'val':
            return self.infos_idx_list
        else:
            if self.max_starting_idx == 0:
                return self.infos_idx_list
            try:
                start = np.random.randint(low=0,high=self.max_starting_idx)
            except ValueError:
                print("ValueError occured in __call__")
                print(self)
                exit(0)

            return self.infos_idx_list[start:start + self.sample_size]



def getMetaClassOcclusion(class_names,cls_to_occs):
    merged = {}
    for c in class_names:
        for k in cls_to_occs.keys():
            if c in k:
                temp = [x for x in cls_to_occs[k] if type(x)!= dict]
                if temp != []:
                    print(c,k)
                try: 
                    merged[c] += cls_to_occs[k]
                except KeyError:
                    merged[c] = cls_to_occs[k]
    return merged



@DATASETS.register_module()
class SequentialDataset(object):
    """A wrapper used to load nuscenes sequentially [commented]

    Load each scene in the correct order for tracking 

    Class members:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
        class_to_idx (dict): mapping from class name to its list index 
        token_to_infos_id (dict): mapping from a token to its index in the infos list 
        visualize (bool): whether the current dataset is in visualization mode
        occlusionMap (dict): a list of occlusions for objects in each frame
        sample_indices (list[int]): a list os sample indices in sequetial order 
        sceneMap (dict): 
        sample_indices_shuffle (list[int]): These are the indices of each sample in sequential order 
        first_in_scene_lookup_shuffle (list[bool]): 
        sequence_level_indices_shuffle (bool): 
        extra_sequence_multipliter (int): only used in training mode, allows to load more sequences per epoch
            this is helpfull to acheive larger batch size when training with gradient accumulation and an
            epoch based scheduler
    """


    def __init__(self, dataset, sampling_mode, sample_size=16, gpus=1, extra_sequence_multipliter=0, visualize=False, verbose=False, **kwargs):
        """Constructor for the sequential dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset to be class sampled.
            sampling_mode (string): the sampling strategy to be used
            visualize (bool): whether to sample from the dataset in visualization mode or not.
        """
        # print("START SequentialDataset Init")
        self.dataset = build_dataset(dataset)
        self.visualize = visualize
        self.sampling_mode = sampling_mode
        self.gpus = gpus
        self.sample_size = sample_size
        self.extra_sequence_multipliter = extra_sequence_multipliter
        self.rank, self.world_size = get_dist_info()
        self.verbose=verbose
        
        self.CLASSES = self.dataset.CLASSES
        self.class_to_idx = {name: i for i, name in enumerate(self.CLASSES)}
        self.token_to_infos_id = {x['token']:i for i,x in enumerate(self.dataset.data_infos)}

        if visualize:
            print(self.CLASSES)
            self.occlusionMap = self.load_occlusion_map()
            print(self.occlusionMap.keys())
            self.occlusionMap = getMetaClassOcclusion(self.CLASSES,self.occlusionMap)
            print(self.occlusionMap.keys())
            self.sample_indices, self.first_in_scene_lookup, self.track_viz_idx = self._get_sample_indices_viz()
            
        else:
            self.sceneMap = self.load_scene_map()
            self.sample_indices, self.first_in_scene_lookup, self.last_in_scene_lookup, self.sequence_level_indices = self._get_sample_indices()
            assert len(self.sequence_level_indices) == self.sequence_num

            if self.sequence_num % self.gpus == 0:
                self.num_sequences_to_yield = self.sequence_num
            else:
                self.num_sequences_to_yield = self.sequence_num + self.gpus - ( self.sequence_num % self.gpus )

            if self.sampling_mode == 'train' and self.extra_sequence_multipliter > 0:
                print("Adding {} extra sequences to each epoch".format(self.gpus*extra_sequence_multipliter))
                self.num_sequences_to_yield += self.gpus * extra_sequence_multipliter

            self.get_sequence_samplers(sample_size=sample_size)
            self.sample_sequences()

        


        # Maintain MMDetection3D API
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

        print("Infos path: {}".format(self.dataset.ann_file))
        print("Created SequentialDataset in sampling mode {} with {} samples from {} sequences.".format(self.sampling_mode,len(self),self.sequence_num))

        print("####################################################################")
        print("len(self)",len(self))
        print("self.sequence_num",self.sequence_num)
        print("self.num_sequences_to_yield",self.num_sequences_to_yield)
        print("self.sample_size",self.sample_size)
        print(( self.num_sequences_to_yield * self.sample_size ))
        print(self.sampling_mode, self.sampling_mode)
        print("####################################################################")


        if self.sampling_mode == 'train':
            assert len(self) == ( self.num_sequences_to_yield * self.sample_size )
        else:
            assert len(self) == ( self.num_sequences_to_yield * self.max_seq_len )



    def log_msg(self):
        return "[SequentialDataset| rank:{}]".format(self.rank)


    def get_extra_infos_info(self,ori_idx,sample):
        sample['lidar2ego_translation'] = DC(to_tensor(self.dataset.data_infos[ori_idx]['lidar2ego_translation']))
        sample['lidar2ego_rotation'] = DC(to_tensor(self.dataset.data_infos[ori_idx]['lidar2ego_rotation']))
        sample['ego2global_rotation'] = DC(to_tensor(self.dataset.data_infos[ori_idx]['ego2global_rotation']))
        sample['ego2global_translation'] = DC(to_tensor(self.dataset.data_infos[ori_idx]['ego2global_translation']))
        # print(sample)
        # exit(0)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """

        if self.verbose:
            print("{} in __getitem__()".format(self.log_msg()))

        if self.visualize:
            ori_idx = self.sample_indices[idx]
            sample = self.dataset[ori_idx]
            sample['instance_idx'] = self.track_viz_idx[idx]
            sample['tracks'] = self.dataset.data_infos[ori_idx]['tracks']

        elif self.sampling_mode == 'train':
            # print('index used',idx)
            # print(len(self.indices_to_yield))
            ori_idx = self.indices_to_yield[idx]
            sample = self.dataset[ori_idx]
            self.get_extra_infos_info(ori_idx,sample)

            sample['first_in_scene'] = False
            sample['last_in_scene'] = False

            compare_temp = int(np.floor(idx/self.gpus) % self.sample_size)
            if compare_temp == 0:
                sample['first_in_scene'] = True
                sample['last_in_scene'] = False

            elif compare_temp == self.sample_size-1:
                sample['first_in_scene'] = False
                sample['last_in_scene'] = True

            else:
                sample['first_in_scene'] = False
                sample['last_in_scene'] = False
            
            # for k,v in sample.items():
            #     print(k,v)
            # exit(0)
            if self.verbose:
                print("{} exiting __getitem__()".format(self.log_msg()))

            # print(sample)
            # exit(0)
            return sample

        elif self.sampling_mode == 'val':
            ori_idx = self.indices_to_yield[idx]
            sample = self.dataset[ori_idx]
            self.get_extra_infos_info(ori_idx,sample)

            sample['first_in_scene'] = False
            sample['last_in_scene'] = False

            if ori_idx == -1:
                sample['padded'] = True
            else:
                sample['padded'] = False

            compare_temp = int(np.floor(idx/self.gpus) % self.max_seq_len)
            if compare_temp == 0:
                sample['first_in_scene'] = True
                sample['last_in_scene'] = False

            elif compare_temp == self.max_seq_len-1:
                sample['first_in_scene'] = False
                sample['last_in_scene'] = True

            else:
                sample['first_in_scene'] = False
                sample['last_in_scene'] = False


            
            if self.verbose:
                print("{} exiting __getitem__()".format(self.log_msg()))

            return sample


        else:
            ori_idx = self.sample_indices_shuffle[idx]
            sample = self.dataset[ori_idx]
            self.get_extra_infos_info(ori_idx,sample)
            try:
                # sample['tracks'] = self.dataset.data_infos[ori_idx]['tracks']
                sample['first_in_scene'] = self.first_in_scene_lookup[idx]
                sample['last_in_scene'] = self.last_in_scene_lookup[idx]
            except KeyError:
                pass
        
        if self.verbose:
            print("{} exiting __getitem__()".format(self.log_msg()))

        return sample

        
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        # print(self.num_sequences_to_yield * self.sample_size)
        return len(self.indices_to_yield)


    def sample_sequences(self):
        random.shuffle(self.sequence_samplers)
        
        # if self.gpus == 1:
        #     self.indices_to_yield = []
        #     for x in self.sequence_samplers:
        #         self.indices_to_yield += x()
                
        if self.sampling_mode == "val": #DISTRIBUTED TRAINING REQUIRES MATCHING THE SAMPLER FOR SEQUENTIAL DATA
            sequences = []
            for x in range(self.num_sequences_to_yield):
                if x >= self.sequence_num:
                    sequences.append([-1 for x in range(self.max_seq_len)])
                else:
                    sequences.append(self.sequence_samplers[x]())
                                
            self.indices_to_yield = []
            # print(np.ceil(self.sequence_num/self.gpus),self.sequence_num,self.gpus)
            for offset in range(int(self.num_sequences_to_yield/self.gpus)):
                offset = offset * self.gpus
                for seq_idx in range(self.max_seq_len):
                    for gpu_id in range(self.gpus):
                        #attempts to mimic the sampling heuristic but may be wrong
                        self.indices_to_yield.append(
                            sequences[offset+gpu_id][seq_idx]
                            )

        elif self.sampling_mode == "train": #DISTRIBUTED TRAINING REQUIRES MATCHING THE SAMPLER FOR SEQUENTIAL DATA
            sequences = []
            for x in range(self.num_sequences_to_yield):
                if x >= self.sequence_num:
                    seq = np.random.randint(0,self.sequence_num)
                    sequences.append(self.sequence_samplers[seq]())
                else:
                    sequences.append(self.sequence_samplers[x]())

            self.indices_to_yield = []
            # print(np.ceil(self.sequence_num/self.gpus),self.sequence_num,self.gpus)
            for offset in range(int(self.num_sequences_to_yield/self.gpus)):
                offset = offset * self.gpus
                for seq_idx in range(self.sample_size):
                    for gpu_id in range(self.gpus):
                        #attempts to mimic the sampling heuristic but may be wrong
                        self.indices_to_yield.append(
                            sequences[offset+gpu_id][seq_idx]
                        )

        self._set_group_flag()


    def shuffle_sequences(self):
        print("[dataset_wrappers] inside shuffle sequences")
        self.sample_sequences()
        return self.indices_to_yield


    def get_sequence_samplers(self,sample_size=16):
        self.sequence_samplers = []
        self.max_seq_len = np.max([len(v) for v in self.sceneMap.values()])
        assert_count = 0 
        for i,ii in self.sequence_level_indices:
            tempSampler = SequenceSampler(infos_idx_list=self.sample_indices[i:ii], sample_size=sample_size,
                                          max_seq_len=self.max_seq_len,sampling_mode=self.sampling_mode)
            # print(len(tempSampler()))
            assert_count += len(tempSampler())
            self.sequence_samplers.append(
                tempSampler
            )
        
        # print( self.num_sequences_to_yield, self.sample_size)
        if self.sampling_mode == 'train':
            # print("train:",assert_count ,"==", self.sequence_num * self.sample_size)
            assert assert_count == ( self.sequence_num * self.sample_size )
        else:
            # print("test:",assert_count ,"==", self.sequence_num * self.max_seq_len)
            assert assert_count == ( self.sequence_num * self.max_seq_len )



    def _get_sample_indices(self):
        """Rearrange the sample indices so that they are read in sequential order

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of ids from the dataset.
        """
        self.sequence_num = 0
        sample_indices = []
        first_in_scene_lookup = []
        last_in_scene_lookup = []
        sequence_level_indices = []
        for k,v in self.sceneMap.items():# iterate over scenes
            for i,tok in enumerate(v):# iterate over samples in a scene
                try:
                    sample_indices.append(self.token_to_infos_id[tok])
                    if i==0:
                        self.sequence_num += 1
                        first_in_scene_lookup.append(True)
                        last_in_scene_lookup.append(False)

                        if sequence_level_indices != []:
                            sequence_level_indices[-1] = (sequence_level_indices[-1],len(sample_indices)-1)
                            
                        sequence_level_indices.append(len(sample_indices)-1)
                    else:
                        first_in_scene_lookup.append(False)
                        last_in_scene_lookup.append(False)
                        
                    # print(token_to_infos_id[tok])
                except KeyError:
                    pass

            if last_in_scene_lookup != []:
                last_in_scene_lookup[-1] = True

        sequence_level_indices[-1] = (sequence_level_indices[-1],len(sample_indices))
        return sample_indices, first_in_scene_lookup, last_in_scene_lookup,  sequence_level_indices
        

    def _get_sample_indices_viz(self):
        """Rearrange the sample indices so that when they are 
        read in sequential order, they go over different occlusion examples
        
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of ids from the dataset.
        """
        sample_indices = []
        first_in_scene_lookup = []
        track_viz_idx = []
        for x in self.occlusionMap[self.viz_class]:
            if len(x['samples']) < 9:
                try:
                    tempList = [self.token_to_infos_id[tok] for tok in x['samples']]
                    tempTrackIdxList = [x['instance_idx'] for tok in x['samples']]
                    sample_indices.extend(tempList)
                    track_viz_idx.extend(tempTrackIdxList)

                except KeyError:
                    # print("Error",x['samples'])
                    pass

        return sample_indices, first_in_scene_lookup, track_viz_idx
        

    def evaluate(self,*args,**kwargs):
        """dataset evalution wrapper"""
        print('improper evaluation ')
        return self.dataset.evaluate(*args,**kwargs)



    def evaluate_tracking(self,results,train=False,neptune=None):
        """evaluates results leveraging the tracking api from NuscenesDatasetTracking"""
        self.token_to_infos_id = {x['token']:i for i,x in enumerate(self.dataset.data_infos)}
        for sample in results:
            sample['pts_bbox']['sample_idx'] = self.token_to_infos_id[sample['pts_bbox']['token']]
        # print(results)
        return self.dataset.evaluate_tracking(self.sample_indices,results,train,neptune=neptune)


    def evaluate_detection(self,results,train=False,neptune=None):
        """evaluates results leveraging the tracking api from NuscenesDatasetTracking"""
        for sample in results:
            sample['pts_bbox']['sample_idx'] = self.token_to_infos_id[sample['pts_bbox']['token']]
        return self.dataset.evaluate_detection(self.sample_indices,results,train)


    def load_occlusion_map(self):
        """Retrieves a map of occlusions sequences per class which 
        allows the user to sample sequences where an occlusion 
        may have taken place to visualize it."""
        filepath = osp.join('/btherien','occlusions.json')#self.dataset.data_root,"occlusions.json")
        try:
            occlusionMap = json.load(open(filepath,'r'))
        except Exception:
            from nuscenes import NuScenes
            nusc = NuScenes(version=self.dataset.version, dataroot=self.dataset.dataset_root, verbose=True)
            from tools.data_converter.nuscenes_tracking_annotator import TrackLabeler
            tl = TrackLabeler(nusc,slow=True,verbose=False,padding=1,
                            visibility_thresh=0, lidar_pts_thresh=16)
            occlusionMap = tl.clsToOcclusion
            json.dump(occlusionMap, open(filepath,'w'))

        return occlusionMap


    def load_scene_map(self):
        """Loads a cached version of the scene map or creates it outright"""
        return load_nusc_scene_map(version=self.dataset.version,data_root=self.dataset.dataset_root)


    def _set_group_flag(self): #taken from custom_3d.py
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        






