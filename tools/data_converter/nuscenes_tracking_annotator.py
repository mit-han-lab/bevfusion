""" Creates tracking IDS for each scene in the dataset

Author: Benjamin therien
"""

import pprint
import numpy as np


class Instance(object):
    """controls the information for one annotation instance in a scene
    
    Args:
        cls (string): the class of the instance
        appearsIndex (np.array): 1 if instance appears 0 otherwise
        
    """
        
    def __init__(self, cls, appearsIndex,id_,token,lidar_pts,visibility):
        super(SceneAnnotator).__init__()
        self.id = id_
        self.token = token
        self.cls = cls
        self.appearsIndex = appearsIndex
        self.lidar_pts = lidar_pts
        self.visibility = visibility
        self.appearances = None
        self.occlusions = None
        
    def getAppearances(self):
        """retrieves the appearace sample, dissapearance 
        sample, and any occlusion smaples"""
        if self.appearances != None:
            return self.appearances
        
        self.appearances = []
        tup = None
        for i,x in enumerate(self.appearsIndex):
            #tup==None means outside of 1 sequences
            if tup == None and x == 0: #
                continue
            elif tup == None and x == 1: #begining of 1 sequence
                tup=(i,)
            elif len(tup) == 1 and x == 0: #end of 1 sequence
                self.appearances.append((tup[0],i-1,))
                tup = None
            elif len(tup) == 1 and x == 1: #continuation of 1 sequence
                continue
            else:
                raise NotImplementedError("Something went wrong in class Instance getTimeStamps() ")
                
        if tup != None:
            self.appearances.append((tup[0],i,))
            
        assert len(self.appearances) > 0
        
        return self.appearances
                
    def getOcclusions(self):
        """returns the intervals where the instance was occluded"""
        if self.occlusions != None:
            return self.occlusions
        
        self.getAppearances()
        
        if len(self.appearances) < 2:
            self.occlusions = []
        else:
            self.occlusions = [(self.appearances[i][1],self.appearances[i+1][0],)
                   for i in range(len(self.appearances)-1)]
        
        return self.occlusions

    def show(self):
        self.getOcclusions()
        print("Class:",self.cls)
        print("Occlusions:")
        pprint.pprint(self.occlusions)
        print("Appearances:")
        pprint.pprint(self.appearances)

class SceneAnnotator(object):
    """controls the tracking annotations for one scene
    
        Attributes:
            slow:
            sampleList: Ordered list of samples in the scene.
            annotationList: Ordered list of annotation tokens for each sample.
            instanceMap: Mapping from instance token to UID.
            instanceList: Ordered list of track IDs in each sample.
            instanceMapID: Mapping from trak ID to instance token.
            instances: List of instance objects for each instance in the current object.
            clsToInstance: Mapping from classes to lists of corresponding instances.
            clsToOcclusion: Mapping from classes to occlusions.
    """
    
    def _str(self):
        return "Class SceneAnnotator : "
    
    def _print(self,string):
        if self.verbose:
            print("{}{}".format(self._str(),string))
        
    
    def __init__(self, nusc, sceneTok, slow=False, verbose=False, padding=0,
                 visibility_thresh=0, lidar_pts_thresh=0):
        super(SceneAnnotator).__init__()
        self.visibility = {1:(0,40),2:(40,60),3:(60,80),4:(80,100)}
        self.padding = padding
        self.slow = slow
        self.verbose = verbose
        self.nusc = nusc
        self.token = sceneTok
        self.sampleList = self.getSampleList()
        self.annotationList = self.getAnnotationList()
        self.instanceList = self.buildInstanceIndex()
        if self.slow:
            self.instanceMapID = {v:k for k,v in self.instanceMap.items()}
            self._print('instance grid shape:{}'.format(self.instanceGrid.shape))
            self._print('instance map len:{}'.format(len(self.instanceMap)))
            self._print('instance map:{}'.format(self.instanceMap))
            self.instances = [Instance(cls=self.instanceCls[i],
                                       appearsIndex=self.instanceGrid[:,i],
                                       id_=i,
                                       token=self.instanceMapID[i],
                                       lidar_pts=self.lidarPtsGrid[:,i],
                                       visibility=self.visibilityGrid[:,i])
                             for i in range(len(self.instanceMap)-1)]
            self.clsToInstance = self.getClsToInstance()
            
            self.clsToOcclusion = {
                cls:self.getOcclusions(cls,
                                       padding=self.padding,
                                       visibility_thresh=visibility_thresh, 
                                       lidar_pts_thresh=lidar_pts_thresh) 
                for cls in self.clsToInstance
            }
        
    def getOcclusions(self,cls,padding=0,visibility_thresh=0,lidar_pts_thresh=0):
        """Retrieves occluded instances of cls type in the scene.
        
        Args:
            cls
            padding
            visibility_thresh
            lidar_pts_thresh
        """
        try:
            instances = self.clsToInstance[cls]
        except KeyError:
            self._print("getOcclusions(): no class of type {} ".format(cls))
            return []
        
        occs = []
        for i in self.instances:
            if i.cls != cls:
                continue
                
            o = i.getOcclusions()
            if o == []:
                continue
            else:
                for tup in o:
                    if i.lidar_pts[tup[0]] < lidar_pts_thresh:
                        continue
                    if i.visibility[tup[0]] < visibility_thresh:
                        continue 
                        
                    start = np.max([tup[0] - padding + 1, 0])
                    end = np.min([tup[1] + padding,len(self.sampleList)])
                    occs.append({
                        'instance_idx':i.id,
                        'instance':i.token,
                        'instance_class':i.cls,
                        'samples':[self.sampleList[ii] for ii in range(start,end)]
                    })
                    
                    if self.verbose:
                        if tup[0] == tup[1]:
                            print('[ tup[0] == tup[1] ]:\ntup:{}\nstart:{}\nend:{}\nrange:{}'.format(
                                    tup,start,end,[x for x in range(start,end)]
                            ))
                        elif tup[0] > tup[1]:
                            print('[ error ], tup[0] > tup[1]: this should never happen')
                        elif tup[0] < tup[1]:
                            pass
    #                             print('[ tup[0] < tup[1] ] \ntup:{}\nstart:{}\nend:{}\nrange:{}'.format(tup,start,end,[x for x in range(start,end)]))
                        else:
                            print("ERROR")
        return occs
    
    def getAnnotations(self):
        """Creates a dictionary mapping sample Tokens to instance Lists."""
        return {self.sampleList[i]:x for i,x in enumerate(self.instanceList)}
    
    def getAnnotationsTTE(self):
        """Creates a dictionary mapping sample Tokens to instance Lists."""
        return {self.sampleList[i]:x for i,x in enumerate(self.TTESup)}
        
        
    def getSampleList(self):
        """Retrieves an ordered list of sample tokens for 
        each sample in the scene."""
        x = self.nusc.get('scene',self.token)
        sampleList = [x['first_sample_token']]
        nextToken = x['first_sample_token']
        while True:
            my_sample = self.nusc.get('sample', nextToken)
            if my_sample['next'] == '':
                break
            else:
                nextToken = my_sample['next']
                sampleList.append(nextToken)
        return sampleList

    
    def getAnnotationList(self):
        """Creats a list of lists of annotations 
        for each sample in the scene
        
        returns:
            annotationList (List[List[annotation Token]]): list of 
                annotations tokens for each sample
        """
        annotationList = []
        for sampleTok in self.sampleList:
            x = self.nusc.get('sample',sampleTok)
            annotationList.append(x['anns'])
        return annotationList
            
    
    def buildInstanceIndex(self):
        """Creates an index of each instance (track) in the 
        current scene and returns list of lists corresponding 
        to the instance ids of each annotation present in the 
        samples which make up the current scene."""
        self.instanceMap = {}
        self.instanceCls = {}
        
        visibilityListList = []
        instanceLidarListList = []
        instanceListList = []
        for annList in self.annotationList:
            visibilityList = []
            instanceLidarList = []
            instanceList = []
            for ann in annList:
                sampleAnn = self.nusc.get('sample_annotation',ann)
                instanceTok = sampleAnn['instance_token']
                lidar_pts = sampleAnn['num_lidar_pts']
                visibility = sampleAnn['visibility_token']
                
                try:
                    instanceList.append(self.instanceMap[instanceTok])
                    visibilityList.append(visibility)
                    instanceLidarList.append(lidar_pts)
                    
                except KeyError:
                    self.instanceMap[instanceTok] = len(self.instanceMap)
                    instanceList.append(self.instanceMap[instanceTok])
                    visibilityList.append(visibility)
                    instanceLidarList.append(lidar_pts)
                    self.instanceCls[self.instanceMap[instanceTok]] = sampleAnn['category_name']
                    
            instanceListList.append(instanceList)
            visibilityListList.append(visibilityList)
            instanceLidarListList.append(instanceLidarList)
            
        if self.slow:
            
            sampleNum = len(instanceListList)
            instanceNum = len(self.instanceMap)

            temp = np.array([[i,x] for i,y in enumerate(instanceListList) for x in y])
            templidar = np.array([x for i,y in enumerate(instanceLidarListList) for x in y])
            tempvis = np.array([x for i,y in enumerate(visibilityListList) for x in y])

            self.instanceGrid = np.zeros((sampleNum,instanceNum))
            self.lidarPtsGrid = np.zeros((sampleNum,instanceNum))
            self.visibilityGrid = np.zeros((sampleNum,instanceNum))
            
            self.instanceGrid[temp[:,0],temp[:,1]] = 1
            self.lidarPtsGrid[temp[:,0],temp[:,1]] = templidar
            self.visibilityGrid[temp[:,0],temp[:,1]] = tempvis
            
            b = np.flip(self.instanceGrid,axis=0)
            am = np.argmax(b,axis=0)
            am = self.instanceGrid.shape[0] - am -1
            self.last_occurence_index = am
            self.first_occurence_index = np.argmax(self.instanceGrid,axis=0)
            self.createTTESup(instanceListList)
            
        return instanceListList
    
    def createTTESup(self,instanceListList):
        """Creates and ordered list of timesteps 
        remaining for a given track for each scene"""
        outer = []
        for i,x in enumerate(instanceListList):
            inner = []
            for ii,trkid in enumerate(x):
                first = self.first_occurence_index[trkid]
                last = self.last_occurence_index[trkid]
                assert first <= i and i <= last
                inner.append((last - first) - (i - first))
                
            outer.append(inner)
                
        self.TTESup = outer
                
    def getClsToInstance(self):
        """creates a dictionary of class strings
        mapped to instance IDS"""
        clsToInstance = {}
        for _id,cls in self.instanceCls.items():
            try:
                clsToInstance[cls].append(_id)
            except KeyError: 
                clsToInstance[cls] = [_id]
                
        return clsToInstance
        
    
    def show(self):
        print("Instance List : ")
        pprint.pprint(self.instanceList)
        print("Instance Class List : ")
        pprint.pprint(self.instanceCls)
        if self.slow:
            print("Instance Grid : ")
            pprint.pprint(self.instanceGrid)
            for instance in self.instances:
                print()
                instance.show()
        else:
            print('speed: Fast')



class TrackLabeler(object):
    """controls the tracking annotations for many scenes"""
    
    def _str(self):
        return "Class TrackLabeler : "
    
    def _print(self,string):
        if self.verbose:
            print("{}{}".format(self._str,string))
    
    def __init__(self, nusc, slow=False, verbose=False, padding=0,
                visibility_thresh=0, lidar_pts_thresh=0):
        super(TrackLabeler).__init__()
        self.scenes = [s['token'] for s in nusc.scene]
        self.nusc = nusc
        self.trackMap = {}
        if slow:
            self.clsToOcclusion = {}
            self.saList = []
            for scene in self.scenes:
                saTemp = SceneAnnotator(nusc=self.nusc,sceneTok=scene,
                                        slow=slow,verbose=verbose,
                                        padding=padding,
                                        visibility_thresh=visibility_thresh, 
                                        lidar_pts_thresh=lidar_pts_thresh)
                self.saList.append(saTemp)
                self.trackMap.update(saTemp.getAnnotations())
                for k,v in saTemp.clsToOcclusion.items():
                    
                    try:
                        self.clsToOcclusion[k] += v
                    except KeyError:
                        self.clsToOcclusion[k] = v
                
        else:
            for scene in self.scenes:
                saTemp = SceneAnnotator(nusc=self.nusc,sceneTok=scene,slow=slow,
                                        verbose=verbose,padding=padding,
                                        visibility_thresh=visibility_thresh, 
                                        lidar_pts_thresh=lidar_pts_thresh)
                self.trackMap.update(saTemp.getAnnotations())
            
    def getTracks(self, sampleTok):
        return self.trackMap[sampleTok]
            
            
        
class TrackLabeler(object):
    """Controls the tracking annotations for many scenes."""
    
    def _str(self):
        return "Class TrackLabeler : "
    
    def _print(self,string):
        if self.verbose:
            print("{}{}".format(self._str,string))
    
    def __init__(self, 
                 nusc, 
                 slow=False, 
                 verbose=False, 
                 padding=0,
                 visibility_thresh=0, 
                 lidar_pts_thresh=0):
        """
        
        """
        super(TrackLabeler).__init__()
        self.scenes = [s['token'] for s in nusc.scene]
        self.nusc = nusc
        self.trackMap = {}
        self.TTEMap = {}
        if slow:
            self.clsToOcclusion = {}
            self.saList = []
            for scene in self.scenes:
                saTemp = SceneAnnotator(nusc=self.nusc,
                                        sceneTok=scene,
                                        slow=slow,
                                        verbose=verbose,
                                        padding=padding,
                                        visibility_thresh=visibility_thresh, 
                                        lidar_pts_thresh=lidar_pts_thresh)
                self.saList.append(saTemp)
                self.trackMap.update(saTemp.getAnnotations())
                self.TTEMap.update(saTemp.getAnnotationsTTE())
                for k,v in saTemp.clsToOcclusion.items():
                    try:
                        self.clsToOcclusion[k] += v
                    except KeyError:
                        self.clsToOcclusion[k] = v
                
        else:
            for scene in self.scenes:
                saTemp = SceneAnnotator(nusc=self.nusc,
                                        sceneTok=scene,
                                        slow=slow,
                                        verbose=verbose,
                                        padding=padding,
                                        visibility_thresh=visibility_thresh, 
                                        lidar_pts_thresh=lidar_pts_thresh)
                self.trackMap.update(saTemp.getAnnotations())
                self.TTEMap.update(saTemp.getAnnotationsTTE())
                
            
    def getTracks(self, sampleTok):
        """Returns ordered tracking annotations for a sample"""
        return self.trackMap[sampleTok]
    
    def getTTE(self, sampleTok):
        """Returns ordered tracking annotations for a sample"""
        return self.TTEMap[sampleTok]
            
            
        