# coding: utf-8

import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.cluster import DBSCAN
from itertools import combinations
from multiprocessing import Pool
#from tqdm import tqdm_notebook as tqdm

base_path = '/home/jfpuget/Kaggle/TrackML/'

class Clusterer(object):   
    
    def __init__(self, eps, max_cluster, scan_center, quality_threshold, cnt, event):
        self.eps_ = eps
        self.max_cluster_ = max_cluster
        self.scan_center_ = scan_center
        self.quality_threshold_ = quality_threshold
        self.cnt_ = cnt
        self.event_ = event
        
    def fit_alpha(self, data, alpha0, z):
        # find tracks originating from (0, 0, z) and diameter 1/alpha0
        cond0 = np.abs(data.rt * alpha0) < 1
        data1 = data[cond0].copy()
        data1['theta0'] = np.arcsin(data1.rt * alpha0) 
        data1['rt1'] = data1.theta0 / alpha0
        data1['theta'] = data1.theta_base +  data1.theta0

        data1['xcos'] = np.cos(data1.theta)
        data1['ysin'] = np.sin(data1.theta)
        data1['zr'] = (np.arcsinh((data1.z - z) / (0.7 * data1.rt1)) / 3.5)
        dfs = data1[['xcos', 'ysin', 'zr']]
        clusters0 = DBSCAN(eps=self.eps_, 
                            min_samples=2, 
                            metric='euclidean', 
                        n_jobs=1).fit(dfs).labels_   
        clusters = np.zeros(data.shape[0])
        clusters[cond0] = clusters0 + 1
        
        # merge track candidates form DBSCAN with previously found track candidates
        maxs1 = data['s1'].max()
        clusters[clusters > 0] += maxs1
        data['s2'] = clusters
        data['N2'] = count_module(data, 's2')
        data.loc[data.N2 < 2, 's2'] = 0
        data.loc[data.N2 > self.max_cluster_, 's2'] = 0
        data['Q2'] = get_all_layer_quality(data, 's2', self.cnt_)
        data['WN2'] = data.N2 * data.Q2
        data.loc[data.WN2 <= data.WN1, 's2'] = 0
        data['N2'] = count_module(data, 's2')
        data['WN2'] = data.N2 * data.Q2
        cond = (  (data['WN2'] > data['WN1']) \
                & (data['Q2'] > self.quality_threshold_) \
                & (data['N2'] < self.max_cluster_)
               ) 
        data.loc[cond, 's1'] = data.loc[cond, 's2']
        data.loc[cond, 'Q1'] = data.loc[cond, 'Q2']
        
        # update current tracks statistics
        data['N1'] = count_module(data, 's1')
        cond = ((data['N1'] >= self.max_cluster_) )
        data.loc[cond, 'N1'] = 0
        data.loc[cond, 's1'] = 0
        data.loc[cond, 'Q1'] = 0
        data['WN1'] = data.N1 * data.Q1

        data['track_id'] = data['s1']
        return data
    
    def fit_predict(self, data, n_iter):
        # initialize track statistics with no tracks
        data['theta_base'] = np.arctan2(data.y, data.x)
        data['rt'] = np.sqrt(data.x**2 + data.y**2)       
        data['theta0'] = 0.0
        data['theta'] = 0.0
        data['rt1'] = 0.0
        data['zr'] = 0.0        
        data['s1'] = data.track_id        
        data['Q1'] = get_all_layer_quality(data, 's1', self.cnt_)
        data['N1'] = count_module(data, 's1')
        data['WN1'] = data.N1 * data.Q1

        scan_center = self.scan_center_

        np.random.seed(0)                
        mm = 1
        
        # loop over randoly chose track parameters
        for ii in range(n_iter): #tqdm(range(n_iter)):
            n = np.random.randint(0, len(scan_center))
            alpha0 = scan_center[n, 0]
            n = np.random.randint(0, len(scan_center))
            z = scan_center[n, 1]
            data = self.fit_alpha(data, mm * alpha0, z)
            mm = - mm
            if ii % 1000 == 0:
                print(self.event_, '%05d' % ii)        
        return data
    
def count_module(dfh, col):
    # count the number of volumes traversed by track candidates define by values in column col
    dfmod = dfh.groupby([col, 'volume_id', 'layer_id']).hit_id.count()                
    dfmod = dfmod.to_frame('n_volume_layer').reset_index().groupby(col).n_volume_layer.count().reset_index()
    dfmod = dfh[[col]].merge(dfmod[[col, 'n_volume_layer']], how='left', on=col)
    dfmod1 = dfh.groupby([col, 'volume_id', 'layer_id', 'module_id']).hit_id.count()                
    dfmod1 = dfmod1.to_frame('n_volume_layer_module').reset_index().groupby(col).n_volume_layer_module.count().reset_index()
    dfmod = dfmod.merge(dfmod1[[col, 'n_volume_layer_module']], how='left', on=col)
    dfmod.loc[dfmod[col] == 0, 'n_volume_layer'] = 0
    dfmod.loc[dfmod['n_volume_layer_module'] <= 3, 'n_volume_layer'] = 0
    return dfmod.n_volume_layer.values

def get_all_layer_quality(dfh, col, cnt):
    # computes quality for all track candidates defined by column col
    # cnt is a dictionary containing tracklet frequencies
    
    def get_layer_quality(layers, cnt=cnt):
        # compute smoothed geometric average of tracklet frequencies for one track candidate.
        layers = [str(x) for x in layers]
        if len(layers) <= 3:
            return 0
        quality = ([cnt[' '.join(layers[i:i+4])] for i in range(len(layers) - 3)])
        quality = np.mean(np.log1p(quality))
        return quality

    dfmod = dfh.sort_values(by=[col, 'z'])
    df = dfmod[dfmod[col] > 0].groupby([col]).layer.apply(get_layer_quality).to_frame('layer_quality').reset_index()
    dfh = dfh[[col]].merge(df, how='left', on=col)
    dfh.layer_quality.fillna(0, inplace=True)
    return dfh.layer_quality.values

def get_event(i):
    return 'event000000%03d' % int(i)

def work_sub(param):
    # computes tracks for event i
    (i, num_iter, max_cluster, quality_threshold) = param
    
    event = get_event(i)
    print('event:', event)
    hits = pd.read_csv(base_path+'input/test/'+event + '-hits.csv')
    data = hits
    # only consider hits from inner volumes
    data = data[data.volume_id < 10].copy()
    print(data.shape)
    data['event_id'] = i
    data['track_id'] = 0
    data['layer'] = 100 * data.volume_id + data.layer_id
    
    # load (alpha0, z) values from train events
    scan_center = np.load('../data/scan_center.npy')

    # load tracklet frequencies
    with open('../data/layers_4_center_fix.pkl', 'rb') as file:
        cnt = pkl.load(file)
                
    # compute tracks and save them
    model = Clusterer(eps=0.0022, max_cluster=max_cluster, scan_center=scan_center, cnt=cnt, 
                      quality_threshold=quality_threshold, event=event)
    data = model.fit_predict(data, num_iter)
    data[['event_id', 'hit_id', 'track_id']].to_csv(base_path+'submissions/final_inner/' + event, index=False)
    return i

def main():
    max_cluster = 20
    quality_threshold = 5
    num_iter=25000

    # compute each event tracks in parallel.  
    # number of process in pool should be close to the number of processors
    params = [(i, num_iter, max_cluster, quality_threshold) for i in range(125)]

    if 1: 
        pool = Pool(processes=21, maxtasksperchild=1)
        ls   = pool.map( work_sub, params, chunksize=1 )
        pool.close()
    else:
        ls = [work_sub(param) for param in params]
        
    # computes submission by concatenating each event tracks
    submissions = []
    for i in (range(125)):
        event = get_event(i)
        data0 = pd.read_csv('../submissions/final_inner/' + event)
        submissions.append(data0)

    submission = pd.concat(submissions, axis=0)
    submission.track_id = (submission.track_id).astype('int64')
    submission.to_csv('../submissions/sub_final_inner.csv', index=False)
    

if __name__ == "__main__":
    main()
