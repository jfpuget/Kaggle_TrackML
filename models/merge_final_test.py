
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from trackml.dataset import load_event
from multiprocessing import Pool

base_path = '/home/jfpuget/Kaggle/TrackML/'

def merge_track(data, center, threshold_base, threshold_center, threshold):
    # merges tracks from two models for a given event
    # data['track_id'] contains the first set of tracks
    # center['track_id'] contains the second set of tracks
    # thresholds are used to fitler whoch pairs of tracks should be merged
    
    # save original data columns for output
    data_cols = data.columns
    
    # reindex center to align with data[data.volume_id < 10] index
    data_center = data[data.volume_id < 10]
    data_center = data_center[['hit_id']].merge(center, how='left', on='hit_id')
    data['center'] = 0
    data.loc[data.volume_id < 10, 'center'] = data_center.track_id.values
    
    # computes track overlap
    data['count_both'] = data.groupby(['track_id', 'center']).hit_id.transform('count')    
    data['count_center'] = data.groupby(['center']).hit_id.transform('count')
    data['count_track'] = data.groupby(['track_id']).hit_id.transform('count')
    
    # compute pairs of tracks that overlap more than input thresholds
    mapping = data.groupby(['track_id', 'center']).count_both.max().to_frame('count_max').reset_index()
    data = data.merge(mapping, how='left', on=['track_id', 'center'])
    data['valid'] = ((data.count_max == data.count_both) & \
                     (data.count_both > threshold) &  \
                     (data.count_both > threshold_center*data.count_center) &   \
                     (data.count_both > threshold_base*data.count_track))
    mapping = data[data.valid].groupby(['track_id', 'center'])[['track_id', 'center']].first()
    
    # merge tracks
    data['new_center'] = 0
    for t,c in mapping.itertuples(index=False, name=None):
        data.loc[data.center == c, 'new_center'] = t        
    data.loc[(data.track_id > 0) & (data.new_center == 0), 'new_center'] = data.track_id[(data.track_id > 0) & (data.new_center == 0)].values
    # use remaining tracks for remaining hits
    track_max = data.new_center.max()
    data.loc[data.center > 0, 'center'] += track_max
    data.loc[(data.center > 0) & (data.new_center == 0), 'new_center'] = data.center[(data.center > 0) & (data.new_center == 0)].values
    data.track_id = data.new_center
    return data[data_cols].copy()


def get_event(i):
    return 'event000000%03d' % i

def work_sub(param):
    # merges tracks for event i and saves result into a file
    (i, ) = param
    th_b = 0.16
    th_c = 0.45
    
    event = get_event(i)
    print('event:', event)
    hits, cells = load_event('../input/test/' + event, parts=['hits', 'cells'])
    data = pd.read_csv('../submissions/final/'+event)
    data = data.merge(hits, how='left', on='hit_id')
    inner = pd.read_csv('../submissions/final_inner/'+event)
    data = merge_track(data, inner, th_b, th_c, 1)
    data['event_id'] = i
    data[['event_id', 'hit_id', 'track_id']].to_csv('../submissions/merge_final/' + event +'.csv', 
                                                    index=False)
    return i


def main():
    # merge each event tracks in parallel.  
    # number of process in pool should be close to the number of processors
    params = [(i, ) for i in range(125)]

    if 1: 
        pool = Pool(processes=20, maxtasksperchild=1)
        ls   = pool.map( work_sub, params, chunksize=1 )
        pool.close()
    else:
        ls = [work_sub(param) for param in params]

    # computes submission by concatenating each event tracks
    submissions = []
    for i in tqdm(range(125)):
        event = get_event(i)
        data0 = pd.read_csv('../submissions/merge_final/' + event + '.csv')
        submissions.append(data0)
    
    submission = pd.concat(submissions, axis=0)
    submission.track_id = (submission.track_id).astype('int64')
    submission.to_csv('../submissions/merge_final.csv', index=False)

if __name__ == "__main__":
    main()


