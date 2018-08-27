
# coding: utf-8

import pandas as pd
import numpy as np
import pickle as pkl
from collections import Counter

from trackml.dataset import load_event

def main():
    # load 90 train events data
    data_l = []
    for i in range(10,100):
        event = '../input/train_1/event0000010%d' % i
        print('event:', event)
        hits, cells, particles, truth = load_event(event)
        data = hits
        data = data.merge(truth, how='left', on='hit_id')
        data = data.merge(particles, how='left', on='particle_id')
        
        # keep hits from tracks orginating from vertex
        data['rv'] = np.sqrt(data.vx**2 + data.vy**2)
        data = data[(data.rv <= 1) & (data.vz <= 50) & (data.vz >=-50)].copy()
        data = data[data.weight > 0]
        data['event_id'] = i

        data['pt'] = np.sqrt(data.px ** 2 + data.py ** 2)
        
        # use a simple relationship to compute alpha0 from pt, see documentaiton or EDA notebook.
        data['alpha0'] = np.exp(-8.115 - np.log(data.pt))

        data_l.append(data)

    data = pd.concat(data_l, axis=0)

    # compute track level statistics
    df = data.groupby(['event_id', 'particle_id'])[['alpha0', 'vz']].first()
    df = df.dropna()
    np.save('../data/scan_center.npy', df.values)

    # compute tracklet frequencies 
    # tracklets are sub tracks of length 4
    
    # assign a unique layer to each hit
    data['layer'] = 100 * data.volume_id + data.layer_id
    
    # for each track compute a string containing the sequence of layers traversed by the track
    data = data.sort_values(by=['particle_id', 'z']).reset_index(drop=True)
    df = data.groupby(['event_id', 'particle_id']).layer.apply(lambda s: ' '.join([str(i) for i in s]))
    df = df.to_frame('layers')

    # count each tracklet occurences 
    cnt = Counter()
    for x in tqdm(df.itertuples(name=None, index=False)):
        layers = x[0].split()
        for i in range(len(layers) - 3):
            s = ' '.join(layers[i:i+4])
            cnt[s] += 1

    #save result
    with open('../data/layers_4_center_fix.pkl', 'wb') as file:
        pkl.dump(cnt, file)
        
if __name__ == "__main__":
    main()

