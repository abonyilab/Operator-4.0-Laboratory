# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:06:17 2023

@author: Andras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from datetime import datetime as dt


#%% Data 
df = pd.read_csv('Op40_labor_tag_position_2021_09_21.csv', index_col=0)
df['timestamp'] = pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
df =df[df['timestamp'] > dt.fromisoformat('2021-09-21 13:26:30')]
df =df[df['timestamp'] < dt.fromisoformat('2021-09-21 15:08:30')]
df.rename(columns = {'tag_id': 'tagid', 'x_pos': 'posx', 'y_pos': 'posy', 'timestamp': 'timepoint'}, inplace = True)



#%% Visualization of position data
plt.figure(figsize=(15, 15) )
img = plt.imread("layout.png")
plt.imshow(img, extent=[-9,-2, -4, 7.5])
plt.scatter(df['posx'],df['posy'], c=df['tagid'], s=0.5)
plt.xlabel('y', rotation = -270)
plt.ylabel('x', rotation = 270)
plt.show()

#%% Get only coordinates
X = df[['posx','posy']]

#%% Clustering function
def Clustering(data,n_opt):
    gmm = GaussianMixture(n_components=n_opt, init_params='kmeans',)#random_state = 999,)
    gmm.fit(data)
    global labels, scores, maxlh
    labels = gmm.predict(data)  # cluster labels of data points
    scores = gmm.score_samples(data) #probabilities of data points
    maxlh = gmm.score(data) # overall likelihood of the model
    return labels, scores,  maxlh

# %% Model fit 10x -- > max likelihood selection

# In order to determine the probability threshold value, we fit have to fit the model first 
n_opt = 5
tmp = [[],[],[],]
# We fit the model 10 times, storing the results. Finally, the best model is selected
for i in range(10):
    Clustering(X,n_opt)
    tmp[0].append(labels)
    tmp[1].append(scores)
    tmp[2].append(maxlh)

    print('Model evaluation: ', i, '/9')
df['cluster'] = tmp[0][tmp[2].index(max(tmp[2]))]
df['score'] = tmp[1][tmp[2].index(max(tmp[2]))]

fig = plt.figure(figsize=(10, 10) )
plt.scatter(df['posx'],df['posy'], c=df['cluster'], s=0.1)
plt.xlim(-10,0)
plt.ylim(-5,5)
plt.show()


#%% Probability threshold searching based empirical cummulative distribution of sample likelihood
def ThresholdFinder(scorevalues):
     fig, ax = plt.subplots()
     num_bins =  len(X)
     counts, bin_edges = np.histogram(scores, bins=num_bins,) # normed=True)
     cdf = np.cumsum(counts)
     ax.set_ylabel('ECDF')
     ax.set_xlabel('p(x_n)')
     ax.plot(bin_edges[1:], cdf)
     plt.xlim(-10,0)
     plt.show()

ThresholdFinder(X) #threshold selection at knee point

#%% Outlier detection: probabilities of data points compared to threshold value
threshold = -2
def OutlierMarker(data):
    data['cluster'] = labels
    outlier = []
    for i in range(len(data)):
        if scores[i] <= threshold:
            outlier.append(1)
        else:
            outlier.append(0)
    data['outlier'] = outlier 
    

#%% iterative clustering and filtering
n_opt = 5 
X1 = X.copy()
change_list = []
lengths = []
score_means = []


change = 1
#the iteration goes until the extent of change is lower than 1%
# In each step the algorithm fits gmm on the data 2x, and then select the ine with higher overall likelihood
# After model fitting, all point with lower likelihood than threshold are removed
while change > 0.01:
    X1 = X1[['posx', 'posy']]
    prev_datalength = len(X1)
    
    # fit the models and select the best
    tmp = [[],[],[], []]   
    for i in range(2):   
        Clustering(X1,n_opt)
        tmp[0].append(labels)
        tmp[1].append(scores)
        tmp[2].append(maxlh)
    labels = tmp[0][tmp[2].index(max(tmp[2]))]
    scores = tmp[1][tmp[2].index(max(tmp[2]))]   
    
    #outlier detection
    OutlierMarker(X1)
    X1 = X1.drop(X1[ X1['outlier'] == 1 ].index)

    #check the extent of change
    change = 1 -(len(X1)/prev_datalength)
    change_list.append(change)
    lengths.append(len(X1))
    print('change: ', change*100, ' %')
    print('new lentgth of data', len(X1))


#%% Result
plt.figure(figsize=(15, 15) )
img = plt.imread("layout.png")
plt.imshow(img, extent=[-9,-2, -4, 8])
plt.scatter(X1['posx'],X1['posy'], c=X1['cluster'], s=0.5)
plt.xlabel('y', rotation = -270)
plt.ylabel('x', rotation = 270)

