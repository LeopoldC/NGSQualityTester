from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np

def ZscoreSilhouetteQuality(OptDistMat,labels,Tz,NR):

    #Compute Silhouette for each sample (no mean)
    SilMat = silhouette_samples(OptDistMat,labels,metric='precomputed')

    #Compute Silhouette for each sample (no mean) on random labels
    SilMatRand = np.zeros((OptDistMat.shape[0],NR))
    for nr in range(NR):
        labelsRand = labels[np.random.permutation(len(labels))]
        SilMatRand[:,nr] = silhouette_samples(OptDistMat,labelsRand,metric='precomputed')

    #Compute Z-score of Silhouette score compared to random cluster attribution
    SilZ = (SilMat - SilMatRand.mean(axis=1))/SilMatRand.std(axis=1)

    #Remove NaN
    SilZ[~np.isfinite(SilZ)] = 0

    #Keep only the samples with a Z-score above Tz (2.32 == pvalue=1e-3 for normal distrib.)
    IdxGoodSamples = np.where(SilZ>Tz)[0]

    return IdxGoodSamples,SilZ


#-------------------
#       TODO
#-------------------
#Outlier detection methods according to:
# https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561

def DBSCANQuality(OptDistMat,labels):
    '''
        Use DBSCAN to discover outliers
    '''
    MinDistv = np.linspace(0.25,0.75,10)
    MinSmplev = np.arange(2,10)

    for MinDist in MinDistv:
        for MinSmple in MinSmplev:

            DBSCAN(eps=MinDist,
                   min_samples=MinSmple,
                   metric='precomputed',
                   leaf_size=30,
                   n_jobs=-1)

            labels_pred = DBSCAN.fit_predict(OptDistMat)


def IsolationForestQuality(OptDistMat,labels):
    '''
        Use Isolation Forest to discover outliers
    '''
    MinDistv = np.linspace(0.25,0.75,10)
    MinSmplev = np.arange(2,10)

    for MinDist in MinDistv:
        for MinSmple in MinSmplev:

            IF = IsolationForest(n_estimators=100,
                                 max_samples='auto',
                                 contamination=0.1,
                                 max_features=1.0,
                                 bootstrap=False,
                                 n_jobs=-1)

            IF.fit(OptDistMat)
