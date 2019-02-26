from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np

import QualityTester as QT

def ZscoreSilhouetteQuality(OptDistMat,labels,Tz,NR):

    #Compute Silhouette for each sample (no mean)
    SilMat = silhouette_samples(OptDistMat,labels,metric='precomputed')

    #Compute Silhouette for each sample (no mean) on random labels
    SilMatRand = np.zeros((OptDistMat.shape[0],NR))
    for nr in range(NR):
        labelsRand = labels[np.random.permutation(len(labels))]
        SilMatRand[:,nr] = silhouette_samples(OptDistMat,labelsRand,metric='precomputed')

    #Compute Z-score of Silhouette score compared to random cluster attribution
    m = SilMatRand.mean(axis=1)
    s = SilMatRand.std(axis=1)

    SilZ = (SilMat - m)/s

    #Remove NaN
    SilZ[~np.isfinite(SilZ)] = 0

    #Keep only the samples with a Z-score above Tz (2.32 == pvalue=1e-3 for normal distrib.)
    IdxGoodSamples = np.where(SilZ>Tz)[0]

    return IdxGoodSamples,SilZ

def ZscoreConnectivityQuality(OptDistMat,labels,Tz,NR):

    #Compute Silhouette for each sample (no mean)
    CMat = Connectivity_samples(OptDistMat,labels)

    #Compute Silhouette for each sample (no mean) on random labels
    CMatRand = np.zeros((OptDistMat.shape[0],NR))
    for nr in range(NR):
        labelsRand = labels[np.random.permutation(len(labels))]
        CMatRand[:,nr] = Connectivity_samples(OptDistMat,labelsRand)

    #Compute Z-score of Silhouette score compared to random cluster attribution
    m = CMatRand.mean(axis=1)
    s = CMatRand.std(axis=1)

    CZ = (CMat - m)/s

    #Remove NaN
    CZ[~np.isfinite(CZ)] = 0

    #Keep only the samples with a Z-score above Tz (2.32 == pvalue=1e-3 for normal distrib.)
    IdxGoodSamples = np.where(CZ>Tz)[0]

    return IdxGoodSamples,CZ


def Connectivity_samples(DistMat,labels,NumNeigh=5):
    '''
        Connectivity index as reported in: https://www.jstatsoft.org/article/view/v025i04/v25i04.pdf

        DistMat : samples x samples numpy array - Distance matrix
        labels: Vector of size Nsamples with one different label per class
        PercentNeigh : Percentage of points that will contribute to compute the connectivity score

        --> The Lower the better the clustering (optimal = 0 )
    '''
    Nsamples = DistMat.shape[0]

    NumNeigh = int(NumNeigh)
    # NumNeigh = int(PercentNeigh*Nsamples)

    connectivity_samples = np.zeros(Nsamples)
    for i in range(Nsamples):
        class_i = labels[i]
        idx_sort = np.argsort(DistMat[i,:])[1:NumNeigh+1]
        position_NN_diff_class = np.where(labels[idx_sort]!=class_i)[0]
        for k in position_NN_diff_class:
            connectivity_samples[i] += (1./float(k+1)) #1/(position of the points in different classes)

    connectivity_samples/=np.sum([1/float(i) for i in range(1,NumNeigh+1)])

    return 1-connectivity_samples


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
