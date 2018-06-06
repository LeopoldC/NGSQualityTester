import numpy as np
import os
from sklearn.metrics import silhouette_score,calinski_harabaz_score,silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# plt.style.use('ggplot')

from QualityMesures import *

#-----------------------------------------------------------------------
#           Main Class
#-----------------------------------------------------------------------
'''
How to use:
import QualityTester as QT

#matrix=Samples x Features
#labels=Samples : Integers corresponding to cluster assignement

qt = QT.QualityTester(Binary=False) #Binary=True  if dataset is binary
qt.compute_distances(matrix=matrix,labels=labels,GetMDS=True)
suff = '_Cont' #Suffix for the name of the images
qt.displayInternal(Suffix=suff)
qt.displayMDS(Suffix=suff)
qt.displayNamesScatter(Suffix=suff)

IdxGoodSamples = qt.compute_samples_quality(matrix=matrix,NR=100,Tz=2.32)

qt.display_samples_quality(Suffix=suff)

'''
class QualityTester:
    def __init__(self,Binary=False,displaytop=10):
        '''
            Bool: Boolean that indicate if the data are Binary (0/1)
            or Continuous. The outcome is different distance mesures
        '''

        #----------------------------
        #   Distances
        #----------------------------

        # The input distance matrix should have (Samples x Features) dimensions
        self.AllDistances = {}
        if Binary == False:
            Dist = ['cityblock','cosine','euclidean','l1','l2','manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'hamming', 'minkowski', 'seuclidean', 'sqeuclidean']
        if Binary == True:
            Dist = ['cityblock','cosine','euclidean','l1','l2','manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        for key in Dist:
            def func(x,key=key): return pairwise_distances(x,metric=key)
            self.AllDistances[key] = func

        #Other distances
        # self.AllDistances.update({'MIC':MIC})
        self.AllDistances.update({'Spearman':Spearman})

        #None function, ie no distance function
        def func(x): return x
        self.AllDistances.update({'None':func})

        #----------------------------
        #   Kernels
        #----------------------------

        self.AllKernels = {key:f for f,key in
                               [(CosineKernel,'CosineKernel'),
                                (SigmoidKernel,'SigmoidKernel'),
                                (RBFKernel,'RBFKernel')]}

        #None function, ie no distance function
        def func(x,labels,Distance): return x
        self.AllKernels.update({'None':func})

        #----------------------------
        #   Internal Indexes
        #----------------------------
        self.AllInternIndex = {key:f for f,key in [(log10CalinskiHarabaz,'log10CalinskiHarabaz'),
                                     (Silhouette,'Silhouette'),
                                     # (Dunn,'Dunn'),
                                     (Connectivity,'Connectivity')]}

        self.Keys = {'Distance':self.AllDistances.keys(),
                     'Kernel':self.AllKernels.keys(),
                     'InternIndex':self.AllInternIndex.keys()}
        self.Res={}
        for Ikey in self.Keys['InternIndex']:
            self.Res.update({Ikey:{}})
            # for Dkey in self.Keys['Distance']:
            #     for Kkey in self.Keys['Kernel']:
            #         if Dkey is not 'None' or Kkey is not 'None': #Remove the None/None
            #             self.Res[Ikey][Dkey + ' / ' + Kkey] = []

        # self.Keys.update({'Dist-Kernel':self.Res[self.Keys['InternIndex'][0]].keys()})

        #Display only the top performer
        if displaytop>0:
            self.displaytop = displaytop
        else:
            self.displaytop = len(self.Res[Ikey].keys())


    def compute_distances(self,matrix,labels,GetMDS=True):
        '''
            Compute all distances matrices and kernels on the matrix
            and compute the internal index according to the labels
        '''

        self.matrix = matrix.copy()
        self.labels = labels.copy()

        #Remove null profiles and zeros columns
        self.checkmatrix()

        self.Mat = {}
        self.labels = labels
        self.GetMDS = GetMDS
        if self.GetMDS: self.MDS = {}

        for Dkey in self.Keys['Distance']:
            print '\n' + Dkey
            if self.GetMDS: self.MDS.update({Dkey:{}})

            for Kkey in self.Keys['Kernel']:
                print '\t' + Kkey
                DKMat = self.computeDistanceMatrix(Distance=Dkey,Kernel=Kkey)
                if not np.all(np.isfinite(DKMat)):
                    continue
                if Dkey is not 'None' or Kkey is not 'None':
                    if self.GetMDS: self.MDS[Dkey].update({Kkey:self.computeMDS(DKMat)})

                    for Ikey in self.Keys['InternIndex']:
                        self.Res[Ikey].update({Dkey+' / '+Kkey: self.AllInternIndex[Ikey](DKMat,labels)})
                        print '\t\t' + Ikey + ': ' + str(self.Res[Ikey][Dkey + ' / ' + Kkey])


    def checkmatrix(self):

        #Remove null profiles
        idx = np.where(self.matrix.sum(axis=1)>0)[0]
        self.matrix =self. matrix[idx,:]
        self.labels = self.labels[idx]

        #Reorganise labels to have continuous values
        labels_tmp = self.labels.copy()
        for i,l in enumerate(np.unique(self.labels)):
            labels_tmp[self.labels==l]=i
        self.labels = labels_tmp

        #Remove zeros columns
        s = np.sum(self.matrix,axis=0)
        idx1 = np.where(s>0)[0]
        self.matrix = self.matrix[:,idx1]

    def computeDistanceMatrix(self,Distance,Kernel):
        DMat = self.AllDistances[Distance](self.matrix)
        return self.AllKernels[Kernel](DMat,self.labels,Distance)

    def computeMDS(self,DKMat):
        '''
            Compute the MDS for the fit_computeMDS function
        '''
        mds = MDS(n_components=2,dissimilarity='precomputed')
        return mds.fit_transform(DKMat)
        # tsne = TSNE(n_components=2,metric='precomputed')
        # return tsne.fit_transform(matrix)

    def compute_samples_quality(self,Distance=None,Kernel=None,NR=100,Tz=2.32,Ikey='Silhouette'):
        '''
            Compute the quality of each profiles according to Distance / Kernel
            By default, the Distance/Kernel with the best silhouette score is choosen.

            The quality is computed as the Zscore of the silouette score of each samples
            compared to random cluster attribution

            inputs:
            Tz: Threshold on the zscore
            NR: Number of random shuffling

        '''
        #Compute Optimal distance matrix
        if Distance is None and Kernel is None:

            values = np.array(self.Res[Ikey].values())
            keys = np.array(self.Res[Ikey].keys())

            idx = np.argsort(values)[::-1]
            idx = idx[~np.isnan(values[idx])]

            BestDist = keys[idx[0]].split(' / ')
            Distance = BestDist[0]
            Kernel = BestDist[1]

            print 'Using the best distance / kernel according to '+Ikey+' Index: '
            print '--> (Distance,Kernel) = {}'.format((Distance,Kernel))


        self.OptDistMat = self.computeDistanceMatrix(Distance,Kernel)

        self.IdxGoodSamples,self.SilZ = ZscoreSilhouetteQuality(self.OptDistMat,self.labels,Tz,NR)

        N = len(self.SilZ)
        Nk = len(self.IdxGoodSamples)
        Nrm = N - Nk
        print ''
        print '# Dataset Total: ' + str(N)
        print '# Dataset Kept: {0} ({1} %)'.format(Nk,round(Nk/float(N)*100,1))
        print '# Dataset Removed: {0} ({1} %)'.format(Nrm,round(Nrm/float(N)*100,1))

        return self.IdxGoodSamples

    def display_samples_quality(self,Save=True,Suffix=''):

        mds = MDS(metric='precomputed')
        XY = mds.fit_transform(self.OptDistMat)

        # dc = self.OptDistMat[self.IdxGoodSamples,:]
        # dc = dc[:,self.IdxGoodSamples]
        # XYc = mds.fit_transform(dc)

        labelsGood = np.zeros(len(self.labels))
        labelsGood[self.IdxGoodSamples] = 1

        f,ax = plt.subplots(2,2,figsize=(15,15))
        ax[0,0].scatter(XY[:,0],XY[:,1],c=self.labels,cmap='rainbow')
        ax[0,0].set_title('Original dataset')
        ax[0,0].grid(color='k',linestyle='--',linewidth=0.1)
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticklabels([])

        ax[0,1].scatter(XY[self.IdxGoodSamples,0],XY[self.IdxGoodSamples,1],c=self.labels[self.IdxGoodSamples],cmap='rainbow')
        ax[0,1].set_title('Cleaned dataset')
        ax[0,1].grid(color='k',linestyle='--',linewidth=0.1)
        ax[0,1].set_xticklabels([])
        ax[0,1].set_yticklabels([])

        color=cm.bwr(np.linspace(0,1,2))
        for i in range(2):
            ax[1,0].scatter(XY[labelsGood==i,0],XY[labelsGood==i,1],c=color[i])

        ax[1,0].grid(color='k',linestyle='--',linewidth=0.1)
        ax[1,0].legend(['Low-Quality Samples','High-Quality Samples '],loc=0,fontsize='x-small')
        ax[1,0].set_title('Samples Quality')
        ax[1,0].set_xticklabels([])
        ax[1,0].set_yticklabels([])

        cax = ax[1,1].scatter(XY[:,0],XY[:,1],c=self.SilZ, vmin=-5, vmax=5,cmap='coolwarm')
        ax[1,1].grid(color='k',linestyle='--',linewidth=0.1)
        ax[1,1].set_title('Z-score Silhouette Index')
        ax[1,1].set_xticklabels([])
        ax[1,1].set_yticklabels([])

        colorbar_ax = f.add_axes([0.95, 0.13, 0.02, 0.33])
        f.colorbar(cax, cax=colorbar_ax)
        # cbar = f.colorbar(cbaxes)

        if Save:
            FigPath = os.path.abspath('.') + '/SamplesQuality' + Suffix + '.png'
            plt.savefig(FigPath,dpi=300,format='png')

        plt.show()

    def displayInternal(self,Save=True,Suffix=''):
        '''
            Display Internal Indexes on all matrices
        '''
        width = 0.35 # the width of the bars
        fig, ax = plt.subplots(1,len(self.Keys['InternIndex']),figsize=(45,int(self.displaytop*1.5)))

        for ik,Ikey in enumerate(self.Keys['InternIndex']):
            values = np.array(self.Res[Ikey].values())
            keys = np.array(self.Res[Ikey].keys())
            idx = np.argsort(values)
            idx = idx[~np.isnan(values[idx])]

            if Ikey=='Connectivity': idx = idx[::-1]

            idx = idx[-self.displaytop:]
            sorted_values = values[idx]
            sorted_keys = keys[idx]

            if Ikey=='log10CalinskiHarabaz':sorted_values[sorted_values<0]=0 #Remove negative values for the dipslay

            ind = np.arange(len(idx))

            # for kk,key in enumzerate(sorted_keys):
            ax[ik].barh(ind, sorted_values)

            # if set_log == True: ax[ik].set_xlabel('log10')

            ax[ik].grid(color='k',linestyle='--',linewidth=0.1)
            ax[ik].set_yticks(ind)
            ax[ik].set_yticklabels(sorted_keys)
            ax[ik].set_title(Ikey)

            if Ikey=='Silhouette':
                if sorted_values.min()<0:
                    mini = -sorted_values.max()*1.1
                else:
                    mini = 0
                ax[ik].set_xlim([mini,sorted_values.max()*1.1])



        if Save:
            FigPath = os.path.abspath('.') + '/InternalIndexes' + Suffix + '.png'
            plt.savefig(FigPath,dpi=300,format='png')

        plt.show()

    def displayMDS(self,Save=True,Suffix=''):
        '''
            Display Internal Indexes on all matrices
        '''
        if self.GetMDS==False:
            print 'MDS not computed, please launch "compute_distances" \
                   with option "ComputeMDS=True"'
            return None

        fig, ax = plt.subplots(len(self.Keys['Distance']),len(self.Keys['Kernel']),figsize=(50,100))
        for dk,Dkey in  enumerate(self.Keys['Distance']):
            ax[dk,0].set_ylabel(Dkey)
            for kk,Kkey in  enumerate(self.Keys['Kernel']):
                if Dkey is not 'None' or Kkey is not 'None':
                    try:
                        XY = self.MDS[Dkey][Kkey]
                        ax[dk,kk].scatter(XY[:,0],XY[:,1],c=self.labels)
                        ax[dk,kk].set_xticklabels([])
                        ax[dk,kk].set_yticklabels([])
                        ax[dk,kk].grid(color='k',linestyle='--',linewidth=0.1)
                    except:
                        pass
                if dk==0:
                    ax[0,kk].set_title(Kkey + '\n' + Dkey + ' / ' + Kkey)
                else:
                    ax[dk,kk].set_title(Dkey + ' / ' + Kkey)

        if Save:
            FigPath = os.path.abspath('.') + '/MDS' + Suffix + '.png'
            plt.savefig(FigPath,dpi=300,format='png')

        plt.show()

    def displayNamesScatter(self,Save=True,Suffix='',IkeyX='Silhouette',IkeyY='Connectivity'):
        '''
            Display Internal Indexes on all matrices
            Ikey = CalinskiHarabaz, Silhouette, Connectivity
        '''
        fig, ax = plt.subplots(1,1,figsize=(10,10))

        keys=[]
        for Ikey in [IkeyX,IkeyY]:
            values = np.array(self.Res[Ikey].values())
            keysI = np.array(self.Res[Ikey].keys())
            idx = np.argsort(values)
            idx = idx[~np.isnan(values[idx])]

            if Ikey=='Connectivity': idx = idx[::-1]

            idx = idx[-self.displaytop:]
            keys = keys + list(keysI[idx])

        keys = list(set(keys))

        for key in keys:
            ax.scatter(self.Res[IkeyX][key],
                       self.Res[IkeyY][key],
                       facecolors='none')
            ax.annotate(key,
                        (self.Res[IkeyX][key],
                         self.Res[IkeyY][key]))

        # for dk,Dkey in  enumerate(self.Keys['Distance']):
        #     for kk,Kkey in  enumerate(self.Keys['Kernel']):
        #         if Dkey is not 'None' or Kkey is not 'None':
        #             ax.scatter(self.Res['Silhouette'][Dkey + ' / ' + Kkey],
        #                        self.Res['Connectivity'][Dkey + ' / ' + Kkey],
        #                        facecolors='none',
        #                        )
        #             ax.annotate(Dkey + ' / ' + Kkey,
        #                         (self.Res['Silhouette'][Dkey + ' / ' + Kkey],
        #                         self.Res['Connectivity'][Dkey + ' / ' + Kkey])
        #                         )

        ax.grid(color='k',linestyle='--',linewidth=0.1)
        ax.set_xlabel(IkeyX)
        ax.set_ylabel(IkeyY)
        # ax.set_ylabel('log10 CalinskiHarabaz')

        if Save:
            FigPath = os.path.abspath('.') + '/NamesScatter' + Suffix + '.png'
            plt.savefig(FigPath,dpi=300,format='png')

        plt.show()


#-----------------------------------------------------------------------
#           Internal Indexes
#-----------------------------------------------------------------------

def Connectivity(DistMat,labels,PercentNeigh=0.3):
    '''
        Connectivity index as reported in: https://www.jstatsoft.org/article/view/v025i04/v25i04.pdf

        DistMat : samples x samples numpy array - Distance matrix
        labels: Vector of size Nsamples with one different label per class
        PercentNeigh : Percentage of points that will contribute to compute the connectivity score

        --> The Lower the better the clustering (optimal = 0 )
    '''
    Nsamples = DistMat.shape[0]

    NumNeigh = int(PercentNeigh*Nsamples)

    connectivity = 0
    for i in range(Nsamples):
        class_i = labels[i]
        idx_sort = np.argsort(DistMat[i,:])
        position_NN_diff_class = np.where(labels[idx_sort]!=class_i)[0][:NumNeigh].sum()
        if position_NN_diff_class>0:
            connectivity += (1./position_NN_diff_class) #1/(position of the points in different classes)


    return connectivity


def Dunn(DistMat,labels):
    '''
        Ratio of the smallest distance between clusters and larger distance inside cluster

        --> The Higher the better the clustering

        Dunn is sensible to outliers in the sample -> Not a reliable mesure
    '''

    labels = labels.astype(np.int32)

    idx_labels = [np.where(labels==l)[0] for l in range(labels.max()+1)]

    max_intra = 0
    min_extra = np.inf

    for i,idx in enumerate(idx_labels):
        #Find all points outside the cluster of i
        mask = np.ones(labels.shape[0],dtype=bool)
        mask[idx] = False
        idx_ = np.arange(labels.shape[0])[mask]
        DistMat_i = DistMat[idx,:]
        max_intra = max(max_intra,DistMat_i[:,idx].max())
        min_extra = min(min_extra,DistMat_i[:,idx_].min())


    dunn = min_extra/float(max_intra)

    return dunn


def Silhouette(Distmat,labels):
    s = silhouette_samples(Distmat,labels,metric='precomputed')
    s[np.isnan(s)] = -1 #Nan append when 2 clusters has zeros intra and inter distance
    # if s<0: s=0
    return s.mean()
def CalinskiHarabaz(Distmat,labels):

    #Coordinates Matrix
    mds = MDS(n_components=Distmat.shape[0],dissimilarity='precomputed')
    CoordMat = mds.fit_transform(Distmat)

    labels = labels.astype(np.int32)
    idx_labels = [np.where(labels==l)[0] for l in range(labels.max()+1)]

    ClusterCenter = [CoordMat[idx,:].mean(axis=0) for idx in idx_labels]
    Center = CoordMat.mean(axis=0)
    # overall between-cluster variance
    SSB = sum([len(idx_labels[i])*np.sum((cc-Center)**2) for i,cc in enumerate(ClusterCenter)])
    # overall within-cluster variance
    SSW = sum([np.sum(np.sum((CoordMat[idx,:]-ClusterCenter[i][None,:])**2,axis=0)) for i,idx in enumerate(idx_labels)])

    return SSB/SSW * (Distmat.shape[0]-len(idx_labels))/(len(idx_labels)-1)


def log10CalinskiHarabaz(Distmat,labels):
    CS = CalinskiHarabaz(Distmat,labels)
    return np.log10(CS+1)

'''
     Calinski-Harabasz:
     The Calinski-Harabasz criterion is sometimes called the variance ratio criterion (VRC).

     --> The Higher the better

Silhouette:
    The silhouette value is a measure of how similar an object is to its own cluster (cohesion)
    compared to other clusters (separation).
    --> The higher the better
'''

#-----------------------------------------------------------------------
#           Distances
#-----------------------------------------------------------------------

#Mutual information function based on MIC
from minepy import MINE
def MIC(X):

    mine = MINE(alpha=0.6, c=15)

    Nsamples = X.shape[0]
    MicVect = np.zeros((Nsamples,Nsamples))

    for i in range(Nsamples):
        for j in range(i,Nsamples):
            mine.compute_score(X[i,:],X[j,:])
            MicVect[i,j] = mine.mic()

    MicVect = MicVect + MicVect.T - np.diag(np.diag(MicVect))

    return 1-MicVect #MIC to Distance

#Spearman Distance
from scipy.stats import spearmanr
def Spearman(X):
    rho, _ = spearmanr(X.T)
    return 1-rho

#-----------------------------------------------------------------------
#           Kernels
#-----------------------------------------------------------------------
def CosineKernel(X,labels=None,Distance=None):
    '''
        Return a distance Cosine Kernel
    '''
    # SqNorm = ((X[None,:,:]-X[:,None,:])**2).sum(axis=2)
    Norm = (X**2).sum()
    return 1 - X.dot(X.T)/Norm

def SigmoidKernel(X,labels,Distance=None):
    '''
        Return a distance Sigmoid Kernel
    '''

    if Distance == 'None':
        X = pairwise_distances(X,metric='euclidean')

    gamma = get_gammaSig(X,labels)
    # gamma = 1.0 / X.shape[1]
    c0 = 1
    return 1 - np.tanh(gamma*X + c0)

def RBFKernel(X,labels,Distance=None):
    '''
        Return a distance RBF Kernel but with
        precomputed distance
    '''
    if Distance == 'None':
        X = pairwise_distances(X,metric='euclidean')


    gamma = get_gammaRBF(X,labels)
    c0 = 1
    return 1 - np.exp(-gamma*X)

def get_gammaRBF(DistMat,labels):
    labels = labels.astype(np.int32)

    idx_labels = [np.where(labels==l)[0] for l in range(labels.max()+1)]

    dist_intra = np.zeros(len(idx_labels))

    for i,idx in enumerate(idx_labels):
        d = DistMat[idx,:]
        d = d[:,idx]
        dist_intra[i] = d.mean()

    dmin = dist_intra.min() + 1e-8
    gamma = np.power(dmin,-2)

    return gamma

def get_gammaSig(DistMat,labels):
    labels = labels.astype(np.int32)

    idx_labels = [np.where(labels==l)[0] for l in range(labels.max()+1)]

    dist_extra = np.zeros(len(idx_labels))

    for i,idx in enumerate(idx_labels):
        #Find all points outside the cluster of i
        idx_ = np.ones(labels.shape[0],dtype=bool)
        idx_[idx] = False
        # idx_ = np.arange(labels.shape[0])[mask]
        DistMat_i = DistMat[idx,:]
        dist_extra[i] = DistMat_i[:,idx_].mean()

    dmin = dist_extra.min() + 1e-8
    gamma = np.power(dmin,-2)

    return gamma

#-----------------------------------------------------------------------
#           Stand-Alone Quality Tester
#-----------------------------------------------------------------------

def compute_samples_quality_from_distmat(matrix,distmat,labels,names,NR=100,Tz=2.32,Nkeep=-1,NPeaksMin=[]):
    '''
        Compute the quality of each profiles according to the given distance matrix
        This function is to use when the optimal distance measure is known

        The quality is computed as the Zscore of the silouette score of each samples
        compared to random cluster attribution

        inputs:
        Tz: Threshold on the zscore
        NR: Number of random shuffling
        Nkeep: Number of profiles to keep maximum in each clusters
        (if -1 all good samples are kept)
        names (optional): numpy array with names of the cluster (for the final display)
        NpeaksMin: (vector) Number of peaks minimum in a profile to pass the test
                   (if -1, the median of the number of peaks per clusters divided by 4
                   is taken)

        Output:
        IdxGoodSamples: indexes of the samples which pass the test
        SilZ: Silhouette Zscore of the samples
        SilZClust: Mean of the silhouette zscore over the clusters

    '''

    IdxGoodSamples,SilZ = ZscoreSilhouetteQuality(distmat,labels,Tz,NR)

    SilZClust = np.array([SilZ[np.logical_and(labels==l,SilZ>Tz)].mean() for l in np.unique(labels)])

    NPeaks = matrix.sum(axis=1)
    if len(NPeaksMin)==0:
        NPeaksMin = np.ones(matrix.shape[0])*np.median(NPeaks)/4.
        print 'Minimal number of peaks for a sample: {}'.format(NPeaksMin[0])

    IdxGoodSamples = []
    SilZClust = []
    for l in np.unique(labels):
        idx = np.where(np.logical_and(labels==l,SilZ>Tz))[0]
        idx = idx[NPeaksMin[idx]<NPeaks[idx]]

        if len(idx>0):
            idx = idx[np.argsort(SilZ[idx])]
            if Nkeep>0: idx = idx[-Nkeep:]


            IdxGoodSamples.extend(idx)
            SilZClust.append(SilZ[idx].mean())

    IdxGoodSamples = np.array(IdxGoodSamples)
    SilZClust = np.array(SilZClust)

    #Displays
    N = len(SilZ)
    Nk = len(IdxGoodSamples)
    Nrm = N - Nk
    print ''
    print '# Dataset Total: ' + str(N)
    print '# Dataset Kept: {0} ({1} %)'.format(Nk,round(Nk/float(N)*100,1))
    print '# Dataset Removed: {0} ({1} %)'.format(Nrm,round(Nrm/float(N)*100,1))

    uniq,count = np.unique(labels[IdxGoodSamples],return_counts=True)
    idx = np.argsort(uniq)
    uniq = uniq[idx]
    count = count[idx]
    print '\nKept Datasets'
    print 'Labels\t\tCounts\t\tName\t\tMean Zscore Silhouette'
    for l,n,c,sz in zip(uniq,names[uniq],count,SilZClust):
        print '{0}\t\t{1}\t\t{2}\t\t{3}'.format(l,c,n,sz)

    labelsRm = np.array([l for l in np.unique(labels) if l not in uniq])
    if len(labelsRm)>0:
        labelsRm = np.sort(labelsRm)
        print '\nRemoved Datasets'
        print 'Labels\t\tName'
        for l,n in zip(labelsRm,names[labelsRm]):
            print '{0}\t\t{1}'.format(l,n)


    return IdxGoodSamples,SilZ,SilZClust
