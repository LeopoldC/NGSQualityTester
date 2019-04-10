import numpy as np
import os
from sklearn.metrics import silhouette_score,calinski_harabaz_score,silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from functools import partial

# plt.style.use('ggplot')
from tqdm import tqdm

from adjustText import adjust_text

from QualityMesures import *
import RandomNGS as RNGS

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
    def __init__(self):
        '''
            Test the different distances and define the samples quality
            Binary='Discrete'/'Bool' : Type of data from the input matrix

        '''


        #----------------------------
        #   PCA
        #----------------------------
        def NoneFunc(matrix,binary,NR): return matrix
        self.PCA = {key:f for f,key in
                               [(PCAEmbedding,'PCA'),
                                (NoneFunc,'None')]}

        #----------------------------
        #   Distances
        #----------------------------
        self.AllDistances  = {'City Block':'cityblock','Cosine':'cosine','Euclidean':'euclidean','L2':'l2',
                            'Bray Curtis':'braycurtis', 'Canberra':'canberra', 'Chebyshev':'chebyshev',
                            'Pearson':'correlation', 'Hamming':'hamming', 'Squared Euclidean':'seuclidean',
                            'Dice':'dice', 'Jaccard':'jaccard','Rogers Tanimoto': 'rogerstanimoto',
                            'Russell Rao':'russellrao', 'Sokal Michener':'sokalmichener',
                            'Sokal Sneath':'sokalsneath', 'Yule':'yule'}

        for key in self.AllDistances.keys():
            def skfunc(x,keyd=self.AllDistances[key]):
                return pairwise_distances(x,metric=keyd)
            self.AllDistances[key] = skfunc

        # def Nonefunc(x): return x
        self.AllDistances.update({'Spearman': Spearman})
        self.AllDistances.update({'Mutual Information': MI})

        #Distance for Continuous(False) or Binary (True)
        self.Dist = {   False:[ 'City Block','Cosine','Euclidean','L2','Bray Curtis',
                                'Canberra','Pearson', 'Squared Euclidean',
                                'Mutual Information', 'Chebyshev',
                                'Spearman'],
                        True: self.AllDistances.keys()}


        #----------------------------
        #   Kernels
        #----------------------------

        self.AllKernels = {key:f for f,key in
                               [(CosineKernel,'Cosine Kernel'),
                                (SigmoidKernel,'Sigmoid Kernel'),
                                (RBFKernel,'Exponential Kernel')]}

        #None function, ie no distance function
        def Nonefunc(x,labels): return x
        self.AllKernels.update({'None':Nonefunc})

        #----------------------------------
        #TEMPORAIRE POUR PAPIER
        #----------------------------------
        self.AllKernels = {'None':Nonefunc}



        #----------------------------
        #   Full names
        #----------------------------
        # self.Names = {  'CB':'City Block','Cos':'Cosine','E':'Euclidean','L2':'L2',
        #                 'BC':'Bray Curtis', 'Can':'Canberra', 'Ch':'Chebyshev',
        #                 'P':'Pearson', 'H':'Hamming', 'SE':'Squared Euclidean',
        #                 'D':'Dice', 'J':'jaccard', 'RS': 'Rogers-Tanimoto',
        #                 'RR':'Russell-Rao', 'SM':'Sokal-Michener','SP':'Spearman',
        #                 'SS':'Sokal-Sneath', 'Y':'Yule','MIC':'Mutual Information',
        #                 'CK':'Cosine Kernel','SK':'Sigmoid Kernel',
        #                 'EK':'Exponential Kernel','N':'None'}

        #----------------------------
        #   Internal Indexes
        #----------------------------
        self.AllInternIndex = {key:f for f,key in [
                                     #(log10CalinskiHarabaz,'log10CalinskiHarabaz'),
                                     (Silhouette,'Silhouette'),
                                     # (Dunn,'Dunn'),
                                     (Connectivity,'Connectivity')]}


        #Result dictionnary
        self.Res={Ikey:{} for Ikey in self.AllInternIndex.keys()}
        self.clust_index_filt_evol={Ikey:{} for Ikey in self.AllInternIndex.keys()}
        self.clust_index_filt_score={Ikey:{} for Ikey in self.AllInternIndex.keys()}
        self.clust_index_filt_threshold={Ikey:{} for Ikey in self.AllInternIndex.keys()}

        self.Distmat = {}

        # #Display only the top performer
        # if displaytop>0:
        #     self.displaytop = displaytop
        # else:
        #     self.displaytop = len(self.Res[Ikey].keys())


    def get_optimal_distance(self,matrix,labels,threshold = [0.01, 0.99]):
        '''
            Compute all distances matrices and kernels on the matrix
            and compute the internal index according to the labels

            DR: Dimensionality reduction algorithm:
                    'MDS': multidimensional scaling (fast but noisy)
                    'TSNE': multidimensional scaling (slow but highlight clusters)
        '''

        self.M = matrix.shape[0]

        self.Binary = self.check_if_binary_matrix(matrix)

        #Define outliers
        print( 'Remove outliers defined by anomalous number of reads')
        self.remove_sum_outliers(matrix,threshold = threshold)
        self.matrix = matrix.copy()[self.idxSumKeep]
        self.labels = labels.copy()[self.idxSumKeep]


        print ('\nTesting all distances:')
        for Pkey in self.PCA.keys():
            print ('\n' + Pkey)
            # if Pkey=='PCA': continue
            try:
                PMat = self.PCA[Pkey](matrix=self.matrix,binary=self.Binary,NR = 10)
                self.PCA[Pkey] = partial(Identity,X=PMat) #Return the PCA matrix if already computed

            except Exception as e:
                print(e)
                continue

            for Dkey in self.Dist[self.Binary and Pkey=='None']:
                print( '\t' + Dkey)
                if (Pkey+'-'+ Dkey) in [key.split('-None')[0] for key in self.Distmat.keys()]:
                    continue
                try:
                    self.PDMat = self.AllDistances[Dkey](PMat)

                except Exception as e:
                    print(e)
                    continue

                for Kkey in self.AllKernels.keys():
                    print ('\t\t' + Kkey)
                    Tkey = Pkey+'-'+ Dkey+'-'+Kkey
                    try:
                        self.Distmat[Tkey] = self.AllKernels[Kkey](self.PDMat,self.labels)

                    except Exception as e:
                        print(e)
                        continue

                    for Ikey in self.AllInternIndex.keys():
                        try:
                            #Compute the clustering index score
                            self.Res[Ikey].update({Tkey: self.AllInternIndex[Ikey](self.Distmat[Tkey],self.labels)})
                            self.clust_index_filt_evol[Ikey][Tkey],self.clust_index_filt_score[Ikey][Tkey],self.clust_index_filt_threshold[Ikey][Tkey] = self.ClustIndexFilt(self.Distmat[Tkey],self.labels,Ikey)

                            print ('\t\t\t' + Ikey + ' (Clustering,CF) index: ( {0} , {1} )'.format(self.Res[Ikey][Tkey],self.clust_index_filt_score[Ikey][Tkey]))

                        except Exception as e:
                            print(e)
                            continue

        print ('Get optimal PCA-Distance-Kernel')
        rank={}
        k,c = np.unique(np.hstack([self.Res[Ikey].keys() for Ikey in self.Res]),return_counts=True)
        self.PDKkeys = k[c==len(self.AllInternIndex.keys())] #Keep keys present only in the all indexes

        self.PDKrank = {}

        for Ikey in self.Res.keys():
            values = np.hstack([self.Res[Ikey][PDKkey] for PDKkey in self.PDKkeys])
            idx = np.argsort(values)
            idx = idx[~np.isnan(values[idx])]
            # if Ikey=='Silhouette': idx = idx[::-1]
            idx = idx[::-1]
            self.PDKrank[Ikey] = np.argsort(idx)

        self.PDKrank['Sum'] = np.zeros(len(self.PDKkeys))
        for Ikey in self.Res.keys():
            self.PDKrank['Sum'] += self.PDKrank[Ikey]

        self.SortedPDKkey = {}
        self.SortedPDKkey['Sum'] = self.PDKkeys[np.argsort(self.PDKrank['Sum'])]

        for Ikey in self.Res.keys():
            self.SortedPDKkey[Ikey] = self.PDKkeys[np.argsort(self.PDKrank[Ikey])]


        self.OptimalPDKkey = self.SortedPDKkey['Sum'][0]
        self.OptimalDistance = self.Distmat[self.OptimalPDKkey].copy()

        print ('Optimal (PCA,Distance,Kernel) = {0}'.format(tuple(self.OptimalPDKkey.split('-'))))



    def check_if_binary_matrix(self,matrix):
        #Define if the matrix is boolean or Discrete
        uniq = np.unique(matrix)
        if len(uniq)==2:
            Binary = np.all(uniq==np.array([False,True]))
        else:
            Binary = False

        return Binary


    def remove_sum_outliers(self,matrix,threshold = [0.01, 0.99]):
        '''
            Get the outliers at [0.01,0.99] and remove it from the matrix, labels.
            inputs:
        '''
        #Get outliers
        _,self.idxSumKeep = get_outliers(matrix, threshold = threshold)

        # #Remove outliers frm the matrix labels and names
        # matrix = matrix[self.idxSumKeep,:]
        # labels = labels[self.idxSumKeep]
        #
        # return matrix,labels

    def print_all_distances(self):

        print ('PCA')
        for Pkey in self.PCA.keys():
            print (Pkey)

        print ('\nDistances')
        print ('\tDiscrete Distances')
        for Dkey in self.Dist[False]:
            print (Dkey)

        print ('\tBoolean Distances')
        for Dkey in self.Dist[True]:
            print (Dkey)

        print ('\Kernel')
        for Kkey in self.AllKernels.keys():
            print (Kkey)

    def compute_samples_quality(self,matrix=[],labels=[],Distance='None',Kernel='None',PCA='None',
                                NR=100,Tz=0,names=[],return_Quality=True,threshold = [0.01, 0.99],
                                NumMaxSamples=-1):
        '''
            Compute the quality of each profiles according to PCA / Distance / Kernel
            By default, the Distance/Kernel with the best silhouette score is choosen.

            The quality is computed as the Zscore of the silouette score of each samples
            compared to random cluster attribution

            inputs:
            Tz: Threshold on the zscore
            NR: Number of random shuffling
            NumMaxSamples: Number max of sample per clusters according to the threshold on the z-score
                            (if -1, None)

            Optional inputs(if get_optimal_distance not launched before)
            matrix: (samples x features)
            labels: (samples)
            names: (np.max(labels)+1) : names of the different clusters labeled by 'labels'

            PCA,Distance,Kernel: Strings corresponding to the triple distance (to get all distances
                                    we can compute self.print_all_distances() )

            return_Quality: Return the Quality of the profiles (-np.inf correspond to outliers
                            detected from anomalous sum compared to the others)
            Output:
                if return_Quality==True
                    return IdxGoodSamples of the good samples and the Silhouette Z-score of the matrix


        '''

        if "matrix" not  in self.__dict__:
            if len(matrix)>0:
                self.matrix = matrix
            else:
                print ('Need matrix')
                raise ValueError
        if "labels" not  in self.__dict__:
            if len(labels)>0:
                self.labels = labels
            else:
                print ('Need labels')
                raise ValueError

        #Remove Sum Outlier
        if "idxSumKeep" not  in self.__dict__:
            print ('Remove sum outliers')
            self.remove_sum_outliers(matrix,threshold = threshold)
            self.M = matrix.shape[0]
            self.matrix = self.matrix[self.idxSumKeep,:]
            self.labels = self.labels[self.idxSumKeep]

        #Get or compute distance matrix
        if PCA is 'None' and Distance is 'None' and Kernel is 'None':
            if "OptimalDistance" not  in self.__dict__:
                print ('Error: Need to enter a PCA-Distance-Kernel or launch all measures with compute_distances()')
                raise ValueError

            if "OptimalPDKkey" in self.__dict__:
                print ('Using Optimal (PCA,Distance,Kernel) = {0}'.format(tuple(self.OptimalPDKkey.split('-'))))

        else:
            try:
                self.OptimalDistance = self.Distmat['-'.join([PCA,Distance,Kernel])]
            except:
                print ('Computing Distance matrix')
                self.OptimalDistance = self.computeDistanceMatrix(self.matrix,self.labels,PCA,Distance,Kernel)
                print ('Done!')


        self.IdxGoodSamples,self.SilZ = ZscoreSilhouetteQuality(self.OptimalDistance,self.labels,Tz,NR)

#        self.IdxGoodSamplesConn,self.ConnZ = ZscoreConnectivityQuality(self.OptimalDistance,self.labels,Tz,NR)

        #Display SilZ
        f,ax = plt.subplots(1,2,figsize=(10,5))
        hist, bin_edges = np.histogram(self.SilZ)
        ax[0].bar(bin_edges[:-1],hist)
        ax[0].grid('k',ls='--')
        ax[0].set_xlabel('Z-score')
        ax[0].set_title('Histogram Z-score profiles')
        ax[1].plot(bin_edges[1:],np.cumsum(hist/hist.sum())*100)
        ax[1].grid('k',ls='--')
        ax[1].set_xlabel('Z-score threshold')
        ax[1].set_ylabel('Precentage of samples removed')
        ax[1].set_title('Evolution of the proportion of samples removed with threshold')
        plt.show()

        #Get new labels
        uniq,count = np.unique(self.labels[self.IdxGoodSamples],return_counts=True)
        idx = np.argsort(uniq)
        uniq = uniq[idx]
        count = count[idx]

        if NumMaxSamples>0:
            idxl = [np.where(np.logical_and(self.SilZ>Tz,self.labels==l))[0] for l in uniq]

            idxl = [idx[np.argsort(self.SilZ[idx])][-NumMaxSamples:] for idx in idxl]
            self.IdxGoodSamples = np.hstack(idxl)
            self.SilZClust = np.hstack([np.mean(self.SilZ[idx]) for idx in idxl])
        else:
            self.SilZClust = np.hstack([np.mean(self.SilZ[np.logical_and(self.SilZ>Tz,self.labels==l)]) for l in uniq])

        N = self.M
        Nk = len(self.IdxGoodSamples)
        Nrm = N - Nk
        print ('')
        print ('# Dataset Total: ' + str(N))
        print ('# Dataset Kept: {0} ({1} %)'.format(Nk,round(Nk/float(N)*100,1)))
        print ('# Dataset Removed: {0} ({1} %)'.format(Nrm,round(Nrm/float(N)*100,1)))



        if len(names)==0:
            names = np.array(['Cluster {}'.format(l) for l in np.unique(self.labels)])

        #self.SilZClust = np.array([self.SilZ[self.IdxGoodSamples[self.labels[self.IdxGoodSamples]==l]].mean() for l in np.unique(self.labels)])

        print ('\nKept Datasets')
        print ('Labels\t\tCounts\t\tMean Zscore Silhouette\t\tName')
        for l,n,c,sz in zip(uniq,names[uniq],count,self.SilZClust):
            print ('{0}\t\t{1}\t\t{2}\t\t{3}'.format(l,c,sz,n))

        labelsRm = np.array([l for l in np.unique(self.labels) if l not in uniq])
        if len(labelsRm)>0:
            labelsRm = np.sort(labelsRm)
            print ('\nRemoved Datasets')
            print ('Labels\t\tName')
            for l,n in zip(labelsRm,names[labelsRm]):
                print ('{0}\t\t{1}'.format(l,n))

        if return_Quality==True:
            #Return indexes correspongind to the original matrix
            IdxGoodSamplesO = np.zeros(self.M,dtype=np.bool)
            IdxGoodSamplesO[self.idxSumKeep[self.IdxGoodSamples]] = True
            IdxGoodSamplesO = np.where(IdxGoodSamplesO==True)[0]

            SilZO = np.ones(self.M)*(-np.inf)
            SilZO[self.idxSumKeep] = self.SilZ

            OptimalDistanceO = np.ones((self.M,self.M))*(np.inf)
            OptimalDistanceO[np.ix_(self.idxSumKeep,self.idxSumKeep)] = self.OptimalDistance

            return IdxGoodSamplesO,SilZO,OptimalDistanceO

    def display_samples_quality(self,Save=True,Suffix='',DR='TSNE',names=[],learning_rate=-1,perplexity=-1,Figsize=15):
        '''
            Display the samples quality with the optimal distance measure using
            either MDS or TSNE. For the TSNE a grid seach is performed to find the best
            parameters
            DR: Dimensionality reduction algorithm. Two possibiliites: MDS or TSNE. In
                the case of TSNE, an optimisatuion of the parameters is performed to get the
                best visualization
            names: array of names of the different clusters. If empty, no names will be displayed

        '''

        if "IdxGoodSamples" not  in self.__dict__:
            print ('Launch "compute_samples_quality" first !')
            raise ValueError

        if DR=='MDS':
            mds = MDS(metric='precomputed')
            self.XY = mds.fit_transform(self.OptimalDistance)
        elif DR=='TSNE':
            if learning_rate==-1 and perplexity==-1:
                self.XY = OptimalTSNE(Distmat=self.OptimalDistance,labels=self.labels)
            else:
                model_tsne = manifold.TSNE(n_components=2, random_state=3,perplexity=perplexity,
                                           early_exaggeration=4.0, learning_rate=learning_rate,
                                           n_iter=100000, metric='precomputed')

                self.XY = model_tsne.fit_transform(self.OptimalDistance)

            #Remove far outliers samples for the plot
            # _,idxK = idxRm,idxK = kde_outliers(np.power(self.XY,2.).sum(axis=1), threshold = [0.001, 0.999], w = 2000,Display=False)
            # vectK = np.zeros(self.XY.shape[0],dtype=np.int32)
            # vectK[idxK] = 1

        else:
            print ('Dimesionality Reduction Algorithm not recognized')

        labelsGood = np.zeros(len(self.labels))
        labelsGood[self.IdxGoodSamples] = 1

        f,ax = plt.subplots(2,2,figsize=(Figsize*1.1,Figsize))
        if len(names)==0:
            ax[0,0].scatter(self.XY[:,0],self.XY[:,1],c=self.labels,cmap='rainbow')
            # ax[0,0].scatter(self.XY[:,0][vectK],self.XY[:,1][vectK],c=self.labels,cmap='rainbow')
        else:
            color=cm.rainbow(np.linspace(0,1,np.max(self.labels)+1))
            for l in range(np.max(self.labels)+1):
                ax[0,0].scatter(self.XY[self.labels==l,0],self.XY[self.labels==l,1],c=color[l],label=names[l])
                # Xtext = self.XY[self.labels==l,0].mean(axis=0) + self.XY[self.labels==l,0].std(axis=0)*np.random.rand()
                # Ytext = self.XY[self.labels==l,1].mean(axis=0) + self.XY[self.labels==l,1].std(axis=0)*np.random.rand()
                # ax[0,0].text(Xtext,Ytext,s=names[l], bbox=dict(facecolor=color[l], alpha=0.6))

            lgd = ax[0,0].legend(bbox_to_anchor=(2.55, 1), loc=2, borderaxespad=0.)

        ax[0,0].set_title('Original dataset')
        ax[0,0].grid(color='k',linestyle='--',linewidth=0.1)
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticklabels([])

        if DR=='MDS':
            mds = MDS(metric='precomputed')
            self.XY_Cl = mds.fit_transform(self.OptimalDistance[np.ix_(self.IdxGoodSamples,self.IdxGoodSamples)])
        elif DR=='TSNE':
            if learning_rate==-1 and perplexity==-1:
                self.XY_Cl = OptimalTSNE(Distmat=self.OptimalDistance[np.ix_(self.IdxGoodSamples,self.IdxGoodSamples)],labels=self.labels[self.IdxGoodSamples])
            else:
                model_tsne = manifold.TSNE(n_components=2, random_state=3,perplexity=perplexity,
                                           early_exaggeration=4.0, learning_rate=learning_rate,
                                           n_iter=100000, metric='precomputed')

                self.XY_Cl = model_tsne.fit_transform(self.OptimalDistance[np.ix_(self.IdxGoodSamples,self.IdxGoodSamples)])

            # #Remove far outliers samples for the plot
            # _,idxK = kde_outliers(np.power(self.XY,2.).sum(axis=1), threshold = [0.001, 0.999], w = 2000,Display=False)
            # vectK_Cl = np.zeros(self.XY_Cl.shape[0],dtype=np.int32)
            # vectK_Cl[idxK] = 1

        if len(names)==0:
            ax[0,1].scatter(self.XY_Cl[:,0],self.XY_Cl[:,1],c=self.labels[self.IdxGoodSamples],cmap='rainbow')
            # ax[0,1].scatter(self.XY_Cl[:,0][vectK_Cl],self.XY_Cl[:,1][vectK_Cl],c=self.labels[self.IdxGoodSamples][vectK_Cl],cmap='rainbow')
        else:
            color=cm.rainbow(np.linspace(0,1,np.max(self.labels)+1))
            for l in range(np.max(self.labels)+1):
                ax[0,1].scatter(self.XY_Cl[self.labels[self.IdxGoodSamples]==l,0],self.XY_Cl[self.labels[self.IdxGoodSamples]==l,1],c=color[l])
                    # Xtext = self.XY_Cl[self.labels[self.IdxGoodSamples]==l,0].mean(axis=0) + self.XY_Cl[self.labels[self.IdxGoodSamples]==l,0].std(axis=0)*np.random.rand()
                    # Ytext = self.XY_Cl[self.labels[self.IdxGoodSamples]==l,1].mean(axis=0) + self.XY_Cl[self.labels[self.IdxGoodSamples]==l,1].std(axis=0)*np.random.rand()
                    # ax[0,1].text(Xtext,Ytext,s=names[l], bbox=dict(facecolor=color[l], alpha=0.6))

        #ax[0,1].scatter(XY[self.IdxGoodSamples,0],XY[self.IdxGoodSamples,1],c=self.labels[self.IdxGoodSamples],cmap='rainbow')

        ax[0,1].set_title('Cleaned dataset')
        ax[0,1].grid(color='k',linestyle='--',linewidth=0.1)
        ax[0,1].set_xticklabels([])
        ax[0,1].set_yticklabels([])

        color=cm.bwr(np.linspace(0,1,2))
        for i in range(2):
            ax[1,0].scatter(self.XY[labelsGood==i,0],
                            self.XY[labelsGood==i,1],c=color[i])
            # ax[1,0].scatter(self.XY[labelsGood==i,0][vectK[labelsGood==i]]
            #                 ,self.XY[labelsGood==i,1][vectK[labelsGood==i]],c=color[i])

        ax[1,0].grid(color='k',linestyle='--',linewidth=0.1)
        ax[1,0].legend(['Low-Quality Samples','High-Quality Samples '],loc=0,fontsize='x-small')
        ax[1,0].set_title('Samples Quality')
        ax[1,0].set_xticklabels([])
        ax[1,0].set_yticklabels([])

        cax = ax[1,1].scatter(self.XY[:,0],self.XY[:,1],c=self.SilZ, vmin=-5, vmax=5,cmap='coolwarm')
        # cax = ax[1,1].scatter(self.XY[:,0][vectK],self.XY[:,1][vectK],c=self.SilZ[vectK], vmin=-5, vmax=5,cmap='coolwarm')
        ax[1,1].grid(color='k',linestyle='--',linewidth=0.1)
        ax[1,1].set_title('Z-score Silhouette Index')
        ax[1,1].set_xticklabels([])
        ax[1,1].set_yticklabels([])

        colorbar_ax = f.add_axes([0.95, 0.13, 0.02, 0.33])
        f.colorbar(cax, cax=colorbar_ax)
        # cbar = f.colorbar(cbaxes)

        if Save:
            if len(names)==0:
                plt.savefig('SamplesQuality' + Suffix + '.png',dpi=300,format='png')
            else:
                plt.savefig('SamplesQuality' + Suffix + '.png',dpi=300,format='png',bbox_extra_artists=(lgd,), bbox_inches='tight')


        plt.show()

    def displayInternal(self,Save=True,Suffix=''):
        '''
            Display Internal Indexes on all matrices
        '''
        width = 0.35 # the width of the bars
        fig, ax = plt.subplots(1,len(self.AllInternIndex.keys()),figsize=(len(self.AllInternIndex.keys())*15,int(self.displaytop*1.5)))

        for ik,Ikey in enumerate(self.AllInternIndex.keys()):
            values = np.array(self.Res[Ikey].values())
            keys = np.array(self.Res[Ikey].keys())
            idx = np.argsort(values)
            idx = idx[~np.isnan(values[idx])]

            # if Ikey=='Connectivity': idx = idx[::-1]

            idx = idx[-self.displaytop:]
            sorted_values = values[idx]
            sorted_keys = keys[idx]

            # if Ikey=='log10CalinskiHarabaz':sorted_values[sorted_values<0]=0 #Remove negative values for the dipslay

            ind = np.arange(len(idx))

            # for kk,key in enumerate(sorted_keys):
            ax[ik].barh(ind, sorted_values)

            # if set_log == True: ax[ik].set_xlabel('log10')

            ax[ik].grid(color='k',linestyle='--',linewidth=0.1)
            ax[ik].set_yticks(ind)
            ax[ik].set_yticklabels([self.Names[n] for n in sorted_keys])
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

    def computeDistanceMatrix(self,matrix,labels,PCA,Distance,Kernel):
        '''
            Return a distance matrix from
        '''
        Binary = self.check_if_binary_matrix(matrix)

        if PCA not in self.PCA.keys():
            print ('PCA key doesn t exist')
            print ('Possible Values')
            print (self.print_all_distances())
            raise ValueError

        if Distance not in self.AllDistances.keys():
            print ('Distance key doesn t exist')
            print ('Possible Values')
            print (self.print_all_distances())
            raise ValueError

        if Kernel not in self.AllKernels.keys():
            print ('Kernel key doesn t exist')
            print ('Possible Values')
            print (self.print_all_distances())
            raise ValueError

        PMat = self.PCA[PCA](matrix=matrix,binary=Binary,NR = 10)
        PDMat = self.AllDistances[Distance](PMat)
        PDKMat = self.AllKernels[Kernel](PDMat,labels)

        return PDKMat

    def displayNamesScatter(self,Save=True,Suffix='',numdisplay=20,
                            IkeyX='Silhouette',IkeyY='Connectivity',ImgSize=7,dispNoPCA=True,dispPCA=True):
        '''
            Display Internal Indexes on all matrices
            Ikey = Silhouette, Connectivity
            numdisplay: Number of distances to display
        '''
        fig, ax = plt.subplots(1,1,figsize=(1.5*ImgSize,ImgSize))

        # keys = self.SortedPDKkey['Sum'][:numdisplay]
        # keys = np.hstack([self.SortedPDKkey[Ikey] for Ikey in self.Res.keys()])
        keys = self.SortedPDKkey['Sum']
        #Remove doublon
        kU,idx = np.unique(keys,return_index=True)
        idxU = np.argsort(idx)
        keys = kU[idxU]

        NamesSimple = { 'City Block':'CB','Cosine':'Cos','Euclidean':'E','L2':'L2',
                        'Bray Curtis':'BC', 'Canberra':'Can', 'Chebyshev':'Ch',
                        'Pearson':'P', 'Hamming':'H', 'Squared Euclidean':'SE',
                        'Dice':'D', 'Jaccard':'J','Rogers Tanimoto': 'RS',
                        'Russell Rao':'RR', 'Sokal Michener':'SM',
                        'Sokal Sneath':'SS', 'Yule':'Y',
                        'Mutual Information': 'MI','Spearman': 'SP',
                        'Cosine Kernel':'CK','Sigmoid Kernel':'SK',
                        'Exponential Kernel':'EK','None':'N'}

        keysPCA = [key for key in keys if key.split('-')[0]=='PCA']
        keysNoPCA = [key for key in keys if key.split('-')[0]=='None']

        if dispNoPCA==True:
            if dispPCA==False:
                keysNoPCA = keysNoPCA[:numdisplay]
            else:
                keysNoPCA = keysNoPCA[:numdisplay//2]
            X = np.array([self.Res[IkeyX][key] for key in keysNoPCA])
            Y = np.array([self.Res[IkeyY][key] for key in keysNoPCA])
            ax.plot(X,Y,'o',label='Without PCA')
            #ax.set_xlim([-0.2,0.5])
            #ax.set_ylim([0,1.1])

        if dispPCA==True:
            if dispNoPCA==False:
                keysPCA = keysPCA[:numdisplay]
            else:
                keysPCA = keysPCA[:numdisplay//2]

            X = np.array([self.Res[IkeyX][key] for key in keysPCA])
            Y = np.array([self.Res[IkeyY][key] for key in keysPCA])
            ax.plot(X,Y,'or',label='With PCA')
            #ax.set_xlim([0.05,0.25])
            #ax.set_ylim([0.65,0.85])

        #----------------------------------
        #TEMPORAIRE POUR PAPIER
        # texts = [ax.text(self.Res[IkeyX][key], self.Res[IkeyY][key], "-".join([NamesSimple[ks] for ks in key.split('-')[1:]])) for key in keys]
        texts = []
        if dispNoPCA==True:
            texts = texts + [ax.text(self.Res[IkeyX][key], self.Res[IkeyY][key],NamesSimple[key.split('-')[1]],size='large') for key in keysNoPCA]

        if dispPCA==True:
            texts = texts + [ax.text(self.Res[IkeyX][key], self.Res[IkeyY][key],NamesSimple[key.split('-')[1]],size='large') for key in keysPCA]
        #----------------------------------
        adjust_text(texts)

        ax.grid(color='k',linestyle='--',linewidth=0.1)
        ax.set_xlabel(IkeyX)
        ax.set_ylabel(IkeyY)

        NamesDesc = '\n'.join([ 'Distances\n',
                    'CB: City Block','Cos: Cosine','E: Euclidean','L2: L2',
                    'BC: Bray Curtis', 'Can: Canberra', 'Ch: Chebyshev',
                    'P: Pearson', 'H: Hamming', 'SE: Squared Euclidean',
                    'D: Dice', 'J: Jaccard', 'RS: Rogers-Tanimoto',
                    'RR: Russell-Rao', 'SM: Sokal-Michener','SP: Spearman',
                    'SS: Sokal-Sneath', 'Y: Yule','MI: Mutual Information',
                    '\n\nKernels:\n',
                    'CK: Cosine Kernel','SK: Sigmoid Kernel',
                    'EK: Exponential Kernel',
                    '\nN: None'])

        #txt = ax.text(0.75, 0.1,NamesDesc, transform=plt.gcf().transFigure)
        lgd = ax.legend(bbox_to_anchor=(1.08, 1), loc=2, borderaxespad=0.)
        if Save:
            plt.subplots_adjust(right=0.7)
            plt.savefig('NamesScatter' + Suffix + '.png',dpi=300,format='png',bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.show()

    def WriteDistanceScore(self,Suffix=''):
        '''
            Write the indexes scores of all distances in a file. One file per index.
        '''
        f = open('Optimal_Distance'+Suffix+'.txt','w')
        keys = np.array(self.Res['Silhouette'].keys())
        idx = np.argsort(np.hstack([self.Res['Silhouette'][key] for key in keys]))[::-1]
        keys = keys[idx]

        f.write('\t'.join(['Rank','Distance Measure'] + self.AllInternIndex.keys() ) +'\n')
        for i,key in enumerate(keys):
            f.write('\t'.join([str(i),key] + [str(self.Res[Ikey][key]) for Ikey in self.AllInternIndex.keys()] )+'\n')
        f.close()

    def ClustIndexFilt(self,distmat,labels,Ikey):

        #Sort profiles by their silhouette index
        SilZ = ZscoreSilhouetteQuality(distmat,labels,Tz=0,NR=100)[1]
        idxS = np.argsort(SilZ)
        idxThreshold = np.where(SilZ[idxS]>1.65)[0][0] #Index of 95% filtering
        NS = len(idxS)//10 - 1

        if Ikey!='Silhouette':
            clust_index_filt_evol = np.zeros(NS)
        else:
            clust_index_filt_evol = np.ones(NS) * (-1)

        for i in range(NS):
            try:
                idx = idxS[i*10:]
                distmatC = distmat[np.ix_(idx,idx)]
                labelsC = labels[idx]

                labels_tmp = labelsC.copy()
                for k,l in enumerate(np.unique(labelsC)):
                    labels_tmp[labelsC==l] = k
                labelsC = labels_tmp

                if Ikey=='Silhouette':
                    clust_index_filt_evol[i] = silhouette_score(distmatC,labelsC,metric='precomputed')
                if Ikey=='Connectivity':
                    clust_index_filt_evol[i] = Connectivity(distmatC,labelsC)
                if Ikey=='log10CalinskiHarabaz':
                    clust_index_filt_evol[i] = log10CalinskiHarabaz(distmatC,labelsC)

            except Exception as e:
                print(e)
                continue

        #Compute the associated score
        if Ikey!='Silhouette':
            clust_index_filt_score = clust_index_filt_evol.mean()
        else:
            clust_index_filt_score = (clust_index_filt_evol+1).sum()/(2.*NS)

        return clust_index_filt_evol,clust_index_filt_score,idxThreshold

    def WriteClustIndexFiltScore(self,Suffix=''):
        '''
            Write the filtered clustering index score of all distances in a file. One file per index.
        '''
        for Ikey in self.AllInternIndex.keys():
            print( Ikey)
            idx = np.argsort(self.clust_index_filt_score[Ikey].values())[::-1]
            keys = np.array(self.clust_index_filt_score[Ikey].keys())[idx]
            f = file('Filt_index_'+Ikey+Suffix+'.txt','w')
            f.write('\t'.join(['Distance','Filtered Clustering Index','Number profiles removed at Zscore '+Ikey+' 1.65'])+'\n')
            for key in keys:
                print (key,self.clust_index_filt_score[Ikey][key],self.clust_index_filt_threshold[Ikey][key])
                f.write('\t'.join([key,str(self.clust_index_filt_score[Ikey][key]),str(self.clust_index_filt_threshold[Ikey][key])])+'\n')
            f.close()

    def DisplayClustFiltIndexesEvolution(self,Suffix=''):
        '''
            Display the clustering indexv evolution wen filtering
        '''
        NI = len(self.AllInternIndex.keys())
        f,ax = plt.subplots(1,NI,figsize=(NI*4,3))
        for i,Ikey in enumerate(self.AllInternIndex.keys()):
            idx = np.argsort(self.clust_index_filt_score[Ikey].values())[::-1]
            keys = np.array(self.clust_index_filt_score[Ikey].keys())[idx]

            N = len(self.clust_index_filt_evol[Ikey][key])
            X = np.arange(N-2)/float(N)*100
            for key in keys[:5]:
                ax[i].plot(X,self.clust_index_filt_evol[Ikey][key][:-2],label=key.split('-None')[0])


            #ax[i].set_ylim(ymin=0.45)
            ax[i].legend(fontsize='x-small')
            ax[i].set_xlabel('Percentage of profiles filtered')
            ax[i].set_ylabel(Ikey + ' index of the dataset')
            ax[i].grid(color='k',linestyle='--',linewidth=0.1)

        plt.tight_layout()
        plt.savefig('Clustering_Indexes_evolution_with_Filtering.png',dpi=300, bbox_inches='tight')
        plt.show()

#-----------------------------------------------------------------------
#           Internal Indexes
#-----------------------------------------------------------------------

def Connectivity(DistMat,labels,NumNeigh=5):
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

    connectivity = 0
    for i in range(Nsamples):
        class_i = labels[i]
        idx_sort = np.argsort(DistMat[i,:])[1:NumNeigh+1]
        position_NN_diff_class = np.where(labels[idx_sort]!=class_i)[0]
        for k in position_NN_diff_class:
            connectivity += (1./float(k+1)) #1/(position of the points in different classes)

    connectivity/=Nsamples
    connectivity/=np.sum([1/float(i) for i in range(1,NumNeigh+1)])

    return 1-connectivity


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
    #Nan append when 2 clusters has zeros intra and inter distance
    return s[~np.isnan(s)].mean()

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
    #CS = calinski_harabaz_score(Distmat, labels,metric='precomputed')
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
# from minepy import MINE
# def MIC(X):
#     '''
#     Mutual information from http://science.sciencemag.org/content/334/6062/1518
#     Optimize the 2D histogram binning before Mutual Information
#     Slower but more accurate
#     '''
#     mine = MINE(alpha=0.6, c=15)
#
#     Nsamples = X.shape[0]
#     MicVect = np.zeros((Nsamples,Nsamples))
#
#     for i in range(Nsamples):
#         for j in range(i,Nsamples):
#             mine.compute_score(X[i,:],X[j,:])
#             MicVect[i,j] = mine.mic()
#
#     MicVect = MicVect + MicVect.T - np.diag(np.diag(MicVect))
#
#     return 1-MicVect #MIC to Distance


# from sklearn.metrics import mutual_info_score
# def MI(X):
#     '''
#         Mutual information from
#         https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
#     '''
#     L = len(np.unique(X[0]))
#     mi = np.zeros((X.shape[0],X.shape[0]))
#
#     for i in range(X.shape[0]):
#         #Get Opitmal bins from scipy: (maximum of the Sturges and FD estimators)
#         if L>2:
#             bins_i = np.histogram(X[i], bins='auto')[1] #Other than binary
#         else:
#             bins_i = [0.,1.,2]
#
#         for j in range(i,X.shape[0]):
#             #Get Opitmal bins from scipy: (maximum of the Sturges and FD estimators)
#             if L>2:
#                 bins_j = np.histogram(X[j], bins='auto')[1]  #Other than binary
#             else:
#                 bins_j = [0.,1.,2.]
#
#             #2D histogram
#             c_xy = np.histogram2d(X[i], X[j], [bins_i,bins_j])[0]
#
#             mi[i,j] = mutual_info_score(None, None, contingency=c_xy)
#
#     mi = mi + mi.T - np.diag(np.diag(mi))
#
#     return mi

def MI(X):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    L = len(np.unique(X[0]))
    mi = np.zeros((X.shape[0],X.shape[0]))

    EPS = np.finfo(float).eps

    for i in tqdm(range(X.shape[0])):
        #Get Opitmal bins from scipy: (maximum of the Sturges and FD estimators)
        if L>2:
            bins_i = np.histogram(X[i], bins='auto')[1] #Other than binary
        else:
            bins_i = [0.,1.,2]

        for j in range(i,X.shape[0]):
            #Get Opitmal bins from scipy: (maximum of the Sturges and FD estimators)
            if L>2:
                bins_j = np.histogram(X[j], bins='auto')[1]  #Other than binary
            else:
                bins_j = [0.,1.,2.]

            jh = np.histogram2d(X[i], X[j], [bins_i,bins_j])[0]

            # compute marginal histograms
            jh = jh + EPS
            sh = np.sum(jh)
            jh = jh / sh
            s1 = np.sum(jh, axis=0)#.reshape((-1, jh.shape[0]))
            s2 = np.sum(jh, axis=1)#.reshape((jh.shape[1], -1))

            # Normalised Mutual Information of:
            # Studholme,  jhill & jhawkes (1998).
            # "A normalized entropy measure of 3-D medical image alignment".
            # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
            # if normalized:
            mi[i,j] = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                    / np.sum(jh * np.log(jh))) - 1
            # mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
            #        - np.sum(s2 * np.log(s2)))


    mi = mi + mi.T - np.diag(np.diag(mi))
    return 1 - mi
#########
#TODO
#########
#
# def MI_MP(X):
#     """
#     Computes (normalized) mutual information between two 1D variate from a
#     joint histogram.
#     Parameters
#     ----------
#     x : 1D array
#         first variable
#     y : 1D array
#         second variable
#     sigma: float
#         sigma for Gaussian smoothing of the joint histogram
#     Returns
#     -------
#     nmi: float
#         the computed similariy measure
#     """
#     L = len(np.unique(X[0]))
#     mi = np.zeros((X.shape[0],X.shape[0]))
#
#     EPS = np.finfo(float).eps
#     processes = []
#     recv_end_c, send_end_c = zip(*[Pipe(False) for k in range(X.shape[0])])
#
#     for k in range(X.shape[0]):
#         p = Process(target=GABI_fit_wrapper, args=(self.matrix_c[k],self.gb_c[k],send_end_c[k],k))
#         processes.append(p)
#         p.start()
#
#     try:
#         for process in processes:
#             process.join()
#
#         # gb = [recv_end.recv() for recv_end in recv_end_c]
#         self.gb_c = [recv_end.recv() for recv_end in recv_end_c]
#
#     except KeyboardInterrupt:
#         self.gb_c = [recv_end.recv() for recv_end in recv_end_c]
#
#
#     for i in tqdm(range(X.shape[0])):
#         #Get Opitmal bins from scipy: (maximum of the Sturges and FD estimators)
#         if L>2:
#             bins_i = np.histogram(X[i], bins='auto')[1] #Other than binary
#         else:
#             bins_i = [0.,1.,2]
#
#         for j in range(i,X.shape[0]):
#             #Get Opitmal bins from scipy: (maximum of the Sturges and FD estimators)
#             if L>2:
#                 bins_j = np.histogram(X[j], bins='auto')[1]  #Other than binary
#             else:
#                 bins_j = [0.,1.,2.]
#
#             jh = np.histogram2d(X[i], X[j], [bins_i,bins_j])[0]
#
#             # compute marginal histograms
#             jh = jh + EPS
#             sh = np.sum(jh)
#             jh = jh / sh
#             s1 = np.sum(jh, axis=0)#.reshape((-1, jh.shape[0]))
#             s2 = np.sum(jh, axis=1)#.reshape((jh.shape[1], -1))
#
#             # Normalised Mutual Information of:
#             # Studholme,  jhill & jhawkes (1998).
#             # "A normalized entropy measure of 3-D medical image alignment".
#             # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
#             # if normalized:
#             mi[i,j] = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
#                     / np.sum(jh * np.log(jh))) - 1
#             # mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
#             #        - np.sum(s2 * np.log(s2)))
#
#
#     mi = mi + mi.T - np.diag(np.diag(mi))
#     return 1 - mi
#########
#########

#Spearman Distance
from scipy.stats import spearmanr
def Spearman(X):
    rho, _ = spearmanr(X.T)
    return 1-rho

#-----------------------------------------------------------------------
#           Kernels
#-----------------------------------------------------------------------
def CosineKernel(X,labels=None):
    '''
        Return a distance Cosine Kernel
    '''
    # SqNorm = ((X[None,:,:]-X[:,None,:])**2).sum(axis=2)
    Norm = (X**2).sum()
    return 1 - X.dot(X.T)/Norm

def SigmoidKernel(X,labels):
    '''
        Return a distance Sigmoid Kernel
    '''

    gamma = get_gammaSig(X,labels)
    # gamma = 1.0 / X.shape[1]
    c0 = 1
    return 1 - np.tanh(gamma*X + c0)

def RBFKernel(X,labels,Distance=None):
    '''
        Return a distance RBF Kernel but with
        precomputed distance
    '''
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

    # dmin = dist_intra.min() + 1e-8
    dmin = dist_intra.mean() + 1e-8
    gamma = dmin/2.

    return gamma

def get_gammaSig(DistMat,labels):

    #Coordinates Matrix
    mds = MDS(n_components=DistMat.shape[0],dissimilarity='precomputed')
    CoordMat = mds.fit_transform(DistMat)

    labels = labels.astype(np.int32)
    idx_labels = [np.where(labels==l)[0] for l in range(labels.max()+1)]

    ClusterCenter = [CoordMat[idx,:].mean(axis=0) for idx in idx_labels]
    Center = CoordMat.mean(axis=0)

    # overall between-cluster variance
    SSB = sum([len(idx_labels[i])*np.sum((cc-Center)**2) for i,cc in enumerate(ClusterCenter)])
    gamma = 1./SSB

    return gamma

#-----------------------------------------------------------------------
#           PCA cleaning
#-----------------------------------------------------------------------
def PCAEmbedding(matrix,binary,NR = 10):
    '''
        Rerturn the matrix transformed by PCA.
        The number of components is optimized such as the eigenvalues of the
        matrix are stastistically superior to the eigenvalues of a random model.
        The random model is based on the reconstruction of a matrix

        matrix samples x Features
        binary: True/False if the matrix is binary
        NR: Number of random matrices
    '''
    M,N = matrix.shape
    pca = PCA(n_components=min(M-1,N-1))
    pca.fit(matrix)
    SV = pca.singular_values_


    pcaR = PCA(n_components=pca.n_components)
    SVR = np.zeros((NR,pcaR.n_components))
    for nr in tqdm(range(NR)):
        matrixR = RNGS.RandomNGS(matrix,binary=binary,seed=nr)
        pcaR.fit(matrixR)
        SVR[nr,:] = pcaR.singular_values_

    m = SVR.mean(axis=0)
    s = SVR.std(axis=0)
    SV[s==0]=0
    s[s==0]=1
    NCompOpt = np.max(np.where((SV-m)/s>2.32)[0])

    pca = PCA(n_components=NCompOpt)
    matrixPCA = pca.fit_transform(matrix)

    return matrixPCA
#-----------------------------------------------------------------------
#           Optimal TSNE
#-----------------------------------------------------------------------
def OptimalTSNE(Distmat,labels):
    '''

        Find Optimal TSNE based on the TSNE which produce the highest silhouette score.
        To compute the distance between the 2D profiles, we use Pearson distance,
        which has proven to be a good distance for continuous datasets
    '''
    print('Grid search optimal parameters for t-SNE')
    
    labelsC = labels.copy()
    for i,l in enumerate(np.unique(labels)):
        labelsC[labels==l] = i

    learning_rates = [10,100,1000]
    perplexitys = [5,10,30,50,100,300,500,1000]
    OptSil=-1

    for learning_rate in tqdm(learning_rates,desc='learning rates'):
        for perplexity in perplexitys:
        # for perplexity in tqdm(perplexitys,desc='perplexities'):
            model_tsne = manifold.TSNE(n_components=2, random_state=3,perplexity=perplexity,
                                       early_exaggeration=4.0, learning_rate=learning_rate,
                                       n_iter=100000, metric='precomputed')

            XY = model_tsne.fit_transform(Distmat)
            distmatTSNE = pairwise_distances(XY,metric='cosine')
            Sil = Silhouette(distmatTSNE,labelsC)
            if OptSil<Sil:
                OptSil = Sil
                Opt_model_tsne = model_tsne
                param_opt = [learning_rate,perplexity]

    print('Optimal (learning rate,perplexity) = ({0},{1})'.format(param_opt[0],param_opt[1]))
    XY = Opt_model_tsne.fit_transform(Distmat)

    return XY

#-----------------------------------------------------------------------
#           Get Outliers defined by their number of reads
#-----------------------------------------------------------------------

def kde_sklearn(reads, bandwidth = 1000, **kwargs):
    """
    Kernel Density Estimation with Scikit-learn :
    Estimate the density from the reads distribution
    Entry :
        - reads, as the number of reads per bin
        - bandwith : width of the gaussien parameter (can be modified)
    Output :
        - returns the density as a matrix for each point from 0 to max(reads)
    """
    x_grid = np.linspace(0, np.max(reads), int(np.max(reads)) + 1)
    kde_skl = KernelDensity(kernel = 'gaussian', bandwidth=bandwidth, **kwargs)
    kde_skl.fit(reads)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    res = np.exp(log_pdf) / np.exp(log_pdf).sum()
    return res

def kde_outliers(reads, threshold = [0.01, 0.99], w = 1000,Display=True):
    """
    outliers as given by the kernel density estimator
    Entry :
        - reads : the number of reads per bin
        - threshold : list of [threshold min, threshold max].
    Below threshold min and above threshold max,
    bins are considered as outliers
    If threshold = float between 0 and 1, consider the threshold x and 1-x
    - w : bandwith of the gaussian window (as needed in kde_sklearn)
    Output :
        - positions of outliers : array of bool where True = outlier, False = to be kept
        - limits of outliers as a tuple (limit_below, limite_above)
        - density as estimated by kde_sklearn function
    """
    # Before everything, 0 bins have to be cleaned to evaluate density properly
    # We will then work on reads[reads > 0]
    cleaned = reads[reads > 0]

    # We need first the density estimation
    density = kde_sklearn(cleaned[:, None], bandwidth = w)

    # Then we calculate the cumulative sum
    cum_sum = np.cumsum(density)

    # We can know find the limits
    if type(threshold) == float and threshold < 1 and threshold > 0 :
        threshold = [threshold, 1 - threshold]

    limit_below = np.abs(cum_sum - threshold[0]).argmin()
    limit_above = np.abs(cum_sum - threshold[1]).argmin()

    pos_out = (reads < limit_below) + (reads > limit_above)
    idxRm = np.where(pos_out)[0]
    idxK = np.where(1-pos_out)[0]

    if Display:
        #Display result

        plt.hist(reads,100)
        plt.title('Distribution of the number of reads per samples')
        plt.show()

        print( 'Lower threshold = ' + str(limit_below))
        print( 'Upper threshold = ' + str(limit_above))
        print( 'Number of removed samples = {0} / {1}'.format(len(idxRm),len(pos_out)))

    # Now use reads again to add 0 bins to the outliers
    return idxRm,idxK #,(limit_below, limit_above), density


def get_outliers(matrix, threshold = [0.01, 0.99]):
    """
    Given a mat
    return idx to remove and idx to keep

    """
    # Position of outliers : limitis and density
    #print(mat.sum(axis = 0))
    idxRm,idxK = kde_outliers(matrix.sum(axis=1), threshold = threshold, w = 2000)

    return idxRm,idxK

#-----------------------------------------------------------------------
#           Miscellaneous
#-----------------------------------------------------------------------
def Identity(X,*args,**kwargs):
    '''
        return the key X only
    '''
    return X


# #-----------------------------------------------------------------------
# #           Stand-Alone Quality Tester
# #-----------------------------------------------------------------------
#
# def compute_samples_quality_from_distmat(matrix,distmat,labels,names=[],NR=100,Tz=0,Nkeep=-1,NPeaksMin=[]):
#     '''
#         Compute the quality of each profiles according to the given distance matrix
#         This function is to use when the optimal distance measure is known
#
#         The quality is computed as the Zscore of the silouette score of each samples
#         compared to random cluster attribution
#
#         inputs:
#         Tz: Threshold on the zscore
#         NR: Number of random shuffling
#         Nkeep: Number of profiles to keep maximum in each clusters
#         (if -1 all good samples are kept)
#         names (optional): numpy array with names of the cluster (for the final display)
#         NpeaksMin: (vector) Number of peaks minimum in a profile to pass the test
#                    (if -1, the median of the number of peaks per clusters divided by 4
#                    is taken)
#
#         Output:
#         IdxGoodSamples: indexes of the samples which pass the test
#         SilZ: Silhouette Zscore of the samples
#         SilZClust: Mean of the silhouette zscore over the clusters
#
#     '''
#
#     IdxGoodSamples,SilZ = ZscoreSilhouetteQuality(distmat,labels,Tz,NR)
#
#     SilZClust = np.array([SilZ[np.logical_and(labels==l,SilZ>Tz)].mean() for l in np.unique(labels)])
#
#     NPeaks = matrix.sum(axis=1)
#     if len(NPeaksMin)==0:
#         NPeaksMin = np.ones(matrix.shape[0])*np.median(NPeaks)/4.
#         print 'Minimal number of peaks for a sample: {}'.format(NPeaksMin[0])
#
#     IdxGoodSamples = []
#     SilZClust = []
#     for l in np.unique(labels):
#         idx = np.where(np.logical_and(labels==l,SilZ>Tz))[0]
#         idx = idx[NPeaksMin[idx]<NPeaks[idx]]
#
#         if len(idx>0):
#             idx = idx[np.argsort(SilZ[idx])]
#             if Nkeep>0: idx = idx[-Nkeep:]
#
#
#             IdxGoodSamples.extend(idx)
#             SilZClust.append(SilZ[idx].mean())
#
#     IdxGoodSamples = np.array(IdxGoodSamples)
#     SilZClust = np.array(SilZClust)
#
#     #Displays
#     N = len(SilZ)
#     Nk = len(IdxGoodSamples)
#     Nrm = N - Nk
#     print ''
#     print '# Dataset Total: ' + str(N)
#     print '# Dataset Kept: {0} ({1} %)'.format(Nk,round(Nk/float(N)*100,1))
#     print '# Dataset Removed: {0} ({1} %)'.format(Nrm,round(Nrm/float(N)*100,1))
#
#     uniq,count = np.unique(labels[IdxGoodSamples],return_counts=True)
#     idx = np.argsort(uniq)
#     uniq = uniq[idx]
#     count = count[idx]
#
#     if len(names)==0:
#         names = np.array(['' for _ in np.unique(labels)])
#
#     print '\nKept Datasets'
#     print 'Labels\t\tCounts\t\tName\t\tMean Zscore Silhouette'
#     for l,n,c,sz in zip(uniq,names[uniq],count,SilZClust):
#         print '{0}\t\t{1}\t\t{2}\t\t{3}'.format(l,c,n,sz)
#
#     labelsRm = np.array([l for l in np.unique(labels) if l not in uniq])
#     if len(labelsRm)>0:
#         labelsRm = np.sort(labelsRm)
#         print '\nRemoved Datasets'
#         print 'Labels\t\tName'
#         for l,n in zip(labelsRm,names[labelsRm]):
#             print '{0}\t\t{1}'.format(l,n)
#
#
#     return IdxGoodSamples,SilZ,SilZClust
