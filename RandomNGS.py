import numpy as np

def RandomNGS(matrix, binary,seed):
    '''
        Provide NR randomize versions of the matrix, keeping the probability of counts over
        the lines and colums.

        matrix: samples x features
        binary = True/False If the dataset is binary, to avoid values more than 1
        NR Number of repeats
    '''

    M,N = matrix.shape
    S = int(matrix.sum())
    probamat = proba_matrix(matrix)

    matrixR = np.zeros((M,N))
    np.random.seed(seed)

    if binary==False:
        idxR = np.random.choice(a=np.arange(M*N), size=S, p=probamat)
        matrixR = np.zeros(M*N)
        matrixR[idxR] = 1
        matrixR = matrixR.reshape(M,N)
    else:
        matrixR[:,:] = random_matrix_binary(matrix,probamat)

    return matrixR


def proba_matrix(matrix):
    '''
        Define the probability of each (i,j) case based on the probability of
        the lines and the columns
    '''
    M,N = matrix.shape
    SI = np.sum(matrix,axis=0).astype(np.float32)
    SJ = np.sum(matrix,axis=1).astype(np.float32)
    ST = np.sum(matrix).astype(np.float32)
    probamat = ((SI[np.newaxis,:]*SJ[:,np.newaxis])/ST**2).reshape(M*N)
    return probamat


def random_matrix_binary(matrix,probamat):
    '''
        Prepare a random matrix based on the probability matrix
    '''
    M,N = matrix.shape
    ST = np.sum(matrix)

    #-----------
    #First Pass: Choosing an ensemble of sites (i,j) in the matrix based on the probability of
    # having a (i,j) site in probamat
    #----------

    idx_matrix_R = np.random.choice(a=np.arange(M*N), size=int(ST), p=probamat)

    #-----------
    #Second Pass: Removing iteratively sites (i,j) already picked in order to
    # get as many sites as in the original matrix ( = ST )
    #-----------

    idx_matrix_R = np.unique(idx_matrix_R)

    #looping until there is no doubling index
    while len(idx_matrix_R) < ST-1:
        # print 'Remaining Index to pick = ' + str(ST-len(idx_matrix_R))
        
        #Remaining Index to pick
        idx_remaining = np.arange(M*N)
        idx_remaining[idx_matrix_R] = 0

        probamatC = probamat[idx_remaining>0] #All except idx_non_reccurent
        idx_remaining = idx_remaining[idx_remaining>0]

        idx_matrix_R = np.hstack((np.random.choice(a=idx_remaining, size=int(ST - len(idx_matrix_R)), p=probamatC/probamatC.sum()),idx_matrix_R))

        idx_matrix_R = np.unique(idx_matrix_R)


    matrix_R = np.zeros(M*N)
    matrix_R[idx_matrix_R]=1

    #Go back to the original form
    matrix_R = matrix_R.reshape(M,N)

    return matrix_R
