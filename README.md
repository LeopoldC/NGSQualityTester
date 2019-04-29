# NGSQualityTester

## Some advise
For biological genomic dataset (Dnase-seq, ATAC-seqe, Chip-seq) :
-If your data came from a peaks caller consider it as a binary signal : 1 if there is a peaks on the region, 0 not.
-else : If your data are bed generated from mapping file, use it like this to generate you matrix


You should consider to have at least 4 replicate by condition.

The program check if data is binary or not and apply aprooved measure.




## How to use the code:
```
import QualityTester as QT

#matrix=Samples x Features (pics value on a genomic region at a given resolution, us : 200pb)
#labels=Samples : Integers corresponding to cluster assignement

qt = QT.QualityTester()  
qt.get_optimal_distance(matrix=np.transpose(matrix),labels=label,threshold = [0.05, 0.95])


```
