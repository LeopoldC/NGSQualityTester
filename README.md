# NGSQualityTester

## Some advise


## How to use the code:
```
import QualityTester as QT

#matrix=Samples x Features
#labels=Samples : Integers corresponding to cluster assignement

qt = QT.QualityTester()  
qt.get_optimal_distance(matrix=np.transpose(matrix),labels=label,threshold = [0.05, 0.95])


```
