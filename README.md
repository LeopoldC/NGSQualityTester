#Package needed
In order to use this repository, you need in python 2:
-scikit-learn >= 0.20 
-numpy
-scipy
-hdf5
-matplotlib
-tdqm

# NGSQualityTester

## How to use:
```
import QualityTester as QT

#matrix=Samples x Features
#labels=Samples : Integers corresponding to cluster assignement

qt = QT.QualityTester(Binary=False) #Binary=True  if dataset is binary. By default binary == false
qt.get_optimal_distance(matrix=np.transpose(matrix),labels=label,threshold = [0.05, 0.95])


```
