# NGSQualityTester

## How to use:
```
import QualityTester as QT

#matrix=Samples x Features
#labels=Samples : Integers corresponding to cluster assignement

qt = QT.QualityTester(Binary=False) #Binary=True  if dataset is binary
qt.compute_distances(matrix=matrix,labels=labels,ComputeMDS=True)
suff = '_Cont' #Suffix for the name of the images
qt.displayInternal(Suffix=suff)
qt.displayMDS(Suffix=suff)
qt.displayNamesScatter(Suffix='_Cont')

IdxGoodSamples = qt.compute_samples_quality(matrix=matrix,NR=100)

qt.display_samples_quality(Suffix='_Cont')

```
