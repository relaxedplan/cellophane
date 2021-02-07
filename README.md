# Partial Identification Sets - algorithmic bias auditing when the protected class is unobserved 

A practical challenge for assessing disparity along protected class lines in algorithmic systems is that **protected class membership is often not observed in the data**.

To address this challenge, various methods have been developed to impute the protected class using proxies found in the original dataset. The most famous of these is BISG, which uses surname and geolocation to predict race.

These methods are controversial for various socio-technical and statistical reasons. The main statistical challenge
 is that the estimation of protected class membership from proxies is uncertain and subject to ad-hoc
  modelling choices. 
  
  Bias metrics calculated using these estimation without taking into account this uncertainty will be fundamentally spurious point estimates. 

This auditing package implements algorithms described [here](https://arxiv.org/pdf/1906.00285.pdf) to provide meaningful estimates of disparity, taking into account the uncertainty of estimation. Instead of generating point estimates, it generates a *range* of all possible disparities, known as a partial identification set.
   
## Visualization

In addition to calculating partial identification sets, his package contains plotting functions to easily visualize partial identification sets:

![image info](https://i.ibb.co/DLzB7Ws/download.png)

## Demo

A short demo can be viewed [here](https://github.com/relaxedplan/partial-identification-sets/blob/master/demo.ipynb)

## Docs

Documentation can be viewed [here](https://htmlpreview.github.io/?https://raw.githubusercontent.com/relaxedplan/partial-identification-sets/master/html/fairness/PartialIdentification.html)
