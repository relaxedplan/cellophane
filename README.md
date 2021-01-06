# unaware-audit
A practical challenge for assessing disparity along protected class lines in algorithmic systems is that **protected class membership is often not observed in the data**.

To address this challenge, various methods have been developed to impute the protected class using proxies found in the original dataset. The most famous of these is BISG, which uses surname and geolocation to predict race.

These methods are controversial for various socio-technical and statistical reasons. The main statistical challenge
 is that the imputation of protected class membership from proxies is often unreliable and subject to ad-hoc
  modelling choices. 
  
  Bias metrics calculated using these imputations without taking into account this unreliability will be fundamentally spurious point estimates.

This auditing package is designed to generate disparity assessments in a binary classification context. It takes into
 account the uncertainty of the proxy model to generate a *range* of possible disparities rather than a point using
  methods described at https://arxiv.org/pdf/1906.00285.pdf. These ranges are described as "partial identification
   sets".

For auditing purposes, this package contains plotting functions to easily visualize partial identification sets:


![image info](https://i.ibb.co/DLzB7Ws/download.png)
