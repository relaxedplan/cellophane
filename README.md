# Overview

A practical challenge for assessing disparity along protected class lines in algorithmic systems is that **protected class membership is often not observed in the data**.

To address this challenge, various methods have been developed to impute the protected class using proxies found in the original dataset. The most (in)famous of these is [BISG](https://github.com/cfpb/proxy-methodology), which uses surname and geolocation to predict race.

These methods are controversial for various socio-technical and statistical reasons. The main statistical challenge is that the estimation of protected class membership from proxies is uncertain and subject to ad-hoc modelling choices. 

Bias metrics calculated using proxy-based estimations without taking into account uncertainty will be fundamentally spurious point estimates. Conclusions reached using these point estimates will be vulnerable to challenge.

This auditing package implements algorithms described [here](https://arxiv.org/pdf/1906.00285.pdf) to provide meaningful estimates of disparity, taking into account the uncertainty of estimation. Instead of generating point estimates, it generates a *range* of all possible disparities, known as a partial identification set. A tight set will allow for robust conclusions even though the protected class membership wasn't observed in the primary set. A wide set generally means that the proxies aren't informative enough to draw conclusions.

## Visualization

In addition to calculating partial identification sets, his package contains plotting functions to easily visualize partial identification sets:

![image info](https://i.ibb.co/DLzB7Ws/download.png)

## Installation
To install cellophane, clone, unpack it, and:

```$ python setup.py install```

A pip package is coming shortly.

## Demo

A short demo can be viewed [here](https://github.com/relaxedplan/partial-identification-sets/blob/master/demo/demo.ipynb)

## Docs

Documentation can be viewed [here](https://htmlpreview.github.io/?https://raw.githubusercontent.com/relaxedplan/cellophane/master/html/cellophane/cellophane.html)

Understanding the underlying algorithms is important. Read about them at [Kallus, Mao and Zhou (2020)](https://arxiv.org/pdf/1906.00285.pdf).
