---
title: "**xgbstack**: Estimating weighted density ensembles for prediction "
author: |
   | Evan L Ray^1,2^ and Nicholas G Reich^1^
   |
   | 1. University of Massachusetts-Amherst
   | 2. Mount Holyoke College
output: html_document
bibliography: biblioExample.bib
nocite: | 
  @ref2, @ref3
---

**Keywords**: Ensemble methods, predictive analytics, biostatistics, epidemiology

**Webpages**: https://github.com/reichlab/xgbstack, http://reichlab.io/flusight/

Ensemble methods are commonly used in predictive analytics to combine predictions from multiple distinct models. A large portion of the literature on prediction focuses on classification problems, with less work devoted to creating predictive distributions of a continuous outcome. One fairly straight-forward way to combine predictive densities is to create a weighted average of them. This is sometimes refered to as "stacking". However, estimating optimal model weights for stacked density ensembles is not a straight-forward task. Using gradient boosting, we have developed and implemented in the package **xgbstack** a method to estimate these model weights potentially as a function of the observed data or the predictive densities themselves. Additionally, the method provides flexible options for regularization of weights to prevent overfitting. We present an application of this method to an weighted density ensemble for three component models that can be used to predict measures of influenza season timing and severity in the United States, both at the national and regional levels. In an out-of-sample test phase of prediction, the ensemble methods showed overall performance that was similar to the best of the component models, but offered more consistent performance across seasons. The **xgbstack** package is currently available on GitHub, with additional features under active development, including the incorporation of additional loss-functions.

# References
