# Automatic thresholding to deal with out-of-distribution data

## Short introduction
![Visual abstract](https://github.com/isaacbmt/OOD_filter/blob/master/img/system.png)
Semi-supervised learning (SSL) leverages both labeled and unlabeled data for training models when the labeled data is limited and the unlabeled data is vast. Frequently, the unlabeled data is more widely available than the labeled data, hence this data is used to improve the level of generalization of a model when the labeled data is scarce. However, in real-world settings unlabeled data might depict a different distribution than the labeled dataset distribution. This indicates that the labeled and the unlabeled dataset have distribution mismatch between them. Such problem generally occurs when the source of unlabeled data is different from the labeled data. For instance, we use the medical imaging domain in applications such as the COVID-19 detection using chest X-ray images to assess the impact of distribution mismatch between the labeled and unlabeled dataset. In this work, we propose an automatic thresholding method to filter the out-of-distribution (OOD) in the unlabeled dataset. The filter is implemented using a Gaussian distribution based thresholding method and a clustering algorithm. As a further step, we propose a method to dynamically discern if it is convenient to threshold the observations. Moreover, the proposed filtering method can be used to improve the data quality in a semi-supervised environment.
