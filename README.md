# STD-GAE

We propose a novel Spatio-Temporal Denoising Graph Autoencoder (STD-GAE) framework, to impute missing PV Power Data. STD-GAE exploits temporal correlation, spatial coherence and value dependencies from domain knowledge to recover missing data. It is empowered by two modules. (1) To cope with sparse yet various scenarios of missing data, STD-GAE incorporates a domain-knowledge aware data augmentation module that creates plausible variations of missing data patterns. This generalizes STD-GAE to robust imputation over different seasons and environment. (2) STD-GAE nontrivially integrates spatiotemporal
graph convolution layers (to recover local missing data by observed “neighboring” PV plants) and denoising autoencoder (to recover corrupted data from augmented counterpart) to improve the accuracy of imputation accuracy at PV fleet level.

![Framework](https://user-images.githubusercontent.com/47265586/163632951-93208446-da6b-4609-b675-b40754cfc03b.png)
