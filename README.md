# STD-GAE

The fast growth of the global Photovoltaic (PV) market enables large-scale PV data analytical pipelines for power forecasting and long-term reliability assessment of PV fleets. Nevertheless, the performance of PV data analysis heavily depends on the quality of PV timeseries data. 

This paper proposes a novel Spatio-Temporal Denoising Graph Autoencoder (STD-GAE) framework, to impute missing PV Power Data. STD-GAE exploits temporal correlation, spatial coherence and value dependencies from domain knowledge to recover missing data. It is empowered by two modules. 
  (1) To cope with sparse yet various scenarios of missing data, STD-GAE incorporates a domain-knowledge aware data augmentation module that creates             plausible variations of missing data patterns. This generalizes STD-GAE to robust imputation over different seasons and environment. 
  (2) STD-GAE nontrivially integrates spatiotemporal graph convolution layers (to recover local missing data by observed “neighboring” PV plants) and             denoising autoencoder (to recover corrupted data from augmented counterpart) to improve the accu- racy of imputation accuracy at PV fleet level. 

Using large-scale real data over 98 PV systems, our experimental study shows that STD-GAE achieves a gain from 8.38% to 45.44% in accuracy (MAE), and remains less sensitive to missing rate, different seasons and missing scenarios, compared with state-of-the-art data imputation methods such as MIDA and LRTC-TNN.

![Framework (9)](https://user-images.githubusercontent.com/47265586/163913110-c53052d1-e4b0-4757-b110-8788b6bb1442.png)
