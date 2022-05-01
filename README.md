# Spatio-Temporal Denoising Graph Autoencoder (STD-GAE)

This repository contains the code for the reproducibility of the experiments presented in the paper "Spatio-Temporal Denoising Graph Autoencoders with Data Augmentation for Photovoltaics (PV) Data Imputation". In this paper, we propose a novel Spatio-Temporal Denoising Graph Autoencoder (STD-GAE) framework for PV timeseries data imputation and achieve state-of-the-art results on real-world PV benchmarks.


<h2 align=center>STD-GAE in a nutshell</h2>

Our paper introduces __STD-GAE__, a method and an architecture that exploits temporal correlation, spatial coherence, and value dependencies from domain knowledge to recover missing data. STD-GAE features domain-knowledge aware data augmentation module to create plausible variations of missing data patterns and integrates spatiotemporal graph convolution layers (to recover local missing data by observed “neighboring” PV plants) and denoising autoencoder (to recover corrupted data from augmented counterpart) to improve the accuracy of imputation accuracy at PV fleet level

<p align=center>
  <a href="https://github.com/marshka/sinfony">
    <img src="./grin.png" alt="Logo"/>
  </a>
</p>


![Framework (9)](https://user-images.githubusercontent.com/47265586/163913110-c53052d1-e4b0-4757-b110-8788b6bb1442.png)
