# Representation Learning of Electronic Healthcare Records for Downstream Prediction Tasks: Comparing Deep Techniques

This repository contains code and explanations for the comparison of a phenotype prediction task using three 
different representations of Electronic Health data, all created with deep learning techniques.

The baseline comes from [1], a study proposing four predictive tasks as benchmarks for comparing different learned 
representations of structured EHR data. I compared the results of their phenotype prediction task on a similar
task for two other EHR representations: visit representations produced by Med2Vec [2] and patient representations 
produced by ConvAE [3]. 

The Benchmark study uses variants of LSTM to create patient representations that retain temporal information, and
includes not only demographics and diagnoses, but items from chart and lab events as well. Their phenotype prediction
task-specific dataset predicts for the presence or absence of 25 selected care conditions (groups of related diagnoses)
in the final sequence for a patient, given all previous sequences for that patient.

Med2Vec uses a multi-layer perceptron and Skip-Gram to learn interpretable dense vectors of patient visits, 
represented as demographic data and co-occurring diagnosis codes within temporally-ordered sequences of visits 
per patient. Their initial phenotype prediction task consists of predicting (grouped) diagnoses of a visit given
preceding visit codes within a specified window of time.

ConvAE uses Convolutional Neural Networks and Stacked AutoEncoders on structured EHR data, and NLP techniques on 
unstructured clinical notes to create general-purpose dense-vectors of temporally-ordered patient information.
The original study doesn't present a phenotype prediction task; however, they claim that their all-purpose
representations of patients can be used for such a task.

The repository is split into one directory per study, each of which contains my own code (some of which is code that
I corrected from publicly available repositories for the studies) and notes on how to reproduce the results. 
The common_resources directory contains mappings used by all the models.

This repo is part of my final project for "Deep Learning for Healthcare" for the 
Spring 2021 semester of my Masters degree in Data Science at the University of Illinois Urbana-Champagne.

## References

[1] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9

[2] Edward Choi, Mohammad Taha Bahadori, Elizabeth Searles, Catherine Coffey, Jimeng Sun
<br>Multi-layer Representation Learning for Medical Concepts
<br>https://arxiv.org/pdf/1602.05568.pdf

[3] Landi, I., Glicksberg, B. S., Lee, H. C., Cherng, S., Landi, G., Danieletto, M., Dudley, J. T., Furlanello, C., & Miotto, R. 
<br>Deep representation learning of electronic health records to unlock patient stratification at scale. npj Digit. Med. 3, 96 (2020).
<br>https://www.nature.com/articles/s41746-020-0301-z


