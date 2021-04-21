# MIMIC-3 Benchmark data: analysing the results of the phenotyping task

This repository aggregates the results of the phenotyping prediction task described in [1]. 

The results stem from running the complete Benchmark model for the phenotype prediction task, 
available in the study's [public Github repository](https://github.com/YerevaNN/mimic3-benchmarks).

get_stats.py:
<br>formats phenotype prediction task results so that each of 25 care conditions
is shown in tabular format with its prevalence within the patient data (test and train splits), 
its type (chronic, acute, or both) and the AUC-ROC score it received from the Benchmark evaluation. 

This tabular format gives us an overview of the relationship (if any) between prevalence, type and 
predictive ability concerning individual care conditions.

get_patient_stats.py
<br>simply sums up the number of patients for each split: train, test and evaluation.

## References

[1] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9