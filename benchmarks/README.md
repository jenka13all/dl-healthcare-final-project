# MIMIC-3 Benchmark data: analysing the results of the phenotyping task

This repository aggregates the results of the phenotyping prediction task described in [1]. 

The results stem from running the complete Benchmark model for the phenotype prediction task, 
available in the study's [public Github repository](https://github.com/YerevaNN/mimic3-benchmarks).

To reproduce the evaluation on the results, execute the following commands in order:
   
   cd analyses

   python map_task_nr_to_codes.py
   -> creates common_resources/task_nr_to_phenotype_label.dict used by other models in this repo
   -> creates common_resources/task_nr_to_phenotype_codes.dict used by other models in this repo
   
   python get_visit_stats.py:
   + requires data/phenotyping/train_listfile.csv and /data/phenotyping/test_listfile.csv 
     and /data/phenotyping/evaluation/pheno_results.json files
     created by following the Benchmark instructions for the phenotyping task
   -> creates data/phenotyping/evaluation/results_table.csv
     
   python make_stats_table.py
   + requires common_resources/task_nr_to_phenotype_label.dict
   + requires data/phenotyping/train_listfile.csv and /data/phenotyping/test_listfile.csv 
     and /data/phenotyping/evaluation/pheno_results.json files
     created by following the Benchmark instructions for the phenotyping task
   -> creates data/phenotyping/evaluation/results_table.csv
     
   python get_visit_stats.py
   + requires data/phenotyping/train_listfile.csv and /data/phenotyping/test_listfile.csv 
     created by following the Benchmark instructions for the phenotyping task
   -> creates figures/visit_distribution.png
     

## References

[1] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9