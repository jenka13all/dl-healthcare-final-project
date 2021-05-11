# MIMIC-III data with the Med2Vec architecture for visit representations of EHR patient data

This repository contains code that works with the results of preprocessed MIMIC-III data for use in an experiment
with the Med2Vec architecture, which was originally presented in [1]. I created the pre-processed data using the 
original Theano implementation belonging to the study [1], available at their 
[public Github repository](https://github.com/mp2893/med2vec). 

You can recreate the same data files by following their instructions. I ended up using the data generated to use
the grouped 3-digit ICD9 codes as labels. This was necessary in order to get usable results for predicting
AUC-ROC score for predicting the presence or absence of 25 care-conditions as specified in [2].

I named these pre-processed data "seqs.pkl" and "labels.pkl" and placed them in the "Med2Vec_data" directory 
for modelling, prediction and analysis. I placed the generated "types" files in the "resources" directory, 
as they were useful for mapping integer IDs to real ICD9 codes. The modelling and prediction code is mostly 
based on a Tensorflow implementation of Med2Vec available at 
[this repository](https://github.com/sdwww/Med2Vec_tensorflow/blob/master/Med2VecRunner.py).

The code here is based on the original code, but with multiple changes.
To reproduce my experiment, clone this repository and:

   cd med2vec

   pip install requirements.txt

In Med2VecRunner.py, uncomment the lines for training the model and predicting next visit, as desired. When
predicting next visit, make sure to indicate use of 3-digit ICD9 codes for labelling by setting the "three_digit"
parameter of "predict_next_visit()" to True or False, accordingly.

To train a model on 3-digit ICD9 codes, you first need to create the mapping dictionaries:

    cd analyses

    python map_diagnoses.py {path to MIMIC-III database files}
    -> creates resources/seqs_to_icd.dict
    -> creates resources/labels_to_icd.dict

Now call the following command with your model path of choice. 

    cd ~/

    python Med2VecRunner.py --seq_file=./Med2Vec_data/train_seqs.pkl \
      --label_file=./Med2Vec_data/label_seqs.pkl \
      --model_path=./Med2Vec_model/{your model path}


If you change the paths to the data files (seq_file and label_file), make sure to update the parameters 
appropriately for n_input, n_output, and n_samples in the configuration dictionary (get_config()).

To evaluate the results, set the model path in "analyses/calculate_metrics.py" (within the file) and execute
the following commands in order:

    cd analyses

    python map_diagnoses.py {path to MIMIC-III database files}
    -> creates resources/seqs_to_icd.dict
    -> creates resources/labels_to_icd.dict

    python calculate_metrics.py
    -> creates Med2Vec_model/{model path}/pheno.json

    python get_patient_condition_stats.py
    -> creates resources/care_conditions_train_prevalence.dict
    -> creates resources/care_conditions_test_prevalence.dict

    python make_stats_table.py
    + requires Med2Vec_mode/{model path}/pheno.json
    + requires resources/care_conditions_train_prevalence.dict
    + requires resources/care_conditions_test_prevalence.dict

    -> creates Med2Vec_data/results_table.csv

    python calculate_metrics_correlation.py
    + requires Med2Vec_data/results_table.csv
    -> creates figures/prevalence_pearson.png
     
    python get_code_stats.py
    -> creates figures/pat_codes_distribution.png
    -> creates figures/visit_codes_distribtuion.png

## References

[1] Edward Choi, Mohammad Taha Bahadori, Elizabeth Searles, Catherine Coffey, Jimeng Sun
<br>Multi-layer Representation Learning for Medical Concepts
<br>https://arxiv.org/pdf/1602.05568.pdf

[2] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9
