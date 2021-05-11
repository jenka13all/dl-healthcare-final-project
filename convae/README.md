# MIMIC-3 data with the ConvAE architecture for latent representations of EHR sequences

This repository provides code for formatting MIMIC-3 data for learning the patient representations 
as described in [1]. It assumes that you have access to the MIMIC-3 dataset and that you've preprocessed 
it using the code available at the [MIMIC-3 Benchmarks repository](https://github.com/YerevaNN/mimic3-benchmarks).

Once you've followed the steps 1 - 5 under "Building a benchmark" in the MIMIC-3 Benchmarks repository, 
and generated the phenotyping task-specific dataset under step 6, you will have a data structure 
that can be used in the following steps. The end-result of these steps is patient data formatted for use in
the ConvAE architecture [1].

The original ConvAE architecture code at [http://github.com/landiisotta/convae_architecture](http://github.com/landiisotta/convae_architecture "Original code repository")
provides the ConvAE model, sample EHR patient sequences, and a 200-concept vocabulary created 
from a synthetic dataset. The sample data is useful for a quick start, but the subsequent encodings and
learned model are not valuable for reproducing the study or using the architecture on real data. 
The ConvAE repository also does not provide guidance for formatting real EHR data: this information has to
be gleaned from careful reading of the study, examination of the code, and comparison to the sample data.

In my repository, I attempt to provide this guidance. By following the steps, you should end up with formatted
MIMIC-3 data that can be used for clustering (as in the ConvAE study) or in other downstream prediction tasks. 
I have focused on the phenotype prediction task described in the Benchmarks study. [2]

# Getting started
Clone the repository and switch to the `convae` folder:

   cd convae

   pip install -r requirements.txt

To prepare the data:

1. The following step creates a vocabulary file - "data/cohort_vocab.csv" from MIMIC-III ICD9 diagnoses. The
   PATH parameter should be the path to where your MIMIC-III database files are stored.
   
       python create_mimic3_vocab {PATH TO MIMIC-III CSVs}
       -> creates data/cohort_vocab_icd.csv

2. The following steps create two sequence files - "data/cohort_ehrseq.csv" for patient training data and 
   "data/cohort_test_ehrseq.csv" for patient testing data, using their diagnoses only. The PATH parameter is for 
   the MIMIC-III root patient data and should be something like "mimic3-benchmarks/data/root". 
   Use the COHORT_TYPE parameter "train" for the training cohort, and "test" for the testing cohort.

       python3 create_patient_seqs.py {PATH TO MIMIC-III root patient data} train
       -> creates data/cohort_ehrseq_icd.csv

       python3 create_patient_seqs.py {PATH TO MIMIC-III root patient data} test
       -> creates data/cohort_test_ehrseq_icd.csv

3. Once you have generated these three files - "cohort_vocab.csv", "cohort_ehrseq.csv" and "cohort_test_ehr.csv",
   you can train models on them. If you want to reproduce my experiment, execute the following steps in order
   to create the necessary files for results: a table of 25 care-conditions, their prevalence in the train
   and test populations, and the AUC-ROC score of the model on predicting the presence or absence of each
   care condition in the test set, using the model trained on the training set.
   
   sh learn-patient-representations.sh ./data
   -> creates data/encodings/best_model.pt
   
   python evaluate_from_checkpoint.py
   -> creates data/encodings/test/test_metrics.txt
   -> creates data/encodings/test/target.csv
   -> creates data/encodings/test/predictions.csv
   
   cd analyses
   
   python make_phen_codes_dict.py
   + requires data/cohort_vocab_icd.csv
   -> creates resources/vocab_index_to_phen_task_nr.dict
     
   python get_prevalence.py
   + requires resources/vocab_index_to_phen_task_nr.dict
   + requires data/cohort_ehrseq_icd.csv
   + requires data/cohort_test_ehrseq_icd.csv
   
   -> creates resources/train_prevalence.dict
   -> creates resources/test_prevalence.dict
     
   python calculate_metrics.py
   + requires resources/vocab_index_to_phen_task_nr.dict
   + requires data/encodings/test/target.csv
   + requires data/encodings/test/predictions.csv
   
   -> creates pat_data/care_true.pkl
   -> creates pat_data/care_preds.pkl
   -> creates pat_data/indices_to_skip_no_fp.pkl
   -> creates pat_data/indices_to_skip_no_tp.pkl
   -> creates pat_data/pheno.json
   
   python make_stats_table.py
   + requires pat_data/pheno.json
   + requires resources/train_prevalence.dict
   + requires resources/test_prevalence.dict
   
   -> creates pat_data/results_table.csv

   python calculate_results_correlation.py
   + requires pat_data/results_table.csv
   -> creates figures/prevalence_pearson.png

## References

[1] Landi, I., Glicksberg, B. S., Lee, H. C., Cherng, S., Landi, G., Danieletto, M., Dudley, J. T., Furlanello, C., & Miotto, R. 
<br>Deep representation learning of electronic health records to unlock patient stratification at scale. npj Digit. Med. 3, 96 (2020).
<br>https://www.nature.com/articles/s41746-020-0301-z


[2] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9
