# MIMIC-3 data with the Med2Vec architecture for visit representations of EHR patient data

This repository contains code that works with the results of data-preprocessing MIMIC-3 data for use in the 
Med2Vec architecture as presented in [1]. It also contains code that works with the results of data modelling 
from a separate implementation of Med2Vec available at [this repository](https://github.com/sdwww/Med2Vec_tensorflow/blob/master/Med2VecRunner.py). 

The results of the data preprocessing are pickled files of patient sequences. Each patient sequence consists 
of two or more visits; each visit is represented as a list  containing diagnosis codes for that visit.
The files are "processed.seqs" and "processed.3digitICD9.seqs", available in the "output" folder. "processed.seqs" 
contains ICD9 diagnosis codes with their complete coding. "processed.3digitICD.seqs" contains ICD9 diagnosis codes in 
a shortened 3-digit form. 

For example, two codes of the first visit of the first patient in "processed.seqs" map to the ICD9 codes V43.64 and V43.65. 
The same data in "processed.3digitICD.seqs", however, only maps to one single diagnosis, the simplified 3-digit V43. 
This grouping of diagnoses into 3-digit mappings reduces the total number of diagnoses to classify, 
eliminates redundancy and noise, and shortens training time. The mapping of code IDs to their 
ICD9 descriptions can be found in "processed.types" and "processed.3digitICD9.types", respectively.

I used only the _data-preprocessing_ code available at the [publicly available repository](https://github.com/mp2893/med2vec)
for the study [1], namely process_mimic.py, following the instructions for doing so on their README.md file. 
I double-checked the total numbers of patients, visits, and diagnoses to make sure that we were using the same data set. For creating 
and training the model, evaluation and prediction I used code from a TensorFlow implementation of Med2Vec
available at [this repository](https://github.com/sdwww/Med2Vec_tensorflow/blob/master/Med2VecRunner.py).

This code contained some serious errors that I had to correct. For example, it called the wrong objective function 
for calculating the loss of the embedding (code) representations. The original implementation of the code 
cost function did not work at all (it returned multiple errors). There was an oversight in the code dealing with 
the patient separator ([-1]) when calculating the average cost of learning the representations. The function 
for predicting a visit based on one preceding visit contained a serious mistake: namely, it was using codes of 
the visit _previous_ to the input visit as the target, instead of the codes of the _following_ visit. 
Finally, it was very confusing, so I rewrote it.

I'm providing the necessary files (Med2Vec.py, Med2VecRunner.py, PredictModel.py, and the preprocessed sequence data files) 
with all my corrections in the "tensorflow-implementation" directory. They can be used by providing a path to
the training data sequences.

```
python3 Med2VecRunner.py --seq_file=./Med2Vec_data/train_seqs.pkl --label_file=./Med2Vec_data/train_labels.pkl --model_path=./Med2Vec_model/your_path_here
```

Other parameters, such as number of samples, unique number of codes in the seq.pkl file, unique number of codes
in the label.pkl file, number of windows to predict on, number of epochs, batch size, etc., need to be updated
in the "config" dictionary in the "get_config" function in Med2VecRunner.

I'm also providing code for calculating the statistics of the patient, visit, and codes data, as well as a
script for mapping Med2Vec diagnosis codes to MIMIC-III Benchmark codes. This is useful for making comparisons 
to the phenotype prediction task from the MIMIC-III Benchmark study [2].

## References

[1] Edward Choi, Mohammad Taha Bahadori, Elizabeth Searles, Catherine Coffey, Jimeng Sun
<br>Multi-layer Representation Learning for Medical Concepts
<br>https://arxiv.org/pdf/1602.05568.pdf

[2] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9
