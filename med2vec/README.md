# MIMIC-III data with the Med2Vec architecture for visit representations of EHR patient data

This repository contains code that works with the results of preprocessed MIMIC-III data for use in an experiment
with the Med2Vec architecture, which was originally presented in [1]. I created the pre-processed data using the 
original Theano implementation belonging to the study [1], available at their 
[public Github repository](https://github.com/mp2893/med2vec). 

You can recreate the same data files by following their instructions. I ended up using the data generated to use
the complete (not 3-digit) ICD9 codes as labels. This was necessary in order to correctly map predicted results 
to the 25 care-conditions specified by the phenotyping benchmark task of another study [2].

I named these pre-processed data "seqs.pkl" and "labels.pkl" and placed them in the "Med2Vec_data" directory 
for modelling, prediction and analysis. I placed the generated "types" files in the "resources" directory, 
as they were useful for mapping integer IDs to real ICD9 codes. The modelling and prediction code is mostly 
based on a Tensorflow implementation of Med2Vec available at 
[this repository](https://github.com/sdwww/Med2Vec_tensorflow/blob/master/Med2VecRunner.py).

The original code had to be changed and corrected in order to run my experiment successfully. For example, 
the original code called the wrong objective function for calculating the loss of the embedding (code) 
representations. The original embedding cost function did not work at all (it returned multiple errors). 
There was an oversight in the model training where non-sequence patient separators ([-1]) were included in the
calculation of the average cost for learning the weights. The result of this was that the trained model had 
learned too much noise. The function for predicting a visit based on one preceding visit contained multiple mistakes: 
for one, it switched the medical codes of the input and target visits. The function for creating a multi-hot
encoded representation of the sequence data did not take prediction functionality into account when dealing
with using complete ICD9 codes as labels instead of the 3-digit grouped codes. 

I've marked in the original code where I've made changes, and I rewrote the "predict_next_visit()" function completely.
My own original code includes: 

* splitting data into train and test sets (SplitData.py)
  
* functions for preparing the data for AUC-ROC metric evaluation on care-conditions in the final visit (in Med2VecRunner.py)

* function for calculating the code embedding cost (in Med2Vec.py)

* functions for evaluating the AUC-ROC metric on the care-conditions predicted in each last visit 
  ("analyses/calculate_metrics.py")

* functions for calculating various statistics such as number of patients and care-condition prevalence in 
  test and train subsets, distribution of the number of visits and number of codes, and the creation of
  various pre-processed data for mapping Med2Vec codes to MIMIC-III codes and care-conditions 
  (in the "analyses" directory)
  
* code for allowing the user to specify data files and model path on the command line instead of having to 
  alter the hard-coding in Med2VecRunner.py

To reproduce my experiment, clone this repository and:

```
cd med2vec
pip install requirements.txt
```

In Med2VecRunner.py, uncomment the lines for training the model and predicting next visit, as desired.

To train a model on complete ICD9 codes, the label file is the same as the sequence file. Call the following
command with your model path of choice. 

```
python Med2VecRunner.py --seq_file=./Med2Vec_data/train_seqs.pkl --label_file=./Med2Vec_data/train_seqs.pkl --model_path=./Med2Vec_model/your_path_here
```

If you change the paths to the data files (seq_file and label_file), make sure to update the parameters 
appropriately for n_input, n_output, and n_samples in the configuration dictionary (get_config()), and possibly
the test files called in predict_next_visit(). Other model parameters can be updated in get_config().

## References

[1] Edward Choi, Mohammad Taha Bahadori, Elizabeth Searles, Catherine Coffey, Jimeng Sun
<br>Multi-layer Representation Learning for Medical Concepts
<br>https://arxiv.org/pdf/1602.05568.pdf

[2] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9
