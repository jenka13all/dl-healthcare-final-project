# MIMIC-3 data with the ConvAE architecture for latent representations of EHR sequences

This repository formats MIMIC-3 data for learning the patient representations as described in [1].
It assumes that you have access to the MIMIC-3 dataset and that you've preprocessed it using the code
available at the [MIMIC-3 Benchmarks repository](https://github.com/YerevaNN/mimic3-benchmarks).

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

### Technical Requirements

```
Python 3.6+

```

# Run ConvAE
Download the `convae` folder

```bash
$ git clone ...

$ cd convae
```

The full list of required Python Packages is available in `requrirements.txt` file. It is possible
to install all the dependencies by:

```bash
$ pip install -r requirements.txt 
```

To prepare the data:

1. The following step creates a vocabulary file from D_ICD_DIAGNOSES.csv and D_ITEMS.csv It takes into account
   the fact that the Benchmark study pre-selects 17 medical concepts (mostly vital signs) to group together.
   This grouping reduces noise and focuses on what the Benchmark study authors believe to be the most significant
   features for patient representation.
   
       python process-mimic3.py {PATH TO MIMIC-III CSVs}

2. The following steps create a sequence file for patients of their diagnoses (only). Future code will integrate vital signs 
   and demographics. The PATH parameter is for the MIMIC-III root patient data and should be something like 
   "mimic3-benchmarks/data/root". Use the COHORT_TYPE parameter "train" for the training cohort, and "test" for the
   testing cohort.

       python3 create_diagnosis_seq.py {PATH TO MIMIC-III root patient data} train

       python3 create_diagnosis_seq.py {PATH TO MIMIC-III root patient data} test


## References

[1] Landi, I., Glicksberg, B. S., Lee, H. C., Cherng, S., Landi, G., Danieletto, M., Dudley, J. T., Furlanello, C., & Miotto, R. 
<br>Deep representation learning of electronic health records to unlock patient stratification at scale. npj Digit. Med. 3, 96 (2020).
<br>https://www.nature.com/articles/s41746-020-0301-z


[2] Harutyunyan, H., Khachatrian, H., Kale, D., Ver Steeg, G. & Galstyan, A. 
<br>Multitask learning and benchmarking with clinical time series data
<br>https://www.nature.com/articles/s41597-019-0103-9
