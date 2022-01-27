# Covert speech decoding from EEG signals
## EPFL ML for Science project
Final report: https://github.com/dauvillc/ml_eeg_classification/blob/master/ml_report_final.pdf
## Repo organization
* data: contains the raw data provided by the lab. This normally includes the folders 'clean_data_sub_01', 
  and 'Data_ML_Internship'. However the data is confidential. For this reason, we were only in measure of providing
  the 'ready-to-use' data which we have extracted ourselves. Therefore the data folder is present but empty.
* data_loading: python local package to extract and load the data.
* evaluations: contains the evaluation of each combination of features (frequencies and electrode groups) and hyperparameter for the logistic regressions and random forests, as two CSV files.
* figures: contains some figures used during the study as well as for the final report.
* models: python local package inluding most notably the CNN architectures.
* preprocessing: python local package including all of the preprocessing functions.
* ready_data/ready_data.zip: contains the data already extracted from the files provided by the lab (those which used to
  be under 'data').
* results: contains the results of the CNN cross-validation script, as CSV files.
* scripts: contains the important scripts to reproduce our results, as well as some scripts that were used to build the ready data from the original.
* data_exploration.ipynb: first notebook that we used to explore the initial dataset.
* explore_features.ipynb: Notebook to be used to reproduce the features exploration.

## Requirements
the following packages are required to reproduce our results:
* ```numpy, matplotlib, scikit-learn, scipy, scikit-image, pandas, seaborn, opencv```: usual scientific and visualization packages;
* ```pytorch```: is required to build and train the CNN models. The code is written to work with either GPU or CPU implementations of pytorch.
* ```mne```: was used originally to produce the ready data from the raw data, as well as for the data exploration. This is a package meant to handle nervous signals such as EEG. It should not be required now that the raw data is unavailable.

## Reproduce the results
* **Features optimisation**: from the project's root directory, run:
  ```python scripts/evaluate_features.py model [--day DAY]```
  where model is either "random_forest" or "logistic_regression" and day is an integer between 1 and 5.
  This script evaluates of combinations of a set of frequencies and a set of electrode groups, and writes
  the results in evaluations/ . This does not require pytorch.
 * **Evaluating a model on a single combination**: from the project's root directory, run:
  ```python scripts/evaluate_model model day [--freqs FREQ1 FREQ2 ...] [--brain_areas AREA1 AREA2 ...] [--penalty C] [--max_depth D]```
  This functionality can be preferably accessed via the evaluate_model.evaluate() function, as used in the explore_features notebook.
  * Training the CNN: from the project's root directory run:
  ```python scripts/train_cnn [--baseline]```
  If --baseline is indicated, then the larger network is used, and trained on the full FFT of differences input.
  Otherwise, the input is cropped to the High gamma frequency band, and only the right and left temporal electrodes
  are kept. This scripts uses pytorch, and while few training epochs are computed, it might take some time if used 
  run on CPU.
