# Patient-independent seizure detection based on long-term iEEG and a novel lightweight CNN

## Abstract
Objective. Patient-dependent seizure detection based on intracranial electroencephalography (iEEG) has made significant progress. However, due to the difference in the locations and number of iEEG electrodes used for each patient, patient-independent seizure detection based on iEEG has not been carried out. Additionally, current seizure detection algorithms based on deep learning have outperformed traditional machine learning algorithms in many performance metrics. However, they still have shortcomings of large memory footprints and slow inference speed. Approach. To solve the above problems of the current study, we propose a novel lightweight convolutional neural network model combining the Convolutional Block Attention Module (CBAM). Its performance for patient-independent seizure detection is evaluated on two long-term continuous iEEG datasets: SWEC-ETHZ and TJU-HH. Finally, we reproduce four other patient-independent methods to compare with our method and calculate the memory footprints and inference speed for all methods. Main results. Our method achieves 83.81% sensitivity (SEN) and 85.4% specificity (SPE) on the SWEC-ETHZ dataset and 86.63% SEN and 92.21% SPE on the TJU-HH dataset. In particular, it takes only 11 ms to infer 10 min iEEG (128 channels), and its memory footprint is only 22 kB. Compared to baseline methods, our method not only achieves better patient-independent seizure detection performance but also has a smaller memory footprint and faster inference speed. Significance. To our knowledge, this is the first iEEG-based patient-independent seizure detection study. This facilitates the application of seizure detection algorithms to the future clinic.
## Versions used in our code testing
Note that different versions won't likely cause dependencies issues.
```
pytorch-gpu == 1.12 
```
## Citation
https://iopscience.iop.org/article/10.1088/1741-2552/acb1d9
