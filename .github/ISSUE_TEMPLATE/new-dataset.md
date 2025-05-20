---
name: New Dataset
about: Template to propose a new dataset that is shall be added to TabArena.
title: "[New Dataset] Adding <Name> "
labels: new_dataset
assignees: ''

---

* Metadata file: < Reference to pull request with a .yaml file following https://github.com/TabArena/dataset_curation/blob/main/dataset_creation_scripts/_template.yaml >
* Preprocessing script: < Reference to a .py file containing a preprocessing pipeline to transform data from the raw data source into a format suitable for benchmarking >


## Dataset Checklist 

### Is the data available through an API for automatic downloading, or does the license allow for reuploading the data? 
<TODO>

### What is the sample size? % Covers sample size criterion
<TODO>

### Was the data extracted from another modality (i.e., text, image, time-series)? If yes: Are tabular learning methods a reasonable solution compared to domain-specific methods? (If possible, provide a reference)
<TODO>

### Is there a deterministic function for optimally mapping the features to the target? 
<TODO>

### Was the data generated artificially or from a parameterized simulation?  
<TODO>

### Can you provide a one-sentence user story detailing the benefits of better predictive performance in this task? 
<TODO>

### Were the samples collected over time?  If yes: Is the task about predicting future data, and, if yes, are there distribution shifts for samples collected later?
<TODO>

### Were the samples collected in different groups (i.e., transactions from different customers, patients from multiple hospitals, repeated experimental results from different batches)? If yes: Is the task about predicting samples from unseen groups, and if yes, are distribution shifts of samples from different groups expected?
<TODO>

### Are there known preprocessing techniques already applied to the ‘rawest’ available data version? 
<TODO>

### What preprocessing steps are recommended to conceptualize the task in the preprocessing Python file?
<TODO>

### Do you have any other recommendations for how to use the dataset for benchmarking?
<TODO>
