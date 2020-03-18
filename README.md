# SoftwareKG
This repository accompanies the publication "Investigating software usage in the social sciences with SoftwareKG"

The knowledge graph created based on this code is available at https://data.gesis.org/softwarekg/. 

The goal is to perform information extraction with respect to software names from scientific articles. 
This work follows a Silver Standard learning approach and applies a Bi-LSTM-CRF for sequence tagging. 
The code is structured as follows:
1. `./data_loading` contains all files for loading scientific articles and transforming them from XML to plain text
2. `./silver_standard_generation` contains all files needed to generate a silver standard corpus from the plain text files
3. `./entity_extraction` contains the files needed to train the Bi-LSTM-CRF (and pre-train it on the silver standard corpus). 
4. `./entity_linking` contains the files that match common software names and match them to DBpedia entries. 
5. `./knowledge_graph_exploitation` gives same sample functions of how SoftwareKG can be accessed to reproduce our analyses.

The [SoSciSoCi](https://github.com/f-krueger/SoSciSoCi) data used in this repository is lying in the partner repository . 
In the following it will be explained how (and in what order to run this code).

## Installing Requirments 
First up, all required python and R packages need to installed to run the project.

### R
We ran all R code with **R 3.3.4** but it should also work with a different R version. The necessary packages are: 

```
install.packages(c("rplos", "fulltext", "xml2", "XML", "dplyr", "ggplot2", "lubridate", "tidyverse", "here", "SPARQL", "patchwork"))
```

### Python
Here, we would recommend to build a separate [Anaconda](https://www.anaconda.com/) environment for the project. 

## Reproducing the Results

We use `#` to mark that code is run directly from some shell. 

### Loading Data

First up we retrieve the articles from PLoS in XML format and save them in an intermediate representation as RData. 
The following commands query more than 50.000 articles (without parallelism) which takes quite long. 
```
# Rscript data_loading/replos_retrieve.R
```
Next we need to extract the individual XML files from the data, parse them and extract the relevant sections. 
```
# Rscript data_loading/extract_papers.R
```
And last we perform a simple sentence split with NLTK on the files. 
```
# python data_loading/sent_tokenize.py
```

### Building the Silver Standard

Now we can start to build the silver standard from the new unlabeled data. The output of this step is also available from https://github.com/dave-s477/SoSciSoCi-SSC.

First up we need to get all data we are going to need in one place.
We assume here that the annotated SoSciSoCi data is in a separate repository besides the softwareKG repo.
```
# cp ../SoSciSoCi/Annotation_Data/*.ann ../SoSciSoCi/Annotation_Data/*.txt data/R_loading/SENTS
```
Snorkel can either work with SQLite or Postgres. Parallel processing can only be used in Postgres. If the model is run on the entire corpus setting up Postgres and Configuring Snorkel for it is highly recommended. Which means that the necessary Postgres databases also need to be built before running Snorkel. Otherwise the code will run multiple weeks.   

The next part of the pipeline is written in form of Jupyter Notebooks:
```
# jupyter notebook
```
1. The first file that needs to be run is: `01_partition_data.ipynb` which transforms the data so that it can be loaded into Snorkel. 
2. `02_database_initialization.ipynb` initializes a Snorkel database on which a Snorkel generative model can be build. 
3. `03_distant_supervision.ipynb` and `04_context.ipynb` give some background on the development of the labeling functions. However, they do not need to be run in full because the labeling functions are also given in the file `learning_functions.py` which is used in the next notebook. However, `03_distant_supervision.ipynb` exports some variables used for distant supervision which are used in later notebooks and scripts. This means this notebook has to be run until the exports are done. 
4. `05_generate_model.ipynb` trains the Snorkel generative model. 
5. `06_create_samples.ipynb`

### Training the Information Extraction Model

Next we can start to actually train the Bi-LSTM-CRF model used for information extraction. 
Here the problem is represented as a sequence tagging problem which means we first want to represent the data in BIO format.
It is necessary to run the following command twice with different input flags to produce all required data. 
```
# python entity_extraction/brat_to_bio.py --input-folder ../SoSciSoCi/Annotation_Data/ --output-file data/SoSciSoCi_bio.txt --positive-samples data/positive_samples_overview.json --write-pos
```
```
# python entity_extraction/brat_to_bio.py --input-folder ../SoSciSoCi/Annotation_Data/ --output-file data/SoSciSoCi_bio.txt --positive-samples data/positive_samples_overview.json
```

The data transformation for this step is then again done using jupyter notebooks, so once again:
```
# jupyter notebook
```
1. `01_data_handling.ipynb` which splits the BIO formatted data into train, devel and test set (in the same way the data was split for Snorkel).
2. `02_silver_data_handling.ipynb` creates chunks of the SSC that are used for pre-training.

Now it is time to train the model and extract the data. 
Right now the training process is configured using scripts which call the Python script `./entity_extraction/perform_training.py`. 
All hyper-parameters are fine-tuned within the scripts. (Currently the training is stopped and resumed from checkpoints which is a little inefficient. When we initially built the memory would slowly fill up through the training process and we would get an allocation error. However, this should be fixed now, so this will soon be addressed.

In total we use three scripts: 1 for SSC pre-training, 1 for GSC training (without using an existing checkpoint) and 1 for GSC training from the SSC checkpoint.
There is also a separate prediction script (`predict.py`) that loads an existing model and applies it on new reasoning data. 
It is called form inside the shell script `run_predicition.sh`.

This time we will work from inside the sub-directory.
```
# cd entity_extraction
# python vocabulary_generator.py --train-sets ../data/SoSciSoCi_train_with_pos_ ../../SoSciSoCi-SSC/data/SSC_pos_samples_ ../../SoSciSoCi-SSC/data/SSC_neg_samples_ --devel-set ../data/SoSciSoCi_devel_ --test-set ../data/SoSciSoCi_test_ --out-folder vocabs --dataset-name SoSciSoCi --use-padding  
```
To evaluate the model run:
```
# bash SSC_pre_training.sh 
# bash GSC_training_from_scratch.sh
# bash GSC_training_from_checkpoint.sh
```
To only train the final model for prediction run:
```
# bash SSC_pre_training.sh 
# bash train_prediction_model.sh
```
And to actually run the prediction:
```
# bash run_prediction.sh
```

3. `03_generate_reasoning_data.ipynb`
4. `04_ma_query.ipynb`
