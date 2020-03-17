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

Now we can start to build the silver standard from the new unlabeled data. 
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
