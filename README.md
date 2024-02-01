# Prediction of Customers in Different Parts of Helsinki using Deep Learning

This is home assignment as a part of application process for summer internship at Wolt, done by __Daniel Wohlrath__.

## Table of Contents

1. [Instructions](#instructions)
2. [Presentation](#presentation)
3. [Main Code](#main-code)
----

## Instructions

You can use the explicit specification files ```environment.yml``` to build an identical environment on your machine. It is not cross-platform. The platform it was created in is linux-x86_64.

Please note that on other platforms, the packages specified might not be available or dependencies might be missing for some of the key packages already in the spec.

Use the terminal for the following steps:

Create the environment from the environment.yml file:

```conda env create -f environment.yml```


Before running the main code in ```analysis.ipynb```, make sure that you're using the environment created from 'environment.yml'

## Presentation

Presentation slides in pdf (7 slides excl. title slide) is located in ```presentation.pdf```

## Main Code

All the coding was done in python using standard libraries like tensorflow, sklearn, numpy, pandas, etc. The code is located in a Jupyter Notebook ```analysis.ipynb```. I am importing some self-defined functions from ```preprocessing.py```, which uses similar libraries.

To run the code and see the results, open the Jupyter Notebook and run it cell by cell. Reading the markdown cells for better orientation is recommended. The chapters are roughly:
- Preprocessing data (scaling, clustering)
- Exploratory data analysis (descriptive statistics like counts, histograms, correlation)
- Modeling:
- - getting specific features, train-val-test splitting, scaling
- - Implementation of 5 different predictive models
- - Evaluation on test data
- Conclusions
- Further Development
- Me and My Background at Wolt

