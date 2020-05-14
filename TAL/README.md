# Project TAL - RI

The objective of the project is to develop a system for finding information in a collection of cooking recipes in English. The project consists of 3 steps (see below for details).

The project will be done in groups of 2 students and the final output will include :
- the source code (preferably in Python, in the form of jupyter notebooks)
- the data used (possible external resources, corpus, models, etc.)
- a 6-page report that will explain your method, the experiments carried out and the scores obtained, the distribution of work within the group, a bibliography (references / sites consulted to carry out the work)

# Deadline for project submission: May 24, 2020

# Stage 1 

The objective is to develop a tool for the automatic classification of recipes by geographical origin to Using the list of ingredients. Learning will be done using scikit-learn. 
You will be able to compare different classification algorithms and the use of different types of traits deduced from the list of ingredients (ingredients, number of ingredients, use of lemmatization or desuffixation, application of different frequency thresholds, etc.
It's up to you to find and test features possibly relevant.

# Stage 2 

The best model obtained in step 1 will have to be applied to a new corpus of recipes in order to automatically predict their geographical origin.

# Stage 3

In this step, you index a minimum of 10,000 recipes, among those whose geographical origin you predicted in step 2, in an instance of the Solr search server. 
You will then create a simple full-text search interface, which will also provide the ability to filter the results according to the geographic origin of the recipes. 
Solr's faceted search function will be used for this filtering.