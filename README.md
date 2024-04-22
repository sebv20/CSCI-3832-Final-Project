# CSCI-3832-Final-Project
This is the repo for our final project for the Spring 2024 CSCI-3832 NLP Course at CU Boulder. 

Team Members: Sebastian Vargas, Baxter Romero, Johnny Wilcox, Trajan Pei, Caleb Bettcher

## Project Overview
This project aims to determine the variability of climate change sentiment across several countries. Our study will utilize a dataset containing Twitter data spanning the last 13 years, encompassing tweets from many countries to ensure a broad representation of global sentiment. By first preprocessing our data using exploratory data analysis, we will fine-tune a BERT model on our data and implement a Random Forest Classifier that quantifies the variation in sentiment towards climate change in each selected country in order to compare these findings against the global average sentiment. The deviation from average sentiment serves as our test statistic, allowing us to evaluate the null hypothesis against the alternative.

Null Hypothesis: Sentiment for climate change is the same across all countries, any variation in sentiment analysis is due to chance.

Alternative Hypothesis: Sentiment for climate change is not uniform across countries. 

Test Statistic: Deviation from average sentiment


## Repo Organization
Our repo contains various files that we used for testing and implementing our code. 

In the bert.ipynb file is our final BERT model that we trained. 
In the randomForest.ipynb file is our final RF model that we implemented. 
In our main.py file is the EDA work that we did. 

