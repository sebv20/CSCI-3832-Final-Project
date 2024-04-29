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

## Dataset
Our dataset consists of a subset of approxiamtely 8,500 tweets from a larger dataset linked here: https://www.kaggle.com/datasets/deffro/the-climate-change-twitter-dataset. The hydrated subset of data we used can be found here: (INSERT LINK TO OUR DATASET). Our team leveraged the following columns for our project work:
1. Example
2. Example
3. Example
4. Example
5. Example
6. Example
7. Example

## Coding Environment
1. VSCode
2. Jupyter Notebook

## Library Overview
1. Pandas

  Pandas is currently the most popular software library for data analysis/data science, allowing users to leverage Python in their analysis methods. Some of the key features listed on their website include:
  
    a. Fast and efficient DataFrame object generation
    b. Tools for loading data into in-memory data objects from different file formats
    c. Data alignment and integrated handling of missing data
    
  Specific to our project, our team utilized Pandas for our EDA and Dataset generation. We decided to use this library because all team members were at least somewhat familiar with the documentation, and due to its widespread popularity.

2. Scikit learn

  Scikit-Learn is another very popular library primarily used for machine learning in Python. Its features include:
  
    a. Classification
    b. Clustering
    c. Regression
    d. Dimensionality reduction
    
  Our team decided to use Scikit-Learn because of its thorough documentation and previous experience with the library. It allowed us to conduct ….(Need Example)…. Analysis of our initial data and build powerful conclusions with notable efficiency.

3. Tensorflow

  TensorFlow is an open-source library built for working with deep neural networks, or more complex machine learning algorithms. It is compatible with Javascript, Python, and C++. The library has the following key features:
  
    a. Responsive Construct
    b. Parallel Neural Network Training
    c. Feature Columns
    d. Statistical Distributions
    
  TensorFlow was perfect for our project due to its powerful machine-learning tools as well as its large community which we were able to leverage when implementing it into our code. In particular, we were able to complete …(Need Example)… Analysis by using …(following tool)….
  
4. Keras

  Keras is another open-source machine-learning library built specifically for Python and serves as a high-level neural network interface that runs on top of TensorFlow. These two libraries are often used in conjunction with one another to provide powerful deep-learning capabilities. The key features of this library are as follows:
  
    a. Convolutional and Recurrent Neural Network Support
    b. Normalization
    c. Pooling
    d. Allows users to produce deep models on smartphones
    
  Much as described above, the primary reason our team decided to use Keras was to support the use of TensorFlow and increase the overall efficiency of our processing methods.
 
5. Open Cage

  Open Cage is a simple, yet, powerful Geocoder that allows developers to incorporate location data into their analysis all around the globe. The library uses an API that combines multiple geocoding systems in the background, selecting only the best results from multiple data sources and algorithms so that users don’t have to. Some key features include:
  
    a. Global coverage of location data
    b. Reverse Geocoding
    c. Aggregation of multiple data sources
    
  A separate library that our team considered was Geopy, however, as we began to implement this library into our analysis we quickly discovered that we would have better luck switching to Open Cage. Due to Open Cage’s (BAXTER NEED HELP) we found this library to be much better suited to our needs.


## Pre-Processing
In order to train our models and conduct analysis on our data we conducted the following pre-processing steps:

1. Hydration

   a. Reason why we did this step
   
2. Example

   a. Reason why

3. Example
   
   a. Reason why

## Hyperparameters

## Additional Details

## Results Summary
We want to break down our results into two sections. The first section we will cover our model performance and second section will be our experiment results.

1. Model Performance

  a. BERT: Our BERT model ran for a total of 4 epochs with a run time of just under 4 hours. The final validation set accuracy came out to 87.7%
  
  b. Random Forest: Our Random Forest model consisted of 100 estimators with a run time of under one minute. The final F1 score acheived by the model came out to 82.01%
  

2. Experiment Results

  Our team failed to reject the null hypothesis, finding a p-value of (BLANK) indicating that climate change sentiment on twitter is uniform across different geographies. Some of the interesting findings were (BLANK).


