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
In the randomForest_V2_Final.ipynb file is our final RF model that we implemented. 
In our main.py file is the EDA work that we did. 

## Dataset
Our dataset consists of a subset of approxiamtely 8,500 tweets from a larger dataset linked here: https://www.kaggle.com/datasets/deffro/the-climate-change-twitter-dataset. The hydrated subset of data we used can be found here: (INSERT LINK TO OUR DATASET). Our team leveraged the following columns for our project work:
1. Topic
2. Sentiment
3. Stance
4. Gender
5. Temp_Avg
6. Agressiveness
7. Lattitude
8. Longitude
9. Text (Vectorized)
10. Region


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

  Open Cage is a simple, yet, powerful Geocoder that allows developers to incorporate location data into their analysis all around the globe. The library uses an API that combines multiple geocoding systems in the background, selecting only the best results from multiple data sources and algorithms so that users don't have to. Some key features include:
  
    a. Global coverage of location data
    b. Reverse Geocoding
    c. Aggregation of multiple data sources
    
  A separate library that our team considered was Geopy, however, as we began to implement this library into our analysis we quickly discovered that GeoPy is heavily outdated and had to long of a runtime for our use. Therefore we switched to a newer geocoder that gave us 2000 free API calls a day and progressively ran it until we got all our location based data turned into JSON Full Address.

6. Selenium

  The Selenium python package allows python to interact directly with web drivers installed on the local enviroment and automate web interactions. Some of its key features include:

  a. Compatability with several different web drviers (chrome, firefox, yahoo, etc.)
  b. Simple implementation - not reliant on other packages
  c. Contains functions used to scrape raw html from fetched webpages

  In combination with the Selenium package, we used the chrome web driver which was downloaded from: "https://chromedriver.chromium.org/downloads". Our scraping script is relatively slow (a few hundred tweets per hour), but it is reliable and succesfully get around the twitter dev API pay wall. 

## Pre-Processing
In order to train our models and conduct analysis on our data we conducted the following pre-processing steps:


1. Removal of Non-Geotagged Entries

  a. In the raw dataset from Kaggle, roughly half of the entries do not have information in regards to geotagging. Due to the nature of our project, we found it easier to just remove those entries all together since we would still be left with hundreds of thousands of potential entries.

2. Hydration

   a. Our original dataset does not contain the raw text portion of each tweet. However, it does contain the tweet id.

   b. We first added a new column, "text" to our dataset in order to store retrieved raw text from twitter.

   c. Then we implemented a web to fetch the raw text of a tweet based on its id. If the tweet was unreachable either by rejection from the twitter API or the tweet no longer existed, the text column was marked as Null.
   
3. Outlier and Null Removal

   a. We did this step to reduce noise that could potentially introduce bias or errors leading to our model being over/underfit. We found that are data set was mostly complete, with no outliers and few NaN variables

4. Aggregation of data set into subsets using GeoCoder
   a. We used BirdCages free testing geocoder API to pull the full JSON address using lattitude and longitude, from there we built a function to parse the address and create the columns: Continent, Country, City.
  
   b. From there we were able to see that majority of our data points lie in North America and Europe so we created the Merged Region column by creating list of states within our three subsets and aggregating accordingly.

5. Visualization of Mean Sentiment Scores by Region
   
   a. In order to get an idea of how sentiment relates to countries, continents and regions we created visuals to look at the distribution of average sentiment using the SNS library. We found that most belivers in climate change correlate with negative sentiment.

## Results Summary
We want to break down our results into two sections. The first section we will cover our model performance and second section will be our experiment results.

1. Model Performance

  a. BERT: Our BERT model ran for a total of 4 epochs with a run time of just under 4 hours. The final validation set accuracy came out to 87.7%
  
  b. Random Forest: Our Random Forest model consisted of 100 estimators with a run time of under one minute. The final F1 score acheived by the model came out to 82.01%

  c. Our efforts to test our model with unseen data were unsuccessful due to the itidf vectorization process creating different feature variables.
  

2. Experiment Results

  Our team failed to reject the null hypothesis, finding a p-value of (BLANK) indicating that climate change sentiment on twitter is uniform across different geographies. Some of the interesting findings were (BLANK).


## Reproducing Results

To reproduce our results you will need to have an environment set up to run .ipynb files. You will need python downloaded as well as the packages that we listed above. You may have to "pip install" additional packages as well if you do not have them downloaded. 


Our repo is broken up into two sections: data and src. All of the data we used is in the data folder. In our src folder is all of our source code. 


When you go in the src folder it is split into our hydration scripts folder, main eda, and models folder. 


The twitter scraping can be reproduced by running the "twitter scraping.ipynb" file. 


In our eda folder we have a few files, "full_df_eda.ipynb" contains the most comprehensive version of our data analysis that we completed. You can run this file to reproduce the results as well.


In the models folder we have our BERT and RF models. The final BERT model that we ran is in "bert_v2_final.ipynb". If attempting to reproduce, be aware that it will take several hours and be computationally expensive. The final RF model is in "randomForest_v2_final.ipynb". 

