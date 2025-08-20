# Movie-Recommendation-Analysis
Movie Recommendation Analysis Using Python

## Objective

The objective of this project is to analyze movie performance data to uncover patterns in profitability, language-based gross earnings, and genre-level financial outcomes. By examining correlations between budget, gross revenue, profit, return on investment (ROI), ratings, and audience engagement, the project seeks to identify the key factors driving commercial success. Additionally, recommendation systems will be developed using movie genres, actors, and content similarity to suggest films that align with user preferences. This analysis will provide actionable insights for decision-making in film production, marketing, and distribution strategies.

## Problem Statement:
Perform analysis and Basic Recommendations based on Similar Genres and Movies which Users prefer. Some of the Key Points on which we will be focusing include: 

1.Profitability of Movies 

2.Language based Gross Analysis 

3.Comparison of Gross and Profit for Different Genres, 

4.Recommendation systems based on Actors, Movies, Genres. 

This Project will help us to understand Correlation between these factors.

### Setting up the environment
Installing the required libraries:

•	We need Numpy for mathematical operation

•	Pandas for Dataframe Manipulation

•	Seaborn and Matplotlib for data visualization

•	And finally Jupyter Notebook to build an interactive ipython notebook




### Python Code (Step-by-Step)
Step 1: Import Libraries
   ```python
    # lets import the basic libraries
import numpy as np
import pandas as pd
 
# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
    
# for jupyter notebbook widgets
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import interact_manual

# for interactive shells
from IPython.display import display

# Supress warnings
import warnings
warnings.filterwarnings('ignore')
   
# setting up the chart size and background
plt.rcParams['figure.figsize'] =(16,8)
plt.style.use('fivethirtyeight')
 ```
Step 2: Load and Inspect the Dataset
Python:
 ```python
# let read the dataset
data = pd.read_csv('movie_metadata.csv')
# lets check the shape
print(data.shape)
 ```
Output:

<img width="199" height="32" alt="image" src="https://github.com/user-attachments/assets/716e1260-b91d-4947-99fa-e198a7aee2f4" />

Step 3: Data Cleaning & Preprocessing
Python code:
```python
 # DATA CLEANING
# lets check the data info
data.info()
# lets remove unnecessary columns from the dataset

# use the 'drop()' function to drop the unnecessary columns

data = data.drop(['color','director_facebook_likes','cast_total_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','country','content_rating','movie_imdb_link','aspect_ratio','plot_keywords','facenumber_in_poster'], axis =1)
data.columns
# lets check the rows having high percentage of missing values in the dataset
round(100*(data.isnull().sum()/len(data.index)),2)

# Since 'gross' and 'budget' columns have large number of NaN values, drop all the rows with NaNs at this column using the
# 'isnan' function of Numpy along with a negation '~'

data = data[~np.isnan(data['gross'])]
data = data[~np.isnan(data['budget'])]

# now check the number of missing value column
data.isnull().sum()

# the row for which the sum of Null is less than two are retained

data = data[data.isnull().sum(axis =1) <= 2]
data.isnull().sum()

# lets input the missing values

# using mean for numerical columns
data['num_critic_for_reviews'].fillna(data['num_critic_for_reviews'].mean(),inplace = True)
data['duration'].fillna(data['duration'].mean(),inplace = True)

# using mode for categorical column
data['language'].fillna(data['language'].mode()[0], inplace = True)

# As we know that we cannot use the statistical values for imputing the missing values of actor names, so we will replace the actor names with "Unknown Actor"

data['actor_2_name'].fillna('Unknown Actor', inplace = True)
data['actor_3_name'].fillna('Unknown Actor', inplace = True)

# As we input all the missing values let check the number of the total missing values in the dataset
data.isnull().sum().sum()

# Using mean for numerical columns
data['num_critic_for_reviews'].fillna(data['num_critic_for_reviews'].mean(), inplace=True)
data['duration'].fillna(data['duration'].mean(), inplace=True)

# Using mode for categorical column
data['language'].fillna(data['language'].mode()[0], inplace=True)

# Replacing missing actor names with 'Unknown Actor'
data['actor_2_name'].fillna('Unknown Actor', inplace=True)
data['actor_3_name'].fillna('Unknown Actor', inplace=True)

# Check total missing values
total_missing = data.isnull().sum().sum()
print(f"Total missing values: {total_missing}")

# lets convert the gross and budget from $ to million $ to make our analysis easier
data['gross'] = data['gross']/1000000
data['budget'] = data['budget']/1000000

# lets create a profit column using the budget and gross
data['Profit'] = data['gross'] - data['budget']

# lets also check the name of Top 10 profitable movies
data[['Profit','movie_title']].sort_values(by = 'Profit', ascending = False).head(10)

```
<img width="359" height="262" alt="image" src="https://github.com/user-attachments/assets/1996db25-ca70-48c2-be12-a121f1257e10" />



## Conclusion

The analysis highlights that movie profitability is influenced not only by gross earnings but also by factors such as budget control, language, genre, and audience engagement. High-budget films often achieve strong gross revenues, but sustainable profitability is more evident in genres and languages with consistent ROI. Correlation analysis confirms the importance of audience votes and ratings as reliable indicators of commercial success. Furthermore, the recommendation system demonstrates the value of leveraging content similarity across genres, actors, and directors to enhance user experience. Overall, the project provides actionable insights for data-driven decision-making in film production, marketing, and distribution.
