# Movie-Recommendation-Analysis
Movie Recommendation Analysis Using Python

## Objective

The objective of this project is to analyze movie performance data to uncover patterns in profitability, language-based gross earnings, and genre-level financial outcomes. By examining correlations between budget, gross revenue, profit, return on investment (ROI), ratings, and audience engagement, the project seeks to identify the key factors driving commercial success. Additionally, recommendation systems will be developed using movie genres, actors, and content similarity to suggest films that align with user preferences. This analysis will provide actionable insights for decision-making in film production, marketing, and distribution strategies.

## Problem Statement:
Perform analysis and Basic Recommendations based on Similar Genres and Movies which Users prefer. Some of the Key Points on which we will be focusing include: 

1.Profitability of Movies 

2.Language based Gross Analysis 

3.Comparison of Gross and Profit for Different Genres. 

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
```

### 1.Profitability of Movies

```
# lets create a profit column using the budget and gross
data['Profit'] = data['gross'] - data['budget']

# lets also check the name of Top 10 profitable movies
data[['Profit','movie_title']].sort_values(by = 'Profit', ascending = False).head(10)

```
Output:

<img width="359" height="262" alt="image" src="https://github.com/user-attachments/assets/1996db25-ca70-48c2-be12-a121f1257e10" />


### 2.Language based Gross Analysis
```
### 2.Language based Gross Analysis
# lets check the values in the language column
data['language'].value_counts()

```
Output:

<img width="185" height="327" alt="image" src="https://github.com/user-attachments/assets/e0ee4857-475d-44b6-8a91-562b7d5859b1" />

```
# looking at the above output we can easily observe that out of 3,500 movies only 150 movies are of other languages

# so it is better to keep only two languages that is English and Foreign
def language(x):
    if x == 'English':
        return 'English'
    else:
        return 'Foreign'

# lets apply the function on the language column
data['language'] = data['language'].apply(language)

# lets check the values again
data['language'].value_counts()
```
Output:

<img width="205" height="65" alt="image" src="https://github.com/user-attachments/assets/a2a196f7-89fd-4ebb-8d1b-53e1a0d5aa84" />

```
# lets define a function for categorizing Duration of movies

def duration(x):
    if x <= 120:
        return 'Short'
    else:
        return 'Long'

# lets apply the function on the duration column
data['duration'] = data['duration'].apply(duration)

# lets check the values of Duration column
data['duration'].value_counts()
```
Output:

<img width="203" height="59" alt="image" src="https://github.com/user-attachments/assets/02d934bd-02c8-48b3-a0fe-0be48adffbd3" />


### 3.Comparison of Gross and Profit for Different Genres
```
# lets check the value in the Genres column

data['genres'].value_counts()
```
Output:

<img width="319" height="176" alt="image" src="https://github.com/user-attachments/assets/ff30de3a-71c3-4414-9993-21d9d21f6cde" />

```
data['genres'].str.split('|')[0]
```
Output:

<img width="324" height="31" alt="image" src="https://github.com/user-attachments/assets/4c370b1b-6b95-4bad-ba08-e227ca4049b7" />

```
# we can see from the above output that most of the movies are having a lot of genres
# also, a movie can have so many genres so lets keep four genres

data['Moviegenres'] = data['genres'].str.split('|')
data['Genre1'] = data['Moviegenres'].apply(lambda x: x[0])

# some of the movies have only one genre. In such cases, assign the same genre to 'genre_2 as well

data['Genre2'] = data['Moviegenres'].apply(lambda x: x[1] if len(x) > 1 else x[0])
data['Genre3'] = data['Moviegenres'].apply(lambda x: x[2] if len(x) > 2 else x[0])
data['Genre4'] = data['Moviegenres'].apply(lambda x: x[3] if len(x) > 3 else x[0])

# lets check the head of the data
data[['genres','Genre1','Genre2','Genre3','Genre4']].head(5)
```
Output:

<img width="404" height="146" alt="image" src="https://github.com/user-attachments/assets/ff4c9d7e-9d56-4663-9e8d-c4e789c2520e" />

```
# lets also calculate the social Media Popularity of a Movie

# to calculate popularity of a movie, we can aggregate No. of voted users, No. of users for Reviews and facebook likes.

data['Social_Media_Popularity'] = ((data['num_user_for_reviews']/data['num_voted_users'])*(data['movie_facebook_likes']))/1000000

# lets also check the Top 10 Most Popular Movie on Social Media
x = data[['movie_title','Social_Media_Popularity']].sort_values(by = 'Social_Media_Popularity',
                                                                ascending = False).head(10).reset_index()
print(x)

sns.barplot(x='movie_title', y ='Social_Media_Popularity', data=x, palette = 'magma')
plt.title('Top 10 Most Popular Movies on Social Media',fontsize = 20)
plt.xticks(rotation = 90, fontsize = 14)
plt.xlabel(' ')
plt.show()
```
Output:

<img width="520" height="143" alt="image" src="https://github.com/user-attachments/assets/9a88aff2-1757-426c-88a3-d4952bd84b04" />

<img width="560" height="386" alt="image" src="https://github.com/user-attachments/assets/2c2a6929-8fc0-4e48-91fe-46f8954f0e24" />

```
# Analyzing which Genre is most Bankable

# Lets compare the Gross with Genres

# first group the genres and get max, min, avg gross of the movies of that genre
display(data[['Genre1','gross',]].groupby(['Genre1']).agg(['max','mean','min']).style.background_gradient(cmap = 'Wistia'))

# lets plot these values using lineplot
data[['Genre1','gross',]].groupby(['Genre1']).agg(['max','mean','min']).plot(kind = 'line',color = ['red','black','blue'])
plt.title('Which Genre is Most Bankable?',fontsize =20)
plt.xticks(np.arange(17),['Action','Adventure','Amination','Biography','Comedy','Crime','Documentary','Drama','Family','Fantasy','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','Western'], rotation = 90, fontsize = 15)
plt.ylabel('Gross',fontsize = 15)
plt.xlabel(' ',)
plt.show()
```

Output:

<img width="246" height="360" alt="image" src="https://github.com/user-attachments/assets/d2a21d7d-9daf-4baa-9544-794146c50cbc" />
<img width="628" height="361" alt="image" src="https://github.com/user-attachments/assets/254169e5-7136-4100-a9e4-296458d82cf4" />

```
print('The Most Profitable Movie from each Genre')
display(data.loc[data.groupby(data['Genre1'])['Profit'].idxmax()][['Genre1','movie_title','gross']].style.background_gradient(cmap = 'copper'))
```
Output:

<img width="305" height="329" alt="image" src="https://github.com/user-attachments/assets/23e9be9f-cfb3-4d2f-a8dc-6b29de1ac9c9" />

```
# Loss and Profit Analysis on English and Foreign Movies

# lets covert year into interger
data['title_year'] = data['title_year'].astype('int')

print('Most Profitable years in Box office')
display(data[['title_year','language','Profit']].groupby(['language','title_year']).agg('sum').sort_values(by = 'Profit',
                                                                                                           ascending = False).head(10).style.background_gradient(cmap = 'Greens'))
```
Output:

<img width="213" height="224" alt="image" src="https://github.com/user-attachments/assets/ec4054b1-177e-42e6-a72f-69880b5407fb" />


```
# lets plot then
sns.lineplot(data=data, x='title_year',y ='Profit',hue ='language')
plt.title('Time series for Box office Profit for English vs Foreign Movies', fontsize = 20)
plt.xticks(fontsize = 18)
plt.xlabel(' ')
plt.show()
```

Output:

<img width="628" height="301" alt="image" src="https://github.com/user-attachments/assets/2f72125b-2c9b-4763-bd84-72ea877be850" />
```
print('Movies that Made Huge Losses')
display(data[data['Profit'] < -2000][['movie_title',
                                      'language','Profit']].style.background_gradient(cmap = 'Reds'))
```

Output:

<img width="280" height="158" alt="image" src="https://github.com/user-attachments/assets/4d0e16ec-f454-4859-9a28-f18cf31a4896" />

```
# Gross Comparison of Long and Short Movies

display(data[data['duration'] == 'Long'][['movie_title','duration','gross', 'Profit']].sort_values(by = 'Profit',ascending = False).head(5).style.background_gradient(cmap = 'spring'))

display(data[data['duration'] == 'Short'][['movie_title','duration','gross', 'Profit']].sort_values(by = 'Profit',ascending = False).head(5).style.background_gradient(cmap = 'spring'))
```
Output:
<img width="430" height="281" alt="image" src="https://github.com/user-attachments/assets/98b7e09d-14f0-4252-95b4-3431d67d4850" />

```
sns.barplot(data=data, x= 'duration', y= 'gross', hue ='language', palette ='spring')
plt.title('Gross Comparison')
plt.show()
```
Output:

<img width="676" height="344" alt="image" src="https://github.com/user-attachments/assets/b6ce3bde-dcad-42f7-a1fb-945dd6b53562" />

```
# Association between IMDB Rating and Duration

print('Average IMDB Score for Long Duration Movies is {0:2f}'.format(data[data['duration'] =='Long']['imdb_score'].mean()))
print('Average IMDB Score for Short Duration Movies is {0:2f}'.format(data[data['duration'] =='Short']['imdb_score'].mean()))
```

Output:

<img width="455" height="37" alt="image" src="https://github.com/user-attachments/assets/ebd639a2-005f-43c1-b1be-211d8e88a97c" />

```
print('\nHighest Rated Long Duration Movie\n',
      data[data['duration'] == 'Long'][['movie_title','imdb_score']].sort_values(by = 'imdb_score', ascending = False).head(1))

print('\nHighest Rated Short Duration Movie\n',
      data[data['duration'] == 'Short'][['movie_title','imdb_score']].sort_values(by = 'imdb_score', ascending = False).head(1))
```
Output:

<img width="338" height="109" alt="image" src="https://github.com/user-attachments/assets/8804da8b-b11e-46c0-afb5-a2b79ba2ae17" />

```
sns.boxplot(data=data,x='imdb_score',y='duration',palette = 'copper')
plt.title('IMDB Ratings vs Gross', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
```
Output:

<img width="667" height="335" alt="image" src="https://github.com/user-attachments/assets/6c8f7da4-1609-41c6-819f-fad005232a15" />


```
# Comparing Critically Acclaimed Actors

def query_actors(x):
    # filter rows where actor_1, actor_2, or actor_3 is equal to x
    mask = (
        (data['actor_1_name'] == x) |
        (data['actor_2_name'] == x) |
        (data['actor_3_name'] == x)
    )
    
    # select relevant columns
    result = data.loc[mask, ['movie_title','budget','gross','title_year','genres','language','imdb_score']]
    return result
```

```
query_actors('Meryl Streep')
```

Output:

<img width="589" height="374" alt="image" src="https://github.com/user-attachments/assets/e40ceff6-7e1f-443a-b808-0a9288c1540e" />


```
def actors_report(x):
    # Filter for actor in any of the 3 actor columns
    a = data[data['actor_1_name'] == x]
    b = data[data['actor_2_name'] == x]
    c = data[data['actor_3_name'] == x]
    
    y = pd.concat([a, b, c], ignore_index=True)

    print("Time:", y['title_year'].min(), y['title_year'].max())
    print("Max Gross: {:0.2f} Millions".format(y['gross'].max()))
    print("Avg Gross: {:0.2f} Millions".format(y['gross'].mean()))
    print("Min Gross: {:0.2f} Millions".format(y['gross'].min()))
    print("Number of 100 Million Movies:", (y['gross'] > 100).shape[0])
    print("Avg IMDB Score: {:0.2f}".format(y['imdb_score'].mean()))
    print("Most Common Genres:\n", y['Genre1'].value_counts().head())
    
actors_report('Meryl Streep')
```
Output:

<img width="231" height="191" alt="image" src="https://github.com/user-attachments/assets/d9925efe-781c-4c59-98fe-6351d37f8a99" />

```
def critically_acclaimed_actors(m):
    # Filter all movies where actor appears in any actor column
    a = data[data['actor_1_name'] == m]
    b = data[data['actor_2_name'] == m]
    c = data[data['actor_3_name'] == m]
    
    y = pd.concat([a, b, c], ignore_index=True)

    # Sum critic reviews safely (ignores NaN automatically)
    return int(y['num_critic_for_reviews'].sum())
    
actors_list = ['Brad Pitt', 'Leonardo DiCaprio', 'Tom Cruise']

for actor in actors_list:
    reviews = critically_acclaimed_actors(actor)
    print(f"Number of Critic Reviews for {actor}: {reviews}")
```
Output:
<img width="355" height="54" alt="image" src="https://github.com/user-attachments/assets/0090634a-1c5a-4bd7-906b-a18a9665d6fa" />

```
# Top movies based on Gross, and IMDB
pd.set_option('display.max_rows', 30000)

@interact
def show_movies_more_than(column='imdb_score', score=9.0):
    # filter based on column and score
    x = data.loc[data[column] > score][[
        'title_year','movie_title','director_name',
        'actor_1_name','actor_2_name','actor_3_name',
        'Profit','imdb_score',
    ]]
    
    # sort by IMDB score (highest first)
    x = x.sort_values(by='imdb_score', ascending=False)
    
    # drop duplicate movies by title
    x = x.drop_duplicates(subset='movie_title', keep='first')
    
    return x
```

<img width="752" height="71" alt="image" src="https://github.com/user-attachments/assets/b1ab3c6f-6e6c-4839-b699-87484b3aef25" />

```
pd.set_option('display.max_rows', 30000)

@interact
def show_articles_more_than(column=['budget','gross'], x=1000):
    return data.loc[data[column] > x][['movie_title','duration','gross','Profit','imdb_score']]
```
Output:

<img width="394" height="166" alt="image" src="https://github.com/user-attachments/assets/8671a6f3-c0ee-44ef-bea3-ef57e71979f8" />

### 4.Recommendation systems based on Languages,Actors,Genres. 
#### Recommending Movies Based on Languages

```def recommend_lang(x):
   y = data[['language','movie_title','imdb_score']][data['language'] == x]
   y = y.sort_values(by = 'imdb_score', ascending = False)
   return y.head(15)
   recommend_lang('Foreign')
```
Output:

<img width="338" height="308" alt="image" src="https://github.com/user-attachments/assets/c90d6d2d-0f70-4db3-9f00-9a4b1ef4ec9e" />


#### Recommending Movies based on Actors

```def recommend_movies_on_actors(x):
    # Filter all movies where actor appears in any actor column
    a = data[['movie_title','imdb_score']][data['actor_1_name'] == x]
    b = data[['movie_title','imdb_score']][data['actor_2_name'] == x]
    c = data[['movie_title','imdb_score']][data['actor_3_name'] == x]
    
    y = pd.concat([a, b, c], ignore_index=True)
    y = y.sort_values(by = 'imdb_score', ascending = False)

    return y.head(15)

recommend_movies_on_actors('Tom Cruise')

```

Output:

<img width="325" height="290" alt="image" src="https://github.com/user-attachments/assets/4efaf44c-5cc0-44da-9c1e-1a6e10d71859" />

#### Recommending Movies of similar Genres


```from mlxtend.preprocessing import TransactionEncoder

x = data['genres'].str.split('|')
te = TransactionEncoder()
x = te.fit_transform(x)
x = pd.DataFrame(x,columns = te.columns_)

x.head()
```
Output:

<img width="812" height="183" alt="image" src="https://github.com/user-attachments/assets/e86ba328-b573-4bd9-b88f-b3ff37140abc" />

```genres = x.astype('int')
genres.head()
```
Output:

<img width="809" height="185" alt="image" src="https://github.com/user-attachments/assets/97bd2a8a-95fc-4645-91d6-40cd8b3a58c6" />

```genres.insert(0,'movie_title',data['movie_title'])
genres.head()
```
Output:

<img width="791" height="218" alt="image" src="https://github.com/user-attachments/assets/519aaad8-11c0-41f1-a870-6dbfa4d7690e" />

### Conclusion

The analysis highlights that movie profitability is influenced not only by gross earnings but also by factors such as budget control, language, genre, and audience engagement. High-budget films often achieve strong gross revenues, but sustainable profitability is more evident in genres and languages with consistent ROI. Correlation analysis confirms the importance of audience votes and ratings as reliable indicators of commercial success. Furthermore, the recommendation system demonstrates the value of leveraging content similarity across genres, actors, and directors to enhance user experience. Overall, the project provides actionable insights for data-driven decision-making in film production, marketing, and distribution.
