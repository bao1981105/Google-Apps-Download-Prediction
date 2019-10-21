# <center>Google Play Store Apps Project Proposal</center>
## <center>by Xu Han</center>
## Summary
While many public data sets provided Apple store data, there is not enough analysis into apps in Google Play store, partly because it’s more difficult to scrape data from Google Play store. 

Nowadays, as mobile phones become more popular, people tend to spend more time on their phones and app usage increases a log. While this is a great opportunity for many app developers, it also becomes a challenge for many developers/businesses to develop a popular app. 

In this project, I will try to predict the number of installs (target variable) from some features of the app itself, shown in googleaplystore. I am trying to find out what kind of apps are more popular and tend to stay longer in people’s phones. Since the exact number of installs was not available, but an estimate was given, it is treated as categorical variable, so the problem will be a classification problem.


## Data
The Google Play Store Apps was downloaded from Kaggle [Link](https://www.kaggle.com/lava18/google-play-store-apps). The information was scraped from the Google and the dataset was updated eight months ago. 

The dataset is googleaplystore.csv. This dataset has 10,841 entries, each representing an app listed in Google Store. There are 13 columns in the original dataset.
These columns include:

  * App: the name of the App.
  * Category: the category the App belongs to. Its values include: 1.9, ART_AND_DESIGN, AUTO_AND_VEHICLES, BEAUTY, BOOKS_AND_REFERENCE, BUSINESS, COMICS, COMMUNICATION, DATING, EDUCATION, ENTERTAINMENT, EVENTS, FAMILY, FINANCE, FOOD_AND_DRINK, GAME, HEALTH_AND_FITNESS, HOUSE_AND_HOME, LIBRARIES_AND_DEMO, LIFESTYLE, MAPS_AND_NAVIGATION, MEDICAL, NEWS_AND_MAGAZINES,   PARENTING, PERSONALIZATION, PHOTOGRAPHY, PRODUCTIVITY, SHOPPING, SOCIAL, SPORTS, TOOLS, TRAVEL_AND_LOCAL, VIDEO_PLAYERS, WEATHER. 
  * Rating: numerical value represented using decimal numbers. Ranging from 1 to 5, with 5 being the highest, usually round off to one decimal place such as 4.1. 
  * Number of reviews: number of reviews this app has received.
  * Size: file size of the app. String datatype consisting of an integer number and a unit symbol, some observations are denoted in unit symbol M, megabyte, some are denoted in k, representing kilobyte, while some have the value "vary with device".
  * Number of installs: categorical variable, target variable. Values include: 0, 1+, 5+, 10+, 50+, 100+, 500+, 1000+, 5000+, 10_000+, 50_000+, 100_000+, 500_000+, 5_000_000, 10_000_000+, 50_000_000, 100_000_000+.
  * Type: categorial variable, Paid or Free.
  * Price: numeric value, price of the app, 0 if free, measured in US dollar.
  * Content Rating: categorical variable representing the audience this app is suitable for. Its values include: Everyone, Everyone 10+, Teen, Mature 17+, Adult only 18+, Unrated. From Google's description, Unrated apps are treated like high-maturity apps for the purpose of parental controls until they get a rating. 
  * Genres: categorical variable, similar to Category, but provides more detailed description. Its values include Action, Adventure, Action & Adventure, Arcade, Art & Design, Auto & Vehicles, Beauty, Board, Books & Reference, Brain Games, Business, Card, Casino, Casual, Comics, Communication, Creativity, Dating, Education, Educational, Entertainment, Events, Finance, Food & Drink, Health & Fitness, House & Home, Libraries & Demo, Lifestyle, Maps & Navigation, Medical, Music, Music & Audio, Music & Video, News & Magazines, Parenting, Personalization, Photography, Pretend Play, Productivity, Puzzle, Racing, Role Playing, Shopping, Simulation, Social, Sports, Strategy, Tools, Travel & Local, Trivia, Video Players & Editors, Weather, Word. 
  * Last Updated: last date this app was updated, when this dataset was made.
  * Current Ver: current version of the app. Some values contain "Varies with device".
  * Android Ver: Android operating system requirement. 
  
## Public Work
This dataset has been used to predict App Ratings or perform sentiment analysis. For the App Rating prediction, the author used columns from googleaplystore to perform linear regression, SVR, and random forest models to predict app rating, which he treated as a continues variable. For sentiment analysis, all columns from the googleplaystore_user_reviews have been used to count words and create word cloud. 
  

## Preprocessing

The github repo is [here](https://github.com/bao1981105/Google-Apps-Download-Prediction).

Based on observation, some entries in the original **Genres** column have several tags separated by a semicolon, which implies that when we make these tags as columns, the corresponding App should have value 1 in these columns. So the first step I did was to loop through all 10841 Apps, find the distinct tags, and make them become columns.   

Secondly, I split the dataset into four dataframes: xTrain, xTest, yTrain, yTest, with yTrain and yTest containing only the target variable **Number of installs**. 

Thirdly, OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder are used to transform these dataframes.

  * App: leave it as it is.   
  * Category: OneHotEncoder. Reasoning: Categorical variable without order.
  * Rating: MinMaxScaler. Reasoning: Numeric variable with an upper bound of 5.0.
  * Number of reviews: StandardScaler. Reasoning: this number can be very large and thus does not have upper bound.
  * Size: saved for later.
  * Number of installs: target variable, LabelEncoder. In the original dataset, the exact number of installs was not given. Instead, a range was given, such as 100+, 500+, 10,000+. It does have order, that’s why I treat it as a categorical variable and use LabelEncoder to transform it.
  * Type: has missing values, saved for later.   
  * Price: Mix datatype, has string and numeric value. Saved for later. 
  * Content Rating: OrdinalEncoder. Reasoning: Categorical variable with an order.
  * Genres: OneHotEncoder. Reasoning: Categorical variable without order.
  * Last Updated: saved for later, should be Date datatype instead of Object.
  * Current Ver: Mixed datatype, saved for later.
  * Android Ver: Mixed datatype, saved for later.
  