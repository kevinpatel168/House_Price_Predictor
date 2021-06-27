#!/usr/bin/env python
# coding: utf-8

# # HOUSE PREDICTION USING MACHINE LEARNING AND DATA SCIENCE

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# ## Data Load: Load banglore home prices into a dataframe

# In[3]:


df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()


# In[7]:


df1.shape


# In[8]:


df1.columns


# In[9]:


df1['area_type'].value_counts()


# In[10]:


df1.groupby('area_type')['area_type'].agg('count')


# ## Drop features that are not required to build our model

# In[11]:


df2 = df1.drop(['area_type','society','availability'],axis='columns')
df2.shape


# In[12]:


df2.head(10)


# ## Data Cleaning: Handle NA values

# In[13]:


df2.isnull().sum()


# ## find median of balcony value and missing value is replaced by that median value 

# In[14]:


df2['balcony'].median()


# In[15]:


df2['balcony'].replace(np.NaN,df2['balcony'].median(),inplace=True)


# In[16]:


df2.isnull().sum()


# In[17]:


df2.head(10)


# In[18]:


df3 = df2.dropna()
df3.isnull().sum()


# In[19]:


df3.head(3349)


# In[20]:


df3.shape


# ## Feature Engineering

# In[21]:


df3['size'].unique()


# In[22]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[23]:


df3.head()


# In[24]:


df3[df3.bhk>20]


# ## Explore total_sqft feature

# In[25]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[26]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# ## Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value in the range. There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion. I am going to just drop such corner cases to keep things simple
# 
# 

# In[27]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[28]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4.head(647)


# In[29]:


df4.loc[549]


# ## Feature Engineering
# 
# 

# ## Add new feature called price per square feet

# In[30]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# ## Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations
# 
# 

# In[31]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[32]:


location_stats.values.sum()


# In[33]:


len(location_stats[location_stats>10])


# In[34]:


len(location_stats[location_stats<=10])


# ## Dimensionality Reduction
# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns

# In[35]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[36]:


len(df5.location.unique())


# In[37]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[38]:


df5.head(20)


# ## Outlier Removal Using Business Logic
# As a data scientist when you have a conversation with your business manager (who has expertise in real estate), he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft

# In[39]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[40]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# ## Outlier Removal Using Standard Deviation and Mean

# In[41]:


df6.price_per_sqft.describe()


# ### Here we find that min price per sqft is 267 rs/sqft whereas max is aprox 170000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation

# In[42]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# ### Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

# In[43]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[44]:


plot_scatter_chart(df7,"Hebbal")


# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

# In[45]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[48]:


plot_scatter_chart(df8,"Hebbal")


# In[47]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[49]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# Outlier Removal Using Bathrooms Feature

# In[50]:


df8.bath.unique()


# In[51]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[52]:


df8[df8.bath>10]


# It is unusual to have 2 more bathrooms than number of bedrooms in a home

# In[54]:


df8[df8.bath>df8.bhk+2]


# In[53]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[55]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# Use One Hot Encoding For Location (to convert text data into numerical data)

# In[56]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[57]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[58]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# Build a Model Now...

# In[59]:


df12.shape


# X variable contain only independent variable.so we have only one dependent variable that is price so we drop price column.

# In[60]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[61]:


X.shape


# In[63]:


y = df12.price
y.head()


# In[64]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[65]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# Use K Fold cross validation to measure accuracy of our LinearRegression model

# In[66]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose

# Find best model using GridSearchCV

# In[67]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
        
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.
# 
# Test the model for few properties

# In[68]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[69]:





# In[70]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[71]:


predict_price('Indira Nagar',1000, 2, 2)


# In[72]:


predict_price('Indira Nagar',1000, 3, 3)




