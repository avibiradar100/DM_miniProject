import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set_style('white')

# Split
from sklearn.model_selection import train_test_split

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('Electronics_data.csv')

st.title("Data Analytics for Amazon Reviews")
st.write("")

st.subheader("Dataset:")
st.write(pd.DataFrame(df.head(20)))

st.write("Total Reviews:",df.shape[0])
st.write("Total Columns:",df.shape[1])
st.write("Total number of ratings :",df.rating.nunique())
st.write("Total number of users   :", df.userId.nunique())
st.write("Total number of products  :", df.productId.nunique())

st.subheader("Ratings summary ")
df.describe()['rating']

# Average rating of products
st.subheader("Average rating of products")
ratings = pd.DataFrame(df.groupby('productId')['rating'].mean())
ratings['ratings_count'] = pd.DataFrame(df.groupby('productId')['rating'].count())
ratings['ratings_average'] = pd.DataFrame(df.groupby('productId')['rating'].mean())
st.write(ratings.head(10))

#histogram
ratings['rating'].hist(bins=70)
st.text(" ")
data=ratings['rating'].to_list()
plt.rcParams['figure.figsize'] = [10, 4]
st.write("Histogram")
fig, ax = plt.subplots()
plt.locator_params(nbins = 10)
plt.xlabel('rating')
plt.ylabel("rating_count")
ax.hist(data)
st.pyplot(fig)