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

st.header("Data Analysis of Amazon Reviews")

st.write(pd.DataFrame(df.head(20)))

st.write("Total Reviews:",df.shape[0])
st.write("Total Columns:",df.shape[1])

# Taking subset of the dataset
df = df.iloc[:5000,0:]

print("Total Reviews:",df.shape[0])
print("Total Columns:",df.shape[1])