# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:16:08 2021

@author: Théophile Le Clerc
"""

import streamlit as st

import time
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns




### bande latérale



st.sidebar.title('Menu')

#slider
test_size = st.sidebar.slider('Proportion du jeu de test', 0.1, 0.5, 0.2, 0.1)


#select box
options = ['Logistic Regression','Decision tree Classifier','KNN']

choix = st.sidebar.selectbox('Faire un choix', options = options)




###page principale

#organisation en titre, sous-titre et texte
st.title("Titanic")

st.subheader('Projet de DataScience par Théophile')

st.write("1 paragraphe très court d'introduction")


#données

@st.cache 
def get_data(test_size):
    
    df = pd.read_csv('titanic.csv')
    
    df_sex = df.groupby('Sex').Age.count()
    
    X = df.drop(['Name', 'Sex'], axis = 1)
    
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 56)
    
    return df, df_sex, X_train, X_test, y_train, y_test

df, df_sex, X_train, X_test, y_train, y_test = get_data(test_size)


#plot avec matplotlib ou seaborn
fig = plt.figure(figsize = (10,7))

sns.countplot(df['Sex'])

st.pyplot(fig)

#plot interactif "natif" avec streamlit
st.bar_chart(df_sex)






@st.cache
def train(choix):
    time.sleep(3)
    if choix == options[0]:
        model = LogisticRegression()
        
    elif choix == options[1]:
        model = DecisionTreeClassifier()
    
    else :
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

st.write('Score ' + choix + ' :', train(choix))

























