# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:42:08 2021

@author: Théophile Le Clerc
"""

import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import time



st.sidebar.title('Menu')


options = ['Régression logistique', 'Arbre de décision', 'KNN']
choix = st.sidebar.selectbox('Choix du modèle', options = options)


test_size = st.sidebar.slider('Test size', 0., 1., 0.2, 0.1)



st.title("Projet Titanic")


st.subheader("Par Thibault et Théophile")


st.write("Court paragraphe d'introduction")



df = pd.read_csv("titanic.csv")



st.dataframe(df)

X = df.drop(['Name', 'Sex','Survived'], axis = 1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 512)







st.write('Modèle sélectionné : ', choix)


@st.cache
def train_model(choix):
    if choix == options[0]:
        model = LogisticRegression()
    elif choix == options[1]:
        model = DecisionTreeClassifier()
    else :
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    time.sleep(3)
    return model.score(X_test, y_test)


st.write("L'accuracy du modèle est : ", train_model(choix))


