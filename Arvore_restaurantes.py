import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# Abrindo csv
base = pd.read_csv('restaurantev2.csv')
#base2 = pd.read_csv('/content/sample_data/restaurante.csv', ';', usecols=['Alternativo', 'Bar'])

#Contando quantidade de instâncias
np.unique(base['Conclusao'], return_counts=True)
sns.countplot(x=base['Conclusao'])

#Separando atributos de entrada e de classe
X_prev = base.iloc[:, 0:10].values
y_classe = base.iloc[:, 10].values

#Tratamento de dados categoricos
label_encoder_Alternativo = LabelEncoder()
label_encoder_Bar = LabelEncoder()
label_encoder_SexSab = LabelEncoder()
label_encoder_fome = LabelEncoder()
label_encoder_chuva = LabelEncoder()
label_encoder_Res = LabelEncoder()

X_prev[:, 0] = label_encoder_Alternativo.fit_transform(X_prev[:, 0])
X_prev[:, 1] = label_encoder_Bar.fit_transform(X_prev[:, 1])
X_prev[:, 2] = label_encoder_SexSab.fit_transform(X_prev[:, 2])
X_prev[:, 3] = label_encoder_fome.fit_transform(X_prev[:, 3])
X_prev[:, 4] = label_encoder_chuva.fit_transform(X_prev[:, 4])
X_prev[:, 5] = label_encoder_Res.fit_transform(X_prev[:, 5])
X_prev[:, 6] = label_encoder_Alternativo.fit_transform(X_prev[:, 6])
X_prev[:, 7] = label_encoder_Bar.fit_transform(X_prev[:, 7])
X_prev[:, 9] = label_encoder_SexSab.fit_transform(X_prev[:, 9])

len(np.unique(base['Cliente']))

onehotencoder_restaurante = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')


X_prev = onehotencoder_restaurante.fit_transform(X_prev)


y_classe
y_classe.shape

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_prev, y_classe, test_size=0.20, random_state=23)

X_treino.shape
X_teste.shape

# Decision Tree
modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste)
