import numpy as np
import cv2 as cv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings


def load_dataframe():
    '''
    Carrega um dataframe Pandas com as imagens para o treinamento do modelo
    '''
    dados = {
        "ARQUIVO": [],
        "ROTULO": [],
        "ALVO": [],
        "IMAGEM": [],
    }

    com_mascara = os.listdir("imagens/maskon")
    sem_mascara = os.listdir("imagens/maskoff")

    for arquivo in com_mascara:
        dados["ARQUIVO"].append(f"imagens/maskon/{arquivo}")
        dados["ROTULO"].append(f"Com mascara")
        dados["ALVO"].append(1)
        img = cv.cvtColor(cv.imread(f"imagens/maskon/{arquivo}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEM"].append(img)
        
    for arquivo in sem_mascara:
        dados["ARQUIVO"].append(f"imagens/maskoff/{arquivo}")
        dados["ROTULO"].append(f"Sem mascara")
        dados["ALVO"].append(0)
        img = cv.cvtColor(cv.imread(f"imagens/maskoff/{arquivo}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEM"].append(img)
        
    dataframe = pd.DataFrame(dados)

    return dataframe


def train_test(dataframe):
    '''
    Divide o dataframe em conjunto de treino e teste
    '''
    X = list(dataframe["IMAGEM"])
    y = list(dataframe["ALVO"])

    return train_test_split(X, y, train_size=0.40, random_state=13)


def pca_model(X_train):
    '''
    PCA para extração de features das imagens
    '''
    pca = PCA(n_components=30)
    pca.fit(X_train)
    
    return pca


def knn(X_train, y_train):

    warnings.filterwarnings("ignore")
    '''
    Modelo K-Nearest Neighbors
    '''
    grid_params = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
    }

    knn_model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)

    knn_model.fit(X_train, y_train)

    return knn_model

