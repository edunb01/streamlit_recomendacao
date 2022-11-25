import streamlit as st

import pickle
import json
import datetime as dt

# Data manipulation
import numpy as np
import pandas as pd

# # Data Visualiation
# import matplotlib.pyplot as plt
# import seaborn as sns

# System
# import os
# import glob

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

# # Machine Learning
# from sklearn.neighbors import KNeighborsClassifier

# Deep Learning
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# ajustar para sua máquina
# file_path = './data/solicitacoes_servicos_por_estado24NovNoite.csv'

# columns = ['id_null','region','date','client_id','service_id','device','platform','data_source', 'source','medium','channel_grouping','campaign','keyword','landing_page', 'dias_entre_solicitações','ultimo_servico', 'dia_primeiro_acesso']
# data = pd.read_csv(file_path, names=columns, header=0)

# data = data.drop_duplicates()

# data = data.fillna('primeira solicitação')

# # Há um desbalanceamento na classe de service_id
# # Vamos assumir que o modelo só consegue sugerir os serviços que são 95% mais utilizados
# # Porque não temos dados suficientes para o modelo aprender esses outros 5%

# data = data[data['service_id'].map(data['service_id'].value_counts() >= 30)]
# data = data[data['source'].map(data['source'].value_counts() >= 10)]

# """# Preprocessing"""

# X = data[['region','date','device','platform','source', 'dias_entre_solicitações', 'ultimo_servico', 'dia_primeiro_acesso']]
# y = np.array(data['service_id'])

def date_transform(data: pd.DataFrame):

    data['month'] = data['date'].apply(lambda x: int(str(x)[4:6]))

    dt_datetime = pd.to_datetime(
        data['date'],
        format="%Y-%m-%d")

    primeiro_acesso_datetime = pd.to_datetime(
        data['dia_primeiro_acesso'],
        format="%Y-%m-%d")

    data['day'] = dt_datetime.dt.weekday + 1 # segunda = 1 e domingo = 7
    data['dias_de_uso'] = (dt_datetime - primeiro_acesso_datetime).dt.days

    data = data.drop(columns=['date', 'dia_primeiro_acesso'])


    return data

def bin_dias_solicitações(data: pd.DataFrame):

    bins = pd.IntervalIndex.from_tuples([(-1, 0), (0,7), (7, 30), (30, 180), (180, 365), (365,float('inf'))])

    data['dias_solicitações_bin'] = pd.cut(data['dias_entre_solicitações'], bins=bins)
    data = data.drop(columns='dias_entre_solicitações')

    data['dias_de_uso_bin'] = pd.cut(data['dias_de_uso'], bins=bins)
    data = data.drop(columns='dias_de_uso')


    return data

# preprocess_pipe = make_pipeline(
#     FunctionTransformer(date_transform),
#     FunctionTransformer(bin_dias_solicitações),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown="ignore", sparse=False), ['region','device','platform','source','day', 'month','ultimo_servico', 'dias_solicitações_bin', 'dias_de_uso_bin']),
#         remainder="passthrough")
# )

# """## Classificação Redes Neurais"""

# # train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# # preprocessing

# X_train_proc = preprocess_pipe.fit_transform(X_train)
# X_test_proc = preprocess_pipe.transform(X_test)

# le = LabelEncoder()
# y_train_le = le.fit_transform(y_train)
# y_test_le = le.transform(y_test)
# y_train_cat = to_categorical(y_train_le)
# y_test_cat = to_categorical(y_test_le)

colunas = ['region','date','device','platform','source', 'dias_entre_solicitações', 'ultimo_servico', 'dia_primeiro_acesso']

with open("./data/dicionario_categoricas.json", "r") as read_content:
    dicionario = (json.load(read_content))

region = st.selectbox('Escolha a Região:', dicionario['region'],index=0)
device = st.selectbox('Escolha o aparelho:', dicionario['device'],index=0)
platform = st.selectbox('Escolha a plataforma:', dicionario['platform'],index=0)
source = st.selectbox('Escolha a source:', dicionario['source'],index=0)
ultimo = st.selectbox('Escolha o último serviço:', dicionario['ultimo_servico'],index=0)
date = st.date_input("data do serviço", value=dt.date.fromisoformat(dicionario['date'][0]),
                     min_value=dt.date.fromisoformat(dicionario['date'][0]),
                     max_value=dt.date.fromisoformat(dicionario['date'][1]),key='data')

dias_entre = st.number_input("Escolha o número de dias entre solicitações", min_value=int(dicionario['dias_entre_solicitações'][0]), max_value=int(dicionario['dias_entre_solicitações'][1]), value=1)
dia_primeiro_acesso = st.date_input("Escolha a data do primeiro acesso", value=dt.date.fromisoformat(dicionario['dia_primeiro_acesso'][0]),
                     min_value=dt.date.fromisoformat(dicionario['dia_primeiro_acesso'][0]),
                     max_value=dt.date.fromisoformat(dicionario['dia_primeiro_acesso'][1]),key='dia_primeiro_acesso')

X_novo = pd.DataFrame(columns = colunas)
X_novo.loc[0] = [region,date,device,platform,source,dias_entre,ultimo,dia_primeiro_acesso]
model = keras.models.load_model('./data/ModeloNN/ModeloRedeNeural')
object_preprocessa = open("./data/preprocess_pipe.pickle", "rb")
preprocess_pipe = pickle.load(object_preprocessa)
object_preprocessa.close()
objectRep = open("./data/label_encoder.pickle", "rb")
le = pickle.load(objectRep)
objectRep.close()
y_pred_proba = model.predict(preprocess_pipe.transform(X_novo))
servicos_preditos = le.inverse_transform(np.apply_along_axis(np.argmax,1,y_pred_proba))
st.write(servicos_preditos)
# servicos_observados = le.inverse_transform(np.apply_along_axis(np.argmax,1,y_test_cat))

# st.header("Recomendações de serviço")

# st.write("Nessa primeira versão online podemos escolher o número de uma linha do X teste e ver os valores preditos")

# dados = X_test.reset_index()
# #st.dataframe(dados)

# with st.sidebar:
#     st.image('./header.png')
#     selected_indices = st.selectbox('Select rows:', dados.index,index=0)





st.header("Valor X de teste")
st.dataframe(X_novo)
# st.table(X_test.iloc[selected_indices,:])
# st.header("Serviço Y Verdadeiro")
# st.write(y_test[selected_indices])

# y_pred_proba = model.predict(X_test_proc)
# recomendados = le.classes_[np.argsort(y_pred_proba[selected_indices,:])[-5:]]
# st.subheader("Serviços recomendados")
# st.table(recomendados)
# aleatorios = np.random.randint(0,len(le.classes_),2)

# def header(elementos):
#      st.write(f'<p style="color:#33ff33;font-size:18px;border-radius:2%;">{elementos}</p>', unsafe_allow_html=True)

# header(le.classes_[aleatorios[0]])
# header(le.classes_[aleatorios[1]])
