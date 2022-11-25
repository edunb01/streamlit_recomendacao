import streamlit as st

import pickle
import json
import datetime as dt

# Data manipulation
import numpy as np
import pandas as pd

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
# from tensorflow.keras.utils import to_categorical


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
servicos_preditos = le.inverse_transform(list(reversed(np.argsort(y_pred_proba,1)[0][-5:])))

# servicos_observados = le.inverse_transform(np.apply_along_axis(np.argmax,1,y_test_cat))

st.header("Recomendações de serviço")
st.table(servicos_preditos)
# st.write("Nessa primeira versão online podemos escolher o número de uma linha do X teste e ver os valores preditos")

# with st.sidebar:
#     st.image('./header.png')
#     selected_indices = st.selectbox('Select rows:', dados.index,index=0)






aleatorios = np.random.randint(0,len(le.classes_),2)

def header(elementos):
     for i in elementos:
        st.markdown(f'<p style="color:#33ff33;font-size:18px;border-radius:2%;">{i}</p>', unsafe_allow_html=True)

header(le.inverse_transform(aleatorios))
