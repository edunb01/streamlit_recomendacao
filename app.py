
import streamlit as st


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





columns = ['id_null','region','date','client_id','service_id','device','platform','data_source', 'source','medium','channel_grouping','campaign','keyword','landing_page']
data = pd.read_csv("./data/solicitacoes_servicos_por_estado.csv", names=columns, header=0)


data = data.drop_duplicates()
data = data[data['service_id'].map(data['service_id'].value_counts() >= 400)]

X = data[['region','date','device','platform','source','landing_page']]
y = np.array(data['service_id'])

def date_transform(data: pd.DataFrame):

    data['month'] = data['date'].apply(lambda x: int(str(x)[4:6]))
    data['sin_month'] = data['month'].apply(lambda x: np.sin(2*np.pi*(1-x)/12))
    data['cos_month'] = data['month'].apply(lambda x: np.cos(2*np.pi*(1-x)/12))

    dt_datetime = pd.to_datetime(
        data['date'],
        format="%Y-%m-%d") # provavelmente será necessário realizar um ajuste nesse formato, já que é diferente do anterior
                        # "%Y%m%d%s%Z"

    data['day'] = dt_datetime.dt.weekday + 1 # segunda = 1 e domingo = 7

    data = data.drop(columns=['month','date'])
    #data = data.drop(columns='date')


    return data

preprocess_pipe = make_pipeline(
    FunctionTransformer(date_transform),
    #FunctionTransformer(lat_lon_norm),
    make_column_transformer(
        #(MinMaxScaler(), ['page_views']),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), ['region','device','platform','source','landing_page']), # incluir outras features como 'plataform','channelGrouping'
        remainder="passthrough")
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)

y_train_cat = to_categorical(y_train_le)
y_test_cat = to_categorical(y_test_le)


X_train_proc = preprocess_pipe.fit_transform(X_train)
X_test_proc = preprocess_pipe.transform(X_test)


st.header("Recomendações de serviço")

st.write("Nessa primeira versão online podemos escolher o número de uma linha do X teste e ver os valores preditos")

dados = X_test.reset_index()
#st.dataframe(dados)

selected_indices = st.selectbox('Select rows:', dados.index,index=0)



model = keras.models.load_model('./data/ModeloNN/ModeloRedeNeural')

st.header("Valor X de teste")
st.table(X_test.iloc[selected_indices,:])
st.header("Valor de Y Verdadeiro")
st.write(y_test[selected_indices])

y_pred_proba = model.predict(X_test_proc)
recomendados = le.classes_[np.argsort(y_pred_proba[selected_indices,:])[-5:]]
st.subheader("Valores recomendados")
st.table(recomendados)
