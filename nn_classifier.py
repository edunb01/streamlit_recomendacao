# Imports

# Data manipulation
import numpy as np
import pandas as pd
import pickle
# Preprocessing
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import top_k_accuracy_score

# Deep Learning
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

"""# Data Cleaning


"""

# ajustar para sua máquina
file_path = './data/solicitacoes_servicos_por_estado24NovNoite.csv'

columns = ['id_null','region','date','client_id','service_id','device','platform','data_source', 'source','medium','channel_grouping','campaign','keyword','landing_page', 'dias_entre_solicitações','ultimo_servico', 'dia_primeiro_acesso']
data = pd.read_csv(file_path, names=columns, header=0)

data = data.drop_duplicates()

data = data.fillna('primeira solicitação')

# Há um desbalanceamento na classe de service_id
# Vamos assumir que o modelo só consegue sugerir os serviços que são 95% mais utilizados
# Porque não temos dados suficientes para o modelo aprender esses outros 5%

data = data[data['service_id'].map(data['service_id'].value_counts() >= 30)]
data = data[data['source'].map(data['source'].value_counts() >= 10)]

"""# Preprocessing"""

X = data[['region','date','device','platform','source', 'dias_entre_solicitações', 'ultimo_servico', 'dia_primeiro_acesso']]
y = np.array(data['service_id'])

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

preprocess_pipe = make_pipeline(
    FunctionTransformer(date_transform),
    FunctionTransformer(bin_dias_solicitações),
    make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse=False), ['region','device','platform','source','day', 'month','ultimo_servico', 'dias_solicitações_bin', 'dias_de_uso_bin']),
        remainder="passthrough")
)

"""## Classificação Redes Neurais"""

# train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# preprocessing

X_train_proc = preprocess_pipe.fit_transform(X_train)
X_test_proc = preprocess_pipe.transform(X_test)

with open('./data/preprocess_pipe.pickle', 'wb') as picklefile:
    pickle.dump(preprocess_pipe, picklefile)

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_cat = to_categorical(y_train_le)
y_test_cat = to_categorical(y_test_le)

with open('./data/label_encoder.pickle', 'wb') as picklefile:
    pickle.dump(le, picklefile)




categoricas = ['region','device','platform','source', 'ultimo_servico']
dicionario = dict()
for i in range(len(categoricas)):
    dicionario[categoricas[i]] = list(X_train[categoricas[i]].unique())
    print("____________________________________")

numericas_datas = ['date', 'dias_entre_solicitações', 'dia_primeiro_acesso']
for i in range(len(numericas_datas)):
    dicionario[numericas_datas[i]] = [str(X_train[numericas_datas[i]].min()),str(X_train[numericas_datas[i]].max())]

import json

# assume you have the following dictionary
with open("./data/dicionario_categoricas.json", "w") as write_file:
    json.dump(dicionario, write_file, indent=4)

with open("./data/dicionario_categoricas.json", "r") as read_content:
    print(json.load(read_content))




# f = open("./data/dicionario_categoricas.pkl","r")
# meu_dict = pickle.load(f)
# print(meu_dict)


# print("______________data min")
# print(X_test['date'].min())
# print("______________data max")
# print(X_test['date'].max())



# NN model

def init_model(X_train, y_train, learning_rate=0.001):

    input_shape = X_train.shape[1:][0]
    output_shape = y_train.shape[1]



    model = models.Sequential()

    model.add(layers.Dense(32, activation='relu', input_dim=input_shape))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(output_shape, activation = 'softmax'))

    # Model compilation
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=6)])
    return model

# fitting the model

def fit_model(model: tf.keras.Model, X_train, y_train, batch_size=128, epochs=200, patience=5, verbose=1):

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       mode="auto",
                       restore_best_weights=True)

    history = model.fit(X_train,
                        y_train,
                        validation_split=0.3,
                        shuffle=True,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es],
                        verbose=verbose)

    return model, history

# model = init_model(X_train_proc, y_train_cat, learning_rate=0.001)

# model, history = fit_model(model, X_train_proc, y_train_cat)

# model.save("./data/ModeloNN/ModeloRedeNeural")


# model = keras.models.load_model('./data/ModeloNN/ModeloRedeNeural')
# y_pred_proba = model.predict(X_test_proc)
# servicos_preditos = le.inverse_transform(np.apply_along_axis(np.argmax,1,y_pred_proba))
# servicos_observados = le.inverse_transform(np.apply_along_axis(np.argmax,1,y_test_cat))
# print((servicos_preditos == servicos_observados).mean())

# for i in range(servicos_preditos.shape[0]):
#   recomendados = le.inverse_transform(np.argsort(y_pred_proba[i,:])[-5:])
#   print("_________________________________________________")
#   print(servicos_observados[i])
#   print("____________5_Recomendacoes______________________________")
#   print(recomendados)
#   print("**************************************")
