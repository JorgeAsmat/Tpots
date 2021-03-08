# Librerías
import numpy as np # algebra lineal
import pandas as pd # manipulacion de datos
from scipy.stats import variation # coeficiente de variación
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import math
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import category_encoders as ce
from sklearn.decomposition import PCA
import sklearn.metrics

#Train
Fecha_inicial_train = pd.Timestamp(2016,1,1)
Fecha_final_train   = pd.Timestamp(2021,1,10)

#Test
Fecha_inicial_test  = pd.Timestamp(2019,1,1)
Fecha_final_test    = pd.Timestamp(2020,8,1)

#Prueba
Fecha_inicial_prueba  = pd.Timestamp(2021,1,1) 
Fecha_final_prueba  = pd.Timestamp(2021,12,31)

list_claves = [
                'ARTICULO' , 'DISENO' , 'COMBINACION', 'Fc.Corte' , 'TITULO_U','NUM_CABOS_U','Telar','LIGAMENTO_FONDO','LIGAMENTO_ORILLO',
                   'DIENTES/CM_PEINE','HILOS/DIENTE_FONDO','HILOS/DIENTE_ORILLO','ANCHO_PEINE','ANCHO_CRUDO','%E_URDIMBRE',
                  'TOTAL_HILOS/ANCHO_CRUDO','PASADAS/CM_T1','PORC_PASADAS/CM_T1' ,'GR/MTL_U','GR/MTL_T1','TOTAL_PASADAS','PORC_GR/MTL_U','PORC_GR/MTL_T1','TOTAL_GR/MTL', 'MAQUINA_PINZAS',
                   'NUM_COLORES_U','NUM_COLORES_T','AGUA    ','LUMINOSIDAD_T_1', 'LUMINOSIDAD_U_1', 'LUMINOSIDAD_T_2', 'LUMINOSIDAD_U_2', 'LUMINOSIDAD_T_3',
                   'LUMINOSIDAD_U_3', 'LUMINOSIDAD_T_4', 'LUMINOSIDAD_U_4', 'LUMINOSIDAD_T_5', 'LUMINOSIDAD_U_5','LUMINOSIDAD_T_6', 'LUMINOSIDAD_U_6', 
                   'FACT_COB_U', 'FACT_COB_T','FACT_COB_TOTAL_REAL', 'TUPIDEZ','Ne_prom','CV% Ne_prom','cN/tex_prom','TPI_prom','FT_prom','CV% TPI_prom',
                   'E%_prom','CV% E_prom','CV%R_prom','CVm%_prom','I_prom','PD(-40%)_prom','PD(-50%)_prom','PG(+35%)_prom','PG(+50%)_prom','NEPS(+140%)_prom','NEPS(+200%)_prom',
                   'H_prom','Sh_prom','var_Ne_prom','var_cN/tex_prom','var_TPI_prom','var_E%_prom','%falla_R_prom','%falla_E_prom' , 'CMPX DE PARO POR URDIMBRE'
              ]

list_predictors = [
                  'TITULO_U','NUM_CABOS_U','Telar','LIGAMENTO_FONDO','LIGAMENTO_ORILLO',
                   'DIENTES/CM_PEINE','HILOS/DIENTE_FONDO','HILOS/DIENTE_ORILLO','ANCHO_PEINE','ANCHO_CRUDO','%E_URDIMBRE',
                  'TOTAL_HILOS/ANCHO_CRUDO','PASADAS/CM_T1','PORC_PASADAS/CM_T1' , 'RPM',
                   'GR/MTL_U','GR/MTL_T1','TOTAL_PASADAS','PORC_GR/MTL_U','PORC_GR/MTL_T1','TOTAL_GR/MTL', 'MAQUINA_PINZAS',
                   'NUM_COLORES_U','NUM_COLORES_T','AGUA    ','LUMINOSIDAD_T_1', 'LUMINOSIDAD_U_1', 'LUMINOSIDAD_T_2', 'LUMINOSIDAD_U_2', 'LUMINOSIDAD_T_3',
                   'LUMINOSIDAD_U_3', 'LUMINOSIDAD_T_4', 'LUMINOSIDAD_U_4', 'LUMINOSIDAD_T_5', 'LUMINOSIDAD_U_5','LUMINOSIDAD_T_6', 'LUMINOSIDAD_U_6', 
                   'FACT_COB_U', 'FACT_COB_T','FACT_COB_TOTAL_REAL', 'TUPIDEZ','Ne_prom','CV% Ne_prom','cN/tex_prom','TPI_prom','FT_prom','CV% TPI_prom',
                   'E%_prom','CV% E_prom','CV%R_prom','CVm%_prom','I_prom','PD(-40%)_prom','PD(-50%)_prom','PG(+35%)_prom','PG(+50%)_prom','NEPS(+140%)_prom','NEPS(+200%)_prom',
                   'H_prom','Sh_prom','var_Ne_prom','var_cN/tex_prom','var_TPI_prom','var_E%_prom','%falla_R_prom','%falla_E_prom'
                  ]

list_targets = ['CMPX DE PARO POR URDIMBRE']
#Se leen los datos necesarios para el proyecto
Data_Muestras = pd.read_csv("Muestras.csv")
Data_Total = pd.read_excel('Data_total_analisis.xls')
temp_Data = Data_Total[list(['Fc.Corte']) +  list_predictors + list_targets ]
DataFrame_filtrado = temp_Data.copy()
#Se definen los limites de los datos de entrenamiento
DataFrame_filtrado = DataFrame_filtrado.loc[ (0 < DataFrame_filtrado['CMPX DE PARO POR URDIMBRE'])
                                                         & (DataFrame_filtrado['CMPX DE PARO POR URDIMBRE']<7.5)
                                                        ]
DataFrame_filtrado.shape
DataFrame_filtrado.reset_index(drop=True,inplace=True)
DataFrame_filtrado_total = DataFrame_filtrado.copy()
#Se definen los limites de los datos de muestras
Data_Muestras = Data_Muestras.loc[ (0 < Data_Muestras['CMPX DE PARO POR URDIMBRE'])
                                                         & (Data_Muestras['CMPX DE PARO POR URDIMBRE']<7.5)
                                                        ]
Data_Muestras.shape
Data_Muestras.reset_index(drop=True,inplace=True)
#Se limpian los predictos 
predictores_numericos = [i for i in list_predictors if  'float' in str(DataFrame_filtrado[i].dtype) or 'int' in str(DataFrame_filtrado[i].dtype)]
predictores_categoricos = [i for i in list_predictors if i not in predictores_numericos]
df=DataFrame_filtrado.copy()
#quitar variables correlacionadas
df=DataFrame_filtrado.copy()
# Create correlation matrix
corr_matrix = df[predictores_categoricos+predictores_numericos].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
for i in to_drop:
 if i in predictores_numericos:
   predictores_numericos.remove(i)
 elif i in predictores_categoricos:
   predictores_categoricos.remove(i)
   
import datetime
#Se hara un entremiento entre 2016 a 2018
#La validacion sera de 2018 a 2019
X_train= df[ (Fecha_inicial_train <= df['Fc.Corte'] )
            & (df['Fc.Corte']  < Fecha_final_train ) ]
X_test= df[
           (Fecha_inicial_test <= df['Fc.Corte'] ) &
           (df['Fc.Corte']  < Fecha_final_test)
           ]
y_train = df[list_targets[0]][X_train.index]
y_test = df[list_targets[0]][X_test.index]
#Los datos de prueba final seran del año 2020
X_prueba_campo = df[
                    (Fecha_inicial_prueba <= df['Fc.Corte'] ) &
                    (df['Fc.Corte']  < Fecha_final_prueba)
                    ]
y_prueba_campo = df[list_targets[0]][X_prueba_campo.index]
X=df[list_predictors]
X_muestras = Data_Muestras[list_predictors]
y_muestras = Data_Muestras[list_targets[0]]
#Separamos las variables categoricas
train = X_train[predictores_categoricos]
test = X_test[predictores_categoricos]
#Se les aplica el encoder
encoder_mean = ce.TargetEncoder(cols = predictores_categoricos ,handle_unknown= 'ignore' )
OH_cols_train = encoder_mean.fit_transform(train[predictores_categoricos],y_train)
OH_cols_test = encoder_mean.transform(test[predictores_categoricos],y_test)
# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train[predictores_numericos]
num_X_test = X_test[predictores_numericos]
# Concatenable
X_train_escalable = pd.concat([num_X_train, OH_cols_train], axis=1)
X_test_escalable = pd.concat([num_X_test, OH_cols_test], axis=1)
# se escalan los datos numéricos
scaler = StandardScaler()
Num_X_Scaler_train=pd.DataFrame(scaler.fit_transform(X_train_escalable))
Num_X_Scaler_test=pd.DataFrame(scaler.transform(X_test_escalable))
#Elimina los indices asi que los volvemos a poner
Num_X_Scaler_train.index = X_train.index
Num_X_Scaler_test.index = X_test.index
# Adecuamos la nomeclatura 
OH_X_train = Num_X_Scaler_train
OH_X_test = Num_X_Scaler_test
#Asignamos los nombres
lista_nombres_numericos = X_train_escalable.columns 
lista_nombres = list(lista_nombres_numericos)
OH_X_train.columns = lista_nombres
OH_X_test.columns = lista_nombres
#Separamos las variables categoricas
prueba_campo = X_prueba_campo[predictores_categoricos]
#Se les aplica el encoder
OH_cols_prueba_campo = encoder_mean.transform(prueba_campo[predictores_categoricos],y_prueba_campo)
# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_prueba_campo = X_prueba_campo[predictores_numericos]
# Concatenable
X_prueba_campo_escalable = pd.concat([num_X_prueba_campo, OH_cols_prueba_campo], axis=1)
# se escalan los datos numéricos
Num_X_Scaler_prueba_campo=pd.DataFrame(scaler.transform(X_prueba_campo_escalable))
#Elimina los indices asi que los volvemos a poner
Num_X_Scaler_prueba_campo.index = X_prueba_campo.index
# Adecuamos la nomeclatura 
OH_X_prueba_campo = Num_X_Scaler_prueba_campo
#Asignamos los nombres
lista_nombres_numericos = X_prueba_campo_escalable.columns 
lista_nombres = list(lista_nombres_numericos)
OH_X_prueba_campo.columns = lista_nombres
#USAREMOS LAS PREDICCIONES DEL TOTAL COMO OTRA COLUMNA PARA LA NN
X_cols = pd.DataFrame(encoder_mean.transform(X.iloc[pd.concat([X_train,X_test]).index][predictores_categoricos].astype(str)))
X_cols.index = pd.concat([X_train,X_test]).index
num_X = X.iloc[pd.concat([X_train,X_test]).index][predictores_numericos]
X_escalable = pd.concat([num_X, X_cols], axis=1)
X_total=pd.DataFrame(scaler.transform(X_escalable))
X_total.index = X_escalable.index
#Renombramos a las columnas
lista_nombres_numericos = X_escalable.columns 
lista_nombres = list(lista_nombres_numericos)
X_total.columns = lista_nombres
#Separamos las variables categoricas
muestras_pruebas = X_muestras[predictores_categoricos]
#Se les aplica el encoder
OH_cols_X_muestras_pruebas = encoder_mean.transform(muestras_pruebas[predictores_categoricos],y_muestras)
# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_muestras_pruebas= X_muestras[predictores_numericos]
# Concatenable
X_muestras_pruebas_escalable = pd.concat([num_X_muestras_pruebas, OH_cols_X_muestras_pruebas], axis=1)
# se escalan los datos numéricos
Num_X_Scaler_prueba_campo=pd.DataFrame(scaler.transform(X_muestras_pruebas_escalable))
#Elimina los indices asi que los volvemos a poner
Num_X_Scaler_prueba_campo.index = X_muestras.index
# Adecuamos la nomeclatura 
OH_X_muestras = Num_X_Scaler_prueba_campo
#Asignamos los nombres
lista_nombres_numericos = X_muestras_pruebas_escalable.columns 
lista_nombres = list(lista_nombres_numericos)
OH_X_muestras.columns = lista_nombres
#Evitamos encontrar valores 0
(OH_X_prueba_campo).fillna(0 , inplace= True)
(OH_X_test).fillna(0 , inplace= True)
(X_total).fillna(0 , inplace= True)
(OH_X_muestras).fillna(0 , inplace= True)
#Se planteo el uso de log sobre la variable a predecir para minimizar el error de prediccion con 
#Numeros elevados
y_train = np.log(y_train)
y_test = np.log(y_test)
y_prueba_campo = np.log(y_prueba_campo)
# check tpot version
import tpot
print('tpot: %s' % tpot.__version__)

from tpot import TPOTRegressor
# Definimos el modelo
model = TPOTRegressor(generations=200, population_size=100, scoring='neg_mean_squared_error', verbosity=2, random_state=1, n_jobs=-1)
# Se entrena
model.fit(OH_X_train, y_train)
# exportamos
model.export('tpot_insurance_best_model.py')