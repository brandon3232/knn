import numpy as np
from knn import KNN
from knn import eficiencia
import pandas as pd
import os

path = os.getcwd() #obtenemos la direccion absoluta del directorio actual
golf = pd.read_csv(path+"/GOLF.csv") #cargamos el archivo 

#convertimos las columnas outlook y windy de categoricas a numericas
golf = pd.get_dummies(golf, columns=["outlook", "windy"], drop_first = False)

# se separa el dataset en un conjunto de entrenamiento y uno de prueba
entrenamiento = golf.sample(frac = 0.7142) #0.7142 es el porcentaje para seleccionar 10 de los 14 registros
index_entrenamiento = entrenamiento.index #obtenemos los indices de los registros seleccionados aleatoriamente para el entrenamiento
prueba = golf.drop(index_entrenamiento, axis=0)#del dataset principal eliminamos los que se usaran de entrenamiento quedando los que usaremos de prueba

#guardamos las columnas de clases de los dos conjuntos de datos
clases_entrenamiento = entrenamiento['play'].values
clases_prueba = prueba['play'].values

#se elimina la columna de clases de los dos conjuntos de datos
entrenamiento_sin_clases = entrenamiento.drop(['play'], axis=1).values
prueba_sin_clases = prueba.drop(['play'], axis=1).values


### k =3
clasificador = KNN(k=3)#elejimos el valor de K
clasificador.aprendizaje(entrenamiento_sin_clases, clases_entrenamiento)#entrenamos el modelo
clasificar = clasificador.clasificacion(prueba_sin_clases)#usamos el conjunto de prueba para verificar la presicion de nuestro modelo
print('clases predichas de los puntos y(n):', clasificar)#imprimimos las clases predichas por el modelo
print("clases de los puntos y(n)", clases_prueba)#imprimimos las clases originales de nuestro conjunto de prueba

val = eficiencia(clasificar, clases_prueba)#se llama a la funcion para validar la tasa de error 
print(f"El porcentaje de eficiencia para k = 3 es de {val}")


### k = 5
clasificador = KNN(k=5)
clasificador.aprendizaje(entrenamiento_sin_clases, clases_entrenamiento)
clasificar = clasificador.clasificacion(prueba_sin_clases)
print('\n\nclases predichas de los puntos y(n):', clasificar)
print("clases de los puntos y(n)", clases_prueba)

val = eficiencia(clasificar, clases_prueba)
print(f"El porcentaje de eficiencia para k = 5 es de {val}")


### k = 7
clasificador = KNN(k=7)
clasificador.aprendizaje(entrenamiento_sin_clases, clases_entrenamiento)
clasificar = clasificador.clasificacion(prueba_sin_clases)
print('\n\nclases predichas de los puntos y(n):', clasificar)
print("clases de los puntos y(n)", clases_prueba)

val = eficiencia(clasificar, clases_prueba)
print(f"El porcentaje de eficiencia para k = 7 es de {val}")