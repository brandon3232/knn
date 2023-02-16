import numpy as np
from knn import KNN
from knn import eficiencia
import pandas as pd
import os

#obtenemos la direccion absoluta del directorio actual
path = os.getcwd() 
#cargamos el archivo
golf = pd.read_csv(path+"/GOLF.csv")  

print("dataset")
print(golf)
print("\n")

#convertimos las columnas outlook y windy de categoricas a numericas
golf = pd.get_dummies(golf, columns=["outlook", "windy"], drop_first = False)

# se separa el dataset en un conjunto de entrenamiento y uno de prueba
#0.7142 es el porcentaje para seleccionar 10 de los 14 registros
entrenamiento = golf.sample(frac = 0.7142) 
#obtenemos los indices de los registros seleccionados aleatoriamente para el entrenamiento
index_entrenamiento = entrenamiento.index 
#del dataset principal eliminamos los que se usaran de entrenamiento quedando los que usaremos de prueba
prueba = golf.drop(index_entrenamiento, axis=0)

print("conjunto de entrenamiento\n")
print(entrenamiento)
print("\nconjunto de prueba\n")
print(prueba)

#guardamos las columnas de clases de los dos conjuntos de datos
clases_entrenamiento = entrenamiento['play'].values
clases_prueba = prueba['play'].values

#se elimina la columna de clases de los dos conjuntos de datos
entrenamiento_sin_clases = entrenamiento.drop(['play'], axis=1).values
prueba_sin_clases = prueba.drop(['play'], axis=1).values


### k =3
print("\n############################## K = 3 ##############################\n") 
#elejimos el valor de K
clasificador = KNN(k=3)
#entrenamos el modelo
clasificador.aprendizaje(entrenamiento_sin_clases, clases_entrenamiento)
#usamos el conjunto de prueba para verificar la presicion de nuestro modelo
clasificar = clasificador.clasificacion(prueba_sin_clases)

#imprimimos las clases predichas por el modelo
print('\nclases predichas de los puntos y(n):', clasificar)
#imprimimos las clases originales de nuestro conjunto de prueba
print("clases de los puntos y(n)", clases_prueba)

#se llama a la funcion para validar la tasa de error 
val = eficiencia(clasificar, clases_prueba)
print(f"El porcentaje de eficiencia para k = 3 es de {val}")


### k = 5
print("\n############################## K = 5 ##############################\n")
clasificador = KNN(k=5)
clasificador.aprendizaje(entrenamiento_sin_clases, clases_entrenamiento)
clasificar = clasificador.clasificacion(prueba_sin_clases)

print('\n\nclases predichas de los puntos y(n):', clasificar)
print("clases de los puntos y(n)", clases_prueba)

val = eficiencia(clasificar, clases_prueba)
print(f"El porcentaje de eficiencia para k = 5 es de {val}")


### k = 7
print("\n############################## K = 7 ##############################\n")
clasificador = KNN(k=7)
clasificador.aprendizaje(entrenamiento_sin_clases, clases_entrenamiento)
clasificar = clasificador.clasificacion(prueba_sin_clases)

print('\n\nclases predichas de los puntos y(n):', clasificar)
print("clases de los puntos y(n)", clases_prueba)

val = eficiencia(clasificar, clases_prueba)
print(f"El porcentaje de eficiencia para k = 7 es de {val}")