import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def aprendizaje(self,caracteristicas,C):
        self.caracteristicas = caracteristicas # matriz de vectores de catacteristicas
        self.C = C # clases asociadas a cada vector x(n)
        self.n_muestras=caracteristicas.shape[0] # cantidad de muestras
        #print(self.caracteristicas)
        
    def clasificacion(self, Y):
        clases = []
        distanciasOrdenadas = pd.DataFrame()
        distanciasDesordenadas = pd.DataFrame()
        for i in range(Y.shape[0]): # por cada vector y(n) a clasificar
            distancias = np.empty(self.n_muestras)
            for n in range(self.n_muestras): # por cada vector x(n) de caracteristicas
                distancias[n] = EUCLIDIANA(self.caracteristicas[n], Y[i])

            # distancias mas cercanas
            k_distancias = np.argsort(distancias)
            
            # identificar las k distancias - clases
            k_etiquetas = self.C[k_distancias[:self.k]]
            
            distanciasOrdenadas[f'p{i}'] = distancias
            distanciasOrdenadas[f'clase_p{i}'] = self.C
            distanciasDesordenadas[f'p{i}'] = distancias[k_distancias]
            distanciasDesordenadas[f'clase_p{i}'] = self.C[k_distancias]
            
            #votacion
            c = Counter(k_etiquetas).most_common(1)#(5,0)
            clases.append(c[0][0]) # almacenamos la clase asignada al vector y(n)
        
        print('\ndistancias desordenadas para el conjunto de prueba\n')
        print(distanciasOrdenadas)
        print('\ndistancias ordenadas para el conjunto de prueba\n')
        print(distanciasDesordenadas)
        
        return clases

def EUCLIDIANA(x, y):
    return np.sqrt(np.sum((x-y)**2))

def eficiencia(prediccion, clases):
    suma = 0
    for i in range(clases.shape[0]):
        if clases[i] != prediccion[i]:
            suma += 1
    return 1 - (suma / clases.shape[0])