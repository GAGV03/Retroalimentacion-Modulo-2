#Autor: Gustavo Alejandro Gutiérrez Valdes

import numpy as np
import math

errores = []

# Función sigmoide (Activación)
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Hipótesis de regresión logística (Aplicando Sigmoide)
def hipotesis(params, samples):
    valor_acumulado = np.dot(params, samples)
    return sigmoide(valor_acumulado)

# Función de costo: Entropía cruzada
def cross_entropy_loss(params, samples, valor_y):
    sum_error = 0
    for i in range(len(samples)):
        h = hipotesis(params, samples[i])
        h = np.clip(h, 1e-10, 1 - 1e-10) # Manejo de log(0) para evitar errores
        error = -valor_y[i] * np.log(h) - (1 - valor_y[i]) * np.log(1 - h)
        sum_error += error
    error_medio = sum_error / len(samples)
    print("Error medio (entropía cruzada): " + str(error_medio))
    errores.append(error_medio)
    return error_medio

def GradientDescent(params, samples, learn_rate, valor_y):
    params = np.array(params)
    num_samples = len(samples)
    num_caracteristicas = len(params)
    params_nuevos = np.zeros(num_caracteristicas)
    
    for j in range(num_caracteristicas):
        sum_error = 0
        for i in range(num_samples):
            error = hipotesis(params, samples[i]) - valor_y[i]
            sum_error += error * samples[i][j]
        params_nuevos[j] = (1 / num_samples) * sum_error
    
    return params - learn_rate * params_nuevos

# Normalización de los datos
def Normalizacion(samples):
    samples = np.array(samples, dtype=float)
    min_vals = np.min(samples, axis=0)
    max_vals = np.max(samples, axis=0)
    diferencia = max_vals - min_vals

    diferencia = np.clip(diferencia, a_min=1e-8, a_max=None) #Evita la división entre 0

    normalized_samples = (samples - min_vals) / diferencia

    return normalized_samples.tolist(), min_vals, diferencia

def Normalizacion_nuevos_datos(nuevos_datos, min_vals, range_vals):
    nuevos_datos_norm = (np.array(nuevos_datos) - min_vals) / range_vals
    return nuevos_datos_norm

# Regresión logística
def logistic_regression(params, samples, valor_y, learning_rate):
    samples, min_vals, range_vals = Normalizacion(samples)
    
    for i in range(len(samples)):
        samples[i] = [1] + samples[i]

    epochs = 0
    while True:
        oldparams = np.array(params)
        print(f"Epoch #: {epochs}")
        print("Parametros actuales: " + str(params))
        params = GradientDescent(params, samples, learning_rate, valor_y)
        error = cross_entropy_loss(params, samples, valor_y)
        print(params)
        epochs += 1
        # Verifica si los parámetros han cambiado suficientemente
        if np.allclose(oldparams, params, atol=1e-6) or error < 0.01:
            print("ENTRENAMIENTO FINALIZADO")
            break
    #print("Muestras: " + str(samples))
    #print("Parametros finales: " + str(params))
    return params, min_vals, range_vals

if __name__ == "__main__":
    # Coeficiente para las características del estudiante y el término independiente
    params = [0, 0, 0]

    # Conjunto de características de los estudiantes (Puntaje de examen, Promedio acumulado)
    datos_estudiantes = [[85, 9.5], [95, 8.7], [55, 6.2], [90, 9.8], [70, 7.9], [60, 3.5], [40, 6.0], [85, 7.4], [92, 8.6], [88, 8.5], [20,2.5], [30,4.0], [10,3.0]]
    
    # Este es el estado de su admisión. (0 - No admitido / 1 - Admitido)
    admision = [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    
    learning_rate = 0.01

    params_finales, min_vals, range_vals = logistic_regression(params, datos_estudiantes, admision, learning_rate)

    import matplotlib.pyplot as plt
    plt.plot(errores)
    plt.xlabel('Epochs')
    plt.ylabel('Error Medio (Entropía Cruzada)')
    plt.show()

    print("*************************************")
    print("PREDICCIONES PARA EL SET DE TESTING")
    print("*************************************")
    
    contador = 1
    nuevos_estudiantes = [[20, 3.5],[100,10],[95,8.5],[50,6.5],[70,7.0],[80,8.0],[10,10]]
    for estudiante in nuevos_estudiantes:
        nuevo_estudiante_normalizado = Normalizacion_nuevos_datos(estudiante, min_vals, range_vals)
        nuevo_estudiante_normalizado = [1] + nuevo_estudiante_normalizado.tolist()  
        probabilidad = hipotesis(params_finales, nuevo_estudiante_normalizado)
        print(f"{contador}) La probabilidad predicha de que el nuevo estudiante sea admitido es: " + str(probabilidad))
        contador += 1

