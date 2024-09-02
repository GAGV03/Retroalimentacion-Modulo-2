#Autor: Gustavo Alejandro Gutiérrez Valdes

import numpy as np
import pandas as pd

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

#Función de Gradient Descent para la evolución del código
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

#Función de normalización para los nuevos datos con los valores obtenidos del entrenamiento 
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
            print("*************************************")
            print("ENTRENAMIENTO FINALIZADO")
            break
    return params, min_vals, range_vals

if __name__ == "__main__":
    # Coeficiente para las características del estudiante y el término independiente
    params = [0, 0, 0]

    # Conjunto de características de los estudiantes (Puntaje de examen, Promedio acumulado) - Set de testing
    training_data = pd.read_csv('train.csv')
    datos_estudiantes = training_data[['PuntajeExamen','PromedioAcumulado']]
    
    # Este es el estado de su admisión. (0 - No admitido / 1 - Admitido)
    admision = training_data['Admision']
   
    #Este es el learning rate que se utilizará con el modelo
    learning_rate = 0.5

    #Aquí se ejecuta la función del modelo y se obtienen valores que serán utilizados para el testing
    params_finales, min_vals, range_vals = logistic_regression(params, datos_estudiantes, admision, learning_rate)

    #Se gráfica la relación entre los errores y las épocas
    import matplotlib.pyplot as plt
    plt.plot(errores)
    plt.xlabel('Epochs')
    plt.ylabel('Error Medio (Entropía Cruzada)')
    plt.show()

    #Aqui se entra a la etapa de validación
    print("*************************************")
    print("VALIDACIONES DEL MODELO")
    print("*************************************")

    # Inicializar contadores de los cuadrantes de la matriz de confusión
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    #Se lee el archivo con el dataset para la etapa de validación
    validation_data = pd.read_csv('validation.csv')
    datos_validation = validation_data[['PuntajeExamen','PromedioAcumulado']].values
    resultados_validation = validation_data[['Admision']].values
    contador = 1

    #Comparación de las predicciones hechas por el modelo con el valor original
    for estudiante,resultado in zip (datos_validation,resultados_validation):
        estudiante_validation_normalizado = Normalizacion_nuevos_datos(estudiante,min_vals,range_vals)
        estudiante_validation_normalizado = [1] + estudiante_validation_normalizado.tolist()
        probabilidad_validation = hipotesis(params_finales,estudiante_validation_normalizado)
        if probabilidad_validation < 0.5:
            resultado_validation = 0
        else:
            resultado_validation = 1
        contador += 1

        #Actualización de los valores de los cuadrantes de la matriz de confusión
        if resultado_validation == 1 and resultado == 1:
            TP += 1
        elif resultado_validation == 0 and resultado == 0:
            TN += 1
        elif resultado_validation == 1 and resultado == 0:
            FP += 1
        elif resultado_validation == 0 and resultado == 1:
            FN += 1
        
    # Calcular las métricas 
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Matriz de confusión
    matriz_confusion = [[TN, FP], [FN, TP]]

    #Muestra visual de las métricas que evalúan al modelo 
    print("MÉTRICAS DE VALORACIÓN DEL MODELO")
    print("*************************************")
    print(f"Matriz de Confusión: {matriz_confusion}")
    print(f"Precisión: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    #Aqui se entra a la etapa de testing
    print("*************************************")
    print("PREDICCIONES PARA EL SET DE TESTING")
    print("*************************************")
    
    contador = 1

    #Se lee el archivo del dataset que se utilizará para la etapa de testing
    testing_data = pd.read_csv('test.csv')
    nuevos_estudiantes = testing_data[['PuntajeExamen','PromedioAcumulado']].values

    #Se recorren los datos y se realizan predicciones con cada uno
    for estudiante in nuevos_estudiantes:
        nuevo_estudiante_normalizado = Normalizacion_nuevos_datos(estudiante, min_vals, range_vals)
        nuevo_estudiante_normalizado = [1] + nuevo_estudiante_normalizado.tolist()  
        probabilidad = hipotesis(params_finales, nuevo_estudiante_normalizado)
        if probabilidad > 0.5:
            print(f"{contador}) El nuevo estudiante será admitido en la universidad")
        else:
            print(f"{contador}) El nuevo estudiante no será admitido en la universidad ")
        contador += 1

