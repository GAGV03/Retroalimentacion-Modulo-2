#Técnica de aprendizaje maquina sin framework. Gustavo Alejandro Gutiérrez Valdes A01747869

import numpy as np

errores = []

def hipotesis(params, samples):
    valor_acumulado = 0
    for i in range(len(params)):
        valor_acumulado += params[i] * samples[i]
    return valor_acumulado

#Función cost/loss que calcula y muestra los errores obtenido por una iteración de la hipotesis
def MSE(params, samples, valor_y):
    sum_error = 0
    for i in range(len(samples)):
        h = hipotesis(params,samples[i])
        error = h - valor_y[i]
        sum_error += error ** 2
    error_medio = sum_error/len(samples)
    print("Error medio: " + str(error_medio))
    errores.append(error_medio)

def GradientDescent(params,samples,learn_rate,valor_y):
    avance = list(params)
    error = 0
    for j in range (len(params)):
        sum_error = 0
        for i in range (len(samples)):
            error = hipotesis(params,samples[i]) - valor_y[i]
            sum_error = sum_error + error * samples[i][j]
        avance[j] = params[j] - learn_rate*(1/len(samples)) * sum_error
    return avance

def Normalizacion(samples):
    samples = np.array(samples, dtype=float)
    min_vals = np.min(samples, axis=0)
    max_vals = np.max(samples, axis=0)
    range_vals = max_vals - min_vals

    range_vals = np.clip(range_vals, a_min=1e-8, a_max=None)

    normalized_samples = (samples - min_vals) / range_vals

    return normalized_samples.tolist()

def linear_regression(params,samples,valor_y,num_epochs,learning_rate):
    for i in range(len(samples)):
        if isinstance(samples[i],list):
            samples[i] = [1] + samples[i]
        else:
            samples[i] = [1,samples[i]]
    
    print("Antes de normalizar: " + str(samples))
    samples = Normalizacion(samples)
    print("Despues de normalizar: " + str(samples))

    epochs = 0
    while True:
        oldparams = list(params)
        print(f"Epoch #: {epochs}")
        print("Parametros actuales: " + str(params))
        params = GradientDescent(params,samples,learning_rate,valor_y)
        MSE(params,samples,valor_y)
        print(params)
        epochs += 1
        if(oldparams == params or epochs == num_epochs):
            print("ENTRENAMIENTO FINALIZADO")
            print("Muestras: "  + str(samples))
            print("Parametros finales: " + str(params))
            break
    return params

if __name__ == "__main__":
    #Conjunto de características de una casa (Superficie en metros cuadrados, número de recámaras)
    caracteristicas_casa = [[1400, 3], [2400, 7], [1800, 4], [1900, 5], [1300, 2], [1100, 2]]
    precio = [245000, 450000, 320000, 350000, 230000, 200000]
    learning_rate = 0.001

    #Coeficiente para las características de la casa y el término independiente
    params = [0,0,0]

    params_finales = linear_regression(params,caracteristicas_casa,precio,5000,learning_rate)

    import matplotlib.pyplot as plt
    plt.plot(errores)
    plt.xlabel('Epochs')
    plt.ylabel('Error Medio')
    plt.show() 

    nueva_casa = [1,1500,3]
    nueva_casa_normalizada = Normalizacion(nueva_casa)
    prediccion_precio = hipotesis(params_finales,nueva_casa_normalizada)
    print("El precio predicho para la nueva casa es: " + str(prediccion_precio))




