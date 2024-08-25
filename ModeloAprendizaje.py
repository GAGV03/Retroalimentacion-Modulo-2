#Técnica de aprendizaje maquina sin framework. Gustavo Alejandro Gutiérrez Valdes A01747869

#Conjunto de características de una casa (Superficie en metros cuadrados, número de recámaras)
caracteristicas_casa = [[1400, 3], [2400, 7], [1800, 4], [1900, 5], [1300, 2], [1100, 2]]
precio = [245000, 450000, 320000, 350000, 230000, 200000]

#Coeficiente para las características de la casa y el término independiente
params = (0,0,0)

def hipotesis(params, caracteristicas_casa):
    valor_acumulado = 0
    for i in range(len(params)):
        valor_acumulado += params[i] * caracteristicas_casa[i]
    return valor_acumulado


#Función cost/loss que calcula y muestra los errores obtenido por una iteración de la hipotesis
def MSE(params, caracteristicas_casa, precio):
    pass        

def GradientDescent():
    pass

def scaling():
    pass

def linear_regression():
    for i in range:
        caracteristicas_casa[i] = [1] + caracteristicas_casa[i]




if __name__ == "__main__":
    learning_rate = 0.01




