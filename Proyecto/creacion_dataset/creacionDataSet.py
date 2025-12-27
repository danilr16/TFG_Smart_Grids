import os
import subprocess
import time

import numpy as np
import pandas as pd


def crea_carpetas(directorio: str, dias: int, minutos: int) -> None:
    if not (os.path.exists(directorio) and os.path.exists(f'{directorio}\day_1')):
        print(f'ERROR: No se ha podido encontrar el directorio "{directorio}\day_1" con los datos del día 1')
        return

    for i in range(dias):
        directorio_dia: str = f'{directorio}\day_{i + 2}'
        os.mkdir(directorio_dia)
        for j in range(3, minutos, 5):
            directorio_minuto: str = f'{directorio_dia}\minuto{str(j).zfill(4)}'
            os.mkdir(directorio_minuto)
        print('Carpetas creadas para dia ' + str(i + 2))
    print(f'Estructura de carpetas creadas exitosamente en el directorio "{directorio}"')


def generar_load_h_data(directorio: str, media: float, desviacion_estandar: float, dia: int, minute: int) -> None:
    # Lee los datos
    load_h_data_path: str = f'{directorio}\day_1\minuto{str(minute).zfill(4)}\LOAD_H_DATA.txt'
    with open(load_h_data_path, 'r') as archivo:
        lineas: list = archivo.readlines()
    matriz: list = []
    for linea in lineas:
        fila: list = linea.strip().split(', ')  # Crea lista de cada línea eliminando espacios y comas
        matriz.append(fila)  # Agrega la lista de elementos a la matriz
    filas_matrizH: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    columnas_matrizH: list = [1, 2]
    submatrizH: list = [[float(matriz[i][j]) for j in columnas_matrizH] for i in filas_matrizH]

    # Potencia máxima de los nudos con consumo residencial
    P_H: list = [15., 0.276, 0.432, 0.725, 0.55, 0.588, 0.477, 0.331, 15., 0.207]
    Q_H: list = [3.1, 0.069, 0.108, 0.182, 0.138, 0.147, 0.12, 0.083, 3., 0.052]

    # Escribe el nuevo archivo
    new_matrizH: list = []
    Nudos: list = [1, 3, 4, 5, 6, 8, 10, 11, 12, 14]
    fila1: list = ['Nudo', '   P', '   Q']
    new_matrizH.append(fila1)
    for i in range(10):
        numrandom: float = np.random.normal(loc=media, scale=desviacion_estandar)  # Crea numero aleatorio
        modPH: float = submatrizH[i][0] * (1 + numrandom)
        modQH: float = submatrizH[i][1] * (1 + numrandom)

        # Si el nuevo valor al sumarle la potencia es negativo lo sustituye por cero
        val_P = min(P_H[i], max(0.0, modPH))
        val_Q = min(Q_H[i], max(0.0, modQH))
        submatrizH[i][0] = val_P
        submatrizH[i][1] = val_Q

        new_matrizH.append([Nudos[i], submatrizH[i][0], submatrizH[i][1]])

    new_load_h_data_path: str = f'{directorio}\day_{dia + 2}\minuto{str(minute).zfill(4)}\LOAD_H_DATA.txt'
    with open(new_load_h_data_path, 'w') as archivo:
        for fila in new_matrizH:
            linea: str = ', '.join(map(str, fila))
            archivo.write(linea + '\n')


def generar_load_I_data(directorio: str, media: float, desviacion_estandar: float, dia: int, minute: int) -> None:
    # Lee LOAD_I_DATA
    load_I_data_path: str = f'{directorio}\day_1\minuto{str(minute).zfill(4)}\LOAD_I_DATA.txt'
    with open(load_I_data_path, 'r') as archivo:
        lineas: list = archivo.readlines()
    matriz: list = []
    for linea in lineas:
        fila: list = linea.strip().split(', ')
        matriz.append(fila)
    filas_matrizI: list = [1, 2, 3, 4, 5, 6, 7, 8]
    columnas_matrizI: list = [1, 2]
    submatrizI: list = [[float(matriz[i][j]) for j in columnas_matrizI] for i in filas_matrizI]

    # Potencia máxima de los nudos con consumo industrial
    P_I: list = [5., 0.224, 0.077, 0.574, 0.068, 5., 0.032, 0.33]
    Q_I: list = [1., 0.139, 0.048, 0.356, 0.042, 1.7, 0.02, 0.205]

    # Escribe el nuevo archivo
    new_matrizI: list = []
    Nudos: list = [1, 3, 7, 9, 10, 12, 13, 14]
    fila1: list = ['Nudo', '   P', '   Q']
    new_matrizI.append(fila1)
    for i in range(8):
        numrandom: float = np.random.normal(loc=media, scale=desviacion_estandar)  # Crea numero aleatorio
        modPI: float = submatrizI[i][0] * (1 + numrandom)
        modQI: float = submatrizI[i][1] * (1 + numrandom)

        # Si el nuevo valor al sumarle la potencia es negativo lo sustituye por cero
        val_P = min(P_I[i], max(0.0, modPI))
        val_Q = min(Q_I[i], max(0.0, modQI))
        submatrizI[i][0] = val_P
        submatrizI[i][1] = val_Q

        new_matrizI.append([Nudos[i], submatrizI[i][0], submatrizI[i][1]])

    new_load_I_data_path: str = f'{directorio}\day_{dia + 2}\minuto{str(minute).zfill(4)}\LOAD_I_DATA.txt'
    with open(new_load_I_data_path, 'w') as archivo:
        for fila in new_matrizI:
            linea: str = ', '.join(map(str, fila))
            archivo.write(linea + '\n')


def generar_gen_data(directorio: str, media: float, desviacion_estandar: float, dia: int, minute: int) -> None:
    # Lee datos generación
    gen_data_path: str = f'{directorio}\day_1\minuto{str(minute).zfill(4)}\GEN_DATA.txt'
    with open(gen_data_path, 'r') as archivo:
        lineas: list = archivo.readlines()
    matriz: list = []
    for linea in lineas:
        fila: list = linea.strip().split(', ')
        matriz.append(fila)
    filas_matrizG: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    columnas_matrizG: list = [2]
    submatrizG: list = [[float(matriz[i][j]) for j in columnas_matrizG] for i in filas_matrizG]

    # Potencia máxima de los nudos con generación
    P_G: list = [0.02, 0.02, 0.03, 0.6, 0.033, 0.03, 1.5, 0.03, 0.03, 0.310, 0.212, 0.04, 0.2, 0.014, 0.01]

    # Escribe el nuevo archivo GEN
    new_matrizG: list = []
    Nombre: list = ['GP3', 'GP4', 'GP5', 'GB5', 'GFC5', 'GP6', 'GWT7', 'GP8', 'GP9', 'GCHPD9',
                    'GCHPFC9', 'GP10', 'GB10', 'GFC10', 'GP11']
    Nudos: list = [3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11]
    Pmax: list = [0.02, 0.02, 0.03, 0.6, 0.033, 0.03, 1.5, 0.03, 0.03, 0.31, 0.212, 0.04, 0.2, 0.014, 0.01]
    Pmin: list = [0, 0, 0, -0.5, -0.033, 0, 0, 0, 0, 0, 0, 0, -0.2, -0.014, 0]
    fila1: list = ['Nombre', 'Nudo', '   P', '   Pmax', 'Pmin']
    new_matrizG.append(fila1)
    for i in range(15):
        numrandom: float = np.random.normal(loc=media, scale=desviacion_estandar)  # Crea numero aleatorio
        modPG: float = submatrizG[i][0] * (1 + numrandom)

        # Si el nuevo valor al sumarle la potencia es negativo lo sustituye por cero
        val_P = min(P_G[i], max(Pmin[i], modPG))
        submatrizG[i][0] = val_P

        new_matrizG.append([Nombre[i], Nudos[i], submatrizG[i][0], Pmax[i], Pmin[i]])

    new_gen_data_path: str = f'{directorio}\day_{dia + 2}\minuto{str(minute).zfill(4)}\GEN_DATA.txt'
    with open(new_gen_data_path, 'w') as archivo:
        for fila in new_matrizG:
            linea: str = ', '.join(map(str, fila))
            archivo.write(linea + '\n')


def generar_datos_random(directorio: str, minutos: int, media: float, desviacion_estandar: float,
                         seed: int = 99) -> None:
    np.random.seed(seed)
    for dia in range(364):
        for minute in range(3, minutos, 5):
            generar_load_h_data(directorio, media, desviacion_estandar, dia, minute)
            generar_load_I_data(directorio, media, desviacion_estandar, dia, minute)
            generar_gen_data(directorio, media, desviacion_estandar, dia, minute)
        print('Datos aleatorios generados para el día ' + str(dia + 2))
    print('Datos aleatorios generados para el resto de días')


def llamada_gams(directorio: str, directorio_opf: str, dias: int, minutos: int) -> None:
    if not os.path.exists(directorio_opf):
        print('ERROR: Directorio del OPF no encontrado')
    directorio_datos: str = os.path.abspath(directorio)
    os.chdir(directorio_opf)
    """
    for dia in range(dias + 1):  # Ejecuta conv_data para crear todos los .g00
        for minute in range(3, minutos, 5):
            minute_path: str = f'{directorio_datos}\day_{dia + 1}\minuto{str(minute).zfill(4)}'
            comando = f'gams conv_data.gms s={minute_path} Idir={minute_path}'
            subprocess.run(comando, shell=True)
            time.sleep(0.01)
        print('Ficheros g00 creados para el día ' + str(dia))
    print("Ficheros .g00 creados")"""

    for dia in range(dias + 1):  # Ejecuta MinPerd_PF para crear todos los .put
        for minute in range(3, minutos, 5):
            day_path: str = f'{directorio_datos}\day_{dia + 1}'
            minute_path: str = f'{directorio_datos}\day_{dia + 1}\minuto{str(minute).zfill(4)}'
            comando = f'gams MinPerd_Base_perdidas_Qopt_v1.gms r={minute_path} Pdir={day_path} --Num=min{str(minute).zfill(4)}'
            subprocess.run(comando, shell=True)
            time.sleep(0.02)
        print('Ficheros put creados para el día ' + str(dia))
    print("Ficheros .put creados")


def extraer_load_h_data(directorio: str, dia: int, nudos_red: int, minute:int) -> list:
    path = f'{os.path.abspath(directorio)}\day_{dia + 1}\minuto{str(minute).zfill(4)}\LOAD_H_DATA.txt'
    with open(path, 'r') as archivo:
        lineas = archivo.readlines()
        matriz = []
    for linea in lineas:
        fila = linea.strip().split(', ')  # Crea lista de cada línea eliminando espacios y comas
        matriz.append(fila)  # Agrega la lista de elementos a la matriz
    filas_matrizH = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    columnas_matrizH = [1, 2]
    submatrizH = [[float(matriz[i][j]) for j in columnas_matrizH] for i in filas_matrizH]

    # Crea la matriz_dataH con 14 filas(nudos), insertando ceros en los nudos donde no había consumo H
    matriz_dataH = []
    Nudos = [1, 3, 4, 5, 6, 8, 10, 11, 12, 14]
    aux = 0
    for i in range(nudos_red):
        if Nudos[aux] == i + 1:
            matriz_dataH.append([i + 1, submatrizH[aux][0], submatrizH[aux][1]])
            aux += 1
        else:
            matriz_dataH.append([i + 1, 0, 0])
    return matriz_dataH


def extraer_load_I_data(directorio: str, dia: int, nudos_red: int, minute:int) -> list:
    #  Crea la matriz_dataI con 14 filas(nudos), insertando ceros en los nudos donde no había consumo I
    path = f'{os.path.abspath(directorio)}\day_{dia + 1}\minuto{str(minute).zfill(4)}\LOAD_I_DATA.txt'
    with open(path, 'r') as archivo:
        lineas = archivo.readlines()
    matriz = []
    for linea in lineas:
        fila = linea.strip().split(', ')
        matriz.append(fila)
    filas_matrizI = [1, 2, 3, 4, 5, 6, 7, 8]
    columnas_matrizI = [1, 2]
    submatrizI = [[float(matriz[i][j]) for j in columnas_matrizI] for i in filas_matrizI]

    matriz_dataI = []
    Nudos = [1, 3, 7, 9, 10, 12, 13, 14]
    aux = 0
    for i in range(nudos_red):
        if Nudos[aux] == i + 1:
            matriz_dataI.append([i + 1, submatrizI[aux][0], submatrizI[aux][1]])
            aux += 1
        else:
            matriz_dataI.append([i + 1, 0, 0])
    return matriz_dataI


def extraer_gen_data(directorio: str, dia: int, minute:int) -> list:
    # Guarda la P de generacion en submatrizG
    path = f'{os.path.abspath(directorio)}\day_{dia + 1}\minuto{str(minute).zfill(4)}\GEN_DATA.txt'
    with open(path, 'r') as archivo:
        lineas = archivo.readlines()
    matriz = []
    for linea in lineas:
        fila = linea.strip().split(', ')
        matriz.append(fila)
    filas_matrizG = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    columnas_matrizG = [2]
    submatrizG = [[float(matriz[i][j]) for j in columnas_matrizG] for i in filas_matrizG]

    # Se crea matriz_dataG en la que se unen los nudos donde hay varias P generadas
    matriz_dataG = []
    Nudos = [3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11]
    i = 0
    while i < len(Nudos):
        if Nudos[i] in [5, 9, 10]:
            matriz_dataG.append([Nudos[i], submatrizG[i][0] + submatrizG[i + 1][0] + submatrizG[i + 2][0]])
            i += 3
        else:
            matriz_dataG.append([Nudos[i], submatrizG[i][0]])
            i += 1
    return matriz_dataG


def datos_entrada(directorio: str, dias: int, minutos: int) -> pd.DataFrame:
    nudos_red = 14
    in_list = list()
    for dia in range(dias + 1):
        for minute in range(3, minutos, 5):
            matriz_dataH = extraer_load_h_data(directorio, dia, nudos_red, minute)
            matriz_dataI = extraer_load_I_data(directorio, dia, nudos_red, minute)

            # Se crea matriz demanda 14x2. Consumos P y Q en cada nudo
            matriz_dataH = np.array(matriz_dataH)
            matriz_dataI = np.array(matriz_dataI)
            demanda = matriz_dataH[:, 1:3] + matriz_dataI[:, 1:3]

            matriz_dataG = extraer_gen_data(directorio, dia, minute)

            # Generacion con todos los nudos de la red
            generacion = []
            aux = 0
            for i in range(nudos_red):
                if aux < 9 and matriz_dataG[aux][0] == i + 1:
                    generacion.append([i + 1, matriz_dataG[aux][1]])
                    aux += 1
                else:
                    generacion.append([i + 1, 0])

            # A la matriz demanda se le resta la generacion en la columna de P
            generacion = np.array(generacion)
            generacion = generacion[:, 1]
            demanda[:, 0] -= generacion

            # Se almacenan los valores de la matriz en una matriz 1x28 para crear el DataFrame
            valores = []
            for filas in demanda:
                for j in filas:
                    valores.append(j)

            valores = np.array(valores)
            valoresT = valores.reshape(1, 28)

            # Creacion del DataFrame
            dato_minuto = pd.DataFrame(valoresT,
                                       columns=['P1', 'Q1', 'P2', 'Q2', 'P3', 'Q3', 'P4', 'Q4', 'P5', 'Q5', 'P6', 'Q6',
                                                'P7',
                                                'Q7', 'P8', 'Q8', 'P9', 'Q9', 'P10', 'Q10', 'P11', 'Q11', 'P12', 'Q12',
                                                'P13',
                                                'Q13', 'P14', 'Q14'],
                                       index=['day' + str(dia + 1) + '_min' + str(minute).zfill(4)])

            in_list.append(dato_minuto)
        print(f'Datos de entrada leídos del día {dia + 1}')

    return pd.concat(in_list)


def datos_salida(directorio: str, dias: int, minutos: int) -> pd.DataFrame:
    out_list = list()
    for dia in range(dias + 1):
        for minute in range(3, minutos, 5):
            # Lee el .txt y crea la lista lineas donde cada elemento es una linea del archivo
            path = f'{os.path.abspath(directorio)}\day_{dia + 1}\RESULT_Base_min{str(minute).zfill(4)}.put'
            with open(path, 'r') as archivo:
                lineas = archivo.readlines()

            # Creo una matriz donde guardo la linea de la 10 a la 23 donde están los datos de tensiones
            matriz = []
            for i in range(4, 18):
                linea = lineas[i]
                fila = linea.split('          ')  # Crea lista de cada línea eliminando espacios
                matriz.append(fila)
            # Añado perdidas
            fila = lineas[0].split(' ')
            matriz.append(fila)
            fila = lineas[32].split('        ')
            matriz.append(fila)

            # En matriz_V (16x1) guardo los datos de las tensiones pasándolos a flotantes
            filas_matriz = range(len(matriz))
            columnas_matriz = [1]
            matriz_V = [[float(matriz[i][j]) for j in columnas_matriz] for i in filas_matriz]

            # Paso matriz_V a (1x16) y creo el DataFrame
            matriz_V = np.array(matriz_V)
            matriz_V = matriz_V.reshape(1, len(matriz_V))
            dato_minuto = pd.DataFrame(matriz_V,
                                       columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                                'V12',
                                                'V13', 'V14', 'PERD', 'GWT7'],
                                       index=['day' + str(dia + 1) + '_min' + str(minute).zfill(4)])

            out_list.append(dato_minuto)
        print(f'Datos de salida leídos del día {dia + 1}')

    return pd.concat(out_list)


def crear_data_set(directorio: str, directorio_opf: str, directorio_dataset: str, dias: int, minutos: int, media: float = 0,
                   desviacion_estandar: float = 0.1) -> None:
    crea_carpetas(directorio, dias, minutos)
    generar_datos_random(directorio, minutos, media, desviacion_estandar)
    llamada_gams(directorio, directorio_opf, dias, minutos)
    in_data = datos_entrada(directorio, dias, minutos)
    out_data = datos_salida(directorio, dias, minutos)
    data = pd.concat([in_data, out_data], axis=1)
    data.to_csv(directorio_dataset + '/dataset.csv')

if __name__ == '__main__':
    #dir = 'datosDias'
    dir_opf = 'OPF\OPF'
    days = 364
    minutes = 1443
    m = 0
    des = 0.1
    """
    in_data = datos_entrada(dir, days, minutes)
    out_data = datos_salida(dir, days, minutes)
    data = pd.concat([in_data, out_data], axis=1)
    data.to_csv('tmp_dataset/dataset.csv')
    """
    dir2 = 'datosDias2'
    llamada_gams(dir2, dir_opf, 0, minutes)