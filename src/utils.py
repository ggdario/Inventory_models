import numpy as np
import pandas as pd


def get_recommendation(file, headers=None, index=False, sep=',', umbral_est=5.0, umbral_det=15.0):
    """
    Devuelve por pantalla la recomendación deterministico/probabilístico y estacionario/no estacionario
    en base al coeficiente de variación tanto a nivel columna como a nivel fila de los datos de entrada.

    Parameters
    ----------
    file: str, fichero con los valores para la recomendacion
    headers: int, opt, fila (empezando en 0) con los nombres de las columnas
    index: int, opt. Columna con los numeros de fila
    sep: str, opt. Delimitador para separar los valores
    umbral_est: float. Umbral para considerar estacionario
    umbral_det: float. Umbral para considerar deterministico

    Returns
    -------

    """

    # Se lee el archivo
    data = pd.read_csv(file, header=headers, index_col=index, sep=sep)

    # calculo de los coeficientes de variación
    coef_stac = np.round(data.apply(lambda x: np.std(x, ddof=1)/np.mean(x) * 100, axis=1),2)
    coef_deter = np.round(data.apply(lambda x: np.std(x, ddof=1)/np.mean(x) * 100, axis=0),2)

    data['CV'] = coef_stac
    data = data.append(coef_deter, ignore_index=True)
    data.to_csv(f"{file.split('.csv')[0]}_cv.csv",sep=sep)

    # Determinacion de estacionario o no
    if any(coef_stac > umbral_est):
        tipo_est = 'NO ESTACIONARIO'
    else:
        tipo_est = 'ESTACIONARIO'

    # Determinación de determinístico o probabilístico
    if any(coef_deter> umbral_det):
        tipo_modelo = 'PROBABILISTICO'
    else:
        tipo_modelo = 'DETERMINISTICO'

    # Se escribe el resultado en pantalla
    print(f'Se recomienda el uso de un modelo {tipo_modelo} y {tipo_est}')