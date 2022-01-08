import numpy as np
import pandas as pd


def get_recommendation(file, headers=None, index=False, sep=',', umbral_est=5.0, umbral_det=15.0):
    """
    Devuelve por pantalla la recomendación deterministico/probabilístico y estacionario/no estacionario
    en base al coeficiente de variación tanto a nivel columna como a nivel fila de los datos de entrada.

    Parameters
    ----------
    file: str, file where the data is written
    headers: int, opt, row (starting 0) where de column is
    index: int, opt. Column where the labels are
    sep: str, opt. Delimiter for values in file
    umbral_est: float. Umbral para considerar estacionario
    umbral_det: float. Umbral para considerar deterministico

    Returns
    -------

    """
    data = pd.read_csv(file, header=headers, index_col=index, sep=sep)

    coef_stac = np.round(data.apply(lambda x: np.std(x, ddof=1)/np.mean(x) * 100, axis=1),2)
    coef_deter = np.round(data.apply(lambda x: np.std(x, ddof=1)/np.mean(x) * 100, axis=0),2)

    data['CV'] = coef_stac
    data = data.append(coef_deter, ignore_index=True)
    data.to_csv(f"{file.split('.csv')[0]}_cv.csv",sep=sep)
    if any(coef_stac > umbral_est):
        tipo_est = 'NO ESTACIONARIO'
    else:
        tipo_est = 'ESTACIONARIO'

    if any(coef_deter> umbral_det):
        tipo_modelo = 'PROBABILISTICO'
    else:
        tipo_modelo = 'DETERMINISTICO'

    print(f'Se recomienda el uso de un modelo {tipo_modelo} y {tipo_est}')