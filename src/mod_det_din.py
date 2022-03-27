import numpy as np
from pulp import *


def no_setup_eoq():
    """
    Resolución del problema de transporte
    En esta primera versión se cambian los datos dentro de la
    propia función, ya que esta no tiene argumentos. En un futuro se
    sacarán los datos fuera.
    Returns
    -------

    """
    # Etiquetas de los orígenes y destinos
    orig = ["R1", "O1", "R2", "O2", "R3", "O3", "R4", "O4"]
    dest = ["1", "2", "3", "4", "E"]

    # Capacidad de producción de los orígenes
    o_dict = {
        "R1": 90,
        "O1": 50,
        "R2": 100,
        "O2": 60,
        "R3": 120,
        "O3": 80,
        "R4": 110,
        "O4": 70,
    }

    # Demanda de los destinos
    d_dict = {
        "1": 100,
        "2": 190,
        "3": 210,
        "4": 160,
        "E": 20,
    }

    # Costes origen-destino
    costs = [
        # 1 2 3 4 E
        [6, 1.05*6.1, 1.05*6.2, 1.05*6.3, 0],  # R1
        [9, 1.05*9.1, 1.05*9.2, 1.05*9.3, 0],  # O1
        [100000e10, 1.05*6, 1.05*6.1, 1.05*6.2, 0],  # R2
        [100000e10, 1.05*9, 1.05*9.1, 1.05*9.2, 0],  # O2
        [100000e10, 100000e10, 1.05*6, 1.05*6.1, 0],  # R3
        [100000e10, 100000e10, 1.05*9, 1.05*9.1, 0],  # O3
        [100000e10, 100000e10, 100000e10, 1.05*6, 0],  # R4
        [100000e10, 100000e10, 100000e10, 1.05*9, 0],  # O4
    ]

    # Los costes van a un dictionario
    costs = makeDict([orig, dest], costs, 0)

    # Se ponen los datos en una variable
    prob = LpProblem("EOQ_No_Setup", LpMinimize)

    # Creación de todas las posibles rutas
    Routes = [(w, b) for w in orig for b in dest]

    # Diccionario con todas las rutas
    vars = LpVariable.dicts("Ruta", (orig, dest), 0, None, LpInteger)

    # Se incluye la función objetivo
    prob += lpSum([vars[w][b] * costs[w][b] for (w, b) in Routes]), "Suma_de_costes"

    # Máxima cantidad de reparto
    for w in orig:
        prob += lpSum([vars[w][b] for b in dest]) <= o_dict[w], "Suma_de_productos_duera_de_almacen_%s" % w

    # Restricciones de demanda
    for b in dest:
        prob += lpSum([vars[w][b] for w in orig]) >= d_dict[b], "Suma_de_productos_a_destino%s" % b

    # Se escribe el problema en un fichero
    prob.writeLP("Transporte.lp")

    # Resolución del problema
    prob.solve()

    # Estado de la solución por pantalla
    print("Estado:", LpStatus[prob.status])

    # Se escribe cada variable con su solucion
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # Coste total
    print("Coste total = ", value(prob.objective))

def eoq_setup(x_ini, d_array, k_arr, h_arr):
    """
    Modelo EOQ con coste de pedido
    Parameters
    ----------
    x_ini: array con los valores iniciales de resolución
    d_array: demandas iniciales
    k_arr: coste de pedido
    h_arr: coste de almacenaje

    Returns
    -------

    """

    # Aplicacion del paso 1 del algoritmo
    minimos1, z1 = step_1(x_ini, d_array, k_arr[0], h_arr[0])

    mins = []
    zs = []
    mins.append(minimos1)
    zs.append(z1)
    min_ant = minimos1.copy()

    # Aplicación de pasos intermedios del algoritmo
    for i in range(1, len(d_array)-1):

        mini, zi = step_gral(d_array, k_arr[i], h_arr[i], min_ant, i)
        mins.append(mini)
        zs.append(zi)
        min_ant = mini.copy()

    # Aplicacion del paso final
    minf, zf = final_step(d_array[-1], k_arr[-1], h_arr[-1], mini)
    mins.append(minf)
    zs.append(zf)

    z_optimos = get_results(zs, d_array)

    print(f'Las cantidades optimas son {z_optimos}, con un coste de {minf}')


def eoq_setup_simple(d_arr, k_arr, h_arr):
    """
    Modelo con coste de pedido simplificado bajo ciertas condiciones
    de convexidad
    Parameters
    ----------
    d_arr demandas (array)
    k_arr coste de pedido (array)
    h_arr coste de almacenaje (array)

    Returns
    -------

    """
    # Aplicación del paso inicial
    minimos1, z1, x1 = step_1_simple(d_arr, k_arr[0], h_arr[0])

    xs = []
    mins = []
    zs = []
    mins.append(minimos1)
    zs.append(z1)
    xs.append(x1)
    min_ant = minimos1.copy()

    # Aplicación de pasos intermedios
    for i in range(1, len(d_arr)-1):

        mini, zi, xi = step_gral_simple(d_arr, k_arr[i], h_arr[i], min_ant, i)
        mins.append(mini)
        zs.append(zi)
        xs.append(xi)
        min_ant = mini.copy()

    # Aplicación del paso final
    minf, zf, xf = final_step_simple(d_arr[-1], k_arr[-1], h_arr[-1], mini)
    mins.append(minf)
    zs.append(np.array([zf]))
    xs.append(np.array(xf))
    zs_finales = get_results_simple(zs, xs, d_arr)

    # Se escriben los resultados por pantalla
    print(f'Las cantidades óptimas son {zs_finales} con un coste total de {minf}')

def get_results_simple(zs, x_arr, d_arr):
    """
    Obtención de los resultados para el algoritmo EOQ simplificado
    con coste de pedido. Básicamente es revertir la lista y ver
    varias condiciones
    Parameters
    ----------
    zs valores obtenidos
    x_arr valores iniciales
    d_arr demandas

    Returns
    -------

    """

    zs_r = list(reversed(zs))
    zs_finales = []

    x_r = list(reversed(x_arr))
    d_r = list(reversed(d_arr))
    x = 0
    for i in range(len(zs_r)):

        if x == 0:
            zs_finales.append(zs_r[i][0])
            if zs_r[i][0] != 0:
                x = 0
            else:
                x = d_r[i]
        else:
            z = zs_r[i][np.where(x_r[i] == x)][0]
            zs_finales.append(z)
            if z != 0:
                x = 0
            else:
                x = d_r[i]
    return list(reversed(zs_finales))


def get_results(zs, d_arr):
    """
    Obtención de resultados para el modelo EOQ con coste de pedido
    no simplificado. Se hacen listas reversas.
    Parameters
    ----------
    zs Valores de z calculados
    d_arr demandas

    Returns
    -------

    """

    zs = list(reversed(zs))
    d_arr = list(reversed(d_arr))
    print(zs[2])
    xant = 0
    z = zs[0]

    z_finales = []
    z_finales.append(z)
    for i, item in enumerate(d_arr[:-1]):
        x = xant + item - z
        z = zs[i+1][x]
        z_finales.append(z)
        xant = x
    return list(reversed(z_finales))
def coste(z, k):
    """
    Función de coste auxiliar para algunos problemas.
    Debe modificarse en función del problema analizado.
    Parameters
    ----------
    z
    k

    Returns
    -------

    """

    # if z == 0:
    #     c = 0
    # elif z <= 3:
    #     c = 10*z + k
    # else:
    #     c = 30 + 20*(z-3) + k
    # return c
    if z == 0:
        c = 0
    else:
        c = 2*z + k
    return c

def step_1(x_ini, d_array, k, h):
    """
    Paso 1 del algoritmo EOQ con coste de pedido
    no simplificado
    Parameters
    ----------
    x_ini valores iniciales
    d_array demandas
    k coste de pedido
    h coste de almacenamiento

    Returns
    -------

    """
    x = np.arange(sum(d_array[1:]) + 1)
    hx = h * x
    z = x + d_array[0] - x_ini

    c = np.array(list([coste(i, k) for i in z]))
    ct = c + hx

    minimos1 = ct.copy()

    return(minimos1, z)

def step_1_simple(d_array, k, h):
    """
    Paso 1 del algoritmo EOQ con coste de pedido
    simplificado
    Parameters
    ----------
    d_array demandas
    k coste de pedido
    h coste de almacenamiento

    Returns
    -------

    """
    x2 = np.cumsum(d_array[1:])
    x = np.insert(x2, 0, 0)
    hx = h * x
    z = x + d_array[0]

    c = np.array(list([coste(i, k) for i in z]))
    ct = c + hx

    minimos1 = ct.copy()
    return(minimos1, z, x)

def step_gral(d_array, k, h, minimos, i):
    """
    Paso intermedio del algoritmo EOQ con coste de pedido
    no simplificado
    Parameters
    ----------
    d_array demandas
    k coste de pedido
    h coste de almacenamiento
    minimos vector con los minimos anteriores
    i paso en el que se encuentra
    Returns
    -------

    """
    x = np.arange(sum(d_array[i+1:]) + 1)
    hx = h * x
    z = np.arange(max(x) + d_array[i] +1)
    c = np.array(list([coste(i, k) for i in z]))

    ct = np.full((len(x), len(z)), np.inf)

    for r, item in enumerate(x):
        cr = c[0:r+d_array[i]+1]
        for col, val in enumerate(cr):
            ct[r,col] = hx[r] + cr[col] + minimos[r + d_array[i] - z[col]]

    minimos_i = np.min(ct, axis=1)
    z_opt = np.argmin(ct, axis=1)

    return minimos_i, z_opt


def step_gral_simple(d_array, k, h, minimos, i):
    """
    Paso intermedio del algoritmo EOQ con coste de pedido
    simplificado
    Parameters
    ----------
    d_array demandas
    k coste de pedido
    h coste de almacenamiento
    minimos vector con los minimos anteriores
    i paso en el que se encuentra
    Returns
    -------

    """
    x2 = np.cumsum(d_array[i+1:])
    x = np.insert(x2, 0, 0)
    hx = h * x
    z2 = x + d_array[i]
    z = np.insert(z2, 0, 0)
    c = np.array(list([coste(i, k) for i in z]))

    ct = np.full((len(x), len(z)), np.inf)

    for r, item in enumerate(x):
        cr = c[0:r+d_array[i]+1]
        for col, val in enumerate(cr):
            if col == 0:
                ct[r, col] = hx[r] + minimos[r+1]
            elif col == r +1 :
                ct[r,col] = cr[col] + hx[r] + minimos[0]
            else:
                pass
    minimos_i = np.min(ct, axis=1)
    z_opt = z[np.argmin(ct, axis=1)]
    return minimos_i, z_opt, x

def final_step(d, k, h, minimos):
    """
    Paso final del algoritmo EOQ con coste de pedido
    no simplificado
    Parameters
    ----------
    d demanda
    k coste de pedido
    h coste de almacenamiento
    minimos vector con los minimos anteriores
    Returns
    -------

    """
    x = 0
    hx = h * x
    z = np.arange(d + 1)

    c = np.array(list([coste(i, k) for i in z]))
    ct = np.full_like(z, np.inf)

    for i in range(len(z)):

        ct[i] = hx + c[i] + minimos[d-z[i]]

    minimos_f = np.min(ct)
    z_opt_f = np.argmin(ct)

    return(minimos_f, z_opt_f)


def final_step_simple(d, k, h, minimos):
    """
    Paso final del algoritmo EOQ con coste de pedido
    simplificado
    Parameters
    ----------
    d demanda
    k coste de pedido
    h coste de almacenamiento
    minimos vector con los minimos anteriores
    Returns
    -------

    """
    x = 0
    hx = h * x
    z = np.array([0, d])

    c = np.array(list([coste(i, k) for i in z]))
    ct = np.full_like(z, np.inf)

    for i in range(len(z)):

        ct[i] = hx + c[i] + np.flip(minimos)[i]

    minimos_f = np.min(ct)
    z_opt_f = z[np.argmin(ct)]

    return(minimos_f, z_opt_f, x)

if __name__ == '__main__':
    # d = [61, 26, 90, 67]
    # k = [98, 114, 185, 70]
    # h = [1, 1, 1, 1]
    no_setup_eoq()