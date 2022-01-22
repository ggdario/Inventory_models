import math
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def eoq_classic(k, d, h, l, time_unit='dias', descuento=False):

    opt = int(math.sqrt(2 * k * d / h))
    to = int(round(opt / d, 0))

    if to > l:
        reord_point = to
        if not descuento:
            print(f'La política óptima es pedir {opt} unidades cada {reord_point} {time_unit}')
    else:
        n = round(l / to)
        reord_point = l - n * to

        if not descuento:
            print(f'La política óptima es pedir {opt} unidades cuando el inventario caiga a {reord_point * d} unidades')

    coste = k/(opt/d) + h * (opt/2)
    if not descuento:
        print(f'El coste asociado es {coste} por 1 {time_unit}')

    return opt, reord_point

def eoq_descuento(k, d, h, l, q, c1, c2):

    opt_no_desc, reord_no_desc = eoq_classic(k, d, h, l, descuento=True)

    if q < opt_no_desc:
        opt = opt_no_desc

    else:
        tcu1 = c1*d + k*d/opt_no_desc + h*opt_no_desc/2

        a=1
        b = 2*(c2*d - tcu1)/h
        c = 2*k*d/h

        p3 = (-b + math.sqrt(math.pow(b,2) - 4*a*c))/(2*a)

        if q > p3:
            opt = opt_no_desc

        else:
            opt = q

    print(f'Se deben pedir {opt} unidades cuando el nivel caiga a {l * d} unidades')

    return opt

def f_multi(y_arr, k_arr, d_arr, h_arr):

    t1 = zip(k_arr, d_arr, y_arr)
    s1 = np.array([i[0]*i[1]/i[2] for i in t1])

    t2 = zip(h_arr, y_arr)
    s2 = np.array([i[0]*i[1]/2 for i in t2])

    s = s1 + s2

    return float(sum(s))

def constrain(y_arr, a_arr, a_limit):

    term = np.multiply(y_arr, a_arr)

    return a_limit - float(sum(term))

def eoq_multi(k_arr, d_arr, h_arr, a_arr, a_limit):

    k_arr = np.array(k_arr)
    d_arr = np.array(d_arr)
    h_arr = np.array(h_arr)
    a_arr = np.array(a_arr)

    y_arr = np.sqrt(2*k_arr*d_arr/h_arr)

    if constrain(y_arr, a_arr, a_limit) > 0:
        print('Los valores óptimos son: ')
        for i in y_arr: print(f'{round(i,2)}')
        print(f'El coste total es: {round(f_multi(y_arr, k_arr, d_arr, h_arr),2)}')
        print(f'El área total es: {round(sum(np.multiply(y_arr, a_arr)),2)}')

    else:
        bounds = tuple((0, np.inf) for i in y_arr)

        con_dict = {'type': 'ineq', 'fun': lambda x: constrain(x, a_arr, a_limit)}
        ini = np.ones_like(y_arr)

        res = minimize(f_multi, ini, args=(k_arr, d_arr, h_arr), method='trust-constr', tol=1e-6,
                       bounds=bounds, constraints=con_dict, options={'maxiter':200, 'disp':True})

        final = res.x

        print('Los valores óptimos son: ')
        for i in final: print(f'{round(i,2)}')
        print(f'El coste total es: {round(f_multi(final, k_arr, d_arr, h_arr),2)}')
        print(f'El área total es: {round(sum(np.multiply(final, a_arr)),2)}')





if __name__ == '__main__':
    y = [11.55, 20, 24.49]
    k = [10, 5, 15]
    d = [2, 4, 4]
    h = [0.3, 0.1, 0.2]
    a = [1, 1, 1]
    a_limit = 25
    eoq_multi(k, d, h, a, a_limit)