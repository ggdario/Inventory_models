import math


def eoq_classic(k, d, h, l, time_unit='dias'):

    opt = int(math.sqrt(2 * k * d / h))
    to = int(round(opt / d, 0))

    if to > l:
        reord_point = to
        print(f'La política óptima es pedir {opt} unidades cada {reord_point} {time_unit}')
    else:
        n = round(l / to)
        reord_point = l - n * to
        print(f'La política óptima es pedir {opt} unidades cuando el inventario caiga a {reord_point * d} unidades')

    coste = k/(opt/d) + h * (opt/2)
    print(f'El coste asociado es {coste} por 1 {time_unit}')
if __name__ == '__main__':
    eoq_classic(100,100,0.02,8)