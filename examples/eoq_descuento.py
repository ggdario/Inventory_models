import matplotlib.pyplot as plt
import pandas as pd
from src.mod_det import *


h = 0.02
d = 187.5
k = 20
l = 2
c1 = 3
c2 = 2.5
q = 1000

#opt_base, coste_base = eoq_descuento(k, d, h, l, q, c1, c2)

costes = np.arange(2.5, 1.99, -0.05)
umbral = np.arange(1000, 1101, 10)

opt_coste = [eoq_descuento(k, d, h, l, umbral[i], c1, costes[i]) for i in range(len(costes))]
dias_pedido = [j[0]/d for j in opt_coste]
opt_graf = [i[0] for i in opt_coste]
coste_graf = [i[1] for i in opt_coste]
print(dias_pedido)

fig = plt.figure(figsize=(6, 6))
graf = plt.scatter(costes, umbral,
            linewidths=1, alpha=.7,
            edgecolor='k', s = 100,
            c=dias_pedido)
plt.title("Días transcurridos hasta el nivel de pedido")
plt.xlabel("Precio con descuento en $")
plt.ylabel("Umbral de unidades para aplicar el descuento")
plt.colorbar(graf)
plt.show()