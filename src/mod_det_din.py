import numpy as np
from pulp import *


def no_setup_eoq():

    orig = ["R1", "O1", "R2", "O2", "R3", "O3", "R4", "O4"]
    dest = ["1", "2", "3", "4", "E"]

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

    d_dict = {
        "1": 100,
        "2": 190,
        "3": 210,
        "4": 160,
        "E": 20,
    }

    costs = [  # Bars
        # 1 2 3 4 E
        [6, 6.1, 6.2, 6.3, 0],  # R1
        [9, 9.1, 9.2, 9.3, 0],  # O1
        [100000, 6, 6.1, 6.2, 0],  # R2
        [100000, 9, 9.1, 9.2, 0],  # O2
        [100000, 100000, 6, 6.1, 0],  # R3
        [100000, 100000, 9, 9.1, 0],  # O3
        [100000, 100000, 100000, 6, 0],  # R4
        [100000, 100000, 100000, 9, 0],  # O4
    ]

    # The cost data is made into a dictionary
    costs = makeDict([orig, dest], costs, 0)

    # Creates the 'prob' variable to contain the problem data
    prob = LpProblem("EOQ_No_Setup", LpMinimize)

    # Creates a list of tuples containing all the possible routes for transport
    Routes = [(w, b) for w in orig for b in dest]

    # A dictionary called 'Vars' is created to contain the referenced variables(the routes)
    vars = LpVariable.dicts("Route", (orig, dest), 0, None, LpInteger)

    # The objective function is added to 'prob' first
    prob += lpSum([vars[w][b] * costs[w][b] for (w, b) in Routes]), "Sum_of_Transporting_Costs"

    # The supply maximum constraints are added to prob for each supply node (warehouse)
    for w in orig:
        prob += lpSum([vars[w][b] for b in dest]) <= o_dict[w], "Sum_of_Products_out_of_Warehouse_%s" % w

    # The demand minimum constraints are added to prob for each demand node (bar)
    for b in dest:
        prob += lpSum([vars[w][b] for w in orig]) >= d_dict[b], "Sum_of_Products_into_Bar%s" % b

    # The problem data is written to an .lp file
    prob.writeLP("BeerDistributionProblem.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # The optimised objective function value is printed to the screen
    print("Total Cost of Transportation = ", value(prob.objective))

def eoq_setup(x_ini, d_array, k_arr, h_arr):
    minimos1, z1 = step_1(x_ini, d_array, k_arr[0], h_arr[0])

    mins = []
    zs = []
    mins.append(minimos1)
    zs.append(z1)
    min_ant = minimos1.copy()

    for i in range(1, len(d_array)-1):

        mini, zi = step_gral(d_array, k_arr[i], h_arr[i], min_ant, i)
        mins.append(mini)
        zs.append(zi)
        min_ant = mini.copy()

    minf, zf = final_step(d_array[-1], k_arr[-1], h_arr[-1], mini)
    mins.append(minf)
    zs.append(zf)

    z_optimos = get_results(zs, d_array)

    print(f'Las cantidades optimas son {z_optimos}, con un coste de {minf}')

def get_results(zs, d_arr):


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

    if z == 0:
        c = 0
    elif z <= 3:
        c = 10*z + k
    else:
        c = 30 + 20*(z-3) + k
    return c

def step_1(x_ini, d_array, k, h):

    x = np.arange(sum(d_array[1:]) + 1)
    hx = h * x
    z = x + d_array[0] - x_ini

    c = np.array(list([coste(i, k) for i in z]))
    ct = c + hx

    minimos1 = ct.copy()

    return(minimos1, z)

def step_gral(d_array, k, h, minimos, i):

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

def final_step(d, k, h, minimos):

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

if __name__ == '__main__':
    d = [3,2,4]
    k = [3,7,6]
    h = [1,3,2]
    eoq_setup(1, d, k, h)