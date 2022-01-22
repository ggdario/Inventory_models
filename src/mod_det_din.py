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

if __name__ == '__main__':
    no_setup_eoq()