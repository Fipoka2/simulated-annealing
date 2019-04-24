import random, numpy, math
import src.tsp.annealing as tsp
from src.plot.utils import PlotData, PlotUtils
from src.tsp.cooling import CoolingType
from src.tsp.exhaustive import Exhaustive

N = 10

cities = [(x, random.sample(range(100), 2)) for x in range(N)]

matrix = numpy.zeros((N, N))
pairs = [(a, b) for a in cities for b in cities]
for idx, pair in enumerate(pairs):
    a, b = pair
    distance = math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1])
    matrix[a[0], b[0]] = distance
    matrix[b[0], a[0]] = distance
numpy.fill_diagonal(matrix, matrix.max() + 1)

test = tsp.Annealing(CoolingType.CONSTANT, matrix=matrix)
test2 = Exhaustive(matrix)
test3 = tsp.Annealing(CoolingType.MAX_LINEAR, matrix=matrix)

tour2, val = test2.find_best_route()

for i in range(2):
    test.anneal()
    test3.anneal()
tour = test.route
# tour2 = test2.route
tour3 = test3.route

datas = [PlotData([cities[tour[i % N]][1][0] for i in range(N + 1)],
                  [cities[tour[i % N]][1][1] for i in range(N + 1)],
                  color="blue", label="1111"),
         PlotData([cities[tour2[i % N]][1][0] for i in range(N + 1)],
                [cities[tour2[i % N]][1][1] for i in range(N + 1)],
                                                     color="green", label="222"),
         PlotData([cities[tour3[i % N]][1][0] for i in range(N + 1)],
                [cities[tour3[i % N]][1][1] for i in range(N + 1)],
                color="red", label="333")]

PlotUtils.build_plot(datas)
