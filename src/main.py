import src.tsp.exhaustive as tsp
from src.tsp.cooling import CoolingType
#
# test = tsp.Annealing(CoolingType.CONSTANT)
# for i in range(10):
#     print(test._evaluate())
#     test.anneal()
#     print(test._evaluate())

test = tsp.Exhaustive()