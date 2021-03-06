import numpy as np

def F(x):
    ret = np.array((
        0,
        24 - x[17],
        24 - x[17],
        24 - x[18],
        24 - x[18],
        44 - x[19],
        24 - x[19],
        44 - x[20],
        24 - x[20],
        0.05 * x[9] + 5 + x[17] - x[21],
        0.05 * x[10] + 5 + x[18] - x[21],
        0.05 * x[11] + 5 + x[17] - x[22],
        0.05 * x[12] + 5 + x[18] - x[22],
        0.05 * x[13] + 5 + x[19] - x[23],
        0.05 * x[14] + 5 + x[20] - x[23],
        0.05 * x[15] + 5 + x[19] - x[24],
        0.05 * x[16] + 5 + x[20] - x[24],
        x[1] + x[2] - x[9] - x[11],
        x[3] + x[4] - x[10] - x[12],
        x[5] + x[6] - x[13] - x[15],
        x[7] + x[8] - x[14] - x[16],
        x[9] + x[10] + 2 * x[21] - 1000,
        x[11] + x[12] + 2 * x[22] - 1000,
        x[13] + x[14] + 2 * x[23] - 1000,
        x[15] + x[16] + 2 * x[24] - 1000,
    ))
    return ret

from Projection import Projection
import pickle

algo = Projection(0.5, 0.5, max_iter=1e5)
start = np.zeros(25)
print(algo.run(F, start))
print(algo.step)
algo.dump()
#
# algo = Projection(0.5, 0.5, max_iter=1e4)
# start = np.zeros(25)
# print(algo.run(F, start))