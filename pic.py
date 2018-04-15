import pickle
import matplotlib.pyplot as plt

with open("log", "rb") as file:
    M = pickle.load(file)
    Z = pickle.load(file)
    beta = pickle.load(file)
    Y = pickle.load(file)
    alpha = pickle.load(file)
    X = pickle.load(file)

for i in range(1, 25):
    x = []
    for _ in X:
        x.append(_[i])
    plt.plot(x)
plt.show()