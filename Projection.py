import numpy as np

def L2_norm(x):
    return np.linalg.norm(x)

class Projection:
    """Projection Algorithm

    The step of this algorithm is as follows:
        set the param gamma and delta
        get the start point as x0
        while r(x_k)!=0 do
            calc m that satisfies 
                        <F(x_k - gamma^m * r(x_k)), r(x_k)> >= delta * ||r(x_k)||^2
                        m belongs nonnegative integer
                   the method used there is Complete trial and error method
            calc z_k = P(x_k - F(x_k))
            calc beta_k = gamma^m_k
            calc y_k = (1 - beta_k) * x_k + beta_k * z_k
            calc alpha = F(y_k).T * (x_k - y_k) / ||F(y_k)||^2
            calc x_k+1 = P(x_k - alpha * F(y_k))

    The function used in this projection are F(*),P(*), r(*):
        F(x) is the part of VI<F(x*), x-x*> which need set by user
        P(*) is Projection function.
        r(x) = x - P(x - F(x)) 
    """

    def __init__(self,gamma, 
                    delta, 
                    error = 1e-6, 
                    max_iter = 1e5):
        self.gamma = gamma
        self.delta = delta
        self.error = error
        self.max_iter = max_iter

        self.M = None
        self.X = None
        self.Z = None
        self.Y = None
        self.beta = None
        self.alpha = None
        self.F = None
        self.step = None

    def run(self, F, start):
        self.X = [start,]
        self.F = F
        self.M = [None,]
        self.Z = [None,]
        self.Y = [None,]
        self.alpha = [None, ]
        self.beta = [None, ]
        self.step = 0
        while self.step < self.max_iter:
            self.update(self.X[-1])
            self.step += 1
        return self.X[-1]

    def update(self, x):
        F = self.F
        P = self.P
        z = P(x - F(x))

        m, beta,  y = self.search_m(x, z)

        alpha = F(y).dot(x - y) / (L2_norm(y) ** 2)
        x_new = P(x - alpha * F(y))

        self.M.append(m)
        self.Z.append(z)
        self.beta.append(beta)
        self.Y.append(y)
        self.alpha.append(alpha)
        self.X.append(x_new)

    def search_m(self, x, z):
        m = 0
        while True:
            beta = self.gamma ** m
            y = (1 - beta) * x + beta * z
            f = self.F(y)
            e = x - self.P(x - self.F(x))
            if f.dot(e) >= self.delta * (L2_norm(e) ** 2):
                break
            else:
                m += 1
        return beta, m, y

    def P(self, x):
        x_ = x.copy()
        x_[x_ < 0] = 0
        return x_

    def dump(self, filename="log"):
        import pickle
        with open(filename, "wb") as file:
            pickle.dump(self.M, file)
            pickle.dump(self.Z, file)
            pickle.dump(self.beta, file)
            pickle.dump(self.Y, file)
            pickle.dump(self.alpha, file)
            pickle.dump(self.X, file)