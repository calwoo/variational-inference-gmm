import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# data set
N = 300
cluster = 3
means = np.array([2, 5, 9])
stds = np.array([1, 1.5, 0.8])

def generate_data(N, clusters, means, stds):
    data = []
    for i in range(clusters):
        cluster_data = np.random.normal(means[i], stds[i], N)
        data.append(cluster_data)
    return np.concatenate(np.array(data))

data = generate_data(N, cluster, means, stds)

# plot the data
fix, ax = plt.subplots(figsize=(12,3))
sns.distplot(data[:N], color='green', rug=True)
sns.distplot(data[N:2*N], color='orange', rug=True)
sns.distplot(data[2*N:], color='red', rug=True)
plt.show()

class Model:
    def __init__(self, data, num_clusters=3, sigma=1):
        self.data = data
        self.K = num_clusters
        self.n = data.shape[0]
        self.sigma = sigma
        # get model parameters-- these are the things CAVI will update to max ELBO
        self.varphi = np.random.dirichlet(np.randon.random(self.K), self.n)
        self.m = np.random.randint(low=np.min(self.data), high=np.max(self.data), size=self.K).astype(float)
        self.s2 = np.random.random(self.K)

    def elbo(self):
        pass

    def cavi(self):
        pass

    