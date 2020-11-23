import matplotlib.pyplot as plt
import numpy as np
import matplotlib

a = np.load('Diffs_old.npz')
weight = a['weight']
name = a['name']


data = a['feature']

data = np.minimum(data,1)
data = np.sum(data, axis=1)

print(np.sum(data>0))
print(np.sum(data>1))
print(np.sum(data>10))
print(np.sum(data>50))
print(np.max(data))
ind = np.argsort(data)
ind = ind[-5:]
print(name[ind])
print(data[ind])

x,y,z=plt.hist(data, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
print(y)
plt.show()
