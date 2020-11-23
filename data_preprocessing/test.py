import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def a():
    a = np.load('Diffs_old.npz')
    weight = a['weight']
    name = a['name']

    x = a['feature']
    y = np.zeros(len(x))

    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    label = pd.DataFrame(data=y
                         , columns=['target'])
    finalDf = pd.concat([principalDf, label], axis=1)
    print(finalDf)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    targets = ['other', 'suspicious']
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        if target == 'other':
            ax.scatter(finalDf.loc[:, 'principal component 1']
                       , finalDf.loc[:, 'principal component 2']
                       , c=color
                       , s=50)

    ax.legend(targets)
    ax.grid()
    plt.show()


aa = '(55,3)'
import re
l = re.findall(r"\d+\.?\d*",aa)
print(aa)