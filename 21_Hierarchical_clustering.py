
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset= pd.read_csv(r'D:\New folder\Mall_Customers.csv')
x= dataset.iloc[:,[3,4]].values
#using dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram= sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distance')
plt.show()
#train the Hierarchical clustering model on dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')
y_hc= hc.fit_predict(x)
print(y_hc)
#visualising the clusters


plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='cluster 3')
"""plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='cluster 5')"""
plt.title('cluster of customers')
plt.xlabel('annual income(k$)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()