import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from scipy import stats
from pyod.models.knn import KNN  
from pyod.utils.data import generate_data, get_outliers_inliers 

x_train, y_train = generate_data(n_train=300, train_only=True, n_features=2)

outlier_fraction = 0.1

x_outliers, x_inliers = get_outliers_inliers(x_train,y_train)
n_inliers = len(x_inliers)
n_outliers = len(x_outliers)

f1 = x_train[:, [0]].reshape(-1, 1) 
f2 = x_train[:, [1]].reshape(-1, 1) 

xx,yy = np.meshgrid(np.linspace(-10,10,200), np.linspace(-10,10,200))

plt.scatter(f1, f2) 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
 
clf = KNN(contamination = outlier_fraction) 
clf.fit(x_train, y_train) 
scores_pred = clf.decision_function(x_train)*-1

y_pred = clf.predict(x_train) 
n_errors = (y_pred != y_train).sum() 

print('The number of prediction errors are ' + str(n_errors)) 

threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_fraction)

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
Z = Z.reshape(xx.shape)

subplot = plt.subplot(1, 2, 1)
subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)

a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

b = subplot.scatter(x_train[:-n_outliers, 0], x_train[:-n_outliers, 1], c='white', s=20, edgecolor='k')

c = subplot.scatter(x_train[-n_outliers:, 0], x_train[-n_outliers:, 1], c='black', s=20, edgecolor='k')
subplot.axis('tight')

subplot.legend(
    [a.collections[0], b, c],
    ['learned decision function', 'true inliers', 'true outliers'],
    prop=matplotlib.font_manager.FontProperties(size=10),
    loc='lower right'
)

subplot.set_title('K-Nearest Neighbours')
subplot.set_xlim((-10, 10))
subplot.set_ylim((-10, 10))
plt.show()

