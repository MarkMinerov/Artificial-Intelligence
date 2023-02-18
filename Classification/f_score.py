import numpy as np
from sklearn.feature_selection import f_classif

X = np.array([5000, 18000, 47500, 45600, 49500]) # all classes together
X1 = np.array([5000, 18000]) # first class
X2 = np.array([47500, 45600, 49500]) #second class

# algorithm
mu = np.mean(X) # overall mean
mu1 = np.mean(X1) # mean of X1
mu2 = np.mean(X2) # mean of X2
SSm = np.sum(((X-mu)**2)) # SS(mean)
SSf = np.sum((X1-mu1)**2) + np.sum((X2-mu2)**2) # SS(fit)
p_fit = 2 # because we have 2 mean lines for X1 and X2
p_mean = 1 # one line for mean of X
F = ((SSm - SSf)*(X.shape[0]-p_fit))/((p_fit - p_mean)*SSf)
print(F, X.shape)

# or

X = np.array([5000, 18000, 47500, 45600, 49500]).reshape(-1,1) # rotate vector
y = np.array([1,1,0,0,0]) # class for each X value
F,pval = f_classif(X,y)
print(X, F, pval)