import numpy as np


x = np.array([[1,2],[0,4]]).T

y = np.array(np.where(x==0)).T


# z = x[y]
print(y)

