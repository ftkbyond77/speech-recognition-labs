import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = np.array([[1, 2], [3, 4], [5, 6]])
scaled_x = scaler.fit_transform(x)