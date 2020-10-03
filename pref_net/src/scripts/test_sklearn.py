from sklearn.model_selection import train_test_split
import numpy as np

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(train_test_split(X, y, test_size=0.2, random_state=0))