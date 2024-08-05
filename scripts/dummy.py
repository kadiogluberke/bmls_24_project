from xgboost import XGBRegressor
import numpy as np

X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)

model = XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

model.save_model("model.json")
