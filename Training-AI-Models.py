import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv('traversal_cost_data.csv')

# Identify non-numerical features
non_numerical_features = ['type_of_terrain', 'zone_classification', 'time_of_day']

# Apply One-Hot Encoding to non-numerical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[non_numerical_features])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(non_numerical_features))

# Drop the original non-numerical columns and concatenate the encoded features
data = data.drop(non_numerical_features, axis=1)
data = pd.concat([data, encoded_df], axis=1)

# Split the data into features and target
X = data.drop('traversal_cost', axis=1)
y = data['traversal_cost']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)

print("Linear Regression Model:")
print(f"MAE: {mae_linear}")
print(f"MSE: {mse_linear}")
print(f"RMSE: {rmse_linear}")

# Polynomial Regression Model
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print("Polynomial Regression Model:")
print(f"MAE: {mae_poly}")
print(f"MSE: {mse_poly}")
print(f"RMSE: {rmse_poly}")

# Neural Network Model
nn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)

print("Neural Network Model:")
print(f"MAE: {mae_nn}")
print(f"MSE: {mse_nn}")
print(f"RMSE: {rmse_nn}")

# Analysis and Summary
print("\nAnalysis and Summary:")
print(f"Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, RMSE: {rmse_linear}")
print(f"Polynomial Regression - MAE: {mae_poly}, MSE: {mse_poly}, RMSE: {rmse_poly}")
print(f"Neural Network - MAE: {mae_nn}, MSE: {mse_nn}, RMSE: {rmse_nn}")

# Save the best model based on the smallest MSE
best_model = None
best_model_name = ""
best_mse = float('inf')

if mse_linear < best_mse:
    best_mse = mse_linear
    best_model = linear_model
    best_model_name = "linear_model"

if mse_poly < best_mse:
    best_mse = mse_poly
    best_model = poly_model
    best_model_name = "poly_model"
    joblib.dump(poly, 'poly.pkl')  # Save the polynomial features transformer

if mse_nn < best_mse:
    best_mse = mse_nn
    best_model = nn_model
    best_model_name = "nn_model"

joblib.dump(best_model, f'{best_model_name}.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nBest model: {best_model_name} with MSE: {best_mse}")