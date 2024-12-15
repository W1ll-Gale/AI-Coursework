import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

# Transform features to polynomial features of degree 4
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")