import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    
    data = pd.read_csv(filepath)

    non_numerical_features = ['type_of_terrain', 'zone_classification', 'time_of_day']

    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(data[non_numerical_features])

    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(non_numerical_features))

    data = data.drop(non_numerical_features, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    X = data.drop('traversal_cost', axis=1)
    y = data['traversal_cost']

    return X, y

def analyze_target_variable(y):
    
    y_min = y.min()
    y_max = y.max()
    y_mean = y.mean()
    y_std = y.std()
    y_range = y_max - y_min

    print("\nTarget Variable (traversal_cost) Analysis:")
    print(f"Min: {y_min:.2f}")
    print(f"Max: {y_max:.2f}")
    print(f"Mean: {y_mean:.2f}")
    print(f"Std: {y_std:.2f}")
    print(f"Range: {y_range:.2f}")

    return y_range

def train_and_evaluate_models(X_train, X_test, y_train, y_test, y_range):
    results = {}

    results['linear'] = linear_regression(X_train, y_train, X_test, y_test)

    results['poly'] = polynomial_regression(X_train, y_train, X_test, y_test)

    results['nn'] = neural_network_model(X_train, y_train, X_test, y_test)

    return results

def linear_regression(X_train, y_train, X_test, y_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    percent_error_linear = np.mean(np.abs((y_test - y_pred_linear) / y_test)) * 100

    results ={
        'model': linear_model,
        'mae': mae_linear,
        'mse': mse_linear,
        'rmse': rmse_linear,
        'percent_error': percent_error_linear
    }

    print("Linear Regression Model:")
    print(f"MAE: {mae_linear}")
    print(f"MSE: {mse_linear}")
    print(f"RMSE: {rmse_linear}")
    print(f"Percentage Error: {percent_error_linear:.2f}%")

    return results

def polynomial_regression(X_train, y_train, X_test, y_test):
    poly = PolynomialFeatures(degree=4)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    percent_error_poly = np.mean(np.abs((y_test - y_pred_poly) / y_test)) * 100

    results = {
        'model': poly_model,
        'poly': poly,
        'mae': mae_poly,
        'mse': mse_poly,
        'rmse': rmse_poly,
        'percent_error': percent_error_poly
    }

    print("Polynomial Regression Model:")
    print(f"MAE: {mae_poly}")
    print(f"MSE: {mse_poly}")
    print(f"RMSE: {rmse_poly}")
    print(f"Percentage Error: {percent_error_poly:.2f}%")

    return results

def neural_network_model(X_train, y_train, X_test, y_test):
    nn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=2000, learning_rate_init=0.001, random_state=42)
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mse_nn)
    percent_error_nn = np.mean(np.abs((y_test - y_pred_nn) / y_test)) * 100

    results= {
        'model': nn_model,
        'mae': mae_nn,
        'mse': mse_nn,
        'rmse': rmse_nn,
        'percent_error': percent_error_nn
    }

    print("Neural Network Model:")
    print(f"MAE: {mae_nn}")
    print(f"MSE: {mse_nn}")
    print(f"RMSE: {rmse_nn}")
    print(f"Percentage Error: {percent_error_nn:.2f}%")

    return results

def analyze_model_performance(results, y_range):
    print("\nAnalysis and Summary:")
    for model_name, metrics in results.items():
        print(f"{model_name.capitalize()} Regression - MAE: {metrics['mae']}, MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, Percentage Error: {metrics['percent_error']:.2f}%")
        print(f"MAE/Range: {(metrics['mae']/y_range)*100:.2f}%")
        print(f"RMSE/Range: {(metrics['rmse']/y_range)*100:.2f}%")

def save_best_model(results, scaler):
    best_model = None
    best_model_name = ""
    best_mse = float('inf')

    for model_name, metrics in results.items():
        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_model = metrics['model']
            best_model_name = model_name
            if model_name == 'poly':
                joblib.dump(metrics['poly'], 'poly.pkl')  

    joblib.dump(best_model, f'{best_model_name}_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print(f"\nBest model: {best_model_name} with MSE: {best_mse}")

def plot_results(y, X_test, y_test, results):
    plt.figure(figsize=(18, 6))

    # Linear Regression Plot
    if 'linear' in results:
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, results['linear']['model'].predict(X_test), alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Linear Regression')

    # Polynomial Regression Plot
    if 'poly' in results:
        plt.subplot(1, 3, 2)
        plt.scatter(y_test, results['poly']['model'].predict(results['poly']['poly'].transform(X_test)), alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Polynomial Regression')

    # Neural Network Plot
    if 'nn' in results:
        plt.subplot(1, 3, 3)
        plt.scatter(y_test, results['nn']['model'].predict(X_test), alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Neural Network')

    plt.tight_layout()
    plt.show()
    plt.savefig('model_performance.png')

def main():

    X, y = load_and_preprocess_data('traversal_cost_data.csv')
    y_range = analyze_target_variable(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, y_range)

    analyze_model_performance(results, y_range)

    save_best_model(results, scaler)

    plot_results(y, X_test, y_test, results)

if __name__ == "__main__":
    main()