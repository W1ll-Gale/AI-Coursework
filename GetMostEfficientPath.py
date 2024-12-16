import pandas as pd
import numpy as np
import joblib
import heapq
from collections import deque
import time
import os
from sklearn.preprocessing import OneHotEncoder

def load_models():
    scaler = joblib.load('scaler.pkl')
    best_model_name = None
    for model_name in ['linear_model.pkl', 'poly_model.pkl', 'nn_model.pkl']:
        if os.path.exists(model_name):
            best_model_name = model_name
            break
    if best_model_name is None:
        raise FileNotFoundError("No trained model found. Please ensure that the models are trained and saved correctly.")
    best_model = joblib.load(best_model_name)
    poly = joblib.load('poly.pkl') if best_model_name == 'poly_model.pkl' else None
    return scaler, poly, best_model

def estimate_traversal_costs(filepath, output_filepath):
    data = pd.read_csv(filepath)
    scaler, poly, best_model = load_models()

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

    # Standardize the features
    X_scaled = scaler.transform(data)

    # Predict using the best model
    if poly:
        X_poly = poly.transform(X_scaled)
        y_pred = best_model.predict(X_poly)
    else:
        y_pred = best_model.predict(X_scaled)

    # Save predictions to file
    data['estimated_traversal_cost'] = y_pred
    data.to_csv(output_filepath, index=False)

def dijkstra(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    distances = { (i, j): float('inf') for i in range(rows) for j in range(cols) }
    distances[start] = 0
    priority_queue = [(0, start)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0

    while priority_queue:
        current_distance, current_cell = heapq.heappop(priority_queue)
        nodes_explored += 1
        if current_cell == end:
            return distances[end], nodes_explored
        for direction in directions:
            neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                distance = current_distance + grid[neighbor[0]][neighbor[1]]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf'), nodes_explored

def dfs(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    stack = [(start, 0)]
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0

    while stack:
        (current_cell, current_distance) = stack.pop()
        nodes_explored += 1
        if current_cell == end:
            return current_distance, nodes_explored
        if current_cell in visited:
            continue
        visited.add(current_cell)
        for direction in directions:
            neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in visited:
                stack.append((neighbor, current_distance + grid[neighbor[0]][neighbor[1]]))

    return float('inf'), nodes_explored

def bfs(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, 0)])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0

    while queue:
        (current_cell, current_distance) = queue.popleft()
        nodes_explored += 1
        if current_cell == end:
            return current_distance, nodes_explored
        if current_cell in visited:
            continue
        visited.add(current_cell)
        for direction in directions:
            neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in visited:
                queue.append((neighbor, current_distance + grid[neighbor[0]][neighbor[1]]))

    return float('inf'), nodes_explored

def a_star(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    open_set = [(0, start)]
    g_costs = { (i, j): float('inf') for i in range(rows) for j in range(cols) }
    g_costs[start] = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0

    def heuristic(cell, end, scale=2): # 'scale' is the average cost multiplier.
        return scale * (abs(cell[0] - end[0]) + abs(cell[1] - end[1]))
    

    while open_set:
        _, current_cell = heapq.heappop(open_set)
        nodes_explored += 1
        if current_cell == end:
            return g_costs[end], nodes_explored
        for direction in directions:
            neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                tentative_g_cost = g_costs[current_cell] + grid[neighbor[0]][neighbor[1]]
                if tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_cost, neighbor))

    return float('inf'), nodes_explored

def a_star_optimized(grid, start, end, weight=1.5):
    rows, cols = len(grid), len(grid[0])
    open_set = [(0, start)]
    g_costs = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0

    def heuristic(cell, end):
        return weight * (abs(cell[0] - end[0]) + abs(cell[1] - end[1])) # Weighted Manhattan

    while open_set:
        _, current_cell = heapq.heappop(open_set)
        nodes_explored += 1
        if current_cell == end:
            return g_costs[end], nodes_explored
        for direction in directions:
            neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                tentative_g_cost = g_costs[current_cell] + grid[neighbor[0]][neighbor[1]]
                if tentative_g_cost < g_costs.get(neighbor, float('inf')):
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_cost, neighbor))

    return float('inf'), nodes_explored

def main():
    # Estimate traversal costs for the provided grid data
    grid_filepath = 'provided_grid.csv'
    estimated_grid_filepath = 'Estimated_grid.csv'
    estimate_traversal_costs(grid_filepath, estimated_grid_filepath)

    # Load the estimated grid data
    estimated_grid = pd.read_csv(estimated_grid_filepath)

    # Debugging: Check for NaN values in the estimated grid data
    if estimated_grid.isnull().values.any():
        print("Warning: NaN values found in the estimated grid data before pivot.")
        print(estimated_grid[estimated_grid.isnull().any(axis=1)])

    # Add row and column indices if not present
    if 'row' not in estimated_grid.columns or 'col' not in estimated_grid.columns:
        estimated_grid['row'] = estimated_grid.index // estimated_grid.shape[1]
        estimated_grid['col'] = estimated_grid.index % estimated_grid.shape[1]

    # Ensure all combinations of row and col indices are present
    rows = estimated_grid['row'].max() + 1
    cols = estimated_grid['col'].max() + 1
    full_index = pd.MultiIndex.from_product([range(rows), range(cols)], names=['row', 'col'])
    estimated_grid = estimated_grid.set_index(['row', 'col']).reindex(full_index).reset_index()

    # Debugging: Check for NaN values after reindexing
    if estimated_grid.isnull().values.any():
        print("Warning: NaN values found in the estimated grid data after reindexing.")
        print(estimated_grid[estimated_grid.isnull().any(axis=1)])

    # Fill NaN values in 'estimated_traversal_cost' with the maximum traversal cost
    max_traversal_cost = estimated_grid['estimated_traversal_cost'].max()
    estimated_grid['estimated_traversal_cost'].fillna(max_traversal_cost, inplace=True)

    grid = estimated_grid.pivot(index='row', columns='col', values='estimated_traversal_cost').values

    # Debugging: Print the grid to ensure it's correctly formatted
    print("Grid:")
    print(grid)

    # Define start and end points for pathfinding algorithms
    start = (0, 0)  # Example start point
    end = (len(grid) - 1, len(grid[0]) - 1)  # Example end point

    # Find the most efficient delivery path using different algorithms
    start_time = time.perf_counter()
    shortest_path_cost_dijkstra, nodes_explored_dijkstra = dijkstra(grid, start, end)
    end_time = time.perf_counter()
    print(f"Dijkstra's Algorithm - Cost: {shortest_path_cost_dijkstra}, Nodes Explored: {nodes_explored_dijkstra}, Time: {end_time - start_time:.8f} seconds")

    start_time = time.perf_counter()
    shortest_path_cost_dfs, nodes_explored_dfs = dfs(grid, start, end)
    end_time = time.perf_counter()
    print(f"Depth First Search - Cost: {shortest_path_cost_dfs}, Nodes Explored: {nodes_explored_dfs}, Time: {end_time - start_time:.8f} seconds")

    start_time = time.perf_counter()
    shortest_path_cost_bfs, nodes_explored_bfs = bfs(grid, start, end)
    end_time = time.perf_counter()
    print(f"Breadth First Search - Cost: {shortest_path_cost_bfs}, Nodes Explored: {nodes_explored_bfs}, Time: {end_time - start_time:.8f} seconds")

    start_time = time.perf_counter()
    shortest_path_cost_a_star, nodes_explored_a_star = a_star(grid, start, end)
    end_time = time.perf_counter()
    print(f"A* Search Algorithm - Cost: {shortest_path_cost_a_star}, Nodes Explored: {nodes_explored_a_star}, Time: {end_time - start_time:.8f} seconds")

    start_time = time.perf_counter()
    shortest_path_cost_a_star_optimized, nodes_explored_a_star_optimized = a_star_optimized(grid, start, end)
    end_time = time.perf_counter()
    print(f"A* Search Algorithm Optimized - Cost: {shortest_path_cost_a_star_optimized}, Nodes Explored: {nodes_explored_a_star_optimized}, Time: {end_time - start_time:.8f} seconds")



if __name__ == "__main__":
    main()