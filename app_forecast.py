import requests
from datetime import datetime, timedelta
import folium
from folium import plugins
from folium.plugins import HeatMap
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from scipy.spatial import cKDTree
from tqdm import tqdm
from branca.colormap import linear
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics as stats
from dotenv import load_dotenv
import os
#import cupy as cp


def get_forecasted_air_quality(lat, lon, api_key, forecast_time):
    url = f"https://airquality.googleapis.com/v1/forecast:lookup?key={api_key}"
    headers = {
        'Content-Type': 'application/json',
    }
    body = {
        "location": {
            "latitude": lat,
            "longitude": lon
        },
        "period": {
            "startTime": forecast_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endTime": (forecast_time + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        },
        "extraComputations": ["POLLUTANT_CONCENTRATION"],
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        data = response.json()
        hours_info = data.get('hourlyForecasts', [])
        no2_values = []

        for hour in hours_info:
            pollutants = hour.get('pollutants', [])
            for pollutant in pollutants:
                if pollutant['code'].lower() == 'no2':
                    no2 = pollutant.get('concentration', {}).get('value')
                    if no2 is not None:
                        no2_values.append(no2)
        
        avg_no2 = sum(no2_values) / len(no2_values) if no2_values else None
        return avg_no2
    else:
        print(f"Failed to retrieve data: {response.status_code}, {response.text}")
        return None

def get_forecasted_air_quality_parallel(grid_points, api_key, forecast_time):
    results = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        future_to_point = {executor.submit(get_forecasted_air_quality, point[0], point[1], api_key, forecast_time): point for point in grid_points}
        for future in tqdm(as_completed(future_to_point), total=len(grid_points)):
            point = future_to_point[future]
            try:
                data = future.result()
                results.append((point, data))
            except Exception as e:
                print(f"Exception for {point}: {e}")
                results.append((point, None))
    return results

def generate_heatmap(centroids, pollution_levels, route_coordinates=None, map_object=None):
    if map_object is None:
        m = folium.Map(location=[centroids[0][0], centroids[0][1]], zoom_start=13)
    else:
        m = map_object

    heat_data = [[lat, lon, pollution] for (lat, lon), pollution in zip(centroids, pollution_levels)]
    HeatMap(heat_data).add_to(m)

    if route_coordinates:
        # Draw polyline
        folium.PolyLine(locations=route_coordinates, color='blue', weight=5).add_to(m)

        # Add arrows using AntPath plugin
        ant_path = plugins.AntPath(locations=route_coordinates, color='blue', weight=5, delay=800)
        ant_path.add_to(m)
    # Add color legend
    colormap = linear.YlOrRd_09.scale(min(pollution_levels), max(pollution_levels))
    colormap.caption = 'NO2 Concentration'
    colormap.add_to(m)

    return m

def create_grid(start_lat, start_lon, radius_km, resolution_m=resolution):
    lat_lon_step = resolution_m / 111000  # 1 degree latitude/longitude is approximately 111km
    num_steps = int(radius_km * 1000 / resolution_m)
    
    grid_points = []
    for i in range(-num_steps, num_steps + 1):
        for j in range(-num_steps, num_steps + 1):
            grid_lat = start_lat + i * lat_lon_step
            grid_lon = start_lon + j * lat_lon_step
            if geodesic((start_lat, start_lon), (grid_lat, grid_lon)).km <= radius_km:
                grid_points.append((grid_lat, grid_lon))
    
    return grid_points

def assign_pollution_to_nodes(G, grid_points, pollution_values):
    tree = cKDTree(grid_points)
    pollution_data = {}
    for node in G.nodes:
        node_lat, node_lon = G.nodes[node]['y'], G.nodes[node]['x']
        _, idx = tree.query([node_lat, node_lon])
        pollution_data[node] = pollution_values[idx]
    return pollution_data

def calculate_pollution_for_route(G, route, pollution_data):
    total_pollution = stats.mean(pollution_data[node] for node in route if node in pollution_data)
    return total_pollution


# Function to find the optimal route using parallel processing
def find_optimal_route_parallel(G, pollution_data, start_node, start_lat, start_lon, distance_goal_km, exploration_weight=0.5):
    min_score = float('-inf')  # Initialize with negative infinity to find the maximum score
    optimal_route = None

    # List of potential end nodes to consider for route calculation
    end_nodes = list(G.nodes)

    # Function to calculate route pollution and return route, pollution level, and score
    def calculate_route_pollution(end_node):
        try:
            route = nx.shortest_path(G, start_node, end_node, weight='length')
            route_distance = sum(geodesic((G.nodes[route[i]]['y'], G.nodes[route[i]]['x']),
                                          (G.nodes[route[i + 1]]['y'], G.nodes[route[i + 1]]['x'])).km
                                 for i in range(len(route) - 1))*1000
            #route_distance = nx.shortest_path_length(G, start_node, end_node, weight='length')
            if 0.75 * distance_goal_km*1000 <= route_distance <= 1.25 * distance_goal_km*1000:  # Within 10% of the target distance
                visited_nodes = set()
                visited_edges = set()
                #print("Route Distance:", route_distance)
                exploration_reward = calculate_route_reward(route, G, visited_nodes, visited_edges)
                route_pollution = calculate_pollution_for_route(G, route, pollution_data)
                route_score = exploration_reward * exploration_weight - abs(route_distance - distance_goal_km*1000) 
                return route, route_score, route_pollution, route_distance
            else:
                #print("Distance not matched")
                return None, float('-inf'), float('inf'), float('inf')
        except nx.NetworkXNoPath:
            return None, float('-inf'), float('inf'), float('inf')
    max_workers = os.cpu_count()
    print(max_workers)
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(calculate_route_pollution, end_node) for end_node in end_nodes]
        
        # Collect results
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            route, route_score, route_pollution, route_distance = future.result()
            if route:
                results.append((route, route_score, route_pollution, route_distance))
        
        # Sort routes by pollution levels (ascending)
        results.sort(key=lambda x: x[2])
        
        # Select top 10 routes with minimum pollution
        top_10_routes = results[:10]
        
        print("TOP TEN ROUTES:")
        for i, result in enumerate(top_10_routes, start=1):
            route, route_score, route_pollution, route_distance = result
            print(f"Rank {i}:")
            print(f"  Score: {route_score}")
            print(f"  Pollution: {route_pollution}")
            print(f"  Distance: {route_distance}\n")
        # Select the route with the highest score among the top 10
        for route, route_score, route_pollution, route_distance in top_10_routes:
            if route_score > min_score:
                min_score = route_score
                optimal_route = route
                print("Min score updated:", min_score)
                #print(route)

    if optimal_route:
        # Extract the pollution and distance information for the final selected route
        for route, route_score, route_pollution, route_distance in top_10_routes:
            if route == optimal_route:
                print("Optimal Route:", optimal_route, "Pollution:", route_pollution, "Distance:", route_distance, "Score:", route_score)
                break
    else:
        print("No optimal route found.")

    return optimal_route


# Function to find the optimal route using parallel processing on GPU
def find_optimal_route_parallel_gpu(G, pollution_data, start_node, start_lat, start_lon, distance_goal_km, exploration_weight=1.5):
    min_score = float('-inf')  # Initialize with negative infinity to find the maximum score
    optimal_route = None

    # List of potential end nodes to consider for route calculation
    end_nodes = list(G.nodes)

    # Function to calculate route pollution and return route, pollution level, and score
    def calculate_route_pollution(end_node):
        try:
            route = nx.shortest_path(G, start_node, end_node, weight='length')
            route_distance = sum(geodesic((G.nodes[route[i]]['y'], G.nodes[route[i]]['x']),
                                          (G.nodes[route[i + 1]]['y'], G.nodes[route[i + 1]]['x'])).km
                                 for i in range(len(route) - 1))
            
            if route_distance <= 1.25 * distance_goal_km:  # Within 25% of the target distance
                visited_nodes = set()
                visited_edges = set()
                exploration_reward = calculate_route_reward(route, G, visited_nodes, visited_edges)
                route_pollution = calculate_pollution_for_route(G, route, pollution_data)
                route_score = exploration_reward - exploration_weight * route_pollution - abs(route_distance - distance_goal_km)
                return route, route_score, route_pollution, route_distance
            else:
                return None, float('-inf'), float('inf'), float('inf')
        except nx.NetworkXNoPath:
            return None, float('-inf'), float('inf'), float('inf')

    # Use CuPy for parallel execution on GPU
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(calculate_route_pollution, end_node) for end_node in end_nodes]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            route, route_score, route_pollution, route_distance = future.result()
            if route:
                results.append((route, route_score, route_pollution, route_distance))
        
    # Move results to GPU
    results = cp.array(results)
    
    # Sort routes by pollution levels (ascending)
    results = results[cp.argsort(results[:, 2])]
    
    # Select top 10 routes with minimum pollution
    top_10_routes = results[:10]
    
    # Select the route with the highest score among the top 10
    for route, route_score, route_pollution, route_distance in top_10_routes:
        if route_score > min_score:
            min_score = route_score
            optimal_route = route
            print("Min score updated:", min_score)
            print(route)

    if optimal_route:
        # Extract the pollution and distance information for the final selected route
        for route, route_score, route_pollution, route_distance in top_10_routes:
            if route == optimal_route:
                print("Optimal Route:", optimal_route, "Pollution:", route_pollution, "Distance:", route_distance, "Score:", route_score)
                break
    else:
        print("No optimal route found.")

    return optimal_route



# Function to calculate reward for exploring new roads
def calculate_route_reward(route, G, visited_nodes=None, visited_edges=None):
    reward = 0
    if visited_nodes is None:
        visited_nodes = set()
    if visited_edges is None:
        visited_edges = set()

    for i in range(len(route)):
        node = route[i]
        if node not in visited_nodes:
            reward += 1  # Reward visiting new nodes
            visited_nodes.add(node)
        if i < len(route) - 1:
            edge = (route[i], route[i+1])
            if edge not in visited_edges:
                reward += 1  # Reward traveling on new edges
                visited_edges.add(edge)
    #print('Route:', route)
    #print('Reward Value For Route:', reward)
    return reward

def forecast_optimal_route(start_lat, start_lon, forecast_time, distance_goal_km, api_key):
    radius_km = distance_goal_km / 2  # Approximate radius to search for routes
    G = ox.graph_from_point((start_lat, start_lon), dist=radius_km*1000, network_type='all')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    #Set Resolution - Lower Resolution Will = More API Calls. Set in increments of 500m is recommended. Best results are <5km = 500 and 6-10km = 1000
    resolution = 1000

    # Create grid points
    grid_points = create_grid(start_lat, start_lon, radius_km, resolution)
    
    # Get pollution forecast for grid points
    pollution_results = get_forecasted_air_quality_parallel(grid_points, api_key, forecast_time)
    grid_points, pollution_values = zip(*pollution_results)
    pollution_values = [value if value is not None else float('inf') for value in pollution_values]

    # Assign pollution data to nodes
    pollution_data = assign_pollution_to_nodes(G, grid_points, pollution_values)

    # Find the optimal route
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    optimal_route = find_optimal_route_parallel(G, pollution_data, start_node, start_lat, start_lon, distance_goal_km)
    
    if optimal_route:
        route_coordinates = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in optimal_route]
        optimal_route_map = generate_heatmap(route_coordinates, [pollution_data.get(node, 0) for node in optimal_route], route_coordinates)
        optimal_route_map.save(f"optimal_route_forecast_{distance_goal_km}_km.html")
        print(f"Optimal Route ({distance_goal_km} km) saved as low_optimal_route_forecast_{distance_goal_km}_km.html")
    else:
        print("No optimal route found")

# Load environment variables
load_dotenv()

# Example usage
if __name__ == "__main__":
    start_lat = 52.6185500
    start_lon = -1.1285710
    forecast_time = datetime.now() + timedelta(hours=1)
    distance_goal_km = 8  # Desired distance goal in km
    api_key = os.getenv("api_key")  # Replace with your actual API key

    forecast_optimal_route(start_lat, start_lon, forecast_time, distance_goal_km, api_key)
