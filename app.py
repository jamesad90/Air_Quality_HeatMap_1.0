import gpxpy
import requests
import folium
from folium.plugins import HeatMap
from datetime import datetime
from geopy.distance import great_circle
import numpy as np
from tqdm import tqdm  # Add tqdm for progress bar

def get_historical_air_quality(lat, lon, start_time, end_time, api_key):
    url = f"https://airquality.googleapis.com/v1/history:lookup?key={api_key}"
    headers = {
        'Content-Type': 'application/json',
    }
    body = {
        "location": {
            "latitude": lat,
            "longitude": lon
        },
        "period": {
            "startTime": start_time,
            "endTime": end_time
        },
        "extraComputations":  ["POLLUTANT_ADDITIONAL_INFO", "DOMINANT_POLLUTANT_CONCENTRATION", "POLLUTANT_CONCENTRATION"],
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        data = response.json()
        hours_info = data.get('hoursInfo', [])
        no2_values = []
        pm25_values = []
        for hour in hours_info:
            pollutants = hour.get('pollutants', [])
            for pollutant in pollutants:
                if pollutant['code'].lower() == 'no2':
                    no2 = pollutant.get('concentration', {}).get('value')
                    if no2 is not None:
                        no2_values.append(no2)
                elif pollutant['code'].lower() == 'pm25':
                    pm25 = pollutant.get('concentration', {}).get('value')
                    if pm25 is not None:
                        pm25_values.append(pm25)
        
        avg_no2 = sum(no2_values) / len(no2_values) if no2_values else None
        avg_pm25 = sum(pm25_values) / len(pm25_values) if pm25_values else None
        return avg_no2, avg_pm25
    else:
        print(f"Failed to retrieve data: {response.status_code}, {response.text}")
        return None, None

def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    coordinates = []
    timestamps = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coordinates.append((point.latitude, point.longitude))
                timestamps.append(point.time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    print(timestamps)
    return coordinates, timestamps

def compute_500m_centroids(coordinates):
    centroids = []
    current_box = []
    for coord in coordinates:
        if not current_box:
            current_box.append(coord)
            continue

        if great_circle(current_box[0], coord).meters > 250:
            centroid_lat = np.mean([point[0] for point in current_box])
            centroid_lon = np.mean([point[1] for point in current_box])
            centroids.append((centroid_lat, centroid_lon))
            current_box = [coord]
        else:
            current_box.append(coord)

    if current_box:
        centroid_lat = np.mean([point[0] for point in current_box])
        centroid_lon = np.mean([point[1] for point in current_box])
        centroids.append((centroid_lat, centroid_lon))

    return centroids

def calculate_average_exposure(coordinates, timestamps, api_key):
    centroids = compute_500m_centroids(coordinates)
    total_centroids = len(centroids)
    print(f"Total centroids to process: {total_centroids}")
    
    pollution_levels_no2 = []
    pollution_levels_pm25 = []

    for center in tqdm(centroids, desc="Processing centroids"):
        start_time = timestamps[0]  # Assuming timestamps are the same for all points in a cluster
        end_time = timestamps[-1]   # Use actual start and end times of the activity
        no2, pm25 = get_historical_air_quality(center[0], center[1], start_time, end_time, api_key)
        if no2 is not None:
            pollution_levels_no2.append(no2)
        if pm25 is not None:
            pollution_levels_pm25.append(pm25)

    average_no2 = sum(pollution_levels_no2) / len(pollution_levels_no2) if pollution_levels_no2 else 0
    average_pm25 = sum(pollution_levels_pm25) / len(pollution_levels_pm25) if pollution_levels_pm25 else 0
    return average_no2, average_pm25, centroids, pollution_levels_no2, pollution_levels_pm25

def generate_heatmap(centroids, pollution_levels):
    m = folium.Map(location=[centroids[0][0], centroids[0][1]], zoom_start=13)
    heat_data = [[lat, lon, pollution] for (lat, lon), pollution in zip(centroids, pollution_levels)]
    HeatMap(heat_data).add_to(m)
    return m

def plot_route_with_heatmap(coordinates, centroids, pollution_levels_no2, pollution_levels_pm25):
    # Create the base map
    m = folium.Map(location=[coordinates[0][0], coordinates[0][1]], zoom_start=13)
    
    # Plot the route
    folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(m)
    
    # Add NO2 heatmap
    heat_data_no2 = [[lat, lon, pollution] for (lat, lon), pollution in zip(centroids, pollution_levels_no2)]
    HeatMap(heat_data_no2, name="NO2 Heatmap", show=True).add_to(m)

    # Add PM2.5 heatmap
    heat_data_pm25 = [[lat, lon, pollution] for (lat, lon), pollution in zip(centroids, pollution_levels_pm25)]
    HeatMap(heat_data_pm25, name="PM2.5 Heatmap", show=False).add_to(m)
    
    # Add layer control to toggle between heatmaps
    folium.LayerControl().add_to(m)
    
    return m

def main(gpx_file_path, api_key):
    coordinates, timestamps = parse_gpx(gpx_file_path)
    average_no2, average_pm25, centroids, pollution_levels_no2, pollution_levels_pm25 = calculate_average_exposure(coordinates, timestamps, api_key)
    
    print(f"Average NO2 Exposure: {average_no2:.2f}")
    print(f"Average PM2.5 Exposure: {average_pm25:.2f}")

    # Generate route with heatmap
    route_map = plot_route_with_heatmap(coordinates, centroids, pollution_levels_no2, pollution_levels_pm25)
    route_map.save("Run_with_NO2_pollution_heatmap.html")
    print("Pollution Heatmap saved as .html file")
    # route_map = plot_route_with_heatmap(coordinates, centroids, pollution_levels_pm25)
    # route_map.save("Run_with_PM2.5_heatmap.html")
    # print("NO2 Heatmap saved as .html file")

from dotenv import load_dotenv
import os
load_dotenv()

# Example usage
if __name__ == "__main__":
    gpx_file_path = 'Afternoon_Run.gpx'
    api_key = os.getenv("api_key")
    main(gpx_file_path, api_key)
