import requests
from datetime import datetime, timedelta
import time
import os

def download_inca_data(
    start_date: datetime,
    end_date: datetime,
    output_folder: str,
    bbox: str = '45.77222010581118, 8.098133748352293, 49.478175684609575, 17.742270413233744',
    requests_this_hour: int = 0
):
    """
    Download INCA historical weather data in NetCDF format for a given date range and bounding box.
    Saves files in the specified output folder, with each file named by the start date of the data it contains.

    Args:
        start_date (datetime): The start date (inclusive) for data download.
        end_date (datetime): The end date (exclusive) for data download.
        output_folder (str): Directory where downloaded files will be saved.
        bbox (str, optional): Bounding box as 'min_lat, min_lon, max_lat, max_lon'. Defaults to Austria.
        requests_this_hour (int, optional): Number of requests already made this hour. Defaults to 0.

    The function respects the API rate limits (5 requests/sec, 240/hour) and will pause as needed.
    Each day's data is saved as a separate NetCDF file in the output folder.
    It does not check for the max api request size, so ensure the API can handle the requests.
    """
    url_head = "https://dataset.api.hub.geosphere.at/v1/grid/historical/inca-v1-1h-1km"
    os.makedirs(output_folder, exist_ok=True)
    hour_start_time = time.time()
    current = start_date

    while current < end_date:
        next_day = current + timedelta(days=1)
        filename = f'inca_data_{current.strftime("%Y%m%dT%H%M")}.nc'
        filepath = os.path.join(output_folder, filename)
        params_dict = {
            'parameters': ['GL', 'P0', 'RH2M', 'RR', 'T2M', 'TD2M', 'UU', 'VV'],
            'start': current.strftime('%Y-%m-%dT00:00'),
            'end': current.strftime('%Y-%m-%dT00:00'),
            'bbox': bbox,
            'output_format': 'netcdf',
            'filename': filename,
        }

        # Rate limit: 5 requests per second, 240 per hour
        if requests_this_hour >= 240:
            elapsed = time.time() - hour_start_time
            wait_time = max(0, 3600 - elapsed)
            print(f"Hourly rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            requests_this_hour = 0
            hour_start_time = time.time()

        response = requests.get(url_head, params=params_dict)
        print(response.url)
        print(f"Status code: {response.status_code}")

        # Print rate limit info from headers if available
        for header in [
            'x-ratelimit-limit-hour', 'x-ratelimit-limit-second',
            'x-ratelimit-remaining-second', 'x-ratelimit-remaining-hour',
            'ratelimit-reset'
        ]:
            if header in response.headers:
                print(f"{header}: {response.headers[header]}")

        if response.status_code == 429:
            reset = int(response.headers.get('ratelimit-reset', 10))
            print(f"Rate limit exceeded. Waiting {reset} seconds before retrying...")
            time.sleep(reset)
            continue

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filepath}")
        else:
            print(f"Request failed for {params_dict['start']} with status code: {response.status_code}")

        requests_this_hour += 1
        current = next_day
        time.sleep(0.21)  # Ensure no more than 5 requests per second

#Example usage if run as a script:
if __name__ == "__main__":
    start_date = datetime(2024, 9, 2)
    end_date = datetime(2025, 1, 1)
    output_folder = r"path\to\output\folder"  # Change to your desired output folder
    download_inca_data(start_date, end_date, output_folder)
