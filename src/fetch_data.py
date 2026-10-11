import os
from datetime import datetime
from meteostat import Point, Daily

# Dictionary of target cities, coordinates (Lat, Lon, Elevation in meters), and climate zone descriptions
TARGET_CITIES = {
    "islamabad": {
        "name": "Islamabad",
        "point": Point(33.7294, 73.0931, 540),
        "zone": "Humid Subtropical",
        "file_name": "islamabad_daily_temps.csv"
    },
    "lahore": {
        "name": "Lahore",
        "point": Point(31.5204, 74.3587, 217),
        "zone": "Semi-arid",
        "file_name": "lahore_daily_temps.csv"
    },
    "karachi": {
        "name": "Karachi",
        "point": Point(24.8607, 67.0011, 8),
        "zone": "Coastal Arid",
        "file_name": "karachi_daily_temps.csv"
    }
}

def fetch_city_data(city_key: str, start_date: datetime, end_date: datetime, data_dir: str, overwrite: bool = True):
    """
    Fetches historical daily climate data for a specific city using Meteostat and saves to CSV.
    """
    if city_key not in TARGET_CITIES:
        raise ValueError(f"Unknown city '{city_key}'. Choose from: {list(TARGET_CITIES.keys())}")

    city_info = TARGET_CITIES[city_key]
    csv_path = os.path.join(data_dir, city_info["file_name"])

    if not overwrite and os.path.exists(csv_path):
        print(f"[*] Skipping {city_info['name']} (already exists at: {csv_path})")
        return

    print(f"\n[+] Fetching climate data for {city_info['name']} ({city_info['zone']})...")
    data = Daily(city_info["point"], start_date, end_date)
    df = data.fetch()

    if df.empty:
        print(f"[!] Warning: No data returned for {city_info['name']}.")
        return

    # Drop rows where average temperature (tavg) is missing
    initial_count = len(df)
    df = df.dropna(subset=['tavg'])
    dropped = initial_count - len(df)

    # Save to CSV
    df.to_csv(csv_path)

    print(f"    Saved {len(df)} records to {csv_path} (dropped {dropped} rows with missing tavg)")
    print(f"    Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"    Temp Range: Min {df['tavg'].min():.1f}°C, Max {df['tavg'].max():.1f}°C, Mean {df['tavg'].mean():.1f}°C")


def fetch_all_data(start_year: int = 2000, end_year: int = 2023, overwrite: bool = False):
    """
    Fetches historical climate data for all target Pakistani cities.
    """
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)

    print(f"=== Fetching Historical Climate Data ({start_year}-{end_year}) ===")
    for city_key in TARGET_CITIES:
        fetch_city_data(city_key, start, end, data_dir, overwrite=overwrite)

    print("\n=== Data Acquisition Complete ===")


def fetch_islamabad_data():
    """
    Backwards-compatible convenience function for Islamabad.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    fetch_city_data("islamabad", datetime(2000, 1, 1), datetime(2023, 12, 31), data_dir, overwrite=True)


if __name__ == "__main__":
    # Fetch all cities; overwrite=False preserves existing islamabad data if desired,
    # or set overwrite=True to refresh all.
    fetch_all_data(start_year=2000, end_year=2023, overwrite=False)
