# football_api.py

import requests
from datetime import datetime, timedelta
import json

API_KEY = '633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49'
BASE_URL = 'https://api.footystats.org/todays-matches'

def get_matches(date=None):
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    
    params = {
        'key': API_KEY,
        'date': date
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and data['data']:
            return data['data']
        else:
            print(f"No matches found for date: {date}")
            if 'message' in data:
                print(f"API message: {data['message']}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for date: {date}")
        print(f"Response content: {response.text}")
        return []



