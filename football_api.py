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
        # Print request URL for debugging
        request_url = f"{BASE_URL}?key={API_KEY}&date={date}"
        print(f"Making request to: {request_url}")
        
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        
        # Print raw response for debugging
        print(f"Response status code: {response.status_code}")
        print(f"Raw response: {response.text[:500]}...")  # Print first 500 chars
        
        data = response.json()
        
        if 'data' in data:
            matches = data['data']
            print(f"Number of matches found: {len(matches)}")
            
            # Print sample match data
            if matches:
                print("\nSample match data:")
                print(json.dumps(matches[0], indent=2))
            
            return matches
        else:
            print(f"No 'data' field in response. Response keys: {data.keys()}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print(f"Response content: {response.text}")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []

def get_team_stats(team_id):
    """Get detailed team statistics from FootyStats API"""
    team_stats_url = f'https://api.footystats.org/team-stats'
    
    params = {
        'key': API_KEY,
        'team_id': team_id
    }
    
    try:
        print(f"\nFetching team stats for team_id: {team_id}")
        response = requests.get(team_stats_url, params=params)
        response.raise_for_status()
        
        # Print response for debugging
        print(f"Team stats response status: {response.status_code}")
        data = response.json()
        
        if 'data' in data:
            print("Team stats found")
            return data['data']
        else:
            print(f"No team stats data found. Response keys: {data.keys()}")
            return {}
            
    except Exception as e:
        print(f"Error fetching team stats: {str(e)}")
        return {}
