# app.py


import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer


# Get the absolute path of the current file (app.py)
current_file_path = os.path.abspath(__file__)

# Get the directory containing app.py
current_dir = os.path.dirname(current_file_path)

# Get the parent directory (project root)
project_root = current_dir

# Add the src directory to the Python path
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from predictor import FootballPredictor
from football_api import get_matches
from data_loader import load_data, load_additional_data
from data_preprocessor import preprocess_data

# Set up paths
model_chunks_dir = os.path.join(project_root, 'savedmodel', 'model_chunks')
data_dir = os.path.join(project_root, 'data')
additional_data_dir = os.path.join(project_root, 'sd')


# Custom CSS
st.markdown("""
<style>
    body {
        color: #ffffff;
        background-color: #4a4a4a;
    }
    .stApp {
        background-image: linear-gradient(to bottom, #1e3c72 0%, #1e3c72 1%, #2a5298 100%);
    }
    .main {
        padding: 2rem;
    }
    h1 {
        color: #ffd700;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 4px #000000;
    }
    h3 {
        color: #ffffff;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 0.5rem;
        border-radius: 5px;
    }
    .match-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction {
        font-size: 1.5rem;
        color: #ffd700;
        text-align: center;
        margin: 1rem 0;
    }
    .probabilities {
        display: flex;
        justify-content: space-between;
    }
    .probability {
        text-align: center;
        padding: 0.5rem;
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
        flex: 1;
        margin: 0 0.5rem;
    }
    .match-details {
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #cccccc;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .login-header {
        color: #ffd700;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #ffd700;
        color: #1e3c72;
        font-weight: bold;
    }
        .prediction-box, .odds-box, .goal-probs-box, .team-stats-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .prediction-box h4, .odds-box h4, .goal-probs-box h4, .team-stats-box h4 {
        color: #ffd700;
        margin-bottom: 0.5rem;
    }

    .predicted-result {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        color: #ffd700;
        margin-bottom: 1rem;
    }
        .probabilities, .odds, .goal-probs, .team-stats {
        display: flex;
        justify-content: space-between;
    }

    .probability, .odd-item, .goal-prob-item, .team-stat {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 5px;
        margin: 0 0.25rem;
    }

    .team-name, .prob-name {
        font-weight: bold;
        margin-bottom: 0.25rem;
    }

    .prob-value, .odd-value {
        font-size: 1.2rem;
        color: #ffd700;
    }

    .team-stats-box .team-stat {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 1rem;
    }

    .team-stats-box h5 {
        color: #ffd700;
        margin-bottom: 0.5rem;
    }

    .team-stats-box p {
        margin: 0.25rem 0;
    }

    .match-details {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 5px;
        padding: 0.5rem;
        margin-top: 1rem;
    }
</style>

""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    try:
        # Update the path to point to the chunks directory
        chunks_dir = os.path.join(project_root, 'savedmodel', 'model_chunks')
        predictor = FootballPredictor.load_split_model(chunks_dir)
        print("Loaded feature names:", predictor.feature_names)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Detailed error: {e}")
        return None


# Load data
@st.cache_data
def load_cached_data():
    df = load_data(data_dir)
    additional_df = load_additional_data(additional_data_dir)
    return df, additional_df

df, additional_df = load_cached_data()

# Load the predictor model
predictor = load_model()

def create_match_features(home_team, away_team, predictor, df, additional_df):
    def encode_team(team):
        try:
            return predictor.label_encoder.transform([team])[0]
        except ValueError:
            return len(predictor.label_encoder.classes_)

    home_team_encoded = encode_team(home_team)
    away_team_encoded = encode_team(away_team)

    # Get recent matches
    home_recent = df[(df['Team 1'] == home_team) | (df['Team 2'] == home_team)].tail(10).copy()
    away_recent = df[(df['Team 1'] == away_team) | (df['Team 2'] == away_team)].tail(10).copy()

    # Calculate form and goals
    home_form = calculate_form(home_recent, home_team)
    away_form = calculate_form(away_recent, away_team)
    home_goals = calculate_goals(home_recent, home_team)
    away_goals = calculate_goals(away_recent, away_team)

    # Get league encoding
    league_encoding = get_league_encoding(df, home_team, away_team, predictor.label_encoder)

    # Get team stats
    home_stats = get_team_stats(additional_df, home_team)
    away_stats = get_team_stats(additional_df, away_team)

    feature_dict = {
        'Team 1': home_team_encoded,
        'Team 2': away_team_encoded,
        'DayOfWeek': datetime.now().weekday(),
        'Month': datetime.now().month,
        'Year': datetime.now().year,
        'league': league_encoding,
        'Team 1_GoalDifference_Last5': home_goals['diff'],
        'Team 1_TotalGoals_Last5': home_goals['total'],
        'Team 2_GoalDifference_Last5': away_goals['diff'],
        'Team 2_TotalGoals_Last5': away_goals['total'],
        'Team 1_Form': home_form,
        'Team 2_Form': away_form,
        'yellow_cards_home': home_stats['yellow_cards'],
        'red_cards_home': home_stats['red_cards'],
        'goals_home': home_stats['goals'],
        'assists_home': home_stats['assists'],
        'yellow_cards_away': away_stats['yellow_cards'],
        'red_cards_away': away_stats['red_cards'],
        'goals_away': away_stats['goals'],
        'assists_away': away_stats['assists'],
    }

    # Create DataFrame using the columns expected by the model
    match_features = pd.DataFrame([feature_dict])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    match_features = pd.DataFrame(imputer.fit_transform(match_features), columns=match_features.columns)

    return match_features


def calculate_form(recent_matches, team):
    if recent_matches.empty:
        return 0
    
    form = 0
    max_form = len(recent_matches) * 3
    for _, match in recent_matches.iterrows():
        if match['Team 1'] == team:
            result = match['FT'].split('-')
            form += 3 if int(result[0]) > int(result[1]) else (1 if result[0] == result[1] else 0)
        else:
            result = match['FT'].split('-')
            form += 3 if int(result[1]) > int(result[0]) else (1 if result[1] == result[0] else 0)
    return form / max_form

def calculate_goals(recent_matches, team):
    if recent_matches.empty:
        return {'diff': 0, 'total': 0}
    
    goals_for = 0
    goals_against = 0
    for _, match in recent_matches.iterrows():
        result = match['FT'].split('-')
        if match['Team 1'] == team:
            goals_for += int(result[0])
            goals_against += int(result[1])
        else:
            goals_for += int(result[1])
            goals_against += int(result[0])
    
    num_matches = len(recent_matches)
    return {
        'diff': (goals_for - goals_against) / num_matches,
        'total': (goals_for + goals_against) / num_matches
    }

def get_league_encoding(df, home_team, away_team, le):
    recent_matches = df[(df['Team 1'] == home_team) | (df['Team 2'] == home_team) |
                        (df['Team 1'] == away_team) | (df['Team 2'] == away_team)]
    
    if recent_matches.empty:
        return 0
    
    recent_match = recent_matches.iloc[-1]
    try:
        return le.transform([recent_match['league']])[0]
    except ValueError:
        return 0

def get_team_stats(additional_df, team):
    team_data = additional_df[additional_df['player_name'].str.contains(team, case=False, na=False)]
    if team_data.empty:
        return {
            'yellow_cards': 0,
            'red_cards': 0,
            'goals': 0,
            'assists': 0,
        }
    return {
        'yellow_cards': team_data['yellow_cards'].mean(),
        'red_cards': team_data['red_cards'].mean(),
        'goals': team_data['goals'].mean(),
        'assists': team_data['assists'].mean(),
    }

# Hardcoded credentials
VALID_USERNAME = "matchday_wizard"
VALID_PASSWORD = "GoalMaster"

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False


if 'login_submitted' not in st.session_state:
    st.session_state.login_submitted = False

def login(username, password):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        st.session_state.logged_in = True
        return True
    return False


def logout():
    st.session_state.logged_in = False


def show_login_page():
    st.markdown("<h1>⚽ Football Match Predictor ⚽ </h1>", unsafe_allow_html=True)
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='login-header'>Login</h2>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if login(username, password):
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
    
    st.markdown("</div>", unsafe_allow_html=True)


def get_matches_for_days(start_date, end_date):
    all_matches = []
    current_date = start_date
    while current_date <= end_date:
        matches = get_matches(current_date.strftime('%Y-%m-%d'))
        all_matches.extend(matches)
        current_date += timedelta(days=1)
    return all_matches

# Main app
def show_main_app():
    st.markdown("<h1>⚽ Football Match Predictor 🏆</h1>", unsafe_allow_html=True)
    
    if st.button("Logout"):
        logout()
    
    if predictor is None:
        st.error("Unable to load the model. Please check the file paths and try again.")
    else:
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime.now())
        with col2:
            end_date = st.date_input("End date", datetime.now() + timedelta(days=1))

        if start_date > end_date:
            st.error("End date must be after start date")
            return

        # Fetch matches for the selected date range
        matches = get_matches_for_days(start_date, end_date)

        if matches:
            st.markdown(f"## Matches from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Group matches by date
            matches_by_date = {}
            for match in matches:
                match_date = datetime.fromtimestamp(match.get('date_unix', 0)).date()
                if match_date not in matches_by_date:
                    matches_by_date[match_date] = []
                matches_by_date[match_date].append(match)

            # Display matches grouped by date
            for date in sorted(matches_by_date.keys()):
                st.markdown(f"### {date.strftime('%A, %B %d, %Y')}")
                for match in matches_by_date[date]:
                    home_team = match['home_name']
                    away_team = match['away_name']
                    
                    st.markdown(f"""
                    <div class="match-card">
                        <h3>{home_team} vs {away_team}</h3>
                    """, unsafe_allow_html=True)
                    
                    # Create match features and make prediction
                    try:
                        match_features = create_match_features(home_team, away_team, predictor, df, additional_df)
                        predictions, probabilities = predictor.predict(match_features)
                        
                        # Display prediction
                        home_prob, draw_prob, away_prob = probabilities[0]
                        
                        # Determine the predicted result based on the highest probability
                        if home_prob > draw_prob and home_prob > away_prob:
                            predicted_result = home_team
                        elif away_prob > home_prob and away_prob > draw_prob:
                            predicted_result = away_team
                        else:
                            predicted_result = "Draw"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h4>Prediction</h4>
                            <div class="predicted-result">{predicted_result}</div>
                            <div class="probabilities">
                                <div class="probability">
                                    <div class="team-name">{home_team}</div>
                                    <div class="prob-value">{home_prob:.2f}</div>
                                </div>
                                <div class="probability">
                                    <div class="team-name">Draw</div>
                                    <div class="prob-value">{draw_prob:.2f}</div>
                                </div>
                                <div class="probability">
                                    <div class="team-name">{away_team}</div>
                                    <div class="prob-value">{away_prob:.2f}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display match odds
                        st.markdown(f"""
                        <div class="odds-box">
                            <h4>Match Odds</h4>
                            <div class="odds">
                                <div class="odd-item">
                                    <div class="team-name">{home_team} Win</div>
                                    <div class="odd-value">{match.get('odds_ft_1', 'N/A')}</div>
                                </div>
                                <div class="odd-item">
                                    <div class="team-name">Draw</div>
                                    <div class="odd-value">{match.get('odds_ft_x', 'N/A')}</div>
                                </div>
                                <div class="odd-item">
                                    <div class="team-name">{away_team} Win</div>
                                    <div class="odd-value">{match.get('odds_ft_2', 'N/A')}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display goal probabilities
                        st.markdown(f"""
                        <div class="goal-probs-box">
                            <h4>Goal Probabilities</h4>
                            <div class="goal-probs">
                                <div class="goal-prob-item">
                                    <div class="prob-name">Over 2.5 Goals</div>
                                    <div class="prob-value">{match.get('o25_potential', 'N/A')}%</div>
                                </div>
                                <div class="goal-prob-item">
                                    <div class="prob-name">Both Teams to Score</div>
                                    <div class="prob-value">{match.get('btts_potential', 'N/A')}%</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        print(f"Detailed error: {e}")

                    # Display additional match information
                    st.markdown(f"""
                    <div class="match-details">
                        <strong>Competition:</strong> {match.get('competition_id', 'Unknown')}<br>
                        <strong>Date:</strong> {datetime.fromtimestamp(match.get('date_unix', 0)).strftime('%Y-%m-%d %H:%M')}<br>
                        <strong>Stadium:</strong> {match.get('stadium_name', 'Unknown')}<br>
                        <strong>Status:</strong> {match.get('status', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display team statistics
                    st.markdown(f"""
                    <div class="team-stats-box">
                        <h4>Team Statistics</h4>
                        <div class="team-stats">
                            <div class="team-stat">
                                <h5>{home_team}</h5>
                                <p>Home PPG: {match.get('home_ppg', 'N/A')}</p>
                                <p>Overall PPG: {match.get('pre_match_teamA_overall_ppg', 'N/A')}</p>
                                <p>Predicted xG: {match.get('team_a_xg_prematch', 'N/A')}</p>
                            </div>
                            <div class="team-stat">
                                <h5>{away_team}</h5>
                                <p>Away PPG: {match.get('away_ppg', 'N/A')}</p>
                                <p>Overall PPG: {match.get('pre_match_teamB_overall_ppg', 'N/A')}</p>
                                <p>Predicted xG: {match.get('team_b_xg_prematch', 'N/A')}</p>
                            </div>
                        </div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        else:
            st.write(f"No matches found for {date}")



# Main app logic
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()
