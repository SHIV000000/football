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
    /* Base styles */
    .stApp {
        background-color: #f0f2f6;
    }

    /* Login Form Styling */
    .login-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 400px;
        margin: 2rem auto;
    }

    /* Form Elements */
    .stTextInput > div > div {
        background-color: white !important;
    }

    .stTextInput input {
        color: #1a1a1a !important;
        background-color: white !important;
        font-size: 1rem !important;
    }

    .stTextInput > label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Button Styling - Global */
    .stButton > button {
        width: 100% !important;
        height: auto !important;
        padding: 0.75rem 1.5rem !important;
        background-color: #2c5282 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin: 0.5rem 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: #1a365d !important;
        transform: translateY(-1px) !important;
    }

    /* Login Form Submit Button */
    .stForm button[type="submit"] {
        width: 100% !important;
        padding: 0.75rem 1.5rem !important;
        background-color: #2c5282 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        cursor: pointer !important;
    }

    .stForm button[type="submit"]:hover {
        background-color: #1a365d !important;
    }

    /* Prediction Elements */
    .winner-prediction {
        background-color: #48bb78 !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .probability-container {
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 0.5rem !important;
        text-align: center !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .probability-label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        font-size: 1rem !important;
    }

    .probability-value {
        color: #2c5282 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    /* Match Card */
    .match-card {
        background-color: white !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .team-name {
        color: #1a1a1a !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 1rem 0 !important;
        text-align: center !important;
    }

    /* Date Input Styling */
    .stDateInput > label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Calendar styling for both light and dark modes */
    .stDateInput input {
        color: #1a1a1a !important;
        background-color: white !important;
        border: 1px solid #cccccc !important;
    }

    /* Calendar popup styling */
    .react-datepicker {
        background-color: white !important;
        border: 1px solid #cccccc !important;
    }

    .react-datepicker__header {
        background-color: #f0f2f6 !important;
    }

    .react-datepicker__day {
        color: #1a1a1a !important;
    }

    .react-datepicker__day:hover {
        background-color: #e6e6e6 !important;
    }

    .react-datepicker__day--selected {
        background-color: #2196f3 !important;
        color: white !important;
    }

    .react-datepicker__day--keyboard-selected {
        background-color: #2196f3 !important;
        color: white !important;
    }

    /* Progress Bar Container */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .progress-label {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.25rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .progress-fill-home {
        background-color: #48bb78;  /* Green */
    }

    .progress-fill-draw {
        background-color: #ed8936;  /* Orange */
    }

    .progress-fill-away {
        background-color: #3182ce;  /* Blue */
    }

    /* Progress Bar Styling */
    .stProgress > div > div > div {
        height: 8px;
        background-color: #e2e8f0;
    }
    
    /* Home Team Progress */
    .stProgress:nth-of-type(1) > div > div > div > div {
        background-color: #48bb78 !important;
    }
    
    /* Draw Progress */
    .stProgress:nth-of-type(2) > div > div > div > div {
        background-color: #ed8936 !important;
    }
    
    /* Away Team Progress */
    .stProgress:nth-of-type(3) > div > div > div > div {
        background-color: #3182ce !important;
    }
    
    /* Adjust spacing */
    .stProgress {
        margin-bottom: 0.5rem;
    }
    
    /* Team names and percentages */
    .element-container p {
        margin-bottom: 1rem;
        font-weight: 500;
    }

    /* Headers and Text */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Prediction Text */
    .prediction-text {
        color: #1a1a1a !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin: 0.5rem 0 !important;
    }

    /* Confidence Levels */
    .prediction-high, .prediction-medium, .prediction-low {
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .prediction-high {
        background-color: #48bb78 !important;  /* Green */
    }

    .prediction-medium {
        background-color: #ed8936 !important;  /* Orange */
    }

    .prediction-low {
        background-color: #e53e3e !important;  /* Red */
    }

    /* Section Headers */
    .section-header {
        color: #1a1a1a !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
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


# Load the predictor model
predictor = load_model()

def create_match_features_from_api(match_data, predictor):
    home_team = match_data['home_name']
    away_team = match_data['away_name']
    
    # Get team statistics
    home_ppg = float(match_data.get('home_ppg', 0))
    away_ppg = float(match_data.get('away_ppg', 0))
    home_overall_ppg = float(match_data.get('pre_match_teamA_overall_ppg', 0))
    away_overall_ppg = float(match_data.get('pre_match_teamB_overall_ppg', 0))
    
    # Calculate expected goals and normalize them
    home_xg = float(match_data.get('team_a_xg_prematch', 0))
    away_xg = float(match_data.get('team_b_xg_prematch', 0))
    total_xg = home_xg + away_xg
    if total_xg > 0:
        home_xg_norm = home_xg / total_xg
        away_xg_norm = away_xg / total_xg
    else:
        home_xg_norm = 0.5
        away_xg_norm = 0.5
    
    # Calculate relative strengths using both home/away and overall PPG
    home_strength = (home_ppg + home_overall_ppg) / 6  # Scale to 0-1
    away_strength = (away_ppg + away_overall_ppg) / 6
    
    # Convert odds to probabilities
    home_odds = float(match_data.get('odds_ft_1', 2.0))
    away_odds = float(match_data.get('odds_ft_2', 2.0))
    odds_prob_home = 1 / home_odds
    odds_prob_away = 1 / away_odds
    total_odds_prob = odds_prob_home + odds_prob_away
    if total_odds_prob > 0:
        odds_prob_home = odds_prob_home / total_odds_prob
        odds_prob_away = odds_prob_away / total_odds_prob
    
    feature_dict = {
        'Team 1': hash(home_team) % 1000,
        'Team 2': hash(away_team) % 1000,
        'DayOfWeek': datetime.fromtimestamp(match_data['date_unix']).weekday(),
        'Month': datetime.fromtimestamp(match_data['date_unix']).month,
        'Year': datetime.fromtimestamp(match_data['date_unix']).year,
        'league': match_data.get('competition_id', 0),
        'Team 1_GoalDifference_Last5': home_xg_norm - away_xg_norm,
        'Team 1_TotalGoals_Last5': home_xg,
        'Team 2_GoalDifference_Last5': away_xg_norm - home_xg_norm,
        'Team 2_TotalGoals_Last5': away_xg,
        'Team 1_Form': home_strength,
        'Team 2_Form': away_strength,
        'yellow_cards_home': float(match_data.get('cards_potential', 0)) / 2,
        'red_cards_home': 0,
        'goals_home': home_overall_ppg,
        'assists_home': home_xg,
        'yellow_cards_away': float(match_data.get('cards_potential', 0)) / 2,
        'red_cards_away': 0,
        'goals_away': away_overall_ppg,
        'assists_away': away_xg,
    }

    return pd.DataFrame([feature_dict])

def adjust_probabilities(home_prob, draw_prob, away_prob, match_data):
    """Adjust probabilities based on odds and team strengths"""
    # Get odds
    home_odds = float(match_data.get('odds_ft_1', 2.0))
    away_odds = float(match_data.get('odds_ft_2', 2.0))
    draw_odds = float(match_data.get('odds_ft_x', 3.0))
    
    # Convert odds to probabilities
    odds_home_prob = 1 / home_odds
    odds_away_prob = 1 / away_odds
    odds_draw_prob = 1 / draw_odds
    
    # Normalize odds probabilities
    total_odds_prob = odds_home_prob + odds_away_prob + odds_draw_prob
    odds_home_prob /= total_odds_prob
    odds_away_prob /= total_odds_prob
    odds_draw_prob /= total_odds_prob
    
    # Get team strengths
    home_ppg = float(match_data.get('home_ppg', 0))
    away_ppg = float(match_data.get('away_ppg', 0))
    home_overall_ppg = float(match_data.get('pre_match_teamA_overall_ppg', 0))
    away_overall_ppg = float(match_data.get('pre_match_teamB_overall_ppg', 0))
    
    # Calculate form-based probabilities
    total_ppg = home_overall_ppg + away_overall_ppg
    if total_ppg > 0:
        form_home_prob = home_overall_ppg / total_ppg
        form_away_prob = away_overall_ppg / total_ppg
    else:
        form_home_prob = 0.4
        form_away_prob = 0.4
    form_draw_prob = 1 - (form_home_prob + form_away_prob)
    
    # Weights for different factors
    model_weight = 0.4
    odds_weight = 0.4
    form_weight = 0.2
    
    # Calculate final probabilities
    final_home_prob = (home_prob * model_weight + 
                      odds_home_prob * odds_weight + 
                      form_home_prob * form_weight)
    
    final_away_prob = (away_prob * model_weight + 
                      odds_away_prob * odds_weight + 
                      form_away_prob * form_weight)
    
    final_draw_prob = (draw_prob * model_weight + 
                      odds_draw_prob * odds_weight + 
                      form_draw_prob * form_weight)
    
    # Normalize final probabilities
    total = final_home_prob + final_away_prob + final_draw_prob
    final_home_prob /= total
    final_away_prob /= total
    final_draw_prob /= total
    
    # Apply minimum probability thresholds
    min_prob = 0.1
    if final_away_prob < min_prob:
        final_away_prob = min_prob
        excess = (1 - min_prob) / 2
        final_home_prob = excess
        final_draw_prob = excess
    
    return final_home_prob, final_draw_prob, final_away_prob

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
    st.markdown('<h1 class="app-title">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="login-container">
                <h2 style="color: #1a1a1a; text-align: center; margin-bottom: 2rem;">Welcome Back!</h2>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username == "matchday_wizard" and password == "GoalMaster":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")

def get_matches_for_days(start_date, end_date):
    all_matches = []
    current_date = start_date
    while current_date <= end_date:
        matches = get_matches(current_date.strftime('%Y-%m-%d'))
        all_matches.extend(matches)
        current_date += timedelta(days=1)
    return all_matches

def calculate_match_prediction(match):
    """Calculate match prediction using multiple factors"""
    
    # Get basic odds
    home_odds = float(match.get('odds_ft_1', 0))
    draw_odds = float(match.get('odds_ft_x', 0))
    away_odds = float(match.get('odds_ft_2', 0))
    
    # Get team performance metrics
    home_ppg = float(match.get('home_ppg', 0))
    away_ppg = float(match.get('away_ppg', 0))
    home_overall_ppg = float(match.get('pre_match_teamA_overall_ppg', 0))
    away_overall_ppg = float(match.get('pre_match_teamB_overall_ppg', 0))
    
    # Get expected goals (xG)
    home_xg = float(match.get('team_a_xg_prematch', 0))
    away_xg = float(match.get('team_b_xg_prematch', 0))
    
    # Calculate probabilities from odds (if available)
    if all([home_odds, draw_odds, away_odds]):
        # Convert odds to probabilities
        total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
        home_prob_odds = (1/home_odds) / total_prob
        draw_prob_odds = (1/draw_odds) / total_prob
        away_prob_odds = (1/away_odds) / total_prob
    else:
        home_prob_odds = 0.33
        draw_prob_odds = 0.33
        away_prob_odds = 0.33
    
    # Calculate form-based probabilities
    total_ppg = home_ppg + away_ppg
    if total_ppg > 0:
        home_prob_form = home_ppg / total_ppg
        away_prob_form = away_ppg / total_ppg
    else:
        home_prob_form = 0.5
        away_prob_form = 0.5
    
    # Calculate xG-based probabilities
    total_xg = home_xg + away_xg
    if total_xg > 0:
        home_prob_xg = home_xg / total_xg
        away_prob_xg = away_xg / total_xg
    else:
        home_prob_xg = 0.5
        away_prob_xg = 0.5
    
    # Weighted combination of all factors
    # Odds are given highest weight as they incorporate market knowledge
    home_final = (home_prob_odds * 0.5) + (home_prob_form * 0.25) + (home_prob_xg * 0.25)
    away_final = (away_prob_odds * 0.5) + (away_prob_form * 0.25) + (away_prob_xg * 0.25)
    
    # Calculate draw probability (based on odds and typical draw frequency)
    draw_final = draw_prob_odds * 0.6  # Reduce draw probability slightly as they're less common
    
    # Normalize probabilities to sum to 1
    total = home_final + away_final + draw_final
    home_final /= total
    away_final /= total
    draw_final /= total
    
    return home_final, draw_final, away_final

def display_prediction(prediction, confidence):
    """Display prediction with appropriate color based on confidence level"""
    if confidence >= 0.6:
        sentiment_class = "prediction-high"
        confidence_text = "High Confidence"
    elif confidence >= 0.4:
        sentiment_class = "prediction-medium"
        confidence_text = "Medium Confidence"
    else:
        sentiment_class = "prediction-low"
        confidence_text = "Low Confidence"
        
    st.markdown(f"""
        <div class="{sentiment_class}">
            {prediction}
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                {confidence_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_probability_bars(home_prob, draw_prob, away_prob, home_team, away_team):
    """Display probability bars for match outcomes"""
    st.markdown("### Match Outcome Probabilities")
    
    # Create a container with white background
    with st.container():
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin: 1rem 0;">
            </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for the probabilities
        col1, col2, col3 = st.columns(3)
        
        # Home team probability
        with col1:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="color: #48bb78; font-weight: 600; font-size: 1.2rem;">{home_prob:.1%}</span>
                    </div>
                    <div style="color: #1a1a1a; font-weight: 500;">{home_team}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Draw probability
        with col2:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="color: #ed8936; font-weight: 600; font-size: 1.2rem;">{draw_prob:.1%}</span>
                    </div>
                    <div style="color: #1a1a1a; font-weight: 500;">Draw</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Away team probability
        with col3:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="color: #3182ce; font-weight: 600; font-size: 1.2rem;">{away_prob:.1%}</span>
                    </div>
                    <div style="color: #1a1a1a; font-weight: 500;">{away_team}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Add the combined progress bar
        st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0 1rem;">
                <div style="width: 100%; height: 20px; background: #e2e8f0; border-radius: 10px; overflow: hidden; display: flex;">
                    <div style="width: {home_prob * 100}%; height: 100%; background-color: #48bb78;"></div>
                    <div style="width: {draw_prob * 100}%; height: 100%; background-color: #ed8936;"></div>
                    <div style="width: {away_prob * 100}%; height: 100%; background-color: #3182ce;"></div>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-top: 1rem; padding: 0 1rem;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #48bb78; border-radius: 2px; margin-right: 5px;"></div>
                    <span style="font-size: 0.8rem;">Home Win</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #ed8936; border-radius: 2px; margin-right: 5px;"></div>
                    <span style="font-size: 0.8rem;">Draw</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background: #3182ce; border-radius: 2px; margin-right: 5px;"></div>
                    <span style="font-size: 0.8rem;">Away Win</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

def display_match_odds(match_data):
    """Display FootyStats match odds in an organized box."""
    # Display match stats
    stats_html = f"""
    <div style="background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 1px solid #e5e7eb; margin: 20px 0;">
        <h3 style="text-align: center; color: #1f2937; margin-bottom: 20px;">Match Stats</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e5e7eb;">
                <div style="color: #4b5563; font-size: 0.9rem; margin-bottom: 5px;">PPG (Home)</div>
                <div style="color: #1f2937; font-size: 1.2rem; font-weight: 600;">{match_data.get('home_ppg', 'N/A')}</div>
            </div>
            <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e5e7eb;">
                <div style="color: #4b5563; font-size: 0.9rem; margin-bottom: 5px;">xG (Pre-match)</div>
                <div style="color: #1f2937; font-size: 1.2rem; font-weight: 600;">{match_data.get('total_xg_prematch', 'N/A')}</div>
            </div>
            <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e5e7eb;">
                <div style="color: #4b5563; font-size: 0.9rem; margin-bottom: 5px;">PPG (Away)</div>
                <div style="color: #1f2937; font-size: 1.2rem; font-weight: 600;">{match_data.get('away_ppg', 'N/A')}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

    # Display odds title
    st.markdown('<h3 style="text-align: center; color: #1f2937; margin: 30px 0; font-size: 1.5rem;">FootyStats Match Odds</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px;">Match Result & Goals</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><span style="color: #4b5563; font-weight: 500;">Over 2.5:</span> {match_data.get("odds_ft_over25", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><span style="color: #4b5563; font-weight: 500;">BTTS Yes:</span> {match_data.get("odds_btts_yes", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><span style="color: #4b5563; font-weight: 500;">Double Chance 1X:</span> {match_data.get("odds_doublechance_1x", "N/A")}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px;">First Half Markets</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><span style="color: #4b5563; font-weight: 500;">1H Over 0.5:</span> {match_data.get("odds_1st_half_over05", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><span style="color: #4b5563; font-weight: 500;">1H Result 1:</span> {match_data.get("odds_1st_half_result_1", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><span style="color: #4b5563; font-weight: 500;">Team Score First:</span> {match_data.get("odds_team_to_score_first_1", "N/A")}</div>', unsafe_allow_html=True)

    # Additional odds in expandable sections
    with st.expander("📊 More Betting Markets"):
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('<div style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px;">Match Result</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Home Win:</span> {match_data.get("odds_ft_1", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Draw:</span> {match_data.get("odds_ft_x", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Away Win:</span> {match_data.get("odds_ft_2", "N/A")}</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin: 20px 0 15px;">Win to Nil</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Home Win to Nil:</span> {match_data.get("odds_win_to_nil_1", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Away Win to Nil:</span> {match_data.get("odds_win_to_nil_2", "N/A")}</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px;">Over/Under Goals</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Over 1.5:</span> {match_data.get("odds_ft_over15", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Over 3.5:</span> {match_data.get("odds_ft_over35", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">Under 2.5:</span> {match_data.get("odds_ft_under25", "N/A")}</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin: 20px 0 15px;">Double Chance</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">1X:</span> {match_data.get("odds_doublechance_1x", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">12:</span> {match_data.get("odds_doublechance_12", "N/A")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #374151; font-size: 1.1rem; background-color: #f3f4f6; padding: 12px 15px; border-radius: 8px; margin: 5px 0; border: 1px solid #e5e7eb;"><span style="color: #4b5563; font-weight: 500;">X2:</span> {match_data.get("odds_doublechance_x2", "N/A")}</div>', unsafe_allow_html=True)

def add_back_to_top_button():
    """Add a floating back to top button."""
    st.markdown("""
        <style>
            .back-to-top {
                position: fixed;
                bottom: 30px;
                right: 30px;
                background-color: #1f2937;
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                text-decoration: none;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
                z-index: 999;
                transition: background-color 0.3s;
                cursor: pointer;
                font-weight: 500;
            }
            .back-to-top:hover {
                background-color: #374151;
            }
            .prediction-separator {
                border: none;
                height: 2px;
                background: linear-gradient(to right, transparent, #e5e7eb, transparent);
                margin: 30px 0;
            }
        </style>
        
        <a href="#top" class="back-to-top">↑ Back to Top</a>
        
        <script>
            // Smooth scroll to top
            document.querySelector('.back-to-top').addEventListener('click', function(e) {
                e.preventDefault();
                window.scrollTo({top: 0, behavior: 'smooth'});
            });
            
            // Show/hide button based on scroll position
            window.onscroll = function() {
                var button = document.querySelector('.back-to-top');
                if (document.body.scrollTop > 500 || document.documentElement.scrollTop > 500) {
                    button.style.display = "block";
                } else {
                    button.style.display = "none";
                }
            };
        </script>
    """, unsafe_allow_html=True)

def show_main_app():
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)  # Add anchor for back to top
    st.markdown("<h1>⚽ Football Match Predictor ⚽</h1>", unsafe_allow_html=True)
    
    # Add the back to top button
    add_back_to_top_button()
    
    # Logout button at the top
    col1, col2, col3 = st.columns([3,2,3])
    with col2:
        if st.button("Logout", key="logout", help="Click to logout"):
            logout()
            st.rerun()

    # Date selection
    st.markdown("<h3>Select Match Date</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p style="color: #1a1a1a; font-weight: 600; font-size: 1rem;">From Date</p>', unsafe_allow_html=True)
        start_date = st.date_input("", min_value=datetime.now().date(), value=datetime.now().date())
    with col2:
        st.markdown('<p style="color: #1a1a1a; font-weight: 600; font-size: 1rem;">To Date</p>', unsafe_allow_html=True)
        end_date = st.date_input(" ", min_value=start_date, value=start_date + timedelta(days=7))
    
    if start_date and end_date:
        if start_date <= end_date:
            with st.spinner("Fetching matches..."):
                matches = get_matches_for_days(start_date, end_date)
            
            if matches:
                st.markdown('<div class="section-header">Match Predictions</div>', unsafe_allow_html=True)
                
                for match in matches:
                    with st.container():
                        home_team = match.get('home_name', 'Unknown')
                        away_team = match.get('away_name', 'Unknown')
                        
                        st.markdown(f"""
                        <div class="match-card">
                            <div class="team-name">{home_team} vs {away_team}</div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            features = create_match_features_from_api(match, predictor)
                            prediction, probabilities = predictor.predict(features)
                            
                            # Get probabilities
                            home_prob, draw_prob, away_prob = calculate_match_prediction(match)
                            
                            # Determine winner
                            winner = ""
                            max_prob = max(home_prob, draw_prob, away_prob)
                            if max_prob == home_prob:
                                winner = f"{home_team} (Home)"
                                winner_prob = home_prob
                            elif max_prob == away_prob:
                                winner = f"{away_team} (Away)"
                                winner_prob = away_prob
                            else:
                                winner = "Draw"
                                winner_prob = draw_prob
                            
                            # Display winner prediction with confidence level
                            st.markdown(f'<div class="prediction-text">Most Likely Outcome: {winner}</div>', unsafe_allow_html=True)
                            display_prediction(winner, winner_prob)
                            
                            # Display probability bars header
                            st.markdown('<div class="prediction-text">Match Outcome Probabilities</div>', unsafe_allow_html=True)
                            
                            # Display probability bars
                            display_probability_bars(home_prob, draw_prob, away_prob, home_team, away_team)
                            
                            # Display match odds
                            st.markdown("---")
                            display_match_odds(match)
                        
                        except Exception as e:
                            st.error(f"Error processing match: {str(e)}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown('<div class="prediction-separator"></div>', unsafe_allow_html=True)
            else:
                st.warning("No matches found for the selected dates. Please try a different date range.")
        else:
            st.error("End date must be after start date")

# Main app logic
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()
