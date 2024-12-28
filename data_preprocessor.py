# data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import SimpleImputer


def preprocess_data(df, additional_df):
    print(f"Columns in DataFrame: {df.columns.tolist()}")
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert date to datetime and extract useful features
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Team 1'] = le.fit_transform(df['Team 1'])
    df['Team 2'] = le.fit_transform(df['Team 2'])
    df['league'] = le.fit_transform(df['league'])
    
    # Create target variable
    def get_result(score):
        try:
            home, away = map(int, score.split('-'))
            return 'H' if home > away else 'A' if home < away else 'D'
        except:
            return None
    
    df['Result'] = df['FT'].apply(get_result)
    df = df.dropna(subset=['Result'])
    
    # Feature engineering
    df['GoalDifference'] = df['FT'].apply(lambda x: int(x.split('-')[0]) - int(x.split('-')[1]))
    df['TotalGoals'] = df['FT'].apply(lambda x: int(x.split('-')[0]) + int(x.split('-')[1]))
    
    # Calculate rolling averages and other stats
    for team_col in ['Team 1', 'Team 2']:
        for stat in ['GoalDifference', 'TotalGoals']:
            df[f'{team_col}_{stat}_Last5'] = df.groupby(['league', team_col])[stat].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    
    # Create form features
    def get_form(results, n=5):
        return sum([1 if r == 'W' else 0.5 if r == 'D' else 0 for r in results[-n:]])
    
    for team in ['Team 1', 'Team 2']:
        df[f'{team}_Form'] = df.groupby(['league', team])['Result'].transform(
            lambda x: [get_form(x[:i]) for i in range(1, len(x)+1)]
        )
    
    # Add new features from additional_df
    if additional_df is not None:
        df = add_additional_features(df, additional_df)
    
    # features list
    features = ['Team 1', 'Team 2', 'DayOfWeek', 'Month', 'Year', 'league',
                'Team 1_GoalDifference_Last5', 'Team 1_TotalGoals_Last5',
                'Team 2_GoalDifference_Last5', 'Team 2_TotalGoals_Last5',
                'Team 1_Form', 'Team 2_Form',
                'yellow_cards_home', 'red_cards_home', 'goals_home', 'assists_home',
                'yellow_cards_away', 'red_cards_away', 'goals_away', 'assists_away']
    X = df[features]
    y = df['Result']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Normalize numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Check class distribution
    class_distribution = Counter(y)
    print(f"Class distribution before SMOTE: {class_distribution}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Class distribution after SMOTE: {Counter(y_resampled)}")

    return X_resampled, y_resampled, le

def add_additional_features(df, additional_df):
    # Aggregate additional data by team
    team_stats = additional_df.groupby('player_club_id').agg({
        'yellow_cards': 'mean',
        'red_cards': 'mean',
        'goals': 'mean',
        'assists': 'mean',
    }).reset_index()
    
    # Merge with main dataset
    df = pd.merge(df, team_stats, left_on='Team 1', right_on='player_club_id', how='left', suffixes=('', '_home'))
    df = pd.merge(df, team_stats, left_on='Team 2', right_on='player_club_id', how='left', suffixes=('_home', '_away'))
    
    # Fill NaN values with 0 for new columns
    new_columns = ['yellow_cards', 'red_cards', 'goals', 'assists']
    for col in new_columns:
        df[f'{col}_home'] = df[f'{col}_home'].fillna(0)
        df[f'{col}_away'] = df[f'{col}_away'].fillna(0)
    
    # Drop unnecessary columns
    df = df.drop(['player_club_id_home', 'player_club_id_away'], axis=1)
    
    return df

def add_head_to_head_features(df):
    df['h2h_wins'] = df.apply(lambda row: get_head_to_head_wins(row['Team 1'], row['Team 2'], df), axis=1)
    df['h2h_goals'] = df.apply(lambda row: get_head_to_head_goals(row['Team 1'], row['Team 2'], df), axis=1)
    return df

def get_head_to_head_wins(team1, team2, df):
    matches = df[((df['Team 1'] == team1) & (df['Team 2'] == team2)) | 
                 ((df['Team 1'] == team2) & (df['Team 2'] == team1))]
    team1_wins = sum((matches['Team 1'] == team1) & (matches['FT'].str.split('-').str[0].astype(int) > 
                                                     matches['FT'].str.split('-').str[1].astype(int)))
    team2_wins = sum((matches['Team 2'] == team1) & (matches['FT'].str.split('-').str[1].astype(int) > 
                                                     matches['FT'].str.split('-').str[0].astype(int)))
    return team1_wins - team2_wins

def get_head_to_head_goals(team1, team2, df):
    matches = df[((df['Team 1'] == team1) & (df['Team 2'] == team2)) | 
                 ((df['Team 1'] == team2) & (df['Team 2'] == team1))]
    team1_goals = sum(matches[matches['Team 1'] == team1]['FT'].str.split('-').str[0].astype(int)) + \
                  sum(matches[matches['Team 2'] == team1]['FT'].str.split('-').str[1].astype(int))
    team2_goals = sum(matches[matches['Team 1'] == team2]['FT'].str.split('-').str[0].astype(int)) + \
                  sum(matches[matches['Team 2'] == team2]['FT'].str.split('-').str[1].astype(int))
    return team1_goals - team2_goals


