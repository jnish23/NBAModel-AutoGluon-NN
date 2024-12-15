import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
import sqlite3
import time
from requests.exceptions import RequestException
import backoff

# Simple configuration
DB_NAME = 'nba_stats.db'
SEASONS = ['2016-17'
        #    , '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24'
           ]
POSITIONS = ['G', 'F', 'C']

@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def get_defensive_stats(position, date, season):
    stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense='Opponent',
        per_mode_detailed='PerGame',
        season=season, 
        season_type_all_star='Regular Season',
        date_from_nullable=date,
        date_to_nullable=date,
        player_position_abbreviation_nullable=position
    )
    
    df = stats.get_data_frames()[0]
    df['SEASON'] = season
    df['Position'] = position
    df['Date'] = date
    return df

def get_all_defensive_stats(date, season):
    all_data = []
    
    for position in POSITIONS:
        print(f"Fetching data for position {position} on {date}")
        try:
            df = get_defensive_stats(position, date, season)
            all_data.append(df)
        except Exception as e:
            print(f"Error fetching data for position {position} on {date}: {str(e)}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def process_defensive_stats(df):
    if df is None or df.empty:
        return None

    df_pivoted = df.pivot(index=['TEAM_ID', 'TEAM_NAME', 'Date'], 
                          columns='Position', 
                          values=df.columns[4:])
    
    df_pivoted.columns = [f'{stat}_{pos}' for stat, pos in df_pivoted.columns]
    df_pivoted.reset_index(inplace=True)
    
    return df_pivoted

def fetch_game_dates(season):
    # Implement this function to return a list of game dates for the season
    # This is a placeholder and should be replaced with actual implementation
    return []

def main():
    conn = sqlite3.connect(DB_NAME)
    
    for season in SEASONS:
        print(f"Processing season: {season}")
        game_dates = fetch_game_dates(season)
        for date in game_dates:
            raw_data = get_all_defensive_stats(date, season)
            processed_data = process_defensive_stats(raw_data)
            if processed_data is not None:
                processed_data.to_sql('defensive_stats_by_position', conn, if_exists='append', index=False)
            time.sleep(1)  # Basic rate limiting

    conn.close()

if __name__ == "__main__":
    main()