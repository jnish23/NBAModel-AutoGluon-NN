import sqlite3
import numpy as np
import pandas as pd
import time
import random
import logging
import backoff
from pathlib import Path
from requests.exceptions import RequestException, Timeout
from datetime import datetime
from typing import Set, Tuple, Optional, Dict, List
from dataclasses import dataclass
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoreadvancedv2,
    boxscorescoringv2,
    boxscorehustlev2,
    boxscoreplayertrackv2,
    boxscoremiscv2,
    boxscoreusagev2,
    playergamelogs,
    leaguegamelog,
    leaguedashteamstats,
    leaguedashteamshotlocations
)
from sbrscrape import Scoreboard


@dataclass
class TablePair:
    """Class to hold related team and player table information"""
    team_table: str
    player_table: str
    id_column: str
    min_season: str  # Minimum season where this stat is available

class NBADatabaseSync:
    def __init__(self, db_path: str, log_path: str = 'nba_sync.log'):
        """
        Initialize the NBA database synchronization tool
        """
        self.db_path = db_path
        self.setup_logging(log_path)
        self.endpoint_map = {
            'team_advanced_stats': boxscoreadvancedv2.BoxScoreAdvancedV2,
            'team_scoring_stats': boxscorescoringv2.BoxScoreScoringV2,
            'team_hustle_stats': boxscorehustlev2.BoxScoreHustleV2,
            'team_track_stats': boxscoreplayertrackv2.BoxScorePlayerTrackV2,
            'team_miscellaneous_stats': boxscoremiscv2.BoxScoreMiscV2,
            'team_usage_stats': boxscoreusagev2.BoxScoreUsageV2,
            'team_shot_locations': leaguedashteamshotlocations.LeagueDashTeamShotLocations

        }
        
        self.table_pairs = [
            TablePair('team_advanced_stats', 'player_advanced_stats', 'GAME_ID', '2013-14'),
            TablePair('team_scoring_stats', 'player_scoring_stats', 'GAME_ID', '2013-14'),
            TablePair('team_track_stats', 'player_track_stats', 'GAME_ID', '2013-14'),
            TablePair('team_miscellaneous_stats', 'player_miscellaneous_stats', 'GAME_ID', '2013-14'),
            TablePair('team_usage_stats', 'player_usage_stats', 'GAME_ID', '2013-14'),
            TablePair('team_hustle_stats', 'player_hustle_stats', 'gameId', '2015-16'),
            TablePair('defensive_stats_by_position', '', 'GAME_DATE', '2016-17'),
            TablePair('team_shot_location_boxscores', '', 'GAME_DATE', '2014-15')


        ]
        
        self.POSITIONS = ['C', 'F', 'G']
        
        self.sportsbooks = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_ny']
        self.odds_columns = ['home_spread', 'home_spread_odds', 'away_spread', 
                           'away_spread_odds', 'home_ml', 'away_ml', 'total', 
                           'over_odds', 'under_odds']


    def setup_logging(self, log_path: str) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def custom_backoff():
        """
        Custom backoff generator that yields fixed delays
        
        Yields:
        int: Delay in seconds (60, 120, 180)
        """
        yield 60  # First retry after 1 minute
        yield 120 # Second retry after 2 minutes
        yield 180 # Third retry after 3 minutes


    @backoff.on_exception(
        custom_backoff,
        (RequestException, Timeout),
        max_tries=4,
        logger=logging.getLogger('nba_api_logger')
    )
    
    def add_team_basic_boxscores(self, seasons: List[str]) -> None:
        """
        Pull and add basic team boxscores for specified seasons with improved DataFrame handling
        """
        logging.info(f"Adding team basic boxscores for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season {season}")
                    season_boxscores = []
                    
                    # Determine season types
                    season_types = ['Regular Season', 'PlayIn', 'Playoffs'] if season >= '2019-20' else ['Regular Season', 'Playoffs']
                    
                    for season_type in season_types:
                        try:
                            team_boxscores = leaguegamelog.LeagueGameLog(
                                season=season, 
                                season_type_all_star=season_type
                            ).get_data_frames()[0]
                            
                            # Only append if DataFrame is not empty
                            if not team_boxscores.empty:
                                # Remove VIDEO_AVAILABLE column if it exists
                                if 'VIDEO_AVAILABLE' in team_boxscores.columns:
                                    team_boxscores = team_boxscores.drop(columns=['VIDEO_AVAILABLE'])
                                
                                # Remove rows that have NaN values in 'WL' column indicating an incomplete game
                                team_boxscores = team_boxscores.dropna(subset=['WL'])
                                
                                season_boxscores.append(team_boxscores)
                                logging.info(f"Retrieved {len(team_boxscores)} games for {season_type} {season}")
                            else:
                                logging.warning(f"No data retrieved for {season_type} {season}")
                                
                        except Exception as e:
                            logging.error(f"Error retrieving {season_type} data for season {season}: {str(e)}")
                            continue
                        
                        time.sleep(4*random.random())

                    
                    if season_boxscores:
                        # Get column dtypes from first DataFrame to ensure consistency
                        dtype_dict = season_boxscores[0].dtypes.to_dict()
                        
                        # Ensure all DataFrames have the same columns and dtypes
                        processed_boxscores = []
                        for df in season_boxscores:
                            # Ensure all columns exist
                            for col, dtype in dtype_dict.items():
                                if col not in df.columns:
                                    df[col] = pd.Series(dtype=dtype)
                            
                            # Ensure correct dtypes
                            df = df.astype(dtype_dict)
                            processed_boxscores.append(df)
                        
                        # Concatenate with explicit dtype preservation
                        season_df = pd.concat(processed_boxscores, ignore_index=True)
                        season_df['SEASON_YEAR'] = season
                        
                        # Add to database
                        table_name = 'team_basic_stats'
                        season_df.to_sql(table_name, conn, if_exists='append', index=False)
                        
                        # Remove duplicates
                        cur = conn.cursor()
                        cur.execute(f'''
                            DELETE FROM {table_name} 
                            WHERE rowid NOT IN (
                                SELECT max(rowid) 
                                FROM {table_name} 
                                GROUP BY TEAM_ID, GAME_ID
                            )
                        ''')
                        conn.commit()
                        
                        logging.info(f"Successfully added {len(season_df)} games for season {season}")
                    else:
                        logging.warning(f"No valid data retrieved for season {season}")
                    
                    
        except Exception as e:
            logging.error(f"Error in add_team_basic_boxscores: {str(e)}")
            raise

    def add_player_basic_boxscores(self, seasons: List[str]) -> None:
        """
        Pull and add basic player boxscores for specified seasons with improved DataFrame handling
        """
        logging.info(f"Adding player basic boxscores for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season {season}")
                    season_boxscores = []
                    
                    season_types = ['Regular Season', 'PlayIn', 'Playoffs'] if season >= '2019-20' else ['Regular Season', 'Playoffs']
                    
                    for season_type in season_types:
                        try:
                            time.sleep(4*random.random())
                            player_boxscores = playergamelogs.PlayerGameLogs(
                                season_nullable=season,
                                season_type_nullable=season_type
                            ).get_data_frames()[0]
                            
                            if not player_boxscores.empty:
                                # Remove rows that have NaN values in 'WL' column indicating an incomplete game
                                player_boxscores = player_boxscores.dropna(subset=['WL'])

                                season_boxscores.append(player_boxscores)
                                logging.info(f"Retrieved {len(player_boxscores)} player games for {season_type} {season}")
                            else:
                                logging.warning(f"No player data retrieved for {season_type} {season}")
                                
                        except Exception as e:
                            logging.error(f"Error retrieving {season_type} data for season {season}: {str(e)}")
                            continue
                        
                        time.sleep(4*random.random())


                    
                    if season_boxscores:
                        # Get column dtypes from first DataFrame
                        dtype_dict = season_boxscores[0].dtypes.to_dict()
                        
                        # Process each DataFrame
                        processed_boxscores = []
                        for df in season_boxscores:
                            # Ensure all columns exist
                            for col, dtype in dtype_dict.items():
                                if col not in df.columns:
                                    df[col] = pd.Series(dtype=dtype)
                            
                            # Ensure correct dtypes
                            df = df.astype(dtype_dict)
                            processed_boxscores.append(df)
                        
                        # Concatenate with explicit dtype preservation
                        season_df = pd.concat(processed_boxscores, ignore_index=True)
                        season_df['SEASON_YEAR'] = season
                        
                        # Add to database
                        table_name = 'player_basic_stats'
                        season_df.to_sql(table_name, conn, if_exists='append', index=False)
                        
                        # Remove duplicates
                        cur = conn.cursor()
                        cur.execute(f'''
                            DELETE FROM {table_name} 
                            WHERE rowid NOT IN (
                                SELECT max(rowid) 
                                FROM {table_name} 
                                GROUP BY PLAYER_ID, GAME_ID
                            )
                        ''')
                        conn.commit()
                        
                        logging.info(f"Successfully added {len(season_df)} player games for season {season}")
                    else:
                        logging.warning(f"No valid player data retrieved for season {season}")
                                        
        except Exception as e:
            logging.error(f"Error in add_player_basic_boxscores: {str(e)}")
            raise

    def sync_basic_stats(self, seasons: List[str]) -> None:
        """
        Sync both team and player basic stats for specified seasons
        
        Parameters:
        seasons (List[str]): List of seasons to sync
        """
        logging.info(f"Starting basic stats sync for seasons: {seasons}")
        
        try:
            # Sync team stats
            self.add_team_basic_boxscores(seasons)
            
            # Sync player stats
            self.add_player_basic_boxscores(seasons)
            
            logging.info("Basic stats sync completed successfully")
            
        except Exception as e:
            logging.error(f"Error in sync_basic_stats: {str(e)}")
            raise
    
    def get_defensive_stats(self, position: str, date: str, season: str, season_type: str) -> pd.DataFrame:
        """
        Get defensive stats for a specific position on a specific date
        
        Parameters:
        position (str): Position abbreviation ('C', 'F', or 'G')
        date (str): Date in YYYY-MM-DD format
        season (str): Season in YYYY-YY format
        
        Returns:
        pd.DataFrame: Defensive stats for the specified position
        """
        logging.info(f"Fetching defensive stats for position {position} on {date}")
        
        try:            
            stats = leaguedashteamstats.LeagueDashTeamStats(
                measure_type_detailed_defense='Opponent',
                per_mode_detailed='PerGame',
                season=season,
                season_type_all_star=season_type,
                date_from_nullable=date,
                date_to_nullable=date,
                player_position_abbreviation_nullable=position
            )
            
            df = stats.get_data_frames()[0]
            df['SEASON'] = season
            df['POSITION'] = position
            df['GAME_DATE'] = date
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching defensive stats for position {position} on {date}: {str(e)}")
            raise

    def get_all_defensive_stats(self, date: str, season: str, season_type: str) -> Optional[pd.DataFrame]:
        """
        Get defensive stats for all positions on a specific date
        
        Parameters:
        date (str): Date in YYYY-MM-DD format
        season (str): Season in YYYY-YY format
        
        Returns:
        Optional[pd.DataFrame]: Combined defensive stats for all positions
        """
        all_data = []
        
        for position in self.POSITIONS:
            try:
                df = self.get_defensive_stats(position, date, season, season_type)
                all_data.append(df)
                logging.info(f"Successfully fetched data for position {position} on {date}")
            except Exception as e:
                logging.error(f"Failed to fetch data for position {position} on {date}: {str(e)}")
                continue
            
            time.sleep(4*random.random())

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def process_defensive_stats(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Process and pivot defensive stats
        
        Parameters:
        df (Optional[pd.DataFrame]): Raw defensive stats DataFrame
        
        Returns:
        Optional[pd.DataFrame]: Processed defensive stats
        """
        if df is None or df.empty:
            return None

        try:
            # Pivot the DataFrame
            pivot_cols = ['L', 'W_PCT', 'MIN', 'OPP_FGM',
                        'OPP_FGA', 'OPP_FG_PCT', 'OPP_FG3M', 'OPP_FG3A', 'OPP_FG3_PCT',
                        'OPP_FTM', 'OPP_FTA', 'OPP_FT_PCT', 'OPP_OREB', 'OPP_DREB', 'OPP_REB',
                        'OPP_AST', 'OPP_TOV', 'OPP_STL', 'OPP_BLK', 'OPP_BLKA', 'OPP_PF',
                        'OPP_PFD', 'OPP_PTS', 'PLUS_MINUS']
            
            df_pivoted = df.pivot(
                index=['SEASON', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE'],
                columns='POSITION',
                values=pivot_cols
            )
            
            # Flatten column names
            df_pivoted.columns = [f'{stat}_{pos}' for stat, pos in df_pivoted.columns]
            df_pivoted = df_pivoted.reset_index() 
            
            return df_pivoted
            
        except Exception as e:
            logging.error(f"Error processing defensive stats: {str(e)}")
            return None

    def fetch_game_date_and_season_type(self, season: str) -> List[str]:
        """
        Fetch all game dates for a given season
        
        Parameters:
        season (str): Season in YYYY-YY format
        
        Returns:
        List[str]: List of game dates in YYYY-MM-DD format
        """
        try:
            query = """
            SELECT DISTINCT GAME_DATE, SEASON_ID
            FROM team_basic_stats 
            WHERE SEASON_YEAR = ? 
            ORDER BY GAME_DATE
            """
            
            with sqlite3.connect(self.db_path) as conn:
                dates = pd.read_sql_query(query, conn, params=[season])
                
                conditions = [
                    dates['SEASON_ID'].str.startswith('22'),
                    dates['SEASON_ID'].str.startswith('52'),
                    dates['SEASON_ID'].str.startswith('42')
                ]
                choices = ['Regular Season', 'PlayIn', 'Playoffs']
                
                
                dates['SEASON_TYPE'] = np.select(conditions, choices)
                
                dates_and_season_type = dates[['GAME_DATE', 'SEASON_TYPE']].values.tolist()
                
                return dates_and_season_type
                
        except Exception as e:
            logging.error(f"Error fetching game dates for season {season}: {str(e)}")
            return []

    def sync_defensive_stats(self, seasons: List[str]) -> None:
        """
        Sync defensive stats for specified seasons
        
        Parameters:
        seasons (List[str]): List of seasons to sync
        """
        logging.info(f"Starting defensive stats sync for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season: {season}")
                    
                    # Get existing dates
                    existing_dates_query = """
                    SELECT DISTINCT GAME_DATE 
                    FROM defensive_stats_by_position 
                    WHERE SEASON = ?
                    
                    """
                    existing_dates = set(pd.read_sql_query(
                        existing_dates_query, 
                        conn, 
                        params=[season]
                    )['GAME_DATE'].tolist())
                    
                    # Get all game dates and season_types
                    game_dates_and_season_types = self.fetch_game_date_and_season_type(season)
                    
                    for date, season_type in game_dates_and_season_types:
                        if date not in existing_dates:
                            try:
                                raw_data = self.get_all_defensive_stats(date, season, season_type)
                                processed_data = self.process_defensive_stats(raw_data)
                                
                                if processed_data is not None:
                                    processed_data.to_sql(
                                        'defensive_stats_by_position',
                                        conn,
                                        if_exists='append',
                                        index=False
                                    )
                                    logging.info(f"Added defensive stats for {date}")
                                    
                            except Exception as e:
                                logging.error(f"Error processing date {date}: {str(e)}")
                                continue
                    
            logging.info("Defensive stats sync completed")
            
        except Exception as e:
            logging.error(f"Error in defensive stats sync: {str(e)}")
            raise
        
        
    def get_shot_location_stats(self, date: str, season: str, season_type: str) -> Optional[pd.DataFrame]:
        """
        Get shot location stats for a specific date
        """
        logging.info(f"Fetching shot location stats for date {date}")
        
        try:
          
            stats = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                distance_range='By Zone',
                per_mode_detailed='PerGame',
                season=season,
                season_type_all_star=season_type,
                date_from_nullable=date,
                date_to_nullable=date,
            )
            
            df = stats.get_data_frames()[0]
            df['SEASON'] = season
            df['GAME_DATE'] = date
            
            # Process column names
            df.columns = df.columns.map(' '.join)
            df.columns = [col.strip() for col in df.columns]
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching shot location stats for {date}: {str(e)}")
            return None
        

    def sync_shot_location_stats(self, seasons: List[str]) -> None:
        """
        Sync shot location stats for specified seasons
        """
        logging.info(f"Starting shot location stats sync for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season: {season}")
                    
                    # Get existing dates
                    existing_dates_query = """
                    SELECT DISTINCT GAME_DATE
                    FROM team_shot_location_boxscores 
                    WHERE SEASON = ?
                    """
                    existing_dates_df = pd.read_sql_query(
                        existing_dates_query, 
                        conn, 
                        params=[season]
                        )
                    
                    existing_dates_list = existing_dates_df['GAME_DATE'].astype(str).tolist()

                    existing_dates = set(existing_dates_list)

                    # Get all game dates and season types
                    game_dates_and_season_types = self.fetch_game_date_and_season_type(season)
                    
                    for date, season_type in game_dates_and_season_types:
                        if date not in existing_dates:
                            try:
                                df = self.get_shot_location_stats(date, season, season_type)
                                if df is not None:
                                    df.to_sql(
                                        'team_shot_location_boxscores',
                                        conn,
                                        if_exists='append',
                                        index=False
                                    )
                                    logging.info(f"Added shot location stats for {date} ({season_type})")
                            except Exception as e:
                                logging.error(f"Error processing {date} ({season_type}): {str(e)}")
                                continue
                            
                            time.sleep(4*random.random())

                # Remove duplicates
                self.remove_duplicates('team_shot_location_boxscores', is_player_table=False)
                
            logging.info("Shot location stats sync completed")
            
        except Exception as e:
            logging.error(f"Error in shot location stats sync: {str(e)}")
            raise


    def fetch_game_ids(self, season: str) -> Set[str]:
        """
        Fetch all game IDs for a given season using the NBA API with custom backoff
        """
        logging.info(f"Fetching game IDs for season {season}")
        
        try:
           
            # Fetch games using NBA API
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00'  # NBA games only
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            # Filter for NBA games only
            nba_games = games_df.loc[
                games_df['SEASON_ID'].str.startswith(('22', '42', '52'))
            ]
            
            # Get unique game IDs
            game_ids = set(nba_games['GAME_ID'].unique())
            
            logging.info(f"Found {len(game_ids)} unique games for season {season}")
            
            if len(game_ids) == 0:
                logging.warning(f"No games found for season {season}")
                
            return game_ids
            
        except (RequestException, Timeout) as e:
            logging.error(f"API error fetching game IDs for season {season}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error fetching game IDs for season {season}: {str(e)}")
            raise
        
        
        
    def get_game_data(self, table: str, game_id: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fetch game data from NBA API with exponential backoff
        
        Parameters:
        table (str): Table name
        game_id (str): Game ID
        
        Returns:
        Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Player and team DataFrames if successful
        """
        endpoint = self.endpoint_map.get(table)
        if not endpoint:
            logging.warning(f"No endpoint found for table {table}")
            return None
            
        try:
            response = endpoint(game_id=game_id)
            return response.get_data_frames()
        except Exception as e:
            logging.error(f"Error fetching data for game {game_id} from {table}: {str(e)}")
            raise

    def process_game(self, game_id: str, table_pair: TablePair) -> bool:
        """
        Process a single game for a table pair
        
        Parameters:
        game_id (str): Game ID
        table_pair (TablePair): Table pair to process
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            game_data = self.get_game_data(table_pair.team_table, game_id)
            if game_data:
                player_df, team_df = game_data
                
                with sqlite3.connect(self.db_path) as conn:
                    player_df.to_sql(table_pair.player_table, conn, if_exists='append', index=False)
                    team_df.to_sql(table_pair.team_table, conn, if_exists='append', index=False)
                
                logging.info(f"Successfully added data for game {game_id} to {table_pair.team_table}")
                return True
            
            logging.warning(f"No data available for game {game_id} in {table_pair.team_table}")
            return False
            
        except Exception as e:
            logging.error(f"Error processing game {game_id} for {table_pair.team_table}: {str(e)}")
            return False
        
        
    def get_nba_odds(self, date: str) -> pd.DataFrame:
        """Fetch NBA odds data for a specific date using sbrscrape"""
        logging.info(f"Fetching odds data for {date}")
        try:
            sb = Scoreboard(sport='NBA', date=date)
            odds_data = sb.games
            return pd.DataFrame(odds_data)
        except Exception as e:
            logging.error(f"Error fetching odds for {date}: {str(e)}")
            return pd.DataFrame()


    def process_odds_data(self, odds_df: pd.DataFrame, season: str, date: str) -> pd.DataFrame:
        """Process and flatten odds data"""
        if odds_df.empty:
            return odds_df

        try:
            # Flatten nested dictionaries for each sportsbook
            for column in self.odds_columns:
                if column in odds_df.columns:
                    for sportsbook in self.sportsbooks:
                        odds_df[f'{column}_{sportsbook}'] = odds_df[column].apply(
                            lambda x: x.get(sportsbook) if isinstance(x, dict) else None
                        )
            
            # Drop original nested columns
            odds_df = odds_df.drop(columns=self.odds_columns, errors='ignore')
            
            # Add metadata
            odds_df['SEASON'] = season
            odds_df['GAME_DATE'] = date
            
            odds_df = odds_df.drop(columns=['date'])
            
            return odds_df
        except Exception as e:
            logging.error(f"Error processing odds data: {str(e)}")
            return pd.DataFrame()
        

    def sync_odds_data(self, seasons: List[str]) -> None:
        """Sync odds data for specified seasons"""
        logging.info(f"Starting odds sync for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    # Get existing dates
                    existing_dates_query = """
                    SELECT DISTINCT GAME_DATE 
                    FROM nba_odds 
                    WHERE SEASON = ?
                    """
                    existing_dates = set(pd.read_sql_query(
                        existing_dates_query, 
                        conn, 
                        params=[season]
                    )['GAME_DATE'].tolist())
                    
                    
                    
                    # Get all game dates
                    game_dates = self.fetch_game_date_and_season_type(season)
                    
                    for date, _ in game_dates:
                        if date not in existing_dates:
                            try:
                                odds_df = self.get_nba_odds(date)
                                if not odds_df.empty:
                                    processed_df = self.process_odds_data(odds_df, season, date)
                                    if not processed_df.empty:
                                        processed_df.to_sql('nba_odds', conn, if_exists='append', index=False)
                                        logging.info(f"Added odds data for {date}")
                            except Exception as e:
                                logging.error(f"Error processing odds for {date}: {str(e)}")
                                continue
                    
                # Remove duplicates
                self.remove_duplicates('nba_odds', is_player_table=False)
                
            logging.info("Odds sync completed")
            
        except Exception as e:
            logging.error(f"Error in odds sync: {str(e)}")
            raise
        
        

    def remove_duplicates(self, table_name: str, is_player_table: bool = False) -> None:
        """
        Remove duplicate entries from specified table, keeping the latest entry
        
        Parameters:
        table_name (str): Name of the table to deduplicate
        is_player_table (bool): If True, use PLAYER_ID for deduplication, else use TEAM_ID
        """
        logging.info(f"Removing duplicates from {table_name}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:

                # Determine ID column based on table type
                
                if 'hustle' in table_name.lower():
                    id_column = 'personId' if is_player_table else 'teamId'
                    game_id_col = 'gameId'

                elif table_name.lower() in ('defensive_stats_by_position', 'team_shot_location_boxscores'):
                    id_column = 'TEAM_ID'
                    game_id_col = 'GAME_DATE'
                    
                elif table_name.lower() in ('nba_odds'):
                    id_column = 'home_team'
                    game_id_col = 'GAME_DATE'
                    
                    
                else:
                    id_column = 'PLAYER_ID' if is_player_table else 'TEAM_ID'
                    game_id_col = 'GAME_ID' 
                    
                
                # Construct deduplication query
                dedup_query = f"""
                DELETE FROM {table_name}
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM {table_name}
                    GROUP BY {id_column}, {game_id_col}
                )
                """
                
                cursor = conn.cursor()
                
                # Get count before deletion
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count_before = cursor.fetchone()[0]
                
                # Execute deduplication
                cursor.execute(dedup_query)
                conn.commit()
                
                # Get count after deletion
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count_after = cursor.fetchone()[0]
                
                duplicates_removed = count_before - count_after
                logging.info(f"Removed {duplicates_removed} duplicates from {table_name}")
            
        except Exception as e:
            logging.error(f"Error removing duplicates from {table_name}: {str(e)}")
            raise


    def deduplicate_all_tables(self) -> None:
        """
        Remove duplicates from all tables in the database
        """
        logging.info("Starting database-wide deduplication")
        
        # Define tables and their types
        team_tables = [
            'team_basic_stats',
            'team_advanced_stats',
            'team_scoring_stats',
            'team_hustle_stats',
            'team_track_stats',
            'team_miscellaneous_stats',
            'team_usage_stats',
            'defensive_stats_by_position',
            'team_shot_location_boxscores'
        ]
        
        player_tables = [
            'player_basic_stats',
            'player_advanced_stats',
            'player_scoring_stats',
            'player_hustle_stats',
            'player_track_stats',
            'player_miscellaneous_stats',
            'player_usage_stats'
        ]
        
        try:
            # Process team tables
            for table in team_tables:
                self.remove_duplicates(table, is_player_table=False)
                
            # Process player tables
            for table in player_tables:
                self.remove_duplicates(table, is_player_table=True)
                
            logging.info("Database-wide deduplication completed successfully")
            
        except Exception as e:
            logging.error(f"Error during database-wide deduplication: {str(e)}")
            raise

    def process_table_pair(self, season: str, table_pair: TablePair, all_game_ids: Set[str]) -> None:
        """
        Process a single table pair for a season
        
        Parameters:
        season (str): Season in format 'YYYY-YY'
        table_pair (TablePair): Table pair to process
        all_game_ids (Set[str]): Set of all game IDs for the season
        """
        # Skip if season is before the stat was tracked
        if season < table_pair.min_season:
            logging.info(f"Skipping {table_pair.team_table} for season {season} (stat not tracked)")
            return

        logging.info(f"Processing {table_pair.team_table} for season {season}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Skip defensive stats as they're handled separately
                if table_pair.team_table in ('defensive_stats_by_position', 'team_shot_location_boxscores'):
                    logging.info("Skipping defensive_stats_by_position and team_shot_location_boxscores as they are handled separately")
                    return
                    
                # Get existing game IDs
                existing_ids = set(pd.read_sql(
                    f"SELECT DISTINCT {table_pair.id_column} FROM {table_pair.player_table}",
                    conn
                )[table_pair.id_column])
                
                # Find missing games
                missing_ids = all_game_ids - existing_ids
                
                if not missing_ids:
                    logging.info(f"No missing games for {table_pair.team_table}")
                    return
                
                logging.info(f"Found {len(missing_ids)} missing games for {table_pair.team_table}")
                
                # Process missing games
                for game_id in missing_ids:
                    try:
                        success = self.process_game(game_id, table_pair)
                        if success:
                            time.sleep(4*random.random())
                    except Exception as e:
                        logging.error(f"Error processing game {game_id}: {str(e)}")
                
                # Remove duplicates after processing
                self.remove_duplicates(table_pair.team_table, is_player_table=False)
                self.remove_duplicates(table_pair.player_table, is_player_table=True)
                
        except Exception as e:
            logging.error(f"Error processing {table_pair.team_table}: {str(e)}")
            
            
        
    def sync_database(self, seasons: List[str]) -> None:
        """
        Main method to synchronize the database
        """
        start_time = datetime.now()
        logging.info(f"Starting database sync for seasons: {seasons}")
        
        try:
            # First sync basic stats
            self.sync_basic_stats(seasons)
            
            # Then sync defensive stats
            self.sync_defensive_stats(seasons)
            
            # Then sync shot location stats
            self.sync_shot_location_stats(seasons)

            # Sync odds data
            self.sync_odds_data(seasons)

            # Then process detailed stats for each game
            for season in seasons:
                logging.info(f"Processing detailed stats for season {season}")
                
                # Fetch all game IDs for the season
                all_game_ids = self.fetch_game_ids(season)
                logging.info(f"Found {len(all_game_ids)} total games for season {season}")
                
                # Process each table pair
                for table_pair in self.table_pairs:
                    self.process_table_pair(season, table_pair, all_game_ids)
            
            # Final deduplication of all tables
            self.deduplicate_all_tables()
            
        except Exception as e:
            logging.error(f"Error in database sync: {str(e)}")
            raise
            
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Database sync completed in {duration}")
        
# Usage example
if __name__ == "__main__":
    seasons_to_check = ['2024-25']
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    
    syncer = NBADatabaseSync(db_path)

    # Regular sync will now include deduplication

    syncer.sync_database(seasons_to_check)


