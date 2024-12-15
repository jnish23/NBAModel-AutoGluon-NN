import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from time import sleep
from filterpy.kalman import KalmanFilter
from typing import Dict, Tuple
from tqdm import tqdm
from optimal_spans import OptimalSpans
import logging
from queries import TEAM_STATS_QUERY


class NBAFeatureAggregator:
    def __init__(self, db_path: str, spans_config_path: str = None):
        self.db_path = db_path
        self.optimal_spans = OptimalSpans(spans_config_path).spans
        
        
    def fetch_raw_data(self, season_range: Tuple[str, str] = None) -> pd.DataFrame:
        """
        Fetch raw team stats from database
        
        Parameters:
        season_range (Tuple[str, str]): Optional start and end seasons
        """
        try:
            query = TEAM_STATS_QUERY
            params = []
            
            # Modify query based on parameters
            if season_range:
                query = query.replace(
                    "BETWEEN '2016-17' AND '2024-25'",
                    "BETWEEN ? AND ?"
                )
                params.extend(season_range)
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql(query, conn, params=params)
                
        except Exception as e:
            logging.error(f"Error fetching raw data: {str(e)}")
            raise
        
    def store_features(self, df: pd.DataFrame) -> None:
        """Store processed features in database"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('team_aggregated_stats', conn, 
                     if_exists='replace', index=False)                        
            
            
    def select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final feature set"""
        keep_cols = ['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 
                    'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS']
        ewa_cols = [col for col in df.columns if "EWA" in col]
        keep_cols.extend(ewa_cols)
        
        return df[keep_cols]
    
        
    def clean_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize team data.
        
        Performs the following operations:
        1) Converts W/L to binary 1/0 
        2) Standardizes franchise abbreviations to current names
        3) Converts GAME_DATE to datetime
        4) Fills missing values with 0
        
        Args:
            df: Input DataFrame containing team data
            
        Returns:
            Cleaned DataFrame with standardized values
        """
        df = df.copy()
        
        # Convert win/loss to binary
        df['WL'] = (df['WL'] == 'W').astype(int)
        
        # Rename season column
        df = df.rename(columns={'SEASON_YEAR': 'SEASON'})

        # Define team abbreviation mappings
        abbr_mapping = {
            'NJN': 'BKN',
            'CHH': 'CHA', 
            'VAN': 'MEM',
            'NOH': 'NOP',
            'NOK': 'NOP',
            'SEA': 'OKC'
        }
        
        # Update team abbreviations
        df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].replace(abbr_mapping)
        
        # Update matchup strings more efficiently
        for old_abbr, new_abbr in abbr_mapping.items():
            df['MATCHUP'] = df['MATCHUP'].str.replace(old_abbr, new_abbr)
        
        # Convert dates to datetime
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
        
    def prep_for_aggregation_team(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function...
        1) Removes categories that are percentages,
        as we will be averaging them and do not want to average 
        percentages. 
        2) Converts shooting percentage stats into raw values"""

        df['FG2M'] = df['FGM'] - df['FG3M']
        df['FG2A'] = df['FGA'] - df['FG3A']
        df['PTS_2PT_MR'] = (df['PTS'] * df['PCT_PTS_2PT_MR']) #.astype('int8')
        df['AST_2PM'] = (df['FG2M'] * df['PCT_AST_2PM']) #.astype('int8')
        df['AST_3PM'] = (df['FG3M'] * df['PCT_AST_3PM']) #.astype('int8')
        df['UAST_2PM'] = (df['FG2M'] * df['PCT_UAST_2PM']) #.astype('int8')
        df['UAST_3PM'] = (df['FG3M'] * df['PCT_UAST_3PM']) #.astype('int8')

        df['OPP_FG2M_G'] = df['OPP_FGM_G'] - df['OPP_FG3M_G']
        df['OPP_FG2A_G'] = df['OPP_FGA_G'] - df['OPP_FG3A_G']

        df['OPP_FG2M_F'] = df['OPP_FGM_F'] - df['OPP_FG3M_F']
        df['OPP_FG2A_F'] = df['OPP_FGA_F'] - df['OPP_FG3A_F']
        
        df['OPP_FG2M_C'] = df['OPP_FGM_C'] - df['OPP_FG3M_C']
        df['OPP_FG2A_C'] = df['OPP_FGA_C'] - df['OPP_FG3A_C']

        df['POINT_DIFF'] = df['PLUS_MINUS']
        df['RECORD'] = df['WL']
        df['TEAM_SCORE'] = df['PTS']
        
        # percentage_columns = [x for x in df.columns if 'PCT' in x]
        drop_cols= ['OPP_FGM_G', 'OPP_FGA_G', 'OPP_FGM_F', 'OPP_FGA_F', 'OPP_FGM_C', 'OPP_FGA_C']
        #                       'MIN', 'PIE', 'PIE']
            
        df = df.drop(columns = drop_cols)
        
        return df
    
    
    
    def create_matchups(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function makes each row a matchup between 
        team and opp so that I can aggregate stats for a team's opponents"""
        keep_cols = ['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL']
        stat_cols = [x for x in df.columns if x not in keep_cols]
        

        
        matchups = pd.merge(df, df, how='left', on=['GAME_ID'], suffixes=['', '_opp'])
        matchups = matchups.loc[matchups['TEAM_ID'] != matchups['TEAM_ID_opp']]

        matchups = matchups.drop(columns = ['SEASON_opp', 'TEAM_ID_opp', 'TEAM_ABBREVIATION_opp', 'GAME_DATE_opp',
                                            'MATCHUP_opp', 'TEAM_ID_opp', 'WL_opp']
                    )
        
        
        return matchups
    
    
    def apply_optimal_spans(self, df: pd.DataFrame, optimal_spans: dict, grouping_col: str) -> pd.DataFrame:
        # Create a copy of the input DataFrame to avoid modifying the original
        df = df.copy()
        
        # Sort the DataFrame
        df = df.sort_values([grouping_col, 'GAME_DATE'])
        
        # Pre-calculate all EWA columns at once
        ewa_columns = {}
        
        for feature, span in optimal_spans.items():
            grouped = df.groupby(grouping_col)[feature]
            ewa_values = grouped.transform(
                lambda x: x.ewm(span=span, adjust=False).mean().shift(1)
            )
            ewa_columns[f'{feature}_EWA'] = ewa_values
        
        # Combine all new columns at once using concat
        result = pd.concat([df, pd.DataFrame(ewa_columns)], axis=1)
        
        return result

    
    def add_percentage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the following features for both team and opp:
        OREB_PCT, DREB_PCT, REB_PCT, TS_PCT, EFG_PCT, AST_RATIO, TOV_PCT, PIE.
        """
        
        df = df.copy()
        
        df['FG2_PCT'] = df['FG2M'] / df[f'FG2A']
        df['FG3_PCT'] = df['FG3M'] / df[f'FG3A']
        
        df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB_opp'])
        df['OREB_PCT_opp'] = df['OREB_opp'] / (df['OREB_opp'] + df['DREB'])

        df['DREB_PCT'] = df['DREB'] / (df['DREB'] + df['OREB_opp'])
        df['DREB_PCT_opp'] = df['DREB_opp'] / (df['DREB_opp'] + df['OREB'])

        df['REB_PCT'] = df['REB'] / (df['REB'] + df['REB_opp'])
        df['REB_PCT_opp'] = df['REB_opp'] / (df['REB_opp'] + df['REB'])

        df['TS_PCT'] = df['PTS'] / ((2*(df['FG2A'] + df['FG3A']) + 0.44*df['FTA']))
        
        df['TS_PCT_opp'] = df['PTS_opp'] / ((2*(df['FG2A_opp'] + df['FG3A_opp']) + 0.44*df['FTA_opp']))

        df['EFG_PCT'] = (df['FG2M'] + 1.5*df['FG3M']) / (df['FG2A']
                                                                        + df['FG3A'])
        df['EFG_PCT_opp'] = (df['FG2M_opp'] + 1.5*df['FG3M_opp']) / (df['FG2A_opp'] 
                                                                    + df['FG3A_opp'])

        df['AST_RATIO'] = (df['AST'] * 100) / df['PACE']
        df['AST_RATIO_opp'] = (df['AST_opp'] * 100) / df['PACE_opp']

        df['TOV_PCT'] = 100*df['TOV'] / (df['FG2A'] 
                                                + df['FG3A'] 
                                                + 0.44*df['FTA'] 
                                                + df['TOV'])
        
        df['TOV_PCT_opp'] = 100*df['TOV_opp'] / (df['FG2A_opp'] 
                                                + df['FG3A_opp'] 
                                                + 0.44*df['FTA_opp'] 
                                                + df['TOV_opp'])
        
        
        df['PIE'] = ((df['PTS'] + df['FG2M'] + df['FG3M'] + df['FTM'] 
                    - df['FG2A'] - df['FG3A'] - df['FTA'] 
                    + df['DREB'] + df['OREB']/2
                    + df['AST'] + df['STL'] + df['BLK']/2
                    - df['PF'] - df['TOV']) 
                    / (df['PTS'] + df['PTS_opp'] + df['FG2M'] + df['FG2M_opp']
                    + df['FG3M'] + df['FG3M_opp'] + df['FTM'] + df['FTM_opp']
                    - df['FG2A'] - df['FG2A_opp'] - df['FG3A'] - df['FG3A_opp'] 
                        - df['FTA'] - df['FTA_opp'] + df['DREB'] + df['DREB_opp']
                        + (df['OREB']+df['OREB_opp'])/2 + df['AST'] + df['AST_opp']
                        + df['STL'] + df['STL_opp'] + (df['BLK'] + df['BLK_opp'])/2
                        - df['PF'] - df['PF_opp'] - df['TOV'] - df['TOV_opp']))
            
        return df
    
    def add_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to rest and schedule density"""
        df = df.copy()
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

        # Calculate days between games for teams
        df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
        
        # Fill NaN values for first games
        df['DAYS_REST'] = df['DAYS_REST'].fillna(5)
        

        # Back-to-back games
        df['B2B_tm'] = (df['DAYS_REST'] <= 1).astype(int)
        
        return df

    def aggregate_features(self) -> None:
        """Main pipeline to process and store features"""
        
        try:
            logging.info("Starting feature aggregation pipeline")
            
            # 1. Fetch raw data
            df_raw = self.fetch_raw_data()
            logging.info(f"Fetched {len(df_raw)} raw records")
            
            # 2. Process data through pipeline
            df = self.clean_team_data(df_raw)
            df = self.prep_for_aggregation_team(df)
            df = self.create_matchups(df)
            df = self.apply_optimal_spans(df, self.optimal_spans, grouping_col='TEAM_ID')
            df = self.add_percentage_features(df)
            df = self.select_final_features(df)
            df = self.add_schedule_features(df)
            
            # 3. Store results
            self.store_features(df)
            
            logging.info("Feature aggregation completed successfully")
            
        except Exception as e:
            logging.error(f"Error in feature aggregation: {str(e)}")
            raise
        
        
# main.py
if __name__ == "__main__":
    DB_PATH = Path.cwd() / 'data' / 'nba_stats.db'
    aggregator = NBAFeatureAggregator(DB_PATH)
    aggregator.aggregate_features()