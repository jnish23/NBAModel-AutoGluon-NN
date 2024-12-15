import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from nba_api.stats.endpoints import leaguegamelog, playergamelogs, teamgamelogs
from time import sleep
import time
from filterpy.kalman import KalmanFilter
from typing import Dict, Tuple
from tqdm import tqdm

from autogluon.tabular import TabularPredictor, TabularDataset

DB_NAME = 'nba_stats.db'

query = """ SELECT
    tbs.SEASON_YEAR,
    tbs.TEAM_ID,
    tbs.TEAM_ABBREVIATION,
    tbs.GAME_ID,
    tbs.GAME_DATE,
    tbs.MATCHUP,
    tbs.WL,
    tbs.MIN,
    tbs.PTS,
    tbs.FGM,
    tbs.FGA,
    tbs.FG3M,
    tbs.FG3A,
    tbs.FTM,
    tbs.FTA,
    tbs.OREB,
    tbs.DREB,
    tbs.REB,
    tbs.AST,
    tbs.TOV,
    tbs.STL,
    tbs.BLK,
    tbs.PLUS_MINUS,
    tas.OFF_RATING,
    tas.DEF_RATING,
    tas.NET_RATING,
    tas.PACE,
    tas.PIE,
    tss.PCT_PTS_2PT_MR,
    tss.PCT_AST_2PM,
    tss.PCT_UAST_2PM,
    tss.PCT_AST_3PM,
    tss.PCT_UAST_3PM,
    ths.defensiveBoxOuts as DEFENSIVE_BOX_OUTS,
    ths.contestedShots2pt as CONTESTED_SHOTS_2PT,
    ths.contestedShots3pt as CONTESTED_SHOTS_3PT,
    ths.DEFLECTIONS,
    tts.DIST,
    tts.TCHS,
    tts.PASS,
    tts.CFGM,
    tts.CFGA,
    tts.UFGM,
    tts.UFGA,
    tts.DFGM,
    tts.DFGA,
    dsbp.OPP_FGM_C,
    dsbp.OPP_FGA_C,
    dsbp.OPP_FGM_F,
    dsbp.OPP_FGA_F,
    dsbp.OPP_FGM_G,
    dsbp.OPP_FGA_G,
    dsbp.OPP_FG3M_C,
    dsbp.OPP_FG3A_C,
    dsbp.OPP_FG3M_F,
    dsbp.OPP_FG3A_F,
    dsbp.OPP_FG3M_G,
    dsbp.OPP_FG3A_G,
    dsbp.OPP_FTM_C,
    dsbp.OPP_FTM_F,
    dsbp.OPP_FTM_G,
    dsbp.OPP_FTA_C,
    dsbp.OPP_FTA_F,
    dsbp.OPP_FTA_G,
    tms.PTS_OFF_TOV,
    tms.PTS_2ND_CHANCE,
    tms.PTS_FB,
    tms.PTS_PAINT,
    tms.OPP_PTS_OFF_TOV,
    tms.OPP_PTS_2ND_CHANCE,
    tms.OPP_PTS_FB,
    tms.OPP_PTS_PAINT,
    tms.BLK as BLOCKS,
    tms.BLKA as BLOCKED_ATT,
    tms.PF,
    tms.PFD,
    tslb.[Restricted Area FGM] as FGM_RESTRICTED,
    tslb.[Restricted Area FGA] as FGA_RESTRICTED,
    tslb.[In The Paint (Non-RA) FGM] as FGM_PAINT_NON_RA,
    tslb.[In The Paint (Non-RA) FGA] as FGA_PAINT_NON_RA,
    tslb.[Mid-Range FGM] as FGM_MIDRANGE,
    tslb.[Mid-Range FGA] as FGA_MIDRANGE,
    tslb.[Corner 3 FGM] as FGM_CORNER3,
    tslb.[Corner 3 FGA] as FGA_CORNER3,
    tslb.[Above the Break 3 FGM] as FGM_ABOVE_BREAK3,
    tslb.[Above the Break 3 FGA] as FGA_ABOVE_BREAK3,
    CASE WHEN MATCHUP like '%@%' THEN 0 ELSE 1 END as IS_HOME
FROM team_basic_stats tbs
LEFT JOIN team_advanced_stats tas ON tbs.TEAM_ID = tas.TEAM_ID AND tbs.GAME_ID = tas.GAME_ID
LEFT JOIN team_scoring_stats tss ON tbs.TEAM_ID = tss.TEAM_ID AND tbs.GAME_ID = tss.GAME_ID
LEFT JOIN team_hustle_stats ths ON tbs.TEAM_ID = ths.teamId AND tbs.GAME_ID = ths.gameId
LEFT JOIN team_track_stats tts ON tbs.TEAM_ID = tts.TEAM_ID AND tbs.GAME_ID = tts.GAME_ID
LEFT JOIN defensive_stats_by_position dsbp ON tbs.TEAM_ID = dsbp.TEAM_ID AND tbs.GAME_DATE = dsbp.Date
LEFT JOIN team_miscellaneous_stats tms ON tbs.TEAM_ID = tms.TEAM_ID AND tbs.GAME_ID = tms.GAME_ID
LEFT JOIN team_shot_location_boxscores tslb ON tbs.TEAM_ID = tslb.TEAM_ID AND tbs.GAME_DATE = tslb.Date
WHERE tbs.SEASON_YEAR BETWEEN '2016-17' AND '2024-25'
ORDER BY tbs.GAME_DATE, tbs.TEAM_ID"""

conn = sqlite3.connect(DB_NAME)
df_team = pd.read_sql(query, conn)


def clean_team_data(df):
    """This function cleans the team_data
    1) Changes W/L to 1/0 
    2) Changes franchise abbreviations to their most 
    recent abbreviation for consistency
    3) Converts GAME_DATE to datetime object
    4) Creates a binary column 'HOME_GAME'
    5) Removes 3 games where advanced stats were not collected
    """
    df = df.copy()
    df['WL'] = (df['WL'] == 'W').astype(int)
    df = df.rename(columns={'SEASON_YEAR': 'SEASON'})

    abbr_mapping = {'NJN': 'BKN',
                    'CHH': 'CHA',
                    'VAN': 'MEM',
                    'NOH': 'NOP',
                    'NOK': 'NOP',
                    'SEA': 'OKC'}

    df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].replace(abbr_mapping)
    df['MATCHUP'] = df['MATCHUP'].str.replace('NJN', 'BKN')
    df['MATCHUP'] = df['MATCHUP'].str.replace('CHH', 'CHA')
    df['MATCHUP'] = df['MATCHUP'].str.replace('VAN', 'MEM')
    df['MATCHUP'] = df['MATCHUP'].str.replace('NOH', 'NOP')
    df['MATCHUP'] = df['MATCHUP'].str.replace('NOK', 'NOP')
    df['MATCHUP'] = df['MATCHUP'].str.replace('SEA', 'OKC')

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    df = df.fillna(0)
    
    return df


df_team_clean = clean_team_data(df_team)



def prep_for_aggregation_team(df):
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
    
    ## Reorder Columns


    return df


df_team_clean2 = prep_for_aggregation_team(df_team_clean)



def create_matchups(df):
    """This function makes each row a matchup between 
    team and opp so that I can aggregate stats for a team's opponents"""
    keep_cols = ['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL']
    stat_cols = [x for x in df.columns if x not in keep_cols]
    

    
    matchups = pd.merge(df, df, how='left', on=['GAME_ID'], suffixes=['', '_opp'])
    matchups = matchups.loc[matchups['TEAM_ID'] != matchups['TEAM_ID_opp']]

    matchups = matchups.drop(columns = ['SEASON_opp', 'TEAM_ID_opp', 'TEAM_ABBREVIATION_opp', 'GAME_DATE_opp',
                                         'MATCHUP_opp', 'TEAM_ID_opp', 'WL_opp']
                 )
    
    matchups
    
    return matchups


matchups = create_matchups(df_team_clean2)


def calculate_ewa_multiple_spans(df: pd.DataFrame, group_col: str, value_col: str, spans: list, team_level: bool) -> pd.DataFrame:
       
    df = df.sort_values([group_col, 'GAME_DATE'])
    
    ewa_df = pd.DataFrame()
    for span in tqdm(spans):
        col_name = f'EWA_{span}'
        ewa_df[col_name] = df.groupby(group_col)[value_col].transform(
            lambda x: x.ewm(span=span, adjust=False).mean().shift(1)
        )
    
    return ewa_df

team_optimal_spans = {'PTS': 29,
 'FGM': 33,
 'FGA': 32,
 'FG3M': 28,
 'FG3A': 19,
 'FTM': 37,
 'FTA': 37,
 'OREB': 33,
 'DREB': 37,
 'REB': 33,
 'AST': 30,
 'TOV': 39,
 'STL': 36,
 'BLK': 35,
 'PLUS_MINUS': 38,
 'OFF_RATING': 37,
 'DEF_RATING': 43,
 'NET_RATING': 39,
 'PACE': 17,
 'PIE': 37,
 'PCT_PTS_2PT_MR': 20,
 'PCT_AST_2PM': 31,
 'PCT_UAST_2PM': 31,
 'PCT_AST_3PM': 37,
 'PCT_UAST_3PM': 38,
 'DEFENSIVE_BOX_OUTS': 11,
 'CONTESTED_SHOTS_2PT': 24,
 'CONTESTED_SHOTS_3PT': 25,
 'deflections': 30,
 'DIST': 43,
 'TCHS': 31,
 'PASS': 22,
 'CFGM': 29,
 'CFGA': 26,
 'UFGM': 34,
 'UFGA': 32,
 'DFGM': 31,
 'DFGA': 29,
 'OPP_FG3M_C': 10,
 'OPP_FG3A_C': 7,
 'OPP_FG3M_F': 18,
 'OPP_FG3A_F': 11,
 'OPP_FG3M_G': 19,
 'OPP_FG3A_G': 12,
 'OPP_FTM_C': 12,
 'OPP_FTM_F': 20,
 'OPP_FTM_G': 21,
 'OPP_FTA_C': 11,
 'OPP_FTA_F': 18,
 'OPP_FTA_G': 19,
 'PTS_OFF_TOV': 45,
 'PTS_2ND_CHANCE': 37,
 'PTS_FB': 31,
 'PTS_PAINT': 27,
 'OPP_PTS_OFF_TOV': 47,
 'OPP_PTS_2ND_CHANCE': 47,
 'OPP_PTS_FB': 52,
 'OPP_PTS_PAINT': 32,
 'BLOCKS': 35,
 'BLOCKED_ATT': 40,
 'PF': 33,
 'PFD': 29,
 'FGM_RESTRICTED': 25,
 'FGA_RESTRICTED': 24,
 'FGM_PAINT_NON_RA': 35,
 'FGA_PAINT_NON_RA': 27,
 'FGM_MIDRANGE': 21,
 'FGA_MIDRANGE': 16,
 'FGM_CORNER3': 44,
 'FGA_CORNER3': 35,
 'FGM_ABOVE_BREAK3': 31,
 'FGA_ABOVE_BREAK3': 21,
 'FG2M': 26,
 'FG2A': 22,
 'PTS_2PT_MR': 21,
 'AST_2PM': 24,
 'AST_3PM': 31,
 'UAST_2PM': 31,
 'UAST_3PM': 36,
 'OPP_FG2M_G': 11,
 'OPP_FG2A_G': 9,
 'OPP_FG2M_F': 9,
 'OPP_FG2A_F': 8,
 'OPP_FG2M_C': 7,
 'OPP_FG2A_C': 6,
 'MIN_opp': 16,
 'PTS_opp': 32,
 'FGM_opp': 34,
 'FGA_opp': 28,
 'FG3M_opp': 36,
 'FG3A_opp': 28,
 'FTM_opp': 43,
 'FTA_opp': 41,
 'OREB_opp': 50,
 'DREB_opp': 39,
 'REB_opp': 39,
 'AST_opp': 40,
 'TOV_opp': 34,
 'STL_opp': 41,
 'BLK_opp': 40,
 'PLUS_MINUS_opp': 38,
 'OFF_RATING_opp': 43,
 'DEF_RATING_opp': 37,
 'NET_RATING_opp': 39,
 'PACE_opp': 17,
 'PIE_opp': 36,
 'PCT_PTS_2PT_MR_opp': 35,
 'PCT_AST_2PM_opp': 43,
 'PCT_UAST_2PM_opp': 43,
 'PCT_AST_3PM_opp': 44,
 'PCT_UAST_3PM_opp': 44,
 'DEFENSIVE_BOX_OUTS_opp': 10,
 'CONTESTED_SHOTS_2PT_opp': 22,
 'CONTESTED_SHOTS_3PT_opp': 21,
 'deflections_opp': 36,
 'DIST_opp': 48,
 'TCHS_opp': 41,
 'PASS_opp': 39,
 'CFGM_opp': 36,
 'CFGA_opp': 33,
 'UFGM_opp': 44,
 'UFGA_opp': 36,
 'DFGM_opp': 26,
 'DFGA_opp': 24,
 'OPP_FG3M_C_opp': 52,
 'OPP_FG3A_C_opp': 50,
 'OPP_FG3M_F_opp': 36,
 'OPP_FG3A_F_opp': 30,
 'OPP_FG3M_G_opp': 33,
 'OPP_FG3A_G_opp': 30,
 'OPP_FTM_C_opp': 44,
 'OPP_FTM_F_opp': 40,
 'OPP_FTM_G_opp': 46,
 'OPP_FTA_C_opp': 44,
 'OPP_FTA_F_opp': 40,
 'OPP_FTA_G_opp': 42,
 'PTS_OFF_TOV_opp': 47,
 'PTS_2ND_CHANCE_opp': 47,
 'PTS_FB_opp': 52,
 'PTS_PAINT_opp': 32,
 'OPP_PTS_OFF_TOV_opp': 45,
 'OPP_PTS_2ND_CHANCE_opp': 37,
 'OPP_PTS_FB_opp': 31,
 'OPP_PTS_PAINT_opp': 27,
 'BLOCKS_opp': 40,
 'BLOCKED_ATT_opp': 35,
 'PF_opp': 29,
 'PFD_opp': 33,
 'FGM_RESTRICTED_opp': 31,
 'FGA_RESTRICTED_opp': 30,
 'FGM_PAINT_NON_RA_opp': 52,
 'FGA_PAINT_NON_RA_opp': 51,
 'FGM_MIDRANGE_opp': 40,
 'FGA_MIDRANGE_opp': 30,
 'FGM_CORNER3_opp': 48,
 'FGA_CORNER3_opp': 36,
 'FGM_ABOVE_BREAK3_opp': 46,
 'FGA_ABOVE_BREAK3_opp': 35,
 'IS_HOME_opp': 52,
 'FG2M_opp': 31,
 'FG2A_opp': 26,
 'PTS_2PT_MR_opp': 40,
 'AST_2PM_opp': 37,
 'AST_3PM_opp': 38,
 'UAST_2PM_opp': 35,
 'UAST_3PM_opp': 52,
 'OPP_FG2M_G_opp': 40,
 'OPP_FG2A_G_opp': 34,
 'OPP_FG2M_F_opp': 34,
 'OPP_FG2A_F_opp': 33,
 'OPP_FG2M_C_opp': 52,
 'OPP_FG2A_C_opp': 49,
 'POINT_DIFF_opp': 38,
 'RECORD_opp': 6,
 'TEAM_SCORE_opp': 32}


def apply_optimal_spans(df: pd.DataFrame, optimal_spans: dict, grouping_col: str) -> pd.DataFrame:
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


df_team_with_ewa = apply_optimal_spans(matchups, team_optimal_spans, 'TEAM_ID')


def add_percentage_features(df):
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
  
  
  
df_team_with_ewa2 = add_percentage_features(df_team_with_ewa)


keep_cols = ['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS']
ewa_cols = [col for col in df_team_with_ewa2.columns if "EWA" in col]
keep_cols.extend(ewa_cols)

df_team_with_ewa3 = df_team_with_ewa2[keep_cols]


def add_schedule_features(df):
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


team_final_df = add_schedule_features(df_team_with_ewa3)


team_final_df.to_sql('team_final_aggregated', conn=conn, if_exists='replace')






