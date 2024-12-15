import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog, playergamelogs, teamgamelogs
from time import sleep
import time
from typing import Dict, Tuple
from tqdm import tqdm

from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common import space

DB_NAME = Path.cwd() / 'data' / 'nba_stats.db'


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
LEFT JOIN defensive_stats_by_position dsbp ON tbs.TEAM_ID = dsbp.TEAM_ID AND tbs.GAME_DATE = dsbp.GAME_DATE
LEFT JOIN team_miscellaneous_stats tms ON tbs.TEAM_ID = tms.TEAM_ID AND tbs.GAME_ID = tms.GAME_ID
LEFT JOIN team_shot_location_boxscores tslb ON tbs.TEAM_ID = tslb.TEAM_ID AND tbs.GAME_DATE = tslb.GAME_DATE
WHERE tbs.SEASON_YEAR BETWEEN '2016-17' AND '2024-25'
ORDER BY tbs.GAME_DATE, tbs.TEAM_ID"""

conn = sqlite3.connect(DB_NAME)
df_team = pd.read_sql(query, conn)
conn.close()

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
    team and opp"""
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


team_optimal_spans = {"PTS": 29,
    "FGM": 33,
    "FGA": 32,
    "FG3M": 28,
    "FG3A": 19,
    "FTM": 37,
    "FTA": 37,
    "OREB": 33,
    "DREB": 37,
    "REB": 33,
    "AST": 30,
    "TOV": 39,
    "STL": 36,
    "BLK": 35,
    "PLUS_MINUS": 38,
    "OFF_RATING": 37,
    "DEF_RATING": 43,
    "NET_RATING": 39,
    "PACE": 17,
    "PIE": 37,
    "PCT_PTS_2PT_MR": 20,
    "PCT_AST_2PM": 31,
    "PCT_UAST_2PM": 31,
    "PCT_AST_3PM": 37,
    "PCT_UAST_3PM": 38,
    "DEFENSIVE_BOX_OUTS": 11,
    "CONTESTED_SHOTS_2PT": 24,
    "CONTESTED_SHOTS_3PT": 25,
    "deflections": 30,
    "DIST": 43,
    "TCHS": 31,
    "PASS": 22,
    "CFGM": 29,
    "CFGA": 26,
    "UFGM": 34,
    "UFGA": 32,
    "DFGM": 31,
    "DFGA": 29,
    "OPP_FG3M_C": 10,
    "OPP_FG3A_C": 7,
    "OPP_FG3M_F": 18,
    "OPP_FG3A_F": 11,
    "OPP_FG3M_G": 19,
    "OPP_FG3A_G": 12,
    "OPP_FTM_C": 12,
    "OPP_FTM_F": 20,
    "OPP_FTM_G": 21,
    "OPP_FTA_C": 11,
    "OPP_FTA_F": 18,
    "OPP_FTA_G": 19,
    "PTS_OFF_TOV": 45,
    "PTS_2ND_CHANCE": 37,
    "PTS_FB": 31,
    "PTS_PAINT": 27,
    "OPP_PTS_OFF_TOV": 47,
    "OPP_PTS_2ND_CHANCE": 47,
    "OPP_PTS_FB": 52,
    "OPP_PTS_PAINT": 32,
    "BLOCKS": 35,
    "BLOCKED_ATT": 40,
    "PF": 33,
    "PFD": 29,
    "FGM_RESTRICTED": 25,
    "FGA_RESTRICTED": 24,
    "FGM_PAINT_NON_RA": 35,
    "FGA_PAINT_NON_RA": 27,
    "FGM_MIDRANGE": 21,
    "FGA_MIDRANGE": 16,
    "FGM_CORNER3": 44,
    "FGA_CORNER3": 35,
    "FGM_ABOVE_BREAK3": 31,
    "FGA_ABOVE_BREAK3": 21,
    "FG2M": 26,
    "FG2A": 22,
    "PTS_2PT_MR": 21,
    "AST_2PM": 24,
    "AST_3PM": 31,
    "UAST_2PM": 31,
    "UAST_3PM": 36,
    "OPP_FG2M_G": 11,
    "OPP_FG2A_G": 9,
    "OPP_FG2M_F": 9,
    "OPP_FG2A_F": 8,
    "OPP_FG2M_C": 7,
    "OPP_FG2A_C": 6,
    "PTS_opp": 32,
    "FGM_opp": 34,
    "FGA_opp": 28,
    "FG3M_opp": 36,
    "FG3A_opp": 28,
    "FTM_opp": 43,
    "FTA_opp": 41,
    "OREB_opp": 50,
    "DREB_opp": 39,
    "REB_opp": 39,
    "AST_opp": 40,
    "TOV_opp": 34,
    "STL_opp": 41,
    "BLK_opp": 40,
    "PLUS_MINUS_opp": 38,
    "OFF_RATING_opp": 43,
    "DEF_RATING_opp": 37,
    "NET_RATING_opp": 39,
    "PACE_opp": 17,
    "PIE_opp": 36,
    "PCT_PTS_2PT_MR_opp": 35,
    "PCT_AST_2PM_opp": 43,
    "PCT_UAST_2PM_opp": 43,
    "PCT_AST_3PM_opp": 44,
    "PCT_UAST_3PM_opp": 44,
    "DEFENSIVE_BOX_OUTS_opp": 10,
    "CONTESTED_SHOTS_2PT_opp": 22,
    "CONTESTED_SHOTS_3PT_opp": 21,
    "deflections_opp": 36,
    "DIST_opp": 48,
    "TCHS_opp": 41,
    "PASS_opp": 39,
    "CFGM_opp": 36,
    "CFGA_opp": 33,
    "UFGM_opp": 44,
    "UFGA_opp": 36,
    "DFGM_opp": 26,
    "DFGA_opp": 24,
    "OPP_FG3M_C_opp": 52,
    "OPP_FG3A_C_opp": 50,
    "OPP_FG3M_F_opp": 36,
    "OPP_FG3A_F_opp": 30,
    "OPP_FG3M_G_opp": 33,
    "OPP_FG3A_G_opp": 30,
    "OPP_FTM_C_opp": 44,
    "OPP_FTM_F_opp": 40,
    "OPP_FTM_G_opp": 46,
    "OPP_FTA_C_opp": 44,
    "OPP_FTA_F_opp": 40,
    "OPP_FTA_G_opp": 42,
    "PTS_OFF_TOV_opp": 47,
    "PTS_2ND_CHANCE_opp": 47,
    "PTS_FB_opp": 52,
    "PTS_PAINT_opp": 32,
    "OPP_PTS_OFF_TOV_opp": 45,
    "OPP_PTS_2ND_CHANCE_opp": 37,
    "OPP_PTS_FB_opp": 31,
    "OPP_PTS_PAINT_opp": 27,
    "BLOCKS_opp": 40,
    "BLOCKED_ATT_opp": 35,
    "PF_opp": 29,
    "PFD_opp": 33,
    "FGM_RESTRICTED_opp": 31,
    "FGA_RESTRICTED_opp": 30,
    "FGM_PAINT_NON_RA_opp": 52,
    "FGA_PAINT_NON_RA_opp": 51,
    "FGM_MIDRANGE_opp": 40,
    "FGA_MIDRANGE_opp": 30,
    "FGM_CORNER3_opp": 48,
    "FGA_CORNER3_opp": 36,
    "FGM_ABOVE_BREAK3_opp": 46,
    "FGA_ABOVE_BREAK3_opp": 35,
    "FG2M_opp": 31,
    "FG2A_opp": 26,
    "PTS_2PT_MR_opp": 40,
    "AST_2PM_opp": 37,
    "AST_3PM_opp": 38,
    "UAST_2PM_opp": 35,
    "UAST_3PM_opp": 52,
    "OPP_FG2M_G_opp": 40,
    "OPP_FG2A_G_opp": 34,
    "OPP_FG2M_F_opp": 34,
    "OPP_FG2A_F_opp": 33,
    "OPP_FG2M_C_opp": 52,
    "OPP_FG2A_C_opp": 49
}


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



def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features related to rest and schedule density"""
    df = df.copy()
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    # Calculate days between games for teams
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    
    # Fill NaN values for first games
    df['DAYS_REST'] = df['DAYS_REST'].fillna(5)
    
    
    return df


df_team_with_ewa3 = add_schedule_features(df_team_with_ewa2)


def create_matchup_df2(df):
    df_team_merged = pd.merge(df, df, on=['GAME_ID', 'GAME_DATE'], suffixes=('_tm', '_opp'))
    df_team_merged = df_team_merged.loc[df_team_merged['TEAM_ID_tm'] != df_team_merged['TEAM_ID_opp']]
    df_team_merged = df_team_merged.sort_values(['GAME_DATE', 'GAME_ID', 'IS_HOME_tm'])

    keep_cols = ['SEASON_tm', 'TEAM_ID_tm', 'TEAM_ABBREVIATION_tm', 'TEAM_ID_opp', 'TEAM_ABBREVIATION_opp',
                 'GAME_ID', 'GAME_DATE', 'MATCHUP_tm', 'IS_HOME_tm', 'WL_tm', 'PTS_tm', 'PTS_opp', 'DAYS_REST_tm', 'DAYS_REST_opp']
    ewa_cols = [col for col in df_team_merged.columns if "EWA" in col]
    keep_cols.extend(ewa_cols)
    
    df_team_merged = df_team_merged.rename(columns={'EWA_tm': 'EWA_tm_tm', 'EWA_opp': 'EWA_tm_opp'})
    
    return df_team_merged[keep_cols] 


df_matchup_team = create_matchup_df2(df_team_with_ewa3)



final_df = df_matchup_team.dropna()


final_df = final_df.rename(columns ={'SEASON_tm':'SEASON'
                                     ,'TEAM_ID_tm':'TEAM_ID_tm'
                                     ,'TEAM_ABBREVIATION_tm':'TEAM_ABBREVIATION'
                                     ,'MATCHUP_tm':'MATCHUP'})

final_train = final_df[final_df['SEASON'] <= '2023-24']
final_test = final_df[final_df['SEASON'] == '2024-25']

model_path = Path.cwd() / "models" / "AutogluonModels" 
print("model_path:", model_path)

# Prepare the features
ewa_columns = [col for col in final_df.columns if '_EWA' in col]
ewa_columns.extend(['IS_HOME_tm', 'DAYS_REST_tm', 'DAYS_REST_opp'])

X_train = final_train[ewa_columns]
y_train = final_train['PTS_tm']

X_test = final_test[ewa_columns]
y_test = final_test['PTS_tm']


# Define hyperparameter search spaces for different models
hyperparameter_config = {
    'GBM': [{  # LightGBM
        'num_boost_round': 100,
        'num_leaves': space.Int(lower=26, upper=66, default=36),
        'learning_rate': space.Real(lower=5e-3, upper=0.1, default=0.05, log=True),
        'feature_fraction': space.Real(lower=0.5, upper=1.0, default=0.8),
        'min_data_in_leaf': space.Int(lower=10, upper=100, default=20),
        'ag_args': {'name_suffix': 'custom'},
    }],
    'XGB': [{  # XGBoost
        'n_estimators': 100,
        'max_depth': space.Int(lower=3, upper=10, default=6),
        'learning_rate': space.Real(lower=5e-3, upper=0.1, default=0.05, log=True),
        'subsample': space.Real(lower=0.5, upper=1.0, default=0.8),
        'colsample_bytree': space.Real(lower=0.5, upper=1.0, default=0.8),
        'ag_args': {'name_suffix': 'custom'},
    }],
    'CAT': [{  # CatBoost
        'iterations': 100,
        'depth': space.Int(lower=4, upper=10, default=6),
        'learning_rate': space.Real(lower=5e-3, upper=0.1, default=0.05, log=True),
        'subsample': space.Real(lower=0.5, upper=1.0, default=0.8),
        'ag_args': {'name_suffix': 'custom'},
    }],
    'NN_TORCH': [{  # Neural Network
        'num_epochs': space.Int(lower=10, upper=100, default=50),
        'learning_rate': space.Real(lower=1e-4, upper=1e-2, default=1e-3, log=True),
        'dropout_prob': space.Real(lower=0.0, upper=0.5, default=0.1),
        'weight_decay': space.Real(lower=1e-6, upper=1e-3, default=1e-5, log=True),
        'ag_args': {'name_suffix': 'custom'},
    }]
}

# Initialize and train the predictor with hyperparameter tuning
predictor = TabularPredictor(label='PTS_tm'
                             , problem_type='regression'
                             , eval_metric='mean_absolute_error'
                             , path = model_path)

# Train with hyperparameter tuning
predictor.fit(
    train_data=pd.concat([X_train, y_train], axis=1),
    hyperparameters=hyperparameter_config,
    num_bag_folds=5,  # Number of folds for bagging
    num_bag_sets=1,   # Number of repeats of kfold bagging
    num_stack_levels=2,  # Number of stacking levels
    time_limit=3600,  # Time limit in seconds (e.g., 1 hour)
    presets=['optimize_for_deployment', 'high_quality'],
    hyperparameter_tune_kwargs={
        'num_trials': 20,  # Number of HPO trials
        'scheduler': 'local',
        'searcher': 'bayes'  # Can also use 'bayesopt' or 'grid'
    },
    excluded_model_types=['KNN', 'RF', 'XT', 'AG_AUTOMM', 'FASTAI', 'XGB'], # Models to exclude
    save_bag_folds=True
)

# Get leaderboard of all models
leaderboard = predictor.leaderboard()

print("Leaderboard:", leaderboard)


# First get predictions on test data
predictions = predictor.predict(X_test)

# Create DataFrame with actual vs predicted scores
results_df = pd.DataFrame({
    'Team': final_test['TEAM_ABBREVIATION'],
    'Opponent': final_test['TEAM_ABBREVIATION_opp'],
    'Matchup': final_test['MATCHUP'],
    'Home': final_test['IS_HOME_tm'],
    'Actual_Score': y_test,
    'Predicted_Score': predictions
})

results_df.to_csv('prediction_results.csv', index=False)
