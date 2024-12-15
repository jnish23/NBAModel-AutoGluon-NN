import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import List, Dict
from datetime import datetime
from autogluon.tabular import TabularPredictor
from pathlib import Path
from nba_api.stats.endpoints import scoreboardv2
from sbrscrape import Scoreboard
from update_database import NBADatabaseSync  # Add this import

def convert_american_to_decimal(x):
    return np.where(x>0, (100+x)/100, 1+(100.0/-x))     

class NBAPredictionPipeline:
    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.model_path = model_path
        self.model = TabularPredictor.load(self.model_path)


    def prepare_team_features(self, team_id: int, opponent_id: int) -> pd.DataFrame:
        """
        Prepare features for a team against specific opponent using latest aggregated data
        
        Parameters:
        team_id (int): ID of the team to prepare features for
        opponent_id (int): ID of the opposing team
        
        Returns:
        pd.DataFrame: Single row DataFrame with all features for prediction
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get the latest EWA features for both teams
                query = """
                WITH RankedGames AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY TEAM_ID ORDER BY GAME_DATE DESC) as rn
                    FROM team_aggregated_stats
                    WHERE TEAM_ID IN (?, ?)
                )
                SELECT *
                FROM RankedGames
                WHERE rn = 1
                """
                
                latest_stats = pd.read_sql_query(query, conn, params=[team_id, opponent_id])
                
                if latest_stats.empty:
                    logging.error(f"No stats found for teams {team_id} and {opponent_id}")
                    return None
                
                # Separate team and opponent stats
                team_stats = latest_stats[latest_stats['TEAM_ID'] == team_id].iloc[0]
                opp_stats = latest_stats[latest_stats['TEAM_ID'] == opponent_id].iloc[0]
                
                # Get columns that end with _EWA (your exponentially weighted averages)
                ewa_cols = [col for col in latest_stats.columns if col.endswith('_EWA')]
                
                # Create feature dictionary
                features = {}
                
                # Add team EWA features
                for col in ewa_cols:
                    features[f'{col}_tm'] = team_stats[col]
                    features[f'{col}_opp'] = opp_stats[col]
                
                
                # Add any other relevant features
                # features['IS_HOME'] = 1  # This will be 1 for home team row, 0 for away team row
                
                # Convert to DataFrame
                features_df = pd.DataFrame([features])
                
                features_df = features_df[['PTS_EWA_tm', 'FGM_EWA_tm', 'FGA_EWA_tm', 'FG3M_EWA_tm', 'FG3A_EWA_tm', 'FTM_EWA_tm', 
                                           'FTA_EWA_tm', 'OREB_EWA_tm', 'DREB_EWA_tm', 'REB_EWA_tm', 'AST_EWA_tm', 'TOV_EWA_tm',
                                           'STL_EWA_tm', 'BLK_EWA_tm', 'PLUS_MINUS_EWA_tm', 'OFF_RATING_EWA_tm', 'DEF_RATING_EWA_tm', 'NET_RATING_EWA_tm', 'PACE_EWA_tm', 'PIE_EWA_tm', 'PCT_PTS_2PT_MR_EWA_tm', 'PCT_AST_2PM_EWA_tm', 'PCT_UAST_2PM_EWA_tm', 'PCT_AST_3PM_EWA_tm', 'PCT_UAST_3PM_EWA_tm', 'DEFENSIVE_BOX_OUTS_EWA_tm', 'CONTESTED_SHOTS_2PT_EWA_tm', 'CONTESTED_SHOTS_3PT_EWA_tm', 'deflections_EWA_tm', 'DIST_EWA_tm', 'TCHS_EWA_tm', 'PASS_EWA_tm', 'CFGM_EWA_tm', 'CFGA_EWA_tm', 'UFGM_EWA_tm', 'UFGA_EWA_tm', 'DFGM_EWA_tm', 'DFGA_EWA_tm', 'OPP_FG3M_C_EWA_tm', 'OPP_FG3A_C_EWA_tm', 'OPP_FG3M_F_EWA_tm', 'OPP_FG3A_F_EWA_tm', 'OPP_FG3M_G_EWA_tm', 'OPP_FG3A_G_EWA_tm', 'OPP_FTM_C_EWA_tm', 'OPP_FTM_F_EWA_tm', 'OPP_FTM_G_EWA_tm', 'OPP_FTA_C_EWA_tm', 'OPP_FTA_F_EWA_tm', 'OPP_FTA_G_EWA_tm', 'PTS_OFF_TOV_EWA_tm', 'PTS_2ND_CHANCE_EWA_tm', 'PTS_FB_EWA_tm', 'PTS_PAINT_EWA_tm', 'OPP_PTS_OFF_TOV_EWA_tm', 'OPP_PTS_2ND_CHANCE_EWA_tm', 'OPP_PTS_FB_EWA_tm', 'OPP_PTS_PAINT_EWA_tm', 'BLOCKS_EWA_tm', 'BLOCKED_ATT_EWA_tm', 'PF_EWA_tm', 'PFD_EWA_tm', 'FGM_RESTRICTED_EWA_tm', 'FGA_RESTRICTED_EWA_tm', 'FGM_PAINT_NON_RA_EWA_tm', 'FGA_PAINT_NON_RA_EWA_tm', 'FGM_MIDRANGE_EWA_tm', 'FGA_MIDRANGE_EWA_tm', 'FGM_CORNER3_EWA_tm', 'FGA_CORNER3_EWA_tm', 'FGM_ABOVE_BREAK3_EWA_tm', 'FGA_ABOVE_BREAK3_EWA_tm', 'FG2M_EWA_tm', 'FG2A_EWA_tm', 'PTS_2PT_MR_EWA_tm', 'AST_2PM_EWA_tm', 'AST_3PM_EWA_tm', 'UAST_2PM_EWA_tm', 'UAST_3PM_EWA_tm', 'OPP_FG2M_G_EWA_tm', 'OPP_FG2A_G_EWA_tm', 'OPP_FG2M_F_EWA_tm', 'OPP_FG2A_F_EWA_tm', 'OPP_FG2M_C_EWA_tm', 'OPP_FG2A_C_EWA_tm', 'PTS_opp_EWA_tm', 'FGM_opp_EWA_tm', 'FGA_opp_EWA_tm', 'FG3M_opp_EWA_tm', 'FG3A_opp_EWA_tm', 'FTM_opp_EWA_tm', 'FTA_opp_EWA_tm', 'OREB_opp_EWA_tm', 'DREB_opp_EWA_tm', 'REB_opp_EWA_tm', 'AST_opp_EWA_tm', 'TOV_opp_EWA_tm', 'STL_opp_EWA_tm', 'BLK_opp_EWA_tm', 'PLUS_MINUS_opp_EWA_tm', 'OFF_RATING_opp_EWA_tm', 'DEF_RATING_opp_EWA_tm', 'NET_RATING_opp_EWA_tm', 'PACE_opp_EWA_tm', 'PIE_opp_EWA_tm', 'PCT_PTS_2PT_MR_opp_EWA_tm', 'PCT_AST_2PM_opp_EWA_tm', 'PCT_UAST_2PM_opp_EWA_tm', 'PCT_AST_3PM_opp_EWA_tm', 'PCT_UAST_3PM_opp_EWA_tm', 'DEFENSIVE_BOX_OUTS_opp_EWA_tm', 'CONTESTED_SHOTS_2PT_opp_EWA_tm', 'CONTESTED_SHOTS_3PT_opp_EWA_tm', 'deflections_opp_EWA_tm', 'DIST_opp_EWA_tm', 'TCHS_opp_EWA_tm', 'PASS_opp_EWA_tm', 'CFGM_opp_EWA_tm', 'CFGA_opp_EWA_tm', 'UFGM_opp_EWA_tm', 'UFGA_opp_EWA_tm', 'DFGM_opp_EWA_tm', 'DFGA_opp_EWA_tm', 'OPP_FG3M_C_opp_EWA_tm', 'OPP_FG3A_C_opp_EWA_tm', 'OPP_FG3M_F_opp_EWA_tm', 'OPP_FG3A_F_opp_EWA_tm', 'OPP_FG3M_G_opp_EWA_tm', 'OPP_FG3A_G_opp_EWA_tm', 'OPP_FTM_C_opp_EWA_tm', 'OPP_FTM_F_opp_EWA_tm', 'OPP_FTM_G_opp_EWA_tm', 'OPP_FTA_C_opp_EWA_tm', 'OPP_FTA_F_opp_EWA_tm', 'OPP_FTA_G_opp_EWA_tm', 'PTS_OFF_TOV_opp_EWA_tm', 'PTS_2ND_CHANCE_opp_EWA_tm', 'PTS_FB_opp_EWA_tm', 'PTS_PAINT_opp_EWA_tm', 'OPP_PTS_OFF_TOV_opp_EWA_tm', 'OPP_PTS_2ND_CHANCE_opp_EWA_tm', 'OPP_PTS_FB_opp_EWA_tm', 'OPP_PTS_PAINT_opp_EWA_tm', 'BLOCKS_opp_EWA_tm', 'BLOCKED_ATT_opp_EWA_tm', 'PF_opp_EWA_tm', 'PFD_opp_EWA_tm', 'FGM_RESTRICTED_opp_EWA_tm', 'FGA_RESTRICTED_opp_EWA_tm', 'FGM_PAINT_NON_RA_opp_EWA_tm', 'FGA_PAINT_NON_RA_opp_EWA_tm', 'FGM_MIDRANGE_opp_EWA_tm', 'FGA_MIDRANGE_opp_EWA_tm', 'FGM_CORNER3_opp_EWA_tm', 'FGA_CORNER3_opp_EWA_tm', 'FGM_ABOVE_BREAK3_opp_EWA_tm', 'FGA_ABOVE_BREAK3_opp_EWA_tm', 'FG2M_opp_EWA_tm', 'FG2A_opp_EWA_tm', 'PTS_2PT_MR_opp_EWA_tm', 'AST_2PM_opp_EWA_tm', 'AST_3PM_opp_EWA_tm', 'UAST_2PM_opp_EWA_tm', 'UAST_3PM_opp_EWA_tm', 'OPP_FG2M_G_opp_EWA_tm', 'OPP_FG2A_G_opp_EWA_tm', 'OPP_FG2M_F_opp_EWA_tm', 'OPP_FG2A_F_opp_EWA_tm', 'OPP_FG2M_C_opp_EWA_tm', 'OPP_FG2A_C_opp_EWA_tm', 'PTS_EWA_opp', 'FGM_EWA_opp', 'FGA_EWA_opp', 'FG3M_EWA_opp', 'FG3A_EWA_opp', 'FTM_EWA_opp', 'FTA_EWA_opp', 'OREB_EWA_opp', 'DREB_EWA_opp', 'REB_EWA_opp', 'AST_EWA_opp', 'TOV_EWA_opp', 'STL_EWA_opp', 'BLK_EWA_opp', 'PLUS_MINUS_EWA_opp', 'OFF_RATING_EWA_opp', 'DEF_RATING_EWA_opp', 'NET_RATING_EWA_opp', 'PACE_EWA_opp', 'PIE_EWA_opp', 'PCT_PTS_2PT_MR_EWA_opp', 'PCT_AST_2PM_EWA_opp', 'PCT_UAST_2PM_EWA_opp', 'PCT_AST_3PM_EWA_opp', 'PCT_UAST_3PM_EWA_opp', 'DEFENSIVE_BOX_OUTS_EWA_opp', 'CONTESTED_SHOTS_2PT_EWA_opp', 'CONTESTED_SHOTS_3PT_EWA_opp', 'deflections_EWA_opp', 'DIST_EWA_opp', 'TCHS_EWA_opp', 'PASS_EWA_opp', 'CFGM_EWA_opp', 'CFGA_EWA_opp', 'UFGM_EWA_opp', 'UFGA_EWA_opp', 'DFGM_EWA_opp', 'DFGA_EWA_opp', 'OPP_FG3M_C_EWA_opp', 'OPP_FG3A_C_EWA_opp', 'OPP_FG3M_F_EWA_opp', 'OPP_FG3A_F_EWA_opp', 'OPP_FG3M_G_EWA_opp', 'OPP_FG3A_G_EWA_opp', 'OPP_FTM_C_EWA_opp', 'OPP_FTM_F_EWA_opp', 'OPP_FTM_G_EWA_opp', 'OPP_FTA_C_EWA_opp', 'OPP_FTA_F_EWA_opp', 'OPP_FTA_G_EWA_opp', 'PTS_OFF_TOV_EWA_opp', 'PTS_2ND_CHANCE_EWA_opp', 'PTS_FB_EWA_opp', 'PTS_PAINT_EWA_opp', 'OPP_PTS_OFF_TOV_EWA_opp', 'OPP_PTS_2ND_CHANCE_EWA_opp', 'OPP_PTS_FB_EWA_opp', 'OPP_PTS_PAINT_EWA_opp', 'BLOCKS_EWA_opp', 'BLOCKED_ATT_EWA_opp', 'PF_EWA_opp', 'PFD_EWA_opp', 'FGM_RESTRICTED_EWA_opp', 'FGA_RESTRICTED_EWA_opp', 'FGM_PAINT_NON_RA_EWA_opp', 'FGA_PAINT_NON_RA_EWA_opp', 'FGM_MIDRANGE_EWA_opp', 'FGA_MIDRANGE_EWA_opp', 'FGM_CORNER3_EWA_opp', 'FGA_CORNER3_EWA_opp', 'FGM_ABOVE_BREAK3_EWA_opp', 'FGA_ABOVE_BREAK3_EWA_opp', 'FG2M_EWA_opp', 'FG2A_EWA_opp', 'PTS_2PT_MR_EWA_opp', 'AST_2PM_EWA_opp', 'AST_3PM_EWA_opp', 'UAST_2PM_EWA_opp', 'UAST_3PM_EWA_opp', 'OPP_FG2M_G_EWA_opp', 'OPP_FG2A_G_EWA_opp', 'OPP_FG2M_F_EWA_opp', 'OPP_FG2A_F_EWA_opp', 'OPP_FG2M_C_EWA_opp', 'OPP_FG2A_C_EWA_opp', 'PTS_opp_EWA_opp', 'FGM_opp_EWA_opp', 'FGA_opp_EWA_opp', 'FG3M_opp_EWA_opp', 'FG3A_opp_EWA_opp', 'FTM_opp_EWA_opp', 'FTA_opp_EWA_opp', 'OREB_opp_EWA_opp', 'DREB_opp_EWA_opp', 'REB_opp_EWA_opp', 'AST_opp_EWA_opp', 'TOV_opp_EWA_opp', 'STL_opp_EWA_opp', 'BLK_opp_EWA_opp', 'PLUS_MINUS_opp_EWA_opp', 'OFF_RATING_opp_EWA_opp', 'DEF_RATING_opp_EWA_opp', 'NET_RATING_opp_EWA_opp', 'PACE_opp_EWA_opp', 'PIE_opp_EWA_opp', 'PCT_PTS_2PT_MR_opp_EWA_opp', 'PCT_AST_2PM_opp_EWA_opp', 'PCT_UAST_2PM_opp_EWA_opp', 'PCT_AST_3PM_opp_EWA_opp', 'PCT_UAST_3PM_opp_EWA_opp', 'DEFENSIVE_BOX_OUTS_opp_EWA_opp', 'CONTESTED_SHOTS_2PT_opp_EWA_opp', 'CONTESTED_SHOTS_3PT_opp_EWA_opp', 'deflections_opp_EWA_opp', 'DIST_opp_EWA_opp', 'TCHS_opp_EWA_opp', 'PASS_opp_EWA_opp', 'CFGM_opp_EWA_opp', 'CFGA_opp_EWA_opp', 'UFGM_opp_EWA_opp', 'UFGA_opp_EWA_opp', 'DFGM_opp_EWA_opp', 'DFGA_opp_EWA_opp', 'OPP_FG3M_C_opp_EWA_opp', 'OPP_FG3A_C_opp_EWA_opp', 'OPP_FG3M_F_opp_EWA_opp', 'OPP_FG3A_F_opp_EWA_opp', 'OPP_FG3M_G_opp_EWA_opp', 'OPP_FG3A_G_opp_EWA_opp', 'OPP_FTM_C_opp_EWA_opp', 'OPP_FTM_F_opp_EWA_opp', 'OPP_FTM_G_opp_EWA_opp', 'OPP_FTA_C_opp_EWA_opp', 'OPP_FTA_F_opp_EWA_opp', 'OPP_FTA_G_opp_EWA_opp', 'PTS_OFF_TOV_opp_EWA_opp', 'PTS_2ND_CHANCE_opp_EWA_opp', 'PTS_FB_opp_EWA_opp', 'PTS_PAINT_opp_EWA_opp', 'OPP_PTS_OFF_TOV_opp_EWA_opp', 'OPP_PTS_2ND_CHANCE_opp_EWA_opp', 'OPP_PTS_FB_opp_EWA_opp', 'OPP_PTS_PAINT_opp_EWA_opp', 'BLOCKS_opp_EWA_opp', 'BLOCKED_ATT_opp_EWA_opp', 'PF_opp_EWA_opp', 'PFD_opp_EWA_opp', 'FGM_RESTRICTED_opp_EWA_opp', 'FGA_RESTRICTED_opp_EWA_opp', 'FGM_PAINT_NON_RA_opp_EWA_opp', 'FGA_PAINT_NON_RA_opp_EWA_opp', 'FGM_MIDRANGE_opp_EWA_opp', 'FGA_MIDRANGE_opp_EWA_opp', 'FGM_CORNER3_opp_EWA_opp', 'FGA_CORNER3_opp_EWA_opp', 'FGM_ABOVE_BREAK3_opp_EWA_opp', 'FGA_ABOVE_BREAK3_opp_EWA_opp', 'FG2M_opp_EWA_opp', 'FG2A_opp_EWA_opp', 'PTS_2PT_MR_opp_EWA_opp', 'AST_2PM_opp_EWA_opp', 'AST_3PM_opp_EWA_opp', 'UAST_2PM_opp_EWA_opp', 'UAST_3PM_opp_EWA_opp', 'OPP_FG2M_G_opp_EWA_opp', 'OPP_FG2A_G_opp_EWA_opp', 'OPP_FG2M_F_opp_EWA_opp', 'OPP_FG2A_F_opp_EWA_opp', 'OPP_FG2M_C_opp_EWA_opp', 'OPP_FG2A_C_opp_EWA_opp']]
                
                return features_df
                
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for NBA points prediction
        
        Parameters:
        df (pd.DataFrame): DataFrame containing NBA game statistics with EWA metrics
        
        Returns:
        pd.DataFrame: Original dataframe with added engineered features
        """
        # Create copy to avoid modifying original
        df_new = df.copy()
        
        # 1. Offensive Matchup Features
        df_new['OFF_DEF_RATING_DELTA'] = df_new['OFF_RATING_EWA_tm'] - df_new['DEF_RATING_EWA_opp']
        
        # Calculate restricted area FG% safely
        df_new['FG_PCT_RESTRICTED_EWA_tm'] = df_new['FGM_RESTRICTED_EWA_tm'] / df_new['FGA_RESTRICTED_EWA_tm'].replace(0, 1)
        
        df_new['PAINT_SCORING_ADVANTAGE'] = (df_new['FG_PCT_RESTRICTED_EWA_tm'] * df_new['FGA_RESTRICTED_EWA_tm']) / \
                                        (df_new['OPP_FG2M_C_EWA_opp'] + df_new['OPP_FG2M_F_EWA_opp']).replace(0, 1)
        
        df_new['RELATIVE_PACE'] = df_new['PACE_EWA_tm'] / df_new['PACE_EWA_opp'].replace(0, 1)
        
        # 2. Style Matchup Features
        df_new['THREE_PT_RATIO_tm'] = df_new['FG3A_EWA_tm'] / df_new['FGA_EWA_tm'].replace(0, 1)
        df_new['PAINT_RATIO_tm'] = (df_new['FGA_RESTRICTED_EWA_tm'] + df_new['FGA_PAINT_NON_RA_EWA_tm']) / \
                                df_new['FGA_EWA_tm'].replace(0, 1)
        
        df_new['AST_SCORING_DIFF'] = (df_new['PCT_AST_2PM_EWA_tm'] + df_new['PCT_AST_3PM_EWA_tm']) - \
                                    (df_new['PCT_AST_2PM_EWA_opp'] + df_new['PCT_AST_3PM_EWA_opp'])
        
        # 3. Efficiency Metrics
        df_new['EFF_SCORING'] = df_new['PTS_EWA_tm'] / \
                            (2 * (df_new['FGA_EWA_tm'] + 0.44 * df_new['FTA_EWA_tm'])).replace(0, 1)
        
        df_new['POSS_EFFICIENCY'] = df_new['PTS_EWA_tm'] / \
                                (df_new['FGA_EWA_tm'] + df_new['TOV_EWA_tm'] + 0.44 * df_new['FTA_EWA_tm']).replace(0, 1)
        
        # 4. Defensive Impact Features
        df_new['DEF_PRESSURE_SCORE'] = (df_new['CONTESTED_SHOTS_2PT_EWA_tm'] + 
                                    df_new['CONTESTED_SHOTS_3PT_EWA_tm'] + 
                                    df_new['deflections_EWA_tm']) / \
                                    df_new['TCHS_EWA_opp'].replace(0, 1)
        
        df_new['TRANSITION_DEF_EFF'] = df_new['OPP_PTS_FB_EWA_tm'] / df_new['PTS_FB_EWA_opp'].replace(0, 1)
                
        # 6. Advanced Interaction Features
        df_new['OREB_SCORING_IMPACT'] = df_new['OREB_EWA_tm'] * df_new['PTS_2ND_CHANCE_EWA_tm'] / \
                                    df_new['FGA_EWA_tm'].replace(0, 1)
        
        df_new['BALL_MOVEMENT_EFF'] = (df_new['AST_EWA_tm'] * df_new['PASS_EWA_tm']) / \
                                    (df_new['TOV_EWA_tm'] + 1)
        
        # 7. Additional Efficiency Metrics
        df_new['ASSISTED_SCORING_EFF'] = (df_new['PCT_AST_2PM_EWA_tm'] * df_new['FG2M_EWA_tm'] + 
                                        df_new['PCT_AST_3PM_EWA_tm'] * df_new['FG3M_EWA_tm']) / \
                                        df_new['FGA_EWA_tm'].replace(0, 1)
        
        df_new['PAINT_DEF_EFF'] = df_new['OPP_PTS_PAINT_EWA_tm'] / \
                                (df_new['FGA_RESTRICTED_EWA_opp'] + df_new['FGA_PAINT_NON_RA_EWA_opp']).replace(0, 1)
        
        # Replace infinities with 0
        df_new = df_new.replace([np.inf, -np.inf], 0)
        
        # Fill any NaNs with 0
        df_new = df_new.fillna(0)
        
        df_new = df_new[['PTS_EWA_tm',
 'FGM_EWA_tm',
 'FGA_EWA_tm',
 'FG3M_EWA_tm',
 'FG3A_EWA_tm',
 'FTM_EWA_tm',
 'FTA_EWA_tm',
 'OREB_EWA_tm',
 'DREB_EWA_tm',
 'REB_EWA_tm',
 'AST_EWA_tm',
 'TOV_EWA_tm',
 'STL_EWA_tm',
 'BLK_EWA_tm',
 'PLUS_MINUS_EWA_tm',
 'OFF_RATING_EWA_tm',
 'DEF_RATING_EWA_tm',
 'NET_RATING_EWA_tm',
 'PACE_EWA_tm',
 'PIE_EWA_tm',
 'PCT_PTS_2PT_MR_EWA_tm',
 'PCT_AST_2PM_EWA_tm',
 'PCT_UAST_2PM_EWA_tm',
 'PCT_AST_3PM_EWA_tm',
 'PCT_UAST_3PM_EWA_tm',
 'DEFENSIVE_BOX_OUTS_EWA_tm',
 'CONTESTED_SHOTS_2PT_EWA_tm',
 'CONTESTED_SHOTS_3PT_EWA_tm',
 'deflections_EWA_tm',
 'DIST_EWA_tm',
 'TCHS_EWA_tm',
 'PASS_EWA_tm',
 'CFGM_EWA_tm',
 'CFGA_EWA_tm',
 'UFGM_EWA_tm',
 'UFGA_EWA_tm',
 'DFGM_EWA_tm',
 'DFGA_EWA_tm',
 'OPP_FG3M_C_EWA_tm',
 'OPP_FG3A_C_EWA_tm',
 'OPP_FG3M_F_EWA_tm',
 'OPP_FG3A_F_EWA_tm',
 'OPP_FG3M_G_EWA_tm',
 'OPP_FG3A_G_EWA_tm',
 'OPP_FTM_C_EWA_tm',
 'OPP_FTM_F_EWA_tm',
 'OPP_FTM_G_EWA_tm',
 'OPP_FTA_C_EWA_tm',
 'OPP_FTA_F_EWA_tm',
 'OPP_FTA_G_EWA_tm',
 'PTS_OFF_TOV_EWA_tm',
 'PTS_2ND_CHANCE_EWA_tm',
 'PTS_FB_EWA_tm',
 'PTS_PAINT_EWA_tm',
 'OPP_PTS_OFF_TOV_EWA_tm',
 'OPP_PTS_2ND_CHANCE_EWA_tm',
 'OPP_PTS_FB_EWA_tm',
 'OPP_PTS_PAINT_EWA_tm',
 'BLOCKS_EWA_tm',
 'BLOCKED_ATT_EWA_tm',
 'PF_EWA_tm',
 'PFD_EWA_tm',
 'FGM_RESTRICTED_EWA_tm',
 'FGA_RESTRICTED_EWA_tm',
 'FGM_PAINT_NON_RA_EWA_tm',
 'FGA_PAINT_NON_RA_EWA_tm',
 'FGM_MIDRANGE_EWA_tm',
 'FGA_MIDRANGE_EWA_tm',
 'FGM_CORNER3_EWA_tm',
 'FGA_CORNER3_EWA_tm',
 'FGM_ABOVE_BREAK3_EWA_tm',
 'FGA_ABOVE_BREAK3_EWA_tm',
 'FG2M_EWA_tm',
 'FG2A_EWA_tm',
 'PTS_2PT_MR_EWA_tm',
 'AST_2PM_EWA_tm',
 'AST_3PM_EWA_tm',
 'UAST_2PM_EWA_tm',
 'UAST_3PM_EWA_tm',
 'OPP_FG2M_G_EWA_tm',
 'OPP_FG2A_G_EWA_tm',
 'OPP_FG2M_F_EWA_tm',
 'OPP_FG2A_F_EWA_tm',
 'OPP_FG2M_C_EWA_tm',
 'OPP_FG2A_C_EWA_tm',
 'PTS_opp_EWA_tm',
 'FGM_opp_EWA_tm',
 'FGA_opp_EWA_tm',
 'FG3M_opp_EWA_tm',
 'FG3A_opp_EWA_tm',
 'FTM_opp_EWA_tm',
 'FTA_opp_EWA_tm',
 'OREB_opp_EWA_tm',
 'DREB_opp_EWA_tm',
 'REB_opp_EWA_tm',
 'AST_opp_EWA_tm',
 'TOV_opp_EWA_tm',
 'STL_opp_EWA_tm',
 'BLK_opp_EWA_tm',
 'PLUS_MINUS_opp_EWA_tm',
 'OFF_RATING_opp_EWA_tm',
 'DEF_RATING_opp_EWA_tm',
 'NET_RATING_opp_EWA_tm',
 'PACE_opp_EWA_tm',
 'PIE_opp_EWA_tm',
 'PCT_PTS_2PT_MR_opp_EWA_tm',
 'PCT_AST_2PM_opp_EWA_tm',
 'PCT_UAST_2PM_opp_EWA_tm',
 'PCT_AST_3PM_opp_EWA_tm',
 'PCT_UAST_3PM_opp_EWA_tm',
 'DEFENSIVE_BOX_OUTS_opp_EWA_tm',
 'CONTESTED_SHOTS_2PT_opp_EWA_tm',
 'CONTESTED_SHOTS_3PT_opp_EWA_tm',
 'deflections_opp_EWA_tm',
 'DIST_opp_EWA_tm',
 'TCHS_opp_EWA_tm',
 'PASS_opp_EWA_tm',
 'CFGM_opp_EWA_tm',
 'CFGA_opp_EWA_tm',
 'UFGM_opp_EWA_tm',
 'UFGA_opp_EWA_tm',
 'DFGM_opp_EWA_tm',
 'DFGA_opp_EWA_tm',
 'OPP_FG3M_C_opp_EWA_tm',
 'OPP_FG3A_C_opp_EWA_tm',
 'OPP_FG3M_F_opp_EWA_tm',
 'OPP_FG3A_F_opp_EWA_tm',
 'OPP_FG3M_G_opp_EWA_tm',
 'OPP_FG3A_G_opp_EWA_tm',
 'OPP_FTM_C_opp_EWA_tm',
 'OPP_FTM_F_opp_EWA_tm',
 'OPP_FTM_G_opp_EWA_tm',
 'OPP_FTA_C_opp_EWA_tm',
 'OPP_FTA_F_opp_EWA_tm',
 'OPP_FTA_G_opp_EWA_tm',
 'PTS_OFF_TOV_opp_EWA_tm',
 'PTS_2ND_CHANCE_opp_EWA_tm',
 'PTS_FB_opp_EWA_tm',
 'PTS_PAINT_opp_EWA_tm',
 'OPP_PTS_OFF_TOV_opp_EWA_tm',
 'OPP_PTS_2ND_CHANCE_opp_EWA_tm',
 'OPP_PTS_FB_opp_EWA_tm',
 'OPP_PTS_PAINT_opp_EWA_tm',
 'BLOCKS_opp_EWA_tm',
 'BLOCKED_ATT_opp_EWA_tm',
 'PF_opp_EWA_tm',
 'PFD_opp_EWA_tm',
 'FGM_RESTRICTED_opp_EWA_tm',
 'FGA_RESTRICTED_opp_EWA_tm',
 'FGM_PAINT_NON_RA_opp_EWA_tm',
 'FGA_PAINT_NON_RA_opp_EWA_tm',
 'FGM_MIDRANGE_opp_EWA_tm',
 'FGA_MIDRANGE_opp_EWA_tm',
 'FGM_CORNER3_opp_EWA_tm',
 'FGA_CORNER3_opp_EWA_tm',
 'FGM_ABOVE_BREAK3_opp_EWA_tm',
 'FGA_ABOVE_BREAK3_opp_EWA_tm',
 'FG2M_opp_EWA_tm',
 'FG2A_opp_EWA_tm',
 'PTS_2PT_MR_opp_EWA_tm',
 'AST_2PM_opp_EWA_tm',
 'AST_3PM_opp_EWA_tm',
 'UAST_2PM_opp_EWA_tm',
 'UAST_3PM_opp_EWA_tm',
 'OPP_FG2M_G_opp_EWA_tm',
 'OPP_FG2A_G_opp_EWA_tm',
 'OPP_FG2M_F_opp_EWA_tm',
 'OPP_FG2A_F_opp_EWA_tm',
 'OPP_FG2M_C_opp_EWA_tm',
 'OPP_FG2A_C_opp_EWA_tm',
 'PTS_EWA_opp',
 'FGM_EWA_opp',
 'FGA_EWA_opp',
 'FG3M_EWA_opp',
 'FG3A_EWA_opp',
 'FTM_EWA_opp',
 'FTA_EWA_opp',
 'OREB_EWA_opp',
 'DREB_EWA_opp',
 'REB_EWA_opp',
 'AST_EWA_opp',
 'TOV_EWA_opp',
 'STL_EWA_opp',
 'BLK_EWA_opp',
 'PLUS_MINUS_EWA_opp',
 'OFF_RATING_EWA_opp',
 'DEF_RATING_EWA_opp',
 'NET_RATING_EWA_opp',
 'PACE_EWA_opp',
 'PIE_EWA_opp',
 'PCT_PTS_2PT_MR_EWA_opp',
 'PCT_AST_2PM_EWA_opp',
 'PCT_UAST_2PM_EWA_opp',
 'PCT_AST_3PM_EWA_opp',
 'PCT_UAST_3PM_EWA_opp',
 'DEFENSIVE_BOX_OUTS_EWA_opp',
 'CONTESTED_SHOTS_2PT_EWA_opp',
 'CONTESTED_SHOTS_3PT_EWA_opp',
 'deflections_EWA_opp',
 'DIST_EWA_opp',
 'TCHS_EWA_opp',
 'PASS_EWA_opp',
 'CFGM_EWA_opp',
 'CFGA_EWA_opp',
 'UFGM_EWA_opp',
 'UFGA_EWA_opp',
 'DFGM_EWA_opp',
 'DFGA_EWA_opp',
 'OPP_FG3M_C_EWA_opp',
 'OPP_FG3A_C_EWA_opp',
 'OPP_FG3M_F_EWA_opp',
 'OPP_FG3A_F_EWA_opp',
 'OPP_FG3M_G_EWA_opp',
 'OPP_FG3A_G_EWA_opp',
 'OPP_FTM_C_EWA_opp',
 'OPP_FTM_F_EWA_opp',
 'OPP_FTM_G_EWA_opp',
 'OPP_FTA_C_EWA_opp',
 'OPP_FTA_F_EWA_opp',
 'OPP_FTA_G_EWA_opp',
 'PTS_OFF_TOV_EWA_opp',
 'PTS_2ND_CHANCE_EWA_opp',
 'PTS_FB_EWA_opp',
 'PTS_PAINT_EWA_opp',
 'OPP_PTS_OFF_TOV_EWA_opp',
 'OPP_PTS_2ND_CHANCE_EWA_opp',
 'OPP_PTS_FB_EWA_opp',
 'OPP_PTS_PAINT_EWA_opp',
 'BLOCKS_EWA_opp',
 'BLOCKED_ATT_EWA_opp',
 'PF_EWA_opp',
 'PFD_EWA_opp',
 'FGM_RESTRICTED_EWA_opp',
 'FGA_RESTRICTED_EWA_opp',
 'FGM_PAINT_NON_RA_EWA_opp',
 'FGA_PAINT_NON_RA_EWA_opp',
 'FGM_MIDRANGE_EWA_opp',
 'FGA_MIDRANGE_EWA_opp',
 'FGM_CORNER3_EWA_opp',
 'FGA_CORNER3_EWA_opp',
 'FGM_ABOVE_BREAK3_EWA_opp',
 'FGA_ABOVE_BREAK3_EWA_opp',
 'FG2M_EWA_opp',
 'FG2A_EWA_opp',
 'PTS_2PT_MR_EWA_opp',
 'AST_2PM_EWA_opp',
 'AST_3PM_EWA_opp',
 'UAST_2PM_EWA_opp',
 'UAST_3PM_EWA_opp',
 'OPP_FG2M_G_EWA_opp',
 'OPP_FG2A_G_EWA_opp',
 'OPP_FG2M_F_EWA_opp',
 'OPP_FG2A_F_EWA_opp',
 'OPP_FG2M_C_EWA_opp',
 'OPP_FG2A_C_EWA_opp',
 'PTS_opp_EWA_opp',
 'FGM_opp_EWA_opp',
 'FGA_opp_EWA_opp',
 'FG3M_opp_EWA_opp',
 'FG3A_opp_EWA_opp',
 'FTM_opp_EWA_opp',
 'FTA_opp_EWA_opp',
 'OREB_opp_EWA_opp',
 'DREB_opp_EWA_opp',
 'REB_opp_EWA_opp',
 'AST_opp_EWA_opp',
 'TOV_opp_EWA_opp',
 'STL_opp_EWA_opp',
 'BLK_opp_EWA_opp',
 'PLUS_MINUS_opp_EWA_opp',
 'OFF_RATING_opp_EWA_opp',
 'DEF_RATING_opp_EWA_opp',
 'NET_RATING_opp_EWA_opp',
 'PACE_opp_EWA_opp',
 'PIE_opp_EWA_opp',
 'PCT_PTS_2PT_MR_opp_EWA_opp',
 'PCT_AST_2PM_opp_EWA_opp',
 'PCT_UAST_2PM_opp_EWA_opp',
 'PCT_AST_3PM_opp_EWA_opp',
 'PCT_UAST_3PM_opp_EWA_opp',
 'DEFENSIVE_BOX_OUTS_opp_EWA_opp',
 'CONTESTED_SHOTS_2PT_opp_EWA_opp',
 'CONTESTED_SHOTS_3PT_opp_EWA_opp',
 'deflections_opp_EWA_opp',
 'DIST_opp_EWA_opp',
 'TCHS_opp_EWA_opp',
 'PASS_opp_EWA_opp',
 'CFGM_opp_EWA_opp',
 'CFGA_opp_EWA_opp',
 'UFGM_opp_EWA_opp',
 'UFGA_opp_EWA_opp',
 'DFGM_opp_EWA_opp',
 'DFGA_opp_EWA_opp',
 'OPP_FG3M_C_opp_EWA_opp',
 'OPP_FG3A_C_opp_EWA_opp',
 'OPP_FG3M_F_opp_EWA_opp',
 'OPP_FG3A_F_opp_EWA_opp',
 'OPP_FG3M_G_opp_EWA_opp',
 'OPP_FG3A_G_opp_EWA_opp',
 'OPP_FTM_C_opp_EWA_opp',
 'OPP_FTM_F_opp_EWA_opp',
 'OPP_FTM_G_opp_EWA_opp',
 'OPP_FTA_C_opp_EWA_opp',
 'OPP_FTA_F_opp_EWA_opp',
 'OPP_FTA_G_opp_EWA_opp',
 'PTS_OFF_TOV_opp_EWA_opp',
 'PTS_2ND_CHANCE_opp_EWA_opp',
 'PTS_FB_opp_EWA_opp',
 'PTS_PAINT_opp_EWA_opp',
 'OPP_PTS_OFF_TOV_opp_EWA_opp',
 'OPP_PTS_2ND_CHANCE_opp_EWA_opp',
 'OPP_PTS_FB_opp_EWA_opp',
 'OPP_PTS_PAINT_opp_EWA_opp',
 'BLOCKS_opp_EWA_opp',
 'BLOCKED_ATT_opp_EWA_opp',
 'PF_opp_EWA_opp',
 'PFD_opp_EWA_opp',
 'FGM_RESTRICTED_opp_EWA_opp',
 'FGA_RESTRICTED_opp_EWA_opp',
 'FGM_PAINT_NON_RA_opp_EWA_opp',
 'FGA_PAINT_NON_RA_opp_EWA_opp',
 'FGM_MIDRANGE_opp_EWA_opp',
 'FGA_MIDRANGE_opp_EWA_opp',
 'FGM_CORNER3_opp_EWA_opp',
 'FGA_CORNER3_opp_EWA_opp',
 'FGM_ABOVE_BREAK3_opp_EWA_opp',
 'FGA_ABOVE_BREAK3_opp_EWA_opp',
 'FG2M_opp_EWA_opp',
 'FG2A_opp_EWA_opp',
 'PTS_2PT_MR_opp_EWA_opp',
 'AST_2PM_opp_EWA_opp',
 'AST_3PM_opp_EWA_opp',
 'UAST_2PM_opp_EWA_opp',
 'UAST_3PM_opp_EWA_opp',
 'OPP_FG2M_G_opp_EWA_opp',
 'OPP_FG2A_G_opp_EWA_opp',
 'OPP_FG2M_F_opp_EWA_opp',
 'OPP_FG2A_F_opp_EWA_opp',
 'OPP_FG2M_C_opp_EWA_opp',
 'OPP_FG2A_C_opp_EWA_opp',
 'IS_HOME_tm',
 'DAYS_REST_tm',
 'DAYS_REST_opp']]
            
            
            
#             ['PTS_EWA_tm', 'FGM_EWA_tm', 'FGA_EWA_tm', 'FG3M_EWA_tm', 'FG3A_EWA_tm', 'FTM_EWA_tm', 
#                          'FTA_EWA_tm', 'OREB_EWA_tm', 'DREB_EWA_tm', 'REB_EWA_tm', 'AST_EWA_tm', 'TOV_EWA_tm', 
#                          'STL_EWA_tm', 'BLK_EWA_tm', 'PLUS_MINUS_EWA_tm', 'OFF_RATING_EWA_tm', 'DEF_RATING_EWA_tm', 
#                          'NET_RATING_EWA_tm', 'PACE_EWA_tm', 'PIE_EWA_tm', 'PCT_PTS_2PT_MR_EWA_tm', 'PCT_AST_2PM_EWA_tm', 
#                          'PCT_UAST_2PM_EWA_tm', 'PCT_AST_3PM_EWA_tm', 'PCT_UAST_3PM_EWA_tm', 'DEFENSIVE_BOX_OUTS_EWA_tm',
#                          'CONTESTED_SHOTS_2PT_EWA_tm', 'CONTESTED_SHOTS_3PT_EWA_tm', 'deflections_EWA_tm', 'DIST_EWA_tm', 
#                          'TCHS_EWA_tm', 'PASS_EWA_tm', 'CFGM_EWA_tm', 'CFGA_EWA_tm', 'UFGM_EWA_tm', 'UFGA_EWA_tm', 'DFGM_EWA_tm',
#                          'DFGA_EWA_tm', 'OPP_FG3M_C_EWA_tm', 'OPP_FG3A_C_EWA_tm', 'OPP_FG3M_F_EWA_tm', 'OPP_FG3A_F_EWA_tm', 
#                          'OPP_FG3M_G_EWA_tm', 'OPP_FG3A_G_EWA_tm', 'OPP_FTM_C_EWA_tm', 'OPP_FTM_F_EWA_tm', 'OPP_FTM_G_EWA_tm',
#                          'OPP_FTA_C_EWA_tm', 'OPP_FTA_F_EWA_tm', 'OPP_FTA_G_EWA_tm', 'PTS_OFF_TOV_EWA_tm', 'PTS_2ND_CHANCE_EWA_tm',
#                          'PTS_FB_EWA_tm', 'PTS_PAINT_EWA_tm', 'OPP_PTS_OFF_TOV_EWA_tm', 'OPP_PTS_2ND_CHANCE_EWA_tm', 'OPP_PTS_FB_EWA_tm', 
#                          'OPP_PTS_PAINT_EWA_tm', 'BLOCKS_EWA_tm', 'BLOCKED_ATT_EWA_tm', 'PF_EWA_tm', 'PFD_EWA_tm', 'FGM_RESTRICTED_EWA_tm',
#                          'FGA_RESTRICTED_EWA_tm', 'FGM_PAINT_NON_RA_EWA_tm', 'FGA_PAINT_NON_RA_EWA_tm', 'FGM_MIDRANGE_EWA_tm', 
#                          'FGA_MIDRANGE_EWA_tm', 'FGM_CORNER3_EWA_tm', 'FGA_CORNER3_EWA_tm', 'FGM_ABOVE_BREAK3_EWA_tm',
#                          'FGA_ABOVE_BREAK3_EWA_tm', 'FG2M_EWA_tm', 'FG2A_EWA_tm', 'PTS_2PT_MR_EWA_tm', 'AST_2PM_EWA_tm', 
#                          'AST_3PM_EWA_tm', 'UAST_2PM_EWA_tm', 'UAST_3PM_EWA_tm', 'OPP_FG2M_G_EWA_tm', 'OPP_FG2A_G_EWA_tm', 
#                          'OPP_FG2M_F_EWA_tm', 'OPP_FG2A_F_EWA_tm', 'OPP_FG2M_C_EWA_tm', 'OPP_FG2A_C_EWA_tm', 'PTS_opp_EWA_tm', 
#                          'FGM_opp_EWA_tm', 'FGA_opp_EWA_tm', 'FG3M_opp_EWA_tm', 'FG3A_opp_EWA_tm', 'FTM_opp_EWA_tm', 'FTA_opp_EWA_tm',
#                          'OREB_opp_EWA_tm', 'DREB_opp_EWA_tm', 'REB_opp_EWA_tm', 'AST_opp_EWA_tm', 'TOV_opp_EWA_tm', 'STL_opp_EWA_tm',
#                          'BLK_opp_EWA_tm', 'PLUS_MINUS_opp_EWA_tm', 'OFF_RATING_opp_EWA_tm', 'DEF_RATING_opp_EWA_tm', 'NET_RATING_opp_EWA_tm',
#                          'PACE_opp_EWA_tm', 'PIE_opp_EWA_tm', 'PCT_PTS_2PT_MR_opp_EWA_tm', 'PCT_AST_2PM_opp_EWA_tm', 'PCT_UAST_2PM_opp_EWA_tm',
#                          'PCT_AST_3PM_opp_EWA_tm', 'PCT_UAST_3PM_opp_EWA_tm', 'DEFENSIVE_BOX_OUTS_opp_EWA_tm', 'CONTESTED_SHOTS_2PT_opp_EWA_tm',
#                          'CONTESTED_SHOTS_3PT_opp_EWA_tm', 'deflections_opp_EWA_tm', 'DIST_opp_EWA_tm', 'TCHS_opp_EWA_tm', 'PASS_opp_EWA_tm',
#                          'CFGM_opp_EWA_tm', 'CFGA_opp_EWA_tm', 'UFGM_opp_EWA_tm', 'UFGA_opp_EWA_tm', 'DFGM_opp_EWA_tm', 'DFGA_opp_EWA_tm', 
#                          'OPP_FG3M_C_opp_EWA_tm', 'OPP_FG3A_C_opp_EWA_tm', 'OPP_FG3M_F_opp_EWA_tm', 'OPP_FG3A_F_opp_EWA_tm', 'OPP_FG3M_G_opp_EWA_tm',
#                          'OPP_FG3A_G_opp_EWA_tm', 'OPP_FTM_C_opp_EWA_tm', 'OPP_FTM_F_opp_EWA_tm', 'OPP_FTM_G_opp_EWA_tm', 'OPP_FTA_C_opp_EWA_tm',
#                          'OPP_FTA_F_opp_EWA_tm', 'OPP_FTA_G_opp_EWA_tm', 'PTS_OFF_TOV_opp_EWA_tm', 'PTS_2ND_CHANCE_opp_EWA_tm', 'PTS_FB_opp_EWA_tm',
#                          'PTS_PAINT_opp_EWA_tm', 'OPP_PTS_OFF_TOV_opp_EWA_tm', 'OPP_PTS_2ND_CHANCE_opp_EWA_tm', 'OPP_PTS_FB_opp_EWA_tm',
#                          'OPP_PTS_PAINT_opp_EWA_tm', 'BLOCKS_opp_EWA_tm', 'BLOCKED_ATT_opp_EWA_tm', 'PF_opp_EWA_tm', 'PFD_opp_EWA_tm', 
#                          'FGM_RESTRICTED_opp_EWA_tm', 'FGA_RESTRICTED_opp_EWA_tm', 'FGM_PAINT_NON_RA_opp_EWA_tm', 'FGA_PAINT_NON_RA_opp_EWA_tm',
#                          'FGM_MIDRANGE_opp_EWA_tm', 'FGA_MIDRANGE_opp_EWA_tm', 'FGM_CORNER3_opp_EWA_tm', 'FGA_CORNER3_opp_EWA_tm',
#                          'FGM_ABOVE_BREAK3_opp_EWA_tm', 'FGA_ABOVE_BREAK3_opp_EWA_tm', 'FG2M_opp_EWA_tm', 'FG2A_opp_EWA_tm',
#                          'PTS_2PT_MR_opp_EWA_tm', 'AST_2PM_opp_EWA_tm', 'AST_3PM_opp_EWA_tm', 'UAST_2PM_opp_EWA_tm', 'UAST_3PM_opp_EWA_tm', 
#                          'OPP_FG2M_G_opp_EWA_tm', 'OPP_FG2A_G_opp_EWA_tm', 'OPP_FG2M_F_opp_EWA_tm', 'OPP_FG2A_F_opp_EWA_tm',
#                          'OPP_FG2M_C_opp_EWA_tm', 'OPP_FG2A_C_opp_EWA_tm', 'FG_PCT_RESTRICTED_EWA_tm', 'AST_SCORING_DIFF', 'EFF_SCORING',
#                          'POSS_EFFICIENCY', 'DEF_PRESSURE_SCORE', 'TRANSITION_DEF_EFF', 'OREB_SCORING_IMPACT', 
#                          'BALL_MOVEMENT_EFF', 'ASSISTED_SCORING_EFF', 'PAINT_DEF_EFF']
# ]
        
        return df_new


    def get_todays_games(self):
        """
        Get today's NBA games using ScoreboardV2
        
        Returns:
        pd.DataFrame: DataFrame containing game information
        """
        # NBA Team ID to abbreviation mapping
        NBA_TEAMS = {1610612737: 'ATL',
                    1610612751: 'BKN',
                    1610612738: 'BOS',
                    1610612766: 'CHA',
                    1610612741: 'CHI',
                    1610612739: 'CLE',
                    1610612742: 'DAL',
                    1610612743: 'DEN',
                    1610612765: 'DET',
                    1610612744: 'GSW',
                    1610612745: 'HOU',
                    1610612754: 'IND',
                    1610612746: 'LAC',
                    1610612747: 'LAL',
                    1610612763: 'MEM',
                    1610612748: 'MIA',
                    1610612749: 'MIL',
                    1610612750: 'MIN',
                    1610612740: 'NOP',
                    1610612752: 'NYK',
                    1610612760: 'OKC',
                    1610612753: 'ORL',
                    1610612755: 'PHI',
                    1610612756: 'PHX',
                    1610612757: 'POR',
                    1610612758: 'SAC',
                    1610612759: 'SAS',
                    1610612761: 'TOR',
                    1610612762: 'UTA',
                    1610612764: 'WAS'}
        
        try:
            # Get today's date in the format MM/DD/YYYY
            today = datetime.now().strftime('%Y/%m/%d')        
            # Get scoreboard for today
            games = scoreboardv2.ScoreboardV2(game_date=today).game_header.get_data_frame()
            

            
            # Extract relevant columns
            game_info = games[[
                'SEASON',
                'GAME_DATE_EST',
                'GAME_ID', 
                'GAME_STATUS_ID',  # 1: Scheduled, 2: In Progress, 3: Final
                'GAME_STATUS_TEXT',
                'HOME_TEAM_ID',
                'VISITOR_TEAM_ID',
            ]]
            
            # Add team abbreviations
            game_info['HOME_TEAM'] = game_info['HOME_TEAM_ID'].map(NBA_TEAMS)
            game_info['AWAY_TEAM'] = game_info['VISITOR_TEAM_ID'].map(NBA_TEAMS)
            
            return game_info
            
        except Exception as e:
            print(f"Error fetching games: {str(e)}")
            return None
        
    def get_todays_odds(self) -> pd.DataFrame:
        """
        Get NBA odds data for specified date
        
        Parameters:
        date (str): Date in YYYY-MM-DD format
        sportsbook (str): Name of sportsbook to get odds from, defaults to 'draftkings'
        
        Returns:
        pd.DataFrame: DataFrame containing odds information
        """
        try:
            today = datetime.now().strftime('%Y/%m/%d')        

            df_data = []
            sb = Scoreboard(sport='NBA', date=today)
            
            team_abbrev_dict = {
                'Atlanta Hawks': 'ATL',
                'Boston Celtics': 'BOS',
                'Brooklyn Nets': 'BKN', 
                'Charlotte Hornets': 'CHA',
                'Chicago Bulls': 'CHI',
                'Cleveland Cavaliers': 'CLE',
                'Dallas Mavericks': 'DAL',
                'Denver Nuggets': 'DEN',
                'Detroit Pistons': 'DET',
                'Golden State Warriors': 'GSW',
                'Houston Rockets': 'HOU',
                'Indiana Pacers': 'IND',
                'Los Angeles Clippers': 'LAC',
                'Los Angeles Lakers': 'LAL',
                'Memphis Grizzlies': 'MEM',
                'Miami Heat': 'MIA',
                'Milwaukee Bucks': 'MIL',
                'Minnesota Timberwolves': 'MIN',
                'New Orleans Pelicans': 'NOP',
                'New York Knicks': 'NYK',
                'Oklahoma City Thunder': 'OKC',
                'Orlando Magic': 'ORL',
                'Philadelphia 76ers': 'PHI',
                'Phoenix Suns': 'PHX',
                'Portland Trail Blazers': 'POR',
                'Sacramento Kings': 'SAC',
                'San Antonio Spurs': 'SAS',
                'Toronto Raptors': 'TOR',
                'Utah Jazz': 'UTA',
                'Washington Wizards': 'WAS'
            }

            for game in sb.games:
                try:
                    df_data.append({
                        'away_team': game['away_team'],
                        'home_team': game['home_team'],
                        'away_spread': game['away_spread']['draftkings'],
                        'home_spread': game['home_spread']['draftkings'], 
                        'OU': game['total']['draftkings'],
                        'home_moneyline': game['home_ml']['draftkings'],
                        'away_moneyline': game['away_ml']['draftkings'],
                        'game_date': today,
                    })
                except KeyError as e:
                    logging.warning(f"Missing odds data for game: {game}. Error: {str(e)}")
                    continue

            df = pd.DataFrame(df_data)
            
            # Convert moneyline odds to decimal format
            df['away_moneyline'] = convert_american_to_decimal(df['away_moneyline'].astype(int))
            df['home_moneyline'] = convert_american_to_decimal(df['home_moneyline'].astype(int))
            
            # Convert team names to abbreviations
            df['home_team'] = df['home_team'].replace(team_abbrev_dict)
            df['away_team'] = df['away_team'].replace(team_abbrev_dict)

            return df

        except Exception as e:
            logging.error(f"Error getting odds data: {str(e)}")
            return pd.DataFrame()      
          

    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Make predictions for both teams in a game
        
        Parameters:
        home_team_id (int): ID of home team
        away_team_id (int): ID of away team
        
        Returns:
        Dict: Prediction results including predicted scores and win probabilities
        """
        try:
            # Prepare home team features
            home_features = self.prepare_team_features(home_team_id, away_team_id)
            # home_features = self.engineer_features(home_features)
            if home_features is None:
                return None
            
            # Prepare away team features (flip IS_HOME)
            away_features = self.prepare_team_features(away_team_id, home_team_id)
            # away_features = self.engineer_features(away_features)
            if away_features is None:
                return None
            # away_features['IS_HOME'] = 0
            
            # Make predictions
            home_score = self.model.predict(home_features)[0]
            away_score = self.model.predict(away_features)[0]
            
            # Calculate win probability (optional - if your model supports it)
            # win_prob = self.calculate_win_probability(home_score, away_score)
            
            return {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'predicted_home_score': round(home_score, 1),
                'predicted_away_score': round(away_score, 1),
                'predicted_margin': round(home_score - away_score, 1),
                # 'win_probability': win_prob,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error predicting game: {str(e)}")
            return None

    def predict_todays_games(self) -> List[Dict]:
        """
        Get and predict all of today's games
        
        Returns:
        List[Dict]: List of prediction results for each game
        """
        try:
            # Get today's games
            games_df = self.get_todays_games()
            odds_df = self.get_todays_odds()

            if games_df is None or games_df.empty:
                logging.info("No games found for today")
                return []
            
            predictions = []
            for _, game in games_df.iterrows():
                try:
                    home_team_id = game['HOME_TEAM_ID']
                    away_team_id = game['VISITOR_TEAM_ID']
                    home_team = game['HOME_TEAM']
                    away_team = game['AWAY_TEAM']
                    
                    # Get odds for this game
                    game_odds = odds_df[
                        (odds_df['home_team'] == home_team) & 
                        (odds_df['away_team'] == away_team)
                    ]
                    
                    prediction = self.predict_game(home_team_id, away_team_id)
                    if prediction:
                        prediction['game_id'] = game['GAME_ID']
                        prediction['game_date'] = game['GAME_DATE_EST']
                        prediction['home_team'] = home_team
                        prediction['away_team'] = away_team
                        predictions.append(prediction)
                        
                        # Add odds information if available
                        if not game_odds.empty:
                            prediction['home_spread'] = float(game_odds['home_spread'].iloc[0])
                            prediction['away_spread'] = float(game_odds['away_spread'].iloc[0])
                            prediction['home_moneyline'] = float(game_odds['home_moneyline'].iloc[0])
                            prediction['away_moneyline'] = float(game_odds['away_moneyline'].iloc[0])
                            prediction['total_line'] = float(game_odds['OU'].iloc[0])
                        
                        predictions.append(prediction)
                        
                        
                except Exception as e:
                    logging.error(f"Error processing game: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error predicting today's games: {str(e)}")
            return []

    def store_predictions(self, predictions: List[Dict]) -> None:
        """Store predictions in database"""
        if not predictions:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                predictions_df = pd.DataFrame(predictions)
                predictions_df.to_sql('game_predictions', conn, if_exists='append', index=False)
                logging.info(f"Stored {len(predictions)} predictions")
                
        except Exception as e:
            logging.error(f"Error storing predictions: {str(e)}")

# Example usage:
if __name__ == "__main__":
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    model_path = Path.cwd() / 'models' / 'AutogluonModels' / 'ag-20241116_093710'
    
    
    season = ['2024-25']
    
    syncer = NBADatabaseSync(db_path)

    # Regular sync will now include deduplication

    syncer.sync_database(season)

    predictor = NBAPredictionPipeline(db_path = db_path, model_path = model_path)

    
    # Get and store predictions
    predictions = predictor.predict_todays_games()
    

    
    if predictions:
        print("\nPredictions for today's games:")
        for pred in predictions:
            print(f"\nGame ID: {pred['game_id']}")
            print(f"Matchup: HOME: {pred['home_team']} vs AWAY: {pred['away_team']}")
            print(f"Predicted Score: {pred['predicted_home_score']} - {pred['predicted_away_score']}")
            print(f"Predicted Margin: {pred['predicted_margin']}")
            print(f"Spread: {pred.get('home_spread', 'N/A')} (Home)")
            print(f"Moneyline: Home {pred.get('home_moneyline', 'N/A')} / Away {pred.get('away_moneyline', 'N/A')}")
            print(f"Total Line: {pred.get('total_line', 'N/A')}")
    else:
        print("No predictions generated for today")
        
        
    print(predictions)