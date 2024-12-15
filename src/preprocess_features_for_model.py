import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from time import sleep
from filterpy.kalman import KalmanFilter
from typing import Dict, Tuple
from tqdm import tqdm

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

class NBAFeatureAggregator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.optimal_spans = team_optimal_spans  # Your existing optimal spans dictionary
        
    def fetch_raw_data(self) -> pd.DataFrame:
        """Fetch raw team stats from database"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(your_query, conn)  # Your existing query
        return df
    
    def aggregate_features(self) -> None:
        """Main pipeline to create and store aggregated features"""
        try:
            logging.info("Starting feature aggregation pipeline")
            
            # 1. Fetch raw data
            df_raw = self.fetch_raw_data()
            logging.info(f"Fetched {len(df_raw)} raw records")
            
            # 2. Clean and prepare data
            df_clean = clean_team_data(df_raw)
            df_prep = prep_for_aggregation_team(df_clean)
            
            # 3. Create matchups
            matchups = create_matchups(df_prep)
            
            # 4. Apply EWA calculations
            df_with_ewa = apply_optimal_spans(matchups, self.optimal_spans, 'TEAM_ID')
            
            # 5. Add percentage features
            df_with_pct = add_percentage_features(df_with_ewa)
            
            # 6. Select final columns
            keep_cols = ['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 
                        'GAME_DATE', 'MATCHUP', 'WL', 'PTS']
            ewa_cols = [col for col in df_with_pct.columns if "EWA" in col]
            keep_cols.extend(ewa_cols)
            
            df_final = df_with_pct[keep_cols].copy()
            
            # 7. Add schedule features
            team_final_df = add_schedule_features(df_final)
            
            # 8. Store in database
            self.store_aggregated_features(team_final_df)
            
            logging.info("Feature aggregation completed successfully")
            
        except Exception as e:
            logging.error(f"Error in feature aggregation pipeline: {str(e)}")
            raise
    
    def store_aggregated_features(self, df: pd.DataFrame) -> None:
        """Store aggregated features in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store main aggregated features
                df.to_sql('team_aggregated_stats', conn, if_exists='replace', index=False)
                
                # Create a view for latest stats per team
                create_view_sql = """
                CREATE OR REPLACE VIEW team_latest_stats AS
                WITH RankedStats AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY TEAM_ID 
                                         ORDER BY GAME_DATE DESC) as rn
                    FROM team_aggregated_stats
                )
                SELECT 
                    TEAM_ID,
                    TEAM_ABBREVIATION,
                    GAME_DATE as LAST_GAME_DATE,
                    WL as LAST_RESULT,
                    PTS as LAST_POINTS,
                    -- Include all EWA columns
                    -- They will be used for predictions
                    * EXCLUDE(rn, TEAM_ID, TEAM_ABBREVIATION, 
                            GAME_DATE, WL, PTS, SEASON, 
                            GAME_ID, MATCHUP)
                FROM RankedStats
                WHERE rn = 1
                """
                
                conn.execute(create_view_sql)
                conn.commit()
                
                logging.info("Stored aggregated features and created latest stats view")
                
        except Exception as e:
            logging.error(f"Error storing aggregated features: {str(e)}")
            raise

# Usage
if __name__ == "__main__":
    aggregator = NBAFeatureAggregator('nba_stats.db')
    aggregator.aggregate_features()