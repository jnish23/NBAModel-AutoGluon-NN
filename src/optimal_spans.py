import json
from pathlib import Path

class OptimalSpans:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent / 'config' / 'optimal_spans.json'
        self.spans = self.load_spans()
    
    def load_spans(self) -> dict:
        """Load optimal spans from JSON config file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to default spans if file doesn't exist
            self.spans = self.default_spans()
            self.save_spans()  # Create the file for future use
            return self.spans
            
    def save_spans(self) -> None:
        """Save current spans to JSON file"""
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(self.spans, f, indent=2)
    
    def update_spans(self, new_spans: dict) -> None:
        """Update spans and save to file"""
        self.spans.update(new_spans)
        self.save_spans()
    
    @staticmethod
    def default_spans() -> dict:
        """Default optimal spans if no config file exists"""
        return {'PTS': 29,
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
                'OPP_FG2A_C_opp': 49}
    
    def get_span(self, feature: str) -> int:
        """Get optimal span for a feature"""
        return self.spans.get(feature, 20)  # Default to 20 if not found
