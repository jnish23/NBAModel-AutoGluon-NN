TEAM_STATS_QUERY = """
SELECT
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
LEFT JOIN team_advanced_stats tas 
    ON tbs.TEAM_ID = tas.TEAM_ID AND tbs.GAME_ID = tas.GAME_ID
LEFT JOIN team_scoring_stats tss 
    ON tbs.TEAM_ID = tss.TEAM_ID AND tbs.GAME_ID = tss.GAME_ID
LEFT JOIN team_hustle_stats ths 
    ON tbs.TEAM_ID = ths.teamId AND tbs.GAME_ID = ths.gameId
LEFT JOIN team_track_stats tts 
    ON tbs.TEAM_ID = tts.TEAM_ID AND tbs.GAME_ID = tts.GAME_ID
LEFT JOIN defensive_stats_by_position dsbp 
    ON tbs.TEAM_ID = dsbp.TEAM_ID AND tbs.GAME_DATE = dsbp.GAME_DATE
LEFT JOIN team_miscellaneous_stats tms 
    ON tbs.TEAM_ID = tms.TEAM_ID AND tbs.GAME_ID = tms.GAME_ID
LEFT JOIN team_shot_location_boxscores tslb 
    ON tbs.TEAM_ID = tslb.TEAM_ID AND tbs.GAME_DATE = tslb.GAME_DATE
WHERE tbs.SEASON_YEAR BETWEEN '2016-17' AND '2024-25'
ORDER BY tbs.GAME_DATE, tbs.TEAM_ID
"""