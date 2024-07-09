import numpy as np
import pandas as pd


### Load data ###

# Seeds
seeds_m = pd.read_csv('data_2023/MNCAATourneySeeds.csv').query('Season == 2023').reset_index(drop=True)
seeds_w = pd.read_csv('data_2023/WNCAATourneySeeds.csv').query('Season == 2023').reset_index(drop=True)

# Regular season results
box_scores_m = pd.read_csv('data_2023/MRegularSeasonDetailedResults.csv').query('Season == 2023').reset_index(drop=True)
box_scores_w = pd.read_csv('data_2023/WRegularSeasonDetailedResults.csv').query('Season == 2023').reset_index(drop=True)

# Conferences - only used in Women's model
team_conf_w = pd.read_csv('data_2023/WTeamConferences.csv')

# Public rankings - used only in Men's model
ratings_all = pd.read_csv('data_2023/MMasseyOrdinals_thru_Season2023_Day128.csv')

# Submission file
submission_file_start = pd.read_csv('data_2023/SampleSubmission2023.csv')


### Build symmetric data frames

# Function to build dataframes
def build_sym_df(df):
    winners = (df
        .rename(columns=lambda x: x.replace("W", ""))
        .rename(columns=lambda x: x.replace("L", "Opp")))
    
    losers = (df
        .rename(columns=lambda x: x.replace("L", ""))
        .rename(columns=lambda x: x.replace("W", "Opp")))
    
    winners["Win"] = 1
    losers["Win"] = 0

    full_data = pd.concat([winners, losers]).reset_index().drop(columns="index")
    
    # Since the L in "Loc" is intrepreted as the losing team identifier with the above logic:
    return full_data.rename(columns={"Oppoc": "Loc"})

# Create initial dataframes

box_scores_full_m = build_sym_df(box_scores_m)
box_scores_full_w = build_sym_df(box_scores_w)

# Fix box score location, since all locations are still winners location
def fix_loc(Win, Loc):
    if (Win == 0 and Loc == 'H'):
        return 'A'
    elif (Win == 0 and Loc == 'A'):
        return 'H'
    else:
        return 'N'

fix_loc2 = np.vectorize(fix_loc)

win_m = box_scores_full_m["Win"]
win_w = box_scores_full_w["Win"]
loc_m = box_scores_full_m["Loc"]
loc_w = box_scores_full_w["Loc"]

box_scores_full_m["Loc"] = fix_loc2(win_m, loc_m)
box_scores_full_w["Loc"] = fix_loc2(win_w, loc_w)


### Aggregate regular season data ###

# Sum box score numbers to get one record per team
box_score_sum_m = box_scores_full_m.groupby(["Season", "TeamID"]).sum().reset_index()
box_score_sum_w = box_scores_full_w.groupby(["Season", "TeamID"]).sum().reset_index()

# Games played - won't be the same because of conference tournaments
games_played_m = box_scores_full_m.groupby(["Season", "TeamID"]).agg(GP = ("Score", "count")).reset_index()
games_played_w = box_scores_full_w.groupby(["Season", "TeamID"]).agg(GP = ("Score", "count")).reset_index()

box_score_sum_m = box_score_sum_m.merge(games_played_m, on=["Season", "TeamID"])
box_score_sum_w = box_score_sum_w.merge(games_played_w, on=["Season", "TeamID"])

# Calculate metrics used for model
ff_plus_m = (box_score_sum_m
 .assign(
    Poss = lambda x: x.FGA - x.OR + x.TO + .475 * x.FTA,
    TO_pct = lambda x: x.TO / x.Poss,
    OR_pct = lambda x: x.OR / (x.OR + x.OppDR),
 )
 .filter(["Season", "TeamID", "TO_pct", "OR_pct"]))

ff_plus_w = (box_score_sum_w
 .assign(
    Poss = lambda x: x.FGA - x.OR + x.TO + .475 * x.FTA,
    Tempo = lambda x: x.Poss / x.GP,
    EFG_pct = lambda x: ((.5 * x.FGM3) + x.FGM) / x.FGA,
    TO_pct = lambda x: x.TO / x.Poss,
    OR_pct = lambda x: x.OR / (x.OR + x.OppDR),
    FTR = lambda x: x.FTA / x.FGM,
 )
 .filter(["Season", "TeamID", "Tempo", "EFG_pct", "TO_pct", "OR_pct", "FTR"]))


### Add seeds ###

box_scores_seeds_m = ff_plus_m.merge(seeds_m, on=["Season", "TeamID"])
box_scores_seeds_w = ff_plus_w.merge(seeds_w, on=["Season", "TeamID"])

box_scores_seeds_m["Seed"] = box_scores_seeds_m["Seed"].str.extract('([0-9]+)').astype(int)
box_scores_seeds_w["Seed"] = box_scores_seeds_w["Seed"].str.extract('([0-9]+)').astype(int)


### Add tournament specific data

# Mens - KenPom rankings
kenpom = (ratings_all
 .query("SystemName == 'POM' and RankingDayNum == 128 and Season == 2023")
 .filter(["Season", "TeamID", "OrdinalRank"])
 .reset_index(drop=True)
 .rename(columns={"OrdinalRank": "KPRank"}))

full_data_m = box_scores_seeds_m.merge(kenpom, on=["Season", "TeamID"])

# Womens - Conference bids
conf_seeds_w = seeds_w.merge(team_conf_w, on=["Season", "TeamID"]).drop(columns="Seed")

conf_bids_w = (conf_seeds_w
 .groupby(["Season", "ConfAbbrev"])["TeamID"]
 .count()
 .reset_index()
 .rename(columns={"TeamID": "ConfBids"}))

full_data_w = (box_scores_seeds_w
 .merge(conf_seeds_w, on=["Season", "TeamID"])
 .merge(conf_bids_w, on=["Season", "ConfAbbrev"])
 .drop(columns=["ConfAbbrev"])
)

### Create final prediction data using submission file

# Initial submission file processing
submission_file_start[["Season", "TeamID", "OppTeamID"]] = submission_file_start["ID"].str.split('_', expand=True)
submission_file = submission_file_start.drop(columns='Pred').set_index('ID')
submission_file = submission_file.astype('int64')

# Break into Men's and Women's to be combined later
sub_m = submission_file.query('TeamID < 2000')
sub_w = submission_file.query('TeamID > 2000')

# Final Men's data
prediction_data_m = (sub_m
 .reset_index()
 .merge(full_data_m, on=["Season", "TeamID"], how='left')
 .merge(full_data_m.rename(columns={"TeamID":"OppTeamID"}), 
        on=["Season", "OppTeamID"], 
        how='left', suffixes=[None, 'Opp'])
 .query('Seed.notna() and SeedOpp.notna()', engine='python')
 .assign(
   TO = lambda x: x.TO_pct - x.TO_pctOpp,
   OR = lambda x: x.OR_pct - x.OR_pctOpp,
   KPDiff = lambda x: x.KPRank - x.KPRankOpp
 )
 .rename(columns={"SeedOpp": "OppSeed"})
 .filter(["ID", "Seed", "OppSeed", "TO", "OR", "KPDiff"])
 .set_index('ID')
)

# Final Women's Data
prediction_data_w = (sub_w
 .reset_index()
 .merge(full_data_w, on=["Season", "TeamID"], how='left')
 .merge(full_data_w.rename(columns={"TeamID":"OppTeamID"}), 
        on=["Season", "OppTeamID"], 
        how='left', suffixes=[None, 'Opp'])
 .query('Seed.notna() and SeedOpp.notna()', engine='python')
 .assign(
    Tempo = lambda x: x.Tempo - x.TempoOpp,
    EFG = lambda x: x.EFG_pct - x.EFG_pctOpp,
    TO = lambda x: x.TO_pct - x.TO_pctOpp,
    OR = lambda x: x.OR_pct - x.OR_pctOpp,
    FTR = lambda x: x.FTR - x.FTROpp
 )
 .rename(columns={"SeedOpp": "OppSeed"})
 .filter(["ID", "Seed", "OppSeed", "Tempo", "EFG", "TO", "OR", "FTR", "ConfBids"])
 .set_index('ID')
)

prediction_data_m.to_csv('input/prediction_data_m_2023.csv')
prediction_data_w.to_csv('input/prediction_data_w_2023.csv')
