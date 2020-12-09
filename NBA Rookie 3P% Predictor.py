import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

"""
Rookie 3P% Improvement Predictor.

Predicts an NBA player's increase in 3P% from their rookie season to their second season using their draft age,
draft position, and college/rookie season shooting stats. Also lists previous NBA players who have had
similar college and rookie shooting numbers to that player.

"""

# Import and read the data.
NBA_shooting_stats = pd.read_csv("3PT Prediction Model Training Stats.csv")  #Shooting stats from perimeter players drafted from 2010-2018.
rookies2019_20_stats = pd.read_csv("rookies.csv")  #Shooting stats from rookies drafted in 2019.

# The stat we are predicting for: 3P% improvement from first to second NBA season.
y = NBA_shooting_stats['3P% Improve']

# Fill in any blank entries with the mean value
y.fillna(y.mean(), inplace=True)

# The features used to predict 3P% improvement for the Random Forest model:
rf_predictors = ['Draft Position', 'Age', '3P% 1st Season', 'FT% 1st Season', '3P% College Improve',
              'Pre All Star - Post All Star 3P%', '3PA per 100 College', "3pr 1st season"]

# The features used to predict 3P% improvement for the XGB Boost model:
xgb_predictors = ['Draft Position', 'Age', '3P% 1st Season', 'FT% 1st Season', '3P% College Improve',
                  'Pre All Star - Post All Star 3P%', '3PA per 100 College', '3PA per 100 1st season', "FT difference", "3pr 1st season"]

# The shooting stats used to find similar historic shooters to a player.
similarity_stat_predictors = ["3P% 1st Season", "3P% College", "3PA per 100 1st season", "3pr 1st season", "FT% 1st Season"]

# Get the features from our dataframes.
similarity_X = NBA_shooting_stats[similarity_stat_predictors]
rf_X = NBA_shooting_stats[rf_predictors]
xgb_X = NBA_shooting_stats[xgb_predictors]

# Fill in any blank entries with the mean values.
imputer = SimpleImputer(strategy="mean")

imputed_rf_x = pd.DataFrame(imputer.fit_transform(rf_X))
imputed_xgb_X = pd.DataFrame(imputer.fit_transform(xgb_X))
imputed_similarity_X = pd.DataFrame(imputer.fit_transform(similarity_X))

imputed_rf_x.columns = rf_X.columns
imputed_xgb_X.columns = xgb_X.columns
imputed_similarity_X.columns = similarity_X.columns

#The Random Forest and XGB Boost models.
rf_model = RandomForestRegressor(n_estimators=700, random_state=1)
xgb_model = XGBRegressor(random_state=0, n_estimators=600, learning_rate= .03)

def xgb_error():

    """
    Prints out the MAE and Median Absolute Error of the XGB Model using cross validation.
    """
    mean_score = -1 * cross_val_score(xgb_model, imputed_xgb_X, y, cv=5, scoring='neg_mean_absolute_error')
    median_score = -1 * cross_val_score(xgb_model, imputed_xgb_X, y, cv=5, scoring='neg_median_absolute_error')
    print("XGB cross-val mean absolute error:", mean_score.mean())
    print("XGB cross-val median absolute error", median_score.mean())

def rf_error():

    """
   Prints out the MAE and Median Absolute Error of the Random Forest Model using cross validation.
   """
    mean_score = -1 * cross_val_score(rf_model, imputed_rf_x, y, cv=5, scoring='neg_mean_absolute_error')
    median_score = -1 * cross_val_score(rf_model, imputed_rf_x, y, cv=5, scoring='neg_median_absolute_error')
    print("RF cross-val mean absolute error:", mean_score.mean())
    print("RF cross-val median absolute error:", median_score.mean())

def rf_predict_curr(name):

    """
    A function for predicting a current 2019-20 rookie's 3P% shooting improvement from their rookie season to their
    sophomore season using a Random Forests model.
    :param name: A string of the rookie's name.
    Prints out the predicted increase/decrease in 3P%.
    """

    # Get the row of shooting stats from our rookie shooting stats dataframe for the player we are predicting.
    specific_rookie_stats = rookies2019_20_stats.loc[rookies2019_20_stats["Name"] == name, :]

    for col in specific_rookie_stats:
        # If any shooting stat is empty (likely due to an unusable sample size), replace it with the mean value of
        # that shooting stat from our training dataframe.
        if (specific_rookie_stats[col].isnull().any()) & (col != ("3P% Improve")) & (col != ("3P% 2nd season")) & (col != "3PA per 100 2nd season"):
            mean = imputed_rf_x.loc[:, col].mean()
            specific_rookie_stats.loc[rookies2019_20_stats["Name"] == name, col] = mean

    # Fit the model on past rookies.
    rf_model.fit(imputed_rf_x, y)

    # Predict 3P% improvement based on the rookie's stats.
    rookie_prediction = rf_model.predict(specific_rookie_stats[rf_predictors ]) * 100
    if rookie_prediction >= 0:
        print("The model predicts that " + name + "'s 3P% will increase by " + str(rookie_prediction)[1:-1] + "% next season.")
    else:
        print("The model predicts that " + name + "'s 3P% will decrease by " + str(rookie_prediction)[1:-1] + "% next season.")

def xgb_predict_curr(name):

    """
    A function for predicting a current 2019-20 rookie's 3P% shooting improvement from their rookie season to their
    sophomore season using a XGB Boost model.
   :param name: A string of the rookie's name.
   Prints out the predicted increase/decrease in 3P%
   """

    # Get the row of shooting stats from our rookie shooting stats dataframe for the player we are predicting.
    specific_rookie_stats = rookies2019_20_stats.loc[rookies2019_20_stats["Name"] == name, :]

    for col in specific_rookie_stats:
        # If any shooting stat is empty (likely due to an unusable sample size), replace it with the mean value of
        # that shooting stat from our training dataframe.
        if (specific_rookie_stats[col].isnull().any()) & (col != ("3P% Improve")) & (col != ("3P% 2nd season")) & (col != "3PA per 100 2nd season"):
            mean = imputed_xgb_X.loc[:, col].mean()
            specific_rookie_stats.loc[rookies2019_20_stats["Name"] == name, col] = mean

    # Fit the model on past rookies.
    xgb_model.fit(imputed_xgb_X, y)

    # Predict 3P% improvement based on the rookie's stats.
    rookie_prediction = xgb_model.predict(specific_rookie_stats[xgb_predictors]) * 100
    if rookie_prediction >= 0:
        print("The model predicts that " + name + "'s 3P% will increase by " + str(rookie_prediction)[1:-1] + "% next season.")
    else:
        print("The model predicts that " + name + "'s 3P% will decrease by " + str(rookie_prediction)[1:-1] + "% next season.")


def similarities(name):
    """
    A function for finding past shooters that similar college and rookie season shooting stats to a current
    2019-20 rookie using k-means clustering.
    :param name: A string of the rookie's name.
    Prints out the similar past players.
    """

    # Get the shooting stats of the rookie we are finding similar players for.
    rookie_sim_stats = rookies2019_20_stats.loc[rookies2019_20_stats["Name"] == name, :]

    for col in rookie_sim_stats:
        # If any shooting stat is empty (likely due to an unusable sample size), replace it with the mean value of
        # that shooting stat from our training dataframe.
        if (rookie_sim_stats[col].isnull().any()) & (col != ("3P% Improve")) & (col != ("3P% 2nd season")) & (col != ("3PA per 100 2nd season")):
            mean = imputed_rf_x .loc[:, col].mean()
            rookie_sim_stats.loc[rookies2019_20_stats["Name"] == name, col] = mean

    # Get the relevant stats for the rookie we are finding similarities to.
    player_row = rookie_sim_stats.loc[rookies2019_20_stats["Name"] == name]
    player_stats = player_row[["Name", "3P% 1st Season", "3P% College", "3PA per 100 1st season", "3pr 1st season", "FT% 1st Season"]]

    # Get the relevant stats in our training data that we'll be using to find the similar players.
    similarity_stats = imputed_similarity_X
    similarity_stats["Name"] = NBA_shooting_stats["Name"]

    # Set up our K-means clusters.
    km = KMeans(n_clusters=10, max_iter=300, random_state=2)

    # Add our rookie to the dataframe of all our past rookies.
    with_new_player = similarity_stats.append(player_stats)

    # Cluster players.
    df_for_kmeans = with_new_player[similarity_stat_predictors]
    y_predict = km.fit_predict(df_for_kmeans)
    with_new_player['cluster'] = y_predict

    # Find the cluster that the rookie we are predicting for is in.
    cluster = with_new_player.loc[with_new_player["Name"] == name][["cluster"]].values[0]

    # Get the list of players in that cluster.
    similar_players_list = (with_new_player.loc[with_new_player["cluster"] == int(cluster)][["Name"]].values[:-1])
    new_list = []

    # Print out the players in that cluster.
    for x in similar_players_list:
        new_list.append(str(x)[2:-2])
    print(new_list)

# User Input
player_name = input("Enter a NBA player's name: ")

if player_name in rookies2019_20_stats.Name.values:
    # Predicts a Rookie's Shooting improvements using the a Random Forests Model.
    rf_predict_curr(player_name)
    similarities(player_name)
else:
    print("Player does not exist. Check for spelling")
    exit()

