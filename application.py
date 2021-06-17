import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime
from selenium import webdriver
import time
import glob
import os
from webdriver_manager.chrome import ChromeDriverManager
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

application = app = Flask(__name__)

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Homepage
@app.route('/')
def my_form():
    return render_template('form.html')

# NFL game predictor page
@app.route('/nfl-form.html')
def my_nfl_form():
    return render_template('nfl-form.html')

# NFL form submission logic to determine winners of week
@app.route('/nfl-form.html', methods=['POST'])
def my_nfl_form_post():
    # Get NFL week input from user's POST request
    text_input = request.form['text']
    text_input = int(text_input)
    season_year_start_input = request.form['season_year_start']
    season_year_start_input = int(season_year_start_input)

    # Perform prediction process for the inputted week from the user. predictions_df will contain the array of predicted teams to win
    predictions_df = perform_prediction(text_input, season_year_start_input)

    # str_return is the array of strings that are sent to my-form.html once the
    # prediction process is complete
    str_return = []

    # If we didn't get any predictions, then that means there was no training data to use (the season likely hasn't started yet)
    if (len(predictions_df) == 0):
        str_return.append("Cannot predict anything because there is no data for the " + str(season_year_start_input) + "-" + str(season_year_start_input+1) + " season yet. Please wait until week 1 has started.")
        return render_template('nfl-form.html', results=str_return, subtitle="Cannot make predictions", percent_correct = "Percent Correct: 0%")

    # predictions_arr is the array of prediction values (string predictions)
    predictions_arr = str(predictions_df['predictions'][0]).replace(']','').replace('[','').replace('\"','').replace('\'','').split(",")
    print("predictions_arr is: " + str(predictions_arr))

    counter = 0
    # team_names_arr is an array of the teams playing (the first two names are the
    # names of the teams in the first matchup, the second two names are the names
    # of the teams in the second matchup, etc.)
    team_names_arr = str(predictions_df['team_names'][0]).replace(']','').replace('[','').replace('\'','').replace('\"','').replace(' ', '').split(",")
    # Iterate through each prediction and add it to the str_return array
    for prediction in predictions_arr:
        str_return.append(team_names_arr[counter] + ' have a ' + str(round(100*float(prediction), 2)) + ' percent chance of beating ' + team_names_arr[counter+1])
        counter = counter + 2

    # BeautifulSoup code to check which games are complete. If a game is done, then
    # I update the str_return value to show the result (either correct or wrong)
    article_list = []
    prediction_results_list = []
    num_correct = 0
    num_wrong = 0
    week_url = 'https://www.espn.com/nfl/scoreboard/_/year/' + str(season_year_start_input) + '/seasontype/2/week/' + str(text_input)
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(week_url)
    soup_week = BeautifulSoup(driver.page_source, 'html.parser')
    try:
        articles = soup_week.find_all(id = "scoreboard-page")[0].find_all(id = "events")[0].find_all("article")
    except IndexError:
        print("Got an Index Error, so will try retrieving url again")
        time.sleep(15)
        driver.get(week_url)
        soup_week = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup_week.find_all(id = "scoreboard-page")[0].find_all(id = "events")[0].find_all("article")

    # Iterate through each article
    for article in articles:
        team_names = article.find_all("span", {"class": "sb-team-short"})
        if (team_names == []):
            print("This is an article without team names and scores, so skipping")
            continue
        first_team = team_names[0].string
        second_team = team_names[1].string

        if article.find_all("tr", {"class": "sb-linescore"})[0].find_all("th")[0].string == "Final" or article.find_all("tr", {"class": "sb-linescore"})[0].find_all("th")[0].string == "Final/OT":
            print("These teams have played, so adding to Prediction Results List: " + first_team + " vs " + second_team)
        else:
            print("These teams haven't played yet, so not going to add to Prediction Results List: " + first_team + " vs " + second_team)
            continue

        first_team_res = int(article.find_all("td", {"class": "total"})[0].find_all("span")[0].string)
        second_team_res = int(article.find_all("td", {"class": "total"})[1].find_all("span")[0].string)

        counter = 0
        prediction_val = 0.0
        team_matchup_num = 0
        for prediction in predictions_arr:
            if (team_names_arr[counter] == first_team):
                prediction_val = float(prediction)
                break
            counter = counter + 2
            team_matchup_num = team_matchup_num + 1
        # Determine results of predictions
        if first_team_res > second_team_res:
            if float(prediction_val) > 0.5:
                str_return[team_matchup_num] = str_return[team_matchup_num] + " =======> Correct"
                num_correct = num_correct + 1
            else:
                str_return[team_matchup_num] = str_return[team_matchup_num] + " =======> Wrong"
                num_wrong = num_wrong + 1
        elif first_team_res < second_team_res:
            if float(prediction_val) < 0.5:
                str_return[team_matchup_num] = str_return[team_matchup_num] + " =======> Correct"
                num_correct = num_correct + 1
            else:
                str_return[team_matchup_num] = str_return[team_matchup_num] + " =======> Wrong"
                num_wrong = num_wrong + 1
        else:
            print("Teams tied, and my algorithm doesn't determine ties, so ignoring")
            str_return[team_matchup_num] = str_return[team_matchup_num] + " =======> Tie so ignoring"

    # Check how many I got correct during this week
    percent_correct = 0
    if num_correct + num_wrong != 0:
        percent_correct = round(100*(float(num_correct)/float(num_correct + num_wrong)), 2)
    return render_template('nfl-form.html', results=str_return, subtitle="Week " + str(text_input) + " Predictions from the " + str(season_year_start_input) + "-" + str(season_year_start_input+1) + " season", percent_correct = "Percent Correct: " + str(num_correct) + "/" + str(num_correct+num_wrong) + " = " + str(percent_correct) + "% ")

# team_abbrevs returns a dictionary that maps team names to their abbreviation. Must have two dictionaries because the Washington Football team
# used to go by a different name prior to 2020
def team_abbrevs(year):
    if (year <= 2019):
        return {'Cardinals': 'crd', 'Ravens': 'rav', 'Falcons': 'atl', 'Bills': 'buf', 'Panthers': 'car', 'Bengals': 'cin', 'Bears': 'chi', 'Browns': 'cle', 'Cowboys': 'dal', 'Broncos': 'den', 'Lions': 'det', 'Texans': 'htx', 'Packers': 'gnb', 'Colts': 'clt', 'Rams': 'ram', 'Jaguars': 'jax', 'Vikings': 'min', 'Chiefs': 'kan', 'Saints': 'nor', 'Raiders': 'rai', 'Giants': 'nyg', 'Chargers': 'sdg', 'Eagles': 'phi', 'Dolphins': 'mia', '49ers': 'sfo', 'Patriots': 'nwe', 'Seahawks': 'sea', 'Jets': 'nyj', 'Buccaneers': 'tam', 'Steelers': 'pit', 'Redskins': 'was', 'Titans': 'oti'}
    else:
        return {'Cardinals': 'crd', 'Ravens': 'rav', 'Falcons': 'atl', 'Bills': 'buf', 'Panthers': 'car', 'Bengals': 'cin', 'Bears': 'chi', 'Browns': 'cle', 'Cowboys': 'dal', 'Broncos': 'den', 'Lions': 'det', 'Texans': 'htx', 'Packers': 'gnb', 'Colts': 'clt', 'Rams': 'ram', 'Jaguars': 'jax', 'Vikings': 'min', 'Chiefs': 'kan', 'Saints': 'nor', 'Raiders': 'rai', 'Giants': 'nyg', 'Chargers': 'sdg', 'Eagles': 'phi', 'Dolphins': 'mia', '49ers': 'sfo', 'Patriots': 'nwe', 'Seahawks': 'sea', 'Jets': 'nyj', 'Buccaneers': 'tam', 'Steelers': 'pit', 'Washington': 'was', 'Titans': 'oti'}

# column_names is the columns that are used for training data
column_names = ['Result', 'YardsPerOffPlay', 'NetYardsGainedPerPassAttempt',
                'RushingYardsPerAttempt',
                'AvgPlaysPerDrive', 'NetYardsPerDrive', 'AvgPointsScoredPerDrive',
                'OppYardsPerOffPlay', 'OppNetYardsGainedPerPassAttempt',
                'OppRushingYardsPerAttempt',
                'OppAvgPlaysPerDrive', 'OppNetYardsPerDrive', 'OppAvgPointsScoredPerDrive',
                'NextOppYardsPerOffPlay', 'NextOppNetYardsGainedPerPassAttempt',
                'NextOppRushingYardsPerAttempt',
                'NextOppAvgPlaysPerDrive', 'NextOppNetYardsPerDrive', 'NextOppAvgPointsScoredPerDrive',
                'NextOppOppYardsPerOffPlay', 'NextOppOppNetYardsGainedPerPassAttempt',
                'NextOppOppRushingYardsPerAttempt', 'NextOppOppAvgPlaysPerDrive', 'NextOppOppNetYardsPerDrive',
                'NextOppOppAvgPointsScoredPerDrive']

# remove_result_field removes the Result column from the inputted df_predictions dataframe
def remove_result_field(df_predictions):
    df_predictions_edited = df_predictions.copy()
    df_predictions_edited.pop('Result')
    return df_predictions_edited

# get_game_data returns the dataframe with all the scraped data for the team denoted
# by team_abbrev and the team they will play, denoted by next_opp_team_abbrev
def get_game_data(team_abbrev, next_opp_team_abbrev, game_res, season_year_start_input):
    team_url = 'https://www.pro-football-reference.com/teams/' + team_abbrev + '/' + str(season_year_start_input) + '.htm'
    next_opp_team_url = 'https://www.pro-football-reference.com/teams/' + next_opp_team_abbrev + '/' + str(season_year_start_input) + '.htm'
    r = requests.get(team_url)
    r_next_opp = requests.get(next_opp_team_url)
    soup = BeautifulSoup(r.content, 'html5lib')
    soup_next_opp = BeautifulSoup(r_next_opp.content, 'html5lib')
    team_stats = soup.find_all(id = "team_stats")[0].find_all("tbody")[0].find_all("tr")[0]

    opp_stats = soup.find_all(id = "team_stats")[0].find_all("tbody")[0].find_all("tr")[1]
    next_opp_team_stats = soup_next_opp.find_all(id = "team_stats")[0].find_all("tbody")[0].find_all("tr")[0]
    next_opp_opp_stats = soup_next_opp.find_all(id = "team_stats")[0].find_all("tbody")[0].find_all("tr")[1]

    # Get training data from statistics site
    df = pd.DataFrame([[float(game_res),
    get_data(team_stats, "yds_per_play_offense"), get_data(team_stats, "pass_net_yds_per_att"),
    get_data(team_stats, "rush_yds_per_att"), get_data(team_stats, "plays_per_drive"),
    get_data(team_stats, "yds_per_drive"), get_data(team_stats, "points_avg"), get_data(opp_stats, "yds_per_play_offense"),
    get_data(opp_stats, "pass_net_yds_per_att"),
    get_data(opp_stats, "rush_yds_per_att"), get_data(opp_stats, "plays_per_drive"),
    get_data(opp_stats, "yds_per_drive"), get_data(opp_stats, "points_avg"),
    get_data(next_opp_team_stats, "yds_per_play_offense"), get_data(next_opp_team_stats, "pass_net_yds_per_att"),
    get_data(next_opp_team_stats, "rush_yds_per_att"), get_data(next_opp_team_stats, "plays_per_drive"),
    get_data(next_opp_team_stats, "yds_per_drive"), get_data(next_opp_team_stats, "points_avg"), get_data(next_opp_opp_stats, "yds_per_play_offense"),
    get_data(next_opp_opp_stats, "pass_net_yds_per_att"),
    get_data(next_opp_opp_stats, "rush_yds_per_att"), get_data(next_opp_opp_stats, "plays_per_drive"),
    get_data(next_opp_opp_stats, "yds_per_drive"), get_data(next_opp_opp_stats, "points_avg")]], columns=column_names)
    return df

# get_data gets the float value of the td tag with data-stat = data_name
def get_data(team_stats, data_name):
    return float(team_stats.find_all("td", {"data-stat": data_name})[0].string)

# perform_prediction generates the dataframe with the predictions for each matchup
# in the week denoted by predict_week
def perform_prediction(predict_week, season_year_start_input):
    error_results_column_names = ["training_rate", "epoch_val", "error", "team_names", "predictions"]
    error_results = pd.DataFrame([], columns=error_results_column_names)

    # Current optimal pairing is training_rate at 0.0009 and epoch_vals at 5
    training_rates = [0.0009]
    epoch_vals = [5]
    y = predict_week
    week_range_training = 5
    # If you ever want to analyze different training rates and epoch_vals, you
    # should unccomment the below two lines.
    # training_rates = [0.1, 0.01, 0.09, 0.001, 0.009, 0.0001, 0.0009]
    # epoch_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50, 60, 100, 200]

    # glob_res will be empty if a csv with week{y} in it does not exist. If a csv
    # with week{y} in it does exist, then we will use that file rather than
    # redo the predictions
    glob_res = glob.glob(str(season_year_start_input) + "*_week" + str(y) + "_*")
    # glob_res = []
    if not glob_res:
        # article_list represents the list of article tags scraped for a week
        article_list = []
        # df_scraped_training_data is the dataframe with all the training data
        df_scraped_training_data = pd.DataFrame([], columns=column_names)
        # df_predictions is the dataframe that stores the prediction values for
        # each matchup
        df_predictions = pd.DataFrame([], columns=column_names)
        # prediction_team_names is an array of arrays, where each array within
        # the main array contains two strings (two teams playing each other this week)
        prediction_team_names = []

        base_week = y - week_range_training
        if (base_week < 1):
            base_week = 1
        # Iterate from week 1 to the current week to scrape data from all those weeks
        for x in range(base_week, y+1):
            week_url = 'https://www.espn.com/nfl/scoreboard/_/year/' + str(season_year_start_input) + '/seasontype/2/week/' + str(x)
            time.sleep(1)
            driver = webdriver.Chrome(ChromeDriverManager().install())
            driver.get(week_url)
            soup_week = BeautifulSoup(driver.page_source, 'html.parser')
            try:
                articles = soup_week.find_all(id = "scoreboard-page")[0].find_all(id = "events")[0].find_all("article")
            except IndexError:
                print("Got an Index Error, so will try retrieving url again")
                time.sleep(15)
                driver.get(week_url)
                soup_week = BeautifulSoup(driver.page_source, 'html.parser')
                articles = soup_week.find_all(id = "scoreboard-page")[0].find_all(id = "events")[0].find_all("article")

            # Iterate through each article
            for article in articles:
                team_names = article.find_all("span", {"class": "sb-team-short"})
                if (team_names == []):
                    print("This is an article without team names and scores, so skipping")
                    continue
                first_team = team_names[0].string
                second_team = team_names[1].string

                if x == y:
                # if not (article.find_all("tr", {"class": "sb-linescore"})[0].find_all("th")[0].string == "Final" or article.find_all("tr", {"class": "sb-linescore"})[0].find_all("th")[0].string == "Final/OT"):
                    # WILL HAVE TO UPDATE THIS TO ONLY ADD THE RESULT IF THE GAME IS ACTUALLY OVER (COULD BE IN PROGRESS!!!)
                    print("These teams are playing in week " + str(x) + ", so adding to predictions list: " + first_team + " vs " + second_team)
                    # prediction_team_names.append(first_team + " vs. " + second_team)
                    prediction_team_names.append([first_team, second_team])
                    if (len(df_scraped_training_data) == 0):
                        return []
                    df_predictions = df_predictions.append(remove_result_field(get_game_data(team_abbrevs(season_year_start_input)[first_team], team_abbrevs(season_year_start_input)[second_team], '0', season_year_start_input)), ignore_index=True)
                    continue

                if not (article.find_all("tr", {"class": "sb-linescore"})[0].find_all("th")[0].string == "Final" or article.find_all("tr", {"class": "sb-linescore"})[0].find_all("th")[0].string == "Final/OT"):
                    print("These teams haven't played yet, so not adding them to training data: " + first_team + " vs " + second_team)
                    continue

                first_team_res = int(article.find_all("td", {"class": "total"})[0].find_all("span")[0].string)
                second_team_res = int(article.find_all("td", {"class": "total"})[1].find_all("span")[0].string)
                if (first_team_res > second_team_res):
                    df_scraped_training_data = df_scraped_training_data.append(get_game_data(team_abbrevs(season_year_start_input)[first_team], team_abbrevs(season_year_start_input)[second_team], 1, season_year_start_input), ignore_index=True)
                    df_scraped_training_data = df_scraped_training_data.append(get_game_data(team_abbrevs(season_year_start_input)[second_team], team_abbrevs(season_year_start_input)[first_team], 0, season_year_start_input), ignore_index=True)
                elif (second_team_res > first_team_res):
                    df_scraped_training_data = df_scraped_training_data.append(get_game_data(team_abbrevs(season_year_start_input)[first_team], team_abbrevs(season_year_start_input)[second_team], 0, season_year_start_input), ignore_index=True)
                    df_scraped_training_data = df_scraped_training_data.append(get_game_data(team_abbrevs(season_year_start_input)[second_team], team_abbrevs(season_year_start_input)[first_team], 1, season_year_start_input), ignore_index=True)
                else:
                    df_scraped_training_data = df_scraped_training_data.append(get_game_data(team_abbrevs(season_year_start_input)[first_team], team_abbrevs(season_year_start_input)[second_team], 0.5, season_year_start_input), ignore_index=True)
                    df_scraped_training_data = df_scraped_training_data.append(get_game_data(team_abbrevs(season_year_start_input)[second_team], team_abbrevs(season_year_start_input)[first_team], 0.5, season_year_start_input), ignore_index=True)
        print("<=========================== df_scraped_training_data ===========================>")
        print(df_scraped_training_data)
        print("<=========================== end of df_scraped_training_data ===========================>")

        for training_rate in training_rates:
            for epoch_val in epoch_vals:
                print("<=========================== Information for this training_rate, epoch_val pair ===========================>")
                print("<=========================== Training Value Information ===========================>")
                print("Training rate is " + str(training_rate))
                print("Epoch is " + str(epoch_val))
                print("<=========================== End of Training Value Information ===========================>")
                raw_dataset = df_scraped_training_data
                dataset = raw_dataset.copy()
                dataset = dataset.dropna()

                train_dataset = dataset.sample(frac=0.7, random_state=0)
                test_dataset = dataset.drop(train_dataset.index)
                print("<=========================== Training Datasets ===========================>")
                print("train_dataset")
                print(train_dataset)
                print("df_predictions")
                print(df_predictions)
                print("<=========================== End of Training Datasets ===========================>")

                train_features = train_dataset.copy()
                test_features = test_dataset.copy()

                train_labels = train_features.pop('Result')
                test_labels = test_features.pop('Result')
                normalizer = preprocessing.Normalization()
                normalizer.adapt(np.array(train_features))

                def plot_loss(history):
                  plt.plot(history.history['loss'], label='loss')
                  plt.plot(history.history['val_loss'], label='val_loss')
                  plt.ylim([0, 10])
                  plt.xlabel('Epoch')
                  plt.ylabel('Error [Result]')
                  plt.legend()
                  plt.grid(True)
                  plt.show()


                def build_and_compile_model(norm):
                  model = keras.Sequential([
                      norm,
                      layers.Dense(64, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(1)
                  ])

                  model.compile(loss='mean_absolute_error',
                                optimizer=tf.keras.optimizers.Adam(training_rate))
                  return model

                dnn_model = build_and_compile_model(normalizer)
                history = dnn_model.fit(
                    train_features, train_labels,
                    validation_split=0.3,
                    verbose=0, epochs=epoch_val)

                test_results = {}
                test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

                print([training_rate, epoch_val, test_results['dnn_model']])

                test_predictions = dnn_model.predict(test_features).flatten()
                df_predictions_copy = df_predictions.copy()
                df_predictions_copy.pop('Result')
                prediction_guesses = dnn_model.predict(df_predictions_copy)
                prediction_guesses_formatted = []
                for prediction_guess in prediction_guesses:
                    print(prediction_guess[0])
                    prediction_guesses_formatted.append(prediction_guess[0])
                print("near end")
                print(prediction_guesses_formatted)
                prediction_guesses_with_names = []
                error_results = error_results.append(pd.DataFrame([[training_rate, epoch_val, test_results['dnn_model'], prediction_team_names, prediction_guesses_formatted]], columns=error_results_column_names))
                print("<=========================== Error Results for this training_rate, epoch_val pair ===========================>")
                print(error_results)
                print("<=========================== End of Error Results for this training_rate, epoch_val pair ===========================>")
                print("Training rate is " + str(training_rate))
                print("Epoch is " + str(epoch_val))
                print("<=========================== End of Information for this training_rate, epoch_val pair ===========================>")

        print("<=========================== Error Results for ALL training_rate, epoch_val pairs ===========================>")
        print(error_results)
        print("<=========================== End of Error Results for ALL training_rate, epoch_val pairs ===========================>")
        now = datetime.datetime.now()
        error_results.to_csv(str(season_year_start_input) + now.strftime("-%m-%d-%H:%M:%S") + '_week' + str(y) + '_predictions.csv', index = False, header=True)
        return error_results
    else:
        print("This week has already been predicted!")
        print(pd.read_csv(max(glob_res, key=os.path.getctime)))
        return pd.read_csv(max(glob_res, key=os.path.getctime))

@app.route('/nba-form.html')
def my_nba_form():
    return render_template('nba-form.html')

if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    app.run()
