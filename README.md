# nfl-machine-learning
This application uses a machine learning algorithm to predict the outcomes of past and upcoming NFL games

# Environment Setup
First, make sure you have `Chrome` installed on your machine

Second, make sure you have `python3`. You can use this installation guide if you don't have python3: https://realpython.com/installing-python/

Third, clone this repo somewhere on your local system. Cd into your repo and then
run `python3 -m venv /path/to/new/virtual/environment`

Fourth, activate your virtual environment by running `source env/bin/activate`

Finally, run `python3 -m pip install -r requirements.txt`

# Using the application
Activate your virtual environment by doing `source env/bin/activate`

Run `python3 application.py`

Search for http://127.0.0.1:5000/ in your browser. You should see the application's home page

# Predicting NFL games
From the home page of the application, you should click the NFL link.

Then, type the year (e.g if you wanted to predict games from a week in the 2020-2021
season, then type 2020).

Then, type the week of games that you would like to predict. If you have already run this
prediction, your results should be stored locally in a csv file with a name that specifies the
year you predicted, the time it was run, and the week you predicted.

If you haven't run this prediction already, then the machine
learning algorithm will have to run and should be complete after 5-10 minutes. As
mentioned, it will store the results in a csv file, but it will also show you
your results in the browser.
