# nfl-machine-learning
This application uses a machine learning algorithm to predict the outcomes of past and upcoming NFL games. Even if the games you are trying to predict have already occurred (e.g week 10 of the 2017-2018 season), the algorithm will only use data from the 1-5 weeks prior to the games you are predicting. Regarding future games, as long as there is at least one week's worth of data in that season within 1-5 weeks prior to the week you're trying to predict, then the algorithm will be able to run. The results of the prediction will be displayed on your browser and saved to a CSV file. This project was not meant for gambling, but rather, to see if a machine learning algorithm could determine what statistics are useful for analysis without actually knowing what those data points represent. 

# Environment Setup
First, make sure you have `Chrome` installed on your machine

Second, make sure you have `python3`. You can use this installation guide if you don't have python3: https://realpython.com/installing-python/

Third, clone this repo somewhere on your local system. Cd into your repo and then
run `python3 -m venv env`

Fourth, activate your virtual environment by running `source env/bin/activate`

Finally, run `python3 -m pip install -r requirements.txt`

# Using the application
Activate your virtual environment by running `source env/bin/activate`

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
