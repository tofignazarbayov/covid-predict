import telebot
import random
import datetime
import time
from telebot import types
from covid_predict import prediction
from covid_predict import situation
from covid_predict import top10
import pandas as pd
import urllib.request as request
import logging

logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)
logsfile = logging.FileHandler('logs_telegram.txt', encoding = "utf-8")
logger.addHandler(logsfile)

def listener(messages):
    with open('message_logs.txt', 'a', encoding = "utf-8") as file:
        for m in messages:
            if m.content_type == 'text':
                time = datetime.datetime.now().strftime("%d%b%Y%H%M")
                file.write(f"{time}: {m.chat.first_name} [{m.chat.id}]: {m.text}\n")

token = ""
client = telebot.TeleBot(token, threaded=False)
client.set_update_listener(listener)
data_countries = pd.read_csv(request.urlopen('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'))
countries_list = sorted(list(set(data_countries["Country"])))
 
@client.message_handler(commands=["start"])
def welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item = types.KeyboardButton("/start")
    markup.add(item)

    response = client.send_message(message.chat.id, f"Hello!\nI am <b>{client.get_me().first_name}</b>!\nI <b>predict</b> number of people with coronavirus <b>tomorrow</b>!\nI also provide statistics for today!\n<b>Type /start or press start button to try!</b>",
    parse_mode="html", reply_markup=markup)
    
    client.register_next_step_handler(response, branching)

@client.message_handler(commands=["text"])
def branching(message):
    if message.text == "/start":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = types.KeyboardButton("Predictions worldwide")
        item2 = types.KeyboardButton("Predictions for country")
        item3 = types.KeyboardButton("Situation in world")
        item4 = types.KeyboardButton("Situation in country")
        item5 = types.KeyboardButton("Most infected countries")
        markup.add(item1, item2, item3, item4, item5)
    
        response = client.send_message(message.chat.id, f"Please, choose one of the options", parse_mode="html", reply_markup=markup)

        client.register_next_step_handler(response, adressing)
    else:
        client.send_message(message.chat.id, 'Please, choose between available messages')
        welcome(message)

@client.message_handler(commands=["text"])
def adressing(message):
    if message.text == "Predictions worldwide":
        world_prediction(message)
    elif message.text == "Predictions for country":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        for country in countries_list:
            markup.add(country)
        response = client.send_message(message.chat.id, f"Choose country", parse_mode="html", reply_markup=markup)
        client.register_next_step_handler(response, country_prediction)
    elif message.text == "Most infected countries":
        most_infected_countries(message)
    elif message.text == "Situation in world":
        world_situation(message)
    elif message.text == "Situation in country":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        for country in countries_list:
            markup.add(country)
        response = client.send_message(message.chat.id, f"Choose country", parse_mode="html", reply_markup=markup)
        client.register_next_step_handler(response, country_situation)
    else:
        client.send_message(message.chat.id, 'Please, choose between available messages')
        welcome(message)

@client.message_handler(commands=["text"])
def world_prediction(message):
    result = prediction("world", 1, "best_of_2")
    client.send_photo(message.chat.id, photo=open(result["graph"], "rb"))
    client.send_message(message.chat.id, result["message"])
    message.text="/start"
    branching(message)

@client.message_handler(commands=["text"])
def country_prediction(message):
    if message.text in countries_list:
        result = prediction(message.text, 1, "best_of_2")
        client.send_photo(message.chat.id, photo=open(result["graph"], "rb"))
        client.send_message(message.chat.id, result["message"])
        message.text="/start"
        branching(message)
    else:
        client.send_message(message.chat.id, 'Please, choose between available messages')
        welcome(message)

@client.message_handler(commands=["text"])
def most_infected_countries(message):
    result = top10()
    client.send_photo(message.chat.id, photo=open(result["graph"], "rb"))
    message.text="/start"
    branching(message)

@client.message_handler(commands=["text"])
def world_situation(message):
    result = situation("world")
    client.send_photo(message.chat.id, photo=open(result["graph"], "rb"))
    client.send_message(message.chat.id, result["message"])
    message.text="/start"
    branching(message)

@client.message_handler(commands=["text"])
def country_situation(message):
    if message.text in countries_list:
        result = situation(message.text)
        client.send_photo(message.chat.id, photo=open(result["graph"], "rb"))
        client.send_message(message.chat.id, result["message"])
        message.text="/start"
        branching(message)
    else:
        client.send_message(message.chat.id, 'Please, choose between available messages')
        welcome(message)

if __name__ == "__main__":
    while True:
        try:
            client.polling(none_stop=True, timeout=125)
            #client.infinity_polling(True)
        except Exception as e:
            logger.error(e)
            time.sleep(25)
            #client.stop_polling()
