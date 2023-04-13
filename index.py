import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from flask import current_app, render_template
import datetime
from util import css_variables
from io import BytesIO
from data import (
    calc_return,
    format_data_frame,
    get_price_data
)
import flask
from flask import Flask, render_template,jsonify # for web app
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly
import plotly.express as px
import json # for graph plotting in website
# NLTK VADER for sentiment analysis
import nltk
from yahoo_fin import stock_info
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from yahoo_fin import stock_info
import alpaca_demo
app = current_app
matplotlib.use('Agg')
import yfinance as yf
import requests
def plot(prices):

    prices = (
        prices
        .sort_index()
        .apply(np.log)
        .diff()
        .fillna(0.0)
        .cumsum()
        .apply(np.exp)
    )

    return prices.plot()


def export_svg(chart):
    output = BytesIO()
    chart.get_figure().savefig(output, format = "svg")
    return output


def customize_chart(chart):
    fig = plt.gcf()
    css = css_variables()

    fig.set_facecolor(css['color_1'])
    chart.set_xlabel(None)
    chart.set_ylabel("Cumulative return", color = css['color_2'])
    chart.tick_params(color = css['color_2'], labelcolor = css['color_2'], which = "both")
    chart.set_facecolor(css['color_1'])
    for s in chart.spines:
        chart.spines[s].set_color(css['color_2'])

    return chart


#This the part where Sentiment forecasting comes in

# for extracting data from finviz
finviz_url = 'https://finviz.com/quote.ashx?t='

def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    return news_table
	
val=222.68
val2=126.01
val3=89.46
val4=129.03
# parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() 
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])
        
        # Set column names
        columns = ['date', 'time', 'headline']

        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
        
        # Create a pandas datetime object from the strings in 'date' and 'time' column
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        
    return parsed_news_df
        
def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    
            
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)    
        
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    return parsed_and_scored_news

def plot_hourly_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('H').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

def plot_daily_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('D').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later





#@app.route("/")

# def home():


#     hist_data = pd.read_csv('AAPL.csv')
#     hist_data = hist_data.pipe(format_data_frame)

#     symbols = ['SPY', 'EZU', 'IWM', 'EWJ', 'EEM']
#     end = datetime.date.today()
#     start = end - datetime.timedelta(days = 30 * 3)

#     df = get_price_data(symbols, start_date = start, end_date = end)
#     _plot = (
#         df
#         .loc[:, ['date', 'symbol', 'adj_close']]
#         .pivot('date', 'symbol', 'adj_close')
#         .pipe(plot)
#     )
#     _plot = customize_chart(_plot)

#     try:
#         chart = export_svg(_plot)
#     finally:
#         plt.close()

#     ret_d01 = calc_return(df, index = 1)
#     ret_d21 = calc_return(df, index = 21)
#     prices  = (
#         df[['symbol', 'date', 'adj_close']]
#         .rename(
#             columns = {'adj_close' : 'price'}
#         )
#         .merge(ret_d01.reset_index(), how = 'inner', on = ['date', 'symbol'])
#         .rename(columns = {'ret' : 'daily_return'})
#         .merge(ret_d21.reset_index(), how = 'inner', on = ['date', 'symbol'])
#         .rename(columns = {'ret' : 'monthly_return'})
#     )

#     # - styling

#     return_cols = ['daily_return', 'monthly_return']
#     def ret_color(x):
#         color = 'tomato' if x < 0 else 'lightgreen'
#         return 'color: %s' % color

#     prices = (
#         prices
#         .sort_values('monthly_return', ascending = False)
#         .assign(date = lambda df: df['date'].dt.strftime("%Y-%m-%d"))
#         .pipe(format_data_frame)
#         .format("{:,.2f}", subset = ['price'])
#         .applymap(ret_color, subset = return_cols)
#         .format("{:+,.2%}", subset = return_cols)
#     )
#     print('type of prices is: ',type(prices))
#     print('type of hist data is: ',type(hist_data))
#     return render_template(
#         "index.html",
#         hist = hist_data.render(),
#         prices = prices.render(),
#         chart = chart.getvalue().decode('utf8')
#     )

def get_dummyVal():
    dummyVal = 69
    return dummyVal
@app.route("/")

@app.route("/index", methods=['GET', 'POST'])
def index():

    val=222.68
    val2=126.01
    val3=89.46
    val4=129.03

    return render_template('index.html',MSFTval = val, AAPLval = val2,GOOGval = val3, METAval = val4, 
                           MSFTliv = stock_info.get_live_price("MSFT"),
                           AAPLliv = stock_info.get_live_price("AAPL"),
                           GOOGliv = stock_info.get_live_price("GOOG"),
                           METAliv = stock_info.get_live_price("META")
                           )

@app.route('/home', methods=['GET', 'POST'])
def home():
    global stocks
    Val = 69
    #print('this is vAalue of Val: ',Val)
    return render_template("index.html")







from flask import request
@app.route("/quote")
def display_quote():
	# get a stock ticker symbol from the query string
	# default to AAPL
	symbol = request.args.get('symbol', default="AAPL")

	# pull the stock quote
	quote = yf.Ticker(symbol)

	#return the object via the HTTP Response
	return jsonify(quote.info)

# API route for pulling the stock history

@app.route("/history")
def display_history():
	#get the query string parameters
	symbol = request.args.get('symbol', default="AAPL")
	period = request.args.get('period', default="1y")
	interval = request.args.get('interval', default="1mo")

	#pull the quote
	quote = yf.Ticker(symbol)	
	#use the quote to pull the historical data from Yahoo finance
	hist = quote.history(period=period, interval=interval)
	#convert the historical data to JSON
	data = hist.to_json()
	#return the JSON in the HTTP response
	return data

@app.route('/historical_data', methods=['GET', 'POST'])
def historical_data():
    return render_template("historical_data.html")


@app.route('/finance_news', methods=['GET', 'POST'])
def finance_news():
    company = 'AAPL'
    api_key = 'O5HJOI45IZBFVF65'
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={company}&apikey={api_key}&size=10'
    response = requests.get(url)
    data = response.json()
    return render_template('finance_news.html', news=data['feed'])
    #return render_template("finance_news.html")


@app.route('/twitter_feeds', methods=['GET', 'POST'])
def twitter_feeds():
    api_key = 'O5HJOI45IZBFVF65'

    url = f'https://www.alphavantage.co/query?function=TOURNAMENT_PORTFOLIO&season=2021-09&apikey={api_key}'
    # url = 'https://www.alphavantage.co/query?function=TOURNAMENT_PORTFOLIO&season=2021-09&apikey=demo'
    r = requests.get(url)
    data = r.json()
    data = r.json()
    return render_template("twitter_feeds.html", data = data)


@app.route('/working', methods=['GET', 'POST'])
def working():
    
    accItems, positions = alpaca_demo.alpacaTrader()
    print(accItems,positions)
    return render_template("TradingBot.html", accountItems = accItems, positions = positions)


@app.route('/about', methods=['GET', 'POST'])
def about():

    company = request.args.get('symbol', default="AAPL")

    api_key = 'O5HJOI45IZBFVF65'
    #company = 'AAPL'

    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={company}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    parsed_data = json.loads(json.dumps(data))
    return render_template("about.html", company=parsed_data)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template("contact.html")

@app.route('/sentiment1', methods=['GET', 'POST'])
def sentiment1():
    return render_template("sentiment1.html")

@app.route('/sentiment', methods = ['POST'])
def sentiment():

	ticker = flask.request.form['ticker'].upper()
	news_table = get_news(ticker)
	parsed_news_df = parse_news(news_table)
	parsed_and_scored_news = score_news(parsed_news_df)
	fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
	fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)

	graphJSON_hourly = json.dumps(fig_hourly, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
	
	header= "Hourly and Daily Sentiment of {} Stock".format(ticker)
	description = """
	The above chart averages the sentiment scores of {} stock hourly and daily.
	The table below gives each of the most recent headlines of the stock and the negative, neutral, positive and an aggregated sentiment score.
	The news headlines are obtained from the FinViz website.
	Sentiments are given by the nltk.sentiment.vader Python library.
    """.format(ticker)
	return render_template('sentiment.html',graphJSON_hourly=graphJSON_hourly, graphJSON_daily=graphJSON_daily, header=header,table=parsed_and_scored_news.to_html(classes='data'),description=description)
