# Load Database Pkg
import sqlite3
conn = sqlite3.connect('output/data.db', check_same_thread=False)
c = conn.cursor()

# Fxn
def create_page_visited_table():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,timeOfvisit TIMESTAMP)')

def add_page_visited_details(pagename,timeOfvisit):
	c.execute('INSERT INTO pageTrackTable(pagename,timeOfvisit) VALUES(?,?)',(pagename,timeOfvisit))
	conn.commit()

def view_all_page_visited_details():
	c.execute('SELECT * FROM pageTrackTable')
	data = c.fetchall()
	return data


# Fxn To Track Input & Prediction
def create_emotSen_table():
	c.execute('CREATE TABLE IF NOT EXISTS HistoryTable(rawtext TEXT,Sentiment TEXT,sentiment_score NUMBER,timeOfvisit TIMESTAMP)')

def add_prediction_details(raw_text,pred_sentiment,sen_score,timeOfvisit):
	c.execute('INSERT INTO HistoryTable(rawtext,Sentiment,sentiment_score,timeOfvisit) VALUES(?,?,?,?)',(raw_text,pred_sentiment,sen_score,timeOfvisit))
	conn.commit()

def view_all_prediction_details():
	c.execute('SELECT * FROM HistoryTable')
	data = c.fetchall()
	return data