import slack_sdk    as slack
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, Response
from slackeventsapi import SlackEventAdapter
import google.generativeai as genai
import pymysql
# --- Gemini Client ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET_'], '/slack/events', app)

client = slack.WebClient(token=os.environ['SLACK_TOKEN_'])
BOT_ID = client.api_call("auth.test")['user_id']

        
def nl_to_sql(question: str) -> str:
    """Convert natural language to SQL using Gemini"""
    schema = """
    Database: sensor_db
    Table: sensor_data
    Columns:
      - device_id (string)
      - status (string: e.g. SUCCESS, FAILED)
      - reading_time (datetime)
      - temperature (float)
      - vibration (float)
    """
    prompt = f"""
    Convert the following natural language question into a valid MySQL-compatible SQL query for TiDB.
    {schema}
    Question: {question}
    Only output the SQL query, nothing else.
    """

    response = gemini_model.generate_content(prompt)
    sql_query = response.text.strip().strip("```sql").strip("```")
    return sql_query

@ slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    if BOT_ID != user_id:
        sql_query = nl_to_sql(text)
        client.chat_postMessage(
                channel=channel_id,  text="hi" )

if __name__ == "__main__":
    app.run(debug=True, port=3000)



