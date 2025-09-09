import slack_sdk    as slack
import os, tempfile
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, Response
from slackeventsapi import SlackEventAdapter
import pymysql
from vertexai.generative_models import GenerativeModel
import vertexai
from google.cloud import secretmanager


# --- Initialize Vertex AI ---
vertexai.init(project=os.environ['PROJECT_ID'], location="us-central1")

# Load Gemini model from Vertex AI
gemini_model = GenerativeModel("gemini-2.0-flash")



app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET_'], '/slack/events', app)

client = slack.WebClient(token=os.environ['SLACK_TOKEN_'])
BOT_ID = client.api_call("auth.test")['user_id']

        
def nl_to_sql(question: str) -> str:
    """Convert natural language to SQL using Gemini"""
    schema = """
    Database: test
    Table: sensor_data1
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

# --- TiDB Connection using pymysql ---


def get_tidb_connection():
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['SECRET_PROJ_ID']}/secrets/tidb-ssl-ca/versions/latest"
    response = client.access_secret_version(name=name)

    # Write PEM to a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(response.payload.data)
    print(tmp.name)
    tmp.flush()
    return pymysql.connect(
        host=os.environ["TIDB_HOST"],
        port=os.environ["TIDB_PORT"],
        user=os.environ["TIDB_USER"],
        password=os.environ["TIDB_PASSWORD"],
        database=os.environ["TIDB_DATABASE"],
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        ssl_ca=tmp.name
    )

def run_query(sql: str):
    """Run query against TiDB and return results"""
    try:
        print("===========================")
        print("inside try")
        conn = get_tidb_connection()
        print("========================")
        print(conn)
        print("==========================")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        if not rows:
            return "No results found."

        result_text = " | ".join(col_names) + "\n"
        result_text += "\n".join([" | ".join(str(x) for x in row) for row in rows[:10]])
        if len(rows) > 10:
            result_text += f"\n...and {len(rows)-10} more rows."
        return result_text

    except Exception as e:
        return f"Error executing query: {e}"

# Keep a simple cache of processed event_ids
processed_events = set()

@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    event_id = payload.get('event_id')
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    subtype = event.get('subtype')

    # 1. Ignore if already processed (deduplication)
    if event_id in processed_events:
        return
    processed_events.add(event_id)

    # 2. Ignore bot messages
    if subtype == "bot_message" or user_id == BOT_ID:
        return

    if text:
        sql_query = nl_to_sql(text)
        results = run_query(sql_query)
        client.chat_postMessage(channel=channel_id, text=f"SQL:\n```{sql_query}```\n\nResults:\n```{results}```")


if __name__ == "__main__":
    app.run(debug=True, port=3000)












