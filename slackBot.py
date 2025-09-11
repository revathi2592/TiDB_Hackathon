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
      - status (string: e.g. SUCCESS, FAIL)
      - reading_time (datetime)
      - temperature (float)
      - vibration (float)
    """
    prompt = f"""
    Convert the following natural language question into a valid MySQL-compatible SQL query for TiDB.
    {schema}
    Question: {question}
    Only output the SQL query, nothing else.
    select only device_id, status, reading_time, temperature and vibration
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
        port=int(os.environ["TIDB_PORT"]),
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
            return [], col_names

        result_text = " | ".join(col_names) + "\n"
        result_text += "\n".join([" | ".join(str(x) for x in row) for row in rows[:10]])
        if len(rows) > 10:
            result_text += f"\n...and {len(rows)-10} more rows."
        return rows, col_names 

    except Exception as e:
        print("============error==================")
        print(e)
        print("===============================")
        return None, [f"Error executing query: {e}"]


def plot_results(rows, col_names, filename="plot.png"):
    """Create a simple line plot from query results"""
    import pandas as pd

    df = pd.DataFrame(rows, columns=col_names)

    # Ensure numeric columns are available
    if not {"reading_time", "temperature", "vibration"}.issubset(df.columns):
        return None  # Don't plot if required columns are missing

    # Plot temperature & vibration over time
    plt.figure(figsize=(8, 4))
    plt.plot(df["reading_time"], df["temperature"], label="Temperature", marker="o")
    plt.plot(df["reading_time"], df["vibration"], label="Vibration", marker="x")
    plt.xlabel("Reading Time")
    plt.ylabel("Value")
    plt.title("Sensor Readings Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to bytes buffer (so no local file needed)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

            

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

    # Deduplication
    if event_id in processed_events:
        return
    processed_events.add(event_id)

    # Ignore bot messages
    if subtype == "bot_message" or user_id == BOT_ID:
        return

    if text:
        sql_query = nl_to_sql(text)
        rows, col_names = run_query(sql_query)

        if rows is None:
            client.chat_postMessage(channel=channel_id, text=f"SQL:\n```{sql_query}```\n\nError: {col_names[0]}")
            return

        # --- If user asks for a graph ---
        if "plot" in text.lower() or "graph" in text.lower() or "chart" in text.lower():
            buf = plot_results(rows, col_names)
            if buf:
                client.files_upload(
                    channels=channel_id,
                    file=buf,
                    filename="plot.png",
                    title="Sensor Data Plot"
                )
            else:
                client.chat_postMessage(channel=channel_id, text="Could not generate plot for this query.")
        else:
            # Text-based result preview
            result_text = " | ".join(col_names) + "\n"
            result_text += "\n".join([" | ".join(str(x) for x in row) for row in rows[:10]])
            if len(rows) > 10:
                result_text += f"\n...and {len(rows)-10} more rows."

            client.chat_postMessage(channel=channel_id, text=f"SQL:\n```{sql_query}```\n\nResults:\n```{result_text}```")


if __name__ == "__main__":
    app.run(debug=True, port=3000)

















