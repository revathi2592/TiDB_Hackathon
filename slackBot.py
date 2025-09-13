import slack_sdk as slack
import os, tempfile, io
from flask import Flask
from slackeventsapi import SlackEventAdapter
import pymysql
import matplotlib.pyplot as plt
import pandas as pd
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from google.cloud import secretmanager

# --- Initialize Vertex AI ---
vertexai.init(project=os.environ['PROJECT_ID'], location="us-central1")
gemini_model = GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET_'], '/slack/events', app)
client = slack.WebClient(token=os.environ['SLACK_TOKEN_'])
BOT_ID = client.api_call("auth.test")['user_id']

THRESHOLDS = {
    "temperature": {"fail": "temperature < 60 or temperature > 95", "warning": "85.1 <= temperature <= 95", "ok": "otherwise"},
    "vibration": {"fail": "vibration > 1.5", "warning": "1.01 <= vibration <= 1.5", "ok": "otherwise"},
    "pressure": {"fail": "pressure < 1.0 or pressure > 2.5", "warning": "2.01 <= pressure <= 2.5", "ok": "otherwise"}
}

# --- TiDB connection ---
def get_tidb_connection():
    client_sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['SECRET_PROJ_ID']}/secrets/tidb-ssl-ca/versions/latest"
    response = client_sm.access_secret_version(name=name)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(response.payload.data)
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
    try:
        conn = get_tidb_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        if not rows:
            return [], col_names

        # Filter out unwanted columns
        exclude_cols = {"id", "embedding"}
        keep_indexes = [i for i, col in enumerate(col_names) if col not in exclude_cols]
        filtered_col_names = [col_names[i] for i in keep_indexes]
        filtered_rows = [[row[i] for i in keep_indexes] for row in rows]

        return filtered_rows, filtered_col_names
    except Exception as e:
        print("Error:", e)
        return None, [f"Error executing query: {e}"]

# --- Plotting ---
def plot_results(rows, col_names):
    df = pd.DataFrame(rows, columns=col_names)
    if not {"reading_time", "device_id"}.issubset(df.columns):
        return None
    df["reading_time"] = pd.to_datetime(df["reading_time"])
    plt.figure(figsize=(10,5))
    metrics = [m for m in ["temperature", "vibration"] if m in df.columns]
    if not metrics: return None

    for device_id, group in df.groupby("device_id"):
        for metric in metrics:
            marker = "o" if metric=="temperature" else "x"
            plt.plot(group["reading_time"], group[metric], marker=marker, label=f"{device_id} - {metric}")

    plt.xlabel("Reading Time")
    plt.ylabel("Value")
    plt.title("Device Comparison - " + " & ".join(metrics).title())
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# --- Formatting Slack blocks ---
def format_results_blocks(rows, col_names, max_rows=5):
    blocks = [{"type":"header","text":{"type":"plain_text","text":"📊 Sensor Data Results"}}]
    display_rows = rows[:max_rows]
    if not display_rows:
        blocks.append({"type":"section","text":{"type":"mrkdwn","text":"No data found."}})
        return blocks

    for row in display_rows:
        fields = [{"type":"mrkdwn","text":f"*{col}*: {val}"} for col, val in zip(col_names, row)]
        blocks.append({"type":"section","fields":fields})
        blocks.append({"type":"divider"})
    if len(rows)>max_rows:
        blocks.append({"type":"context","elements":[{"type":"mrkdwn","text":f"...and {len(rows)-max_rows} more rows"}]})
    return blocks

# --- NL to SQL ---
def nl_to_sql(question: str):
    schema = """
    Database: test
    Table: sensor_data1
    Columns: device_id, status, reading_time, temperature, vibration, pressure
    """
    prompt = f"""
    Convert this question to valid SQL (MySQL/TiDB) and select device_id, reading_time, temperature, vibration, pressure:
    {schema}
    Question: {question}
    Only return SQL query.
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip().strip("```sql").strip("```")

# --- Vector search ---
def run_vector_search(user_query, top_k=5):
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    embeddings = model.get_embeddings([user_query])
    query_vector = embeddings[0].values
    vector_sql = f"""
    SELECT device_id, reading_time, temperature, vibration, pressure, status,
           1 - (dot_product(embedding, JSON_ARRAY_PACK('{query_vector}')) /
           (norm(embedding) * norm(JSON_ARRAY_PACK('{query_vector}')))) AS similarity
    FROM sensor_data1
    ORDER BY similarity ASC
    LIMIT {top_k};
    """
    rows, col_names = run_query(vector_sql)
    return rows, col_names, vector_sql

# --- Classify query ---
def classify_query(user_query):
    prompt = f"""
    You are a query classifier.
    SQL → structured facts, VECTOR → semantic similarity.
    Query: {user_query}
    Answer:
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# --- Handle query ---
def handle_query(user_query):
    mode = classify_query(user_query)
    if mode=="SQL":
        sql = nl_to_sql(user_query)
        rows, col_names = run_query(sql)
    else:
        rows, col_names, sql = run_vector_search(user_query)

    if not rows:
        table_text = "No matching rows found."
    else:
        table_text = pd.DataFrame(rows, columns=col_names).head(10).to_markdown(index=False)

    thresholds_text = "\n".join([f"- {m}: FAIL if {r['fail']}, WARNING if {r['warning']}, OK if {r['ok']}" 
                                 for m,r in THRESHOLDS.items()])

    prompt = f"""
    You are an assistant analyzing IoT sensor data.
    User query: {user_query}
    Business rules: {thresholds_text}
    Retrieved rows: {table_text}
    Instructions: Explain which metrics caused FAIL/WARNING if applicable.
    """
    response = gemini_model.generate_content(prompt)
    return {"mode": mode, "query": sql, "rows": rows, "cols": col_names, "semantic_answer": response.text.strip()}

# --- Slack message event ---
processed_events = set()

@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    event_id = payload.get('event_id')
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    subtype = event.get('subtype')

    if event_id in processed_events or subtype=="bot_message" or user_id==BOT_ID:
        return
    processed_events.add(event_id)

    if not text:
        return

    # --- If plot requested, skip Gemini/semantic ---
    if "plot" in text.lower() or "graph" in text.lower() or "chart" in text.lower():
        sql_query = nl_to_sql(text)
        rows, col_names = run_query(sql_query)
        buf = plot_results(rows, col_names)
        if buf:
            client.files_upload_v2(
                channel=channel_id,
                initial_comment="Here is your sensor data plot",
                file_uploads=[{"file": buf, "filename": "plot.png", "title": "Sensor Data Plot"}]
            )
        else:
            client.chat_postMessage(channel=channel_id, text="Could not generate plot for this query.")
    else:
        result = handle_query(text)
        client.chat_postMessage(
            channel=channel_id,
            text=f"*Mode*: {result['mode']}\n*Query:*\n```{result['query']}```\n\n💡 {result['semantic_answer']}",
            blocks=format_results_blocks(result["rows"], result["cols"]) if result["rows"] else []
        )

if __name__ == "__main__":
    app.run(debug=True, port=3000)
