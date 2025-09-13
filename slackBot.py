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
import matplotlib.pyplot as plt
import io
from vertexai.language_models import TextEmbeddingModel


# --- Initialize Vertex AI ---
vertexai.init(project=os.environ['PROJECT_ID'], location="us-central1")

# Load Gemini model from Vertex AI
gemini_model = GenerativeModel("gemini-2.0-flash")



app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET_'], '/slack/events', app)

client = slack.WebClient(token=os.environ['SLACK_TOKEN_'])
BOT_ID = client.api_call("auth.test")['user_id']

THRESHOLDS = {
    "temperature": {
        "fail": "temperature < 60 or temperature > 95",
        "warning": "85.1 <= temperature <= 95",
        "ok": "otherwise",
    },
    "vibration": {
        "fail": "vibration > 1.5",
        "warning": "1.01 <= vibration <= 1.5",
        "ok": "otherwise",
    },
    "pressure": {
        "fail": "pressure < 1.0 or pressure > 2.5",
        "warning": "2.01 <= pressure <= 2.5",
        "ok": "otherwise",
    }
}



# --- Step 1: Classify query type ---
def classify_query(user_query):
    prompt = f"""
    You are a query classifier.
    If the query requires structured SQL facts (filters, aggregates, counts) → return "SQL".
    If the query is about semantic similarity or patterns → return "VECTOR".
    Query: {user_query}
    Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

        
# --- Step 2: Run SQL Query (NL → SQL → Execute) ---
def run_sql_query(user_query):
    schema = """
    Database: test
    Table: sensor_data1
    Columns:
      - device_id (string: e.g. device_1, device_2)
      - status (string: e.g. SUCCESS, FAIL)
      - reading_time (datetime)
      - temperature (float)
      - vibration (float)
    """
sql_prompt = f"""
Convert the following natural language question into a valid MySQL-compatible SQL query for TiDB.

Database: test
Table: sensor_data1
Columns:
  - device_id (string: e.g. device_1, device_2)
  - status (string: e.g. SUCCESS, FAIL)
  - reading_time (datetime)
  - temperature (float)
  - vibration (float)
  - pressure (float)

⚠️ Rules:
- Always include `device_id`, `reading_time`, `temperature`, and `vibration` in the SELECT clause.
- You may include other relevant columns (like `status` or `pressure`) if the query requires.
- Do NOT include `id` or `embedding`.
- Return only the SQL query, nothing else.
Question: {user_query}
Answer:
"""

    sql_response = gemini_model.generate_content(sql_prompt)
    sql_query = sql_response.text.strip().strip("```sql").strip("```")

    rows, col_names = run_query(sql_query)
    return rows, col_names, sql_query


# --- Step 3: Vector Search ---
def run_vector_search(user_query, top_k=5):
    # Create embedding for query
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    embeddings = model.get_embeddings([user_query])
    query_vector = embeddings[0].values

    # TiDB ANN search
    vector_sql = f"""
    SELECT device_id, reading_time, status,
           1 - (dot_product(embedding, JSON_ARRAY_PACK('{query_vector}')) /
           (norm(embedding) * norm(JSON_ARRAY_PACK('{query_vector}')))) AS similarity
    FROM sensor_data1
    ORDER BY similarity ASC
    LIMIT {top_k};
    """
    rows, col_names = run_query(vector_sql)
    return rows, col_names, vector_sql

# --- Step 4: Main handler ---
def handle_query(user_query):
    mode = classify_query(user_query)
    if mode == "SQL":
        rows, col_names, sql = run_sql_query(user_query)
    else:
        rows, col_names, sql = run_vector_search(user_query)

    # Format results
    if not rows:
        table_text = "No matching rows found."
    else:
        import pandas as pd
        df = pd.DataFrame(rows, columns=col_names)
        table_text = df.head(10).to_markdown(index=False)

    # Build thresholds text for Gemini
    thresholds_text = "\n".join([
        f"- {metric}: FAIL if {rules['fail']}, WARNING if {rules['warning']}, OK if {rules['ok']}"
        for metric, rules in THRESHOLDS.items()
    ])

    # Prompt Gemini
    prompt = f"""
    You are an assistant analyzing IoT sensor data for devices.

    User query:
    {user_query}

    Business rules for classification:
    {thresholds_text}

    Retrieved rows:
    {table_text}

    Instructions:
    - Apply the above rules exactly when explaining results.
    - If no rows match a failure condition, explicitly answer that the device has not failed.
    - Be concise but clear, explaining which metric(s) caused FAIL/WARNING if applicable.
    """

    response = gemini_model.generate_content(prompt)

    return {
        "mode": mode,
        "query": sql,
        "rows": rows,
        "cols": col_names,
        "semantic_answer": response.text.strip()
    }



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
    """Run query against TiDB and return results (excluding id and embeddings columns)."""
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

        # --- Filter out unwanted columns ---
        exclude_cols = {"id", "embedding"}
        keep_indexes = [i for i, col in enumerate(col_names) if col not in exclude_cols]
        filtered_col_names = [col_names[i] for i in keep_indexes]
        filtered_rows = [[row[i] for i in keep_indexes] for row in rows]

        # (optional) preview text if you still need it
        result_text = " | ".join(filtered_col_names) + "\n"
        result_text += "\n".join([" | ".join(str(x) for x in row) for row in filtered_rows[:10]])
        if len(filtered_rows) > 10:
            result_text += f"\n...and {len(filtered_rows)-10} more rows."

        return filtered_rows, filtered_col_names

    except Exception as e:
        print("============error==================")
        print(e)
        print("===============================")
        return None, [f"Error executing query: {e}"]



def plot_results(rows, col_names, filename="plot.png"):
    """Create a line plot for available metrics over time by device"""
    import pandas as pd

    df = pd.DataFrame(rows, columns=col_names)

    # Require time + device
    if "reading_time" not in df.columns or "device_id" not in df.columns:
        return None  

    # Convert to datetime
    df["reading_time"] = pd.to_datetime(df["reading_time"])

    plt.figure(figsize=(10, 5))

    # Collect metrics that actually exist
    metrics = [m for m in ["temperature", "vibration"] if m in df.columns]
    if not metrics:
        return None

    # Plot per device
    for device_id, group in df.groupby("device_id"):
        for metric in metrics:
            marker = "o" if metric == "temperature" else "x"
            plt.plot(
                group["reading_time"],
                group[metric],
                marker=marker,
                label=f"{device_id} - {metric}"
            )

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



def format_results_table(rows, col_names, max_rows=10):
    """Format query results into a table with markdown for Slack"""
    import pandas as pd

    df = pd.DataFrame(rows, columns=col_names)

    # Limit rows
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        footer = f"\n...and {len(df) - max_rows} more rows."
    else:
        df_display = df
        footer = ""

    # Build table string
    table = df_display.to_markdown(index=False)  # requires tabulate (pandas uses it)
    return f"```{table}```{footer}"


def format_results_blocks(rows, col_names, max_rows=5):
    """Format query results as Slack Block Kit blocks (avoids >2000 char errors)."""
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "📊 Sensor Data Results"}}
    ]

    # Limit rows
    display_rows = rows[:max_rows]
    if not display_rows:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "No data found."}})
        return blocks

    for row in display_rows:
        fields = []
        for col, val in zip(col_names, row):
            field_text = f"*{col}*: {str(val)[:1000]}"  # truncate to stay safe
            fields.append({"type": "mrkdwn", "text": field_text})

        blocks.append({"type": "section", "fields": fields})
        blocks.append({"type": "divider"})

    if len(rows) > max_rows:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"...and {len(rows)-max_rows} more rows"}]
        })

    return blocks


            

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
        result = handle_query(text)

        if result["rows"] is None:
            client.chat_postMessage(
                channel=channel_id,
                text=f"Query:\n```{result['query']}```\n\nError: No results"
            )
            return
        
        # If graph requested
        if "plot" in text.lower() or "graph" in text.lower() or "chart" in text.lower():
            buf = plot_results(result["rows"], result["cols"])  # also corrected key to "cols"
            if buf:
                client.files_upload_v2(
                    channel=channel_id,
                    initial_comment="Here is your sensor data plot ",
                    file_uploads=[
                        {
                            "file": buf,
                            "filename": "plot.png",
                            "title": "Sensor Data Plot"
                        }
                    ]
                )
            else:
                client.chat_postMessage(channel=channel_id, text="Could not generate plot for this query.")
        else:
            client.chat_postMessage(
                channel=channel_id,
                text=f"*Mode*: {result['mode']}\n*Query:*\n```{result['query']}```\n\n💡 {result['semantic_answer']}",
                blocks=format_results_blocks(result["rows"], result["cols"]) if result["rows"] else []
            )



if __name__ == "__main__":
    app.run(debug=True, port=3000)






































