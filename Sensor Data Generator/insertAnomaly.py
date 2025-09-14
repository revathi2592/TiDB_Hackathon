import json
import pymysql
from datetime import datetime
import requests


# --------------------------
# CONFIG
# --------------------------
TIDB = {
    "host": "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "3QfRdCFvT88wpCx.root",
    "password": "W0o0czmbYdCNxx3H",
    "database": "test",
    "ssl_verify_cert": True,
    "ssl_verify_identity": True,
    "ssl_ca": r"C:\Users\rmani@deloitte.com\Documents\TiDB\isrgrootx1.pem"
}

DISTANCE_THRESHOLD = 0.20
DEFAULT_DAYS_WINDOW = 30
VECTOR_DIM = 1536  # Must match your TiDB VECTOR column length

# --------------------------
# DB HELPERS
# --------------------------
def get_conn():
    return pymysql.connect(**TIDB)

def insert_anomaly(conn, rec, match_distance=None, source="manual"):
    """Insert anomaly into TiDB (VECTOR column handled correctly)."""
    with conn.cursor() as cur:
        sql = """
        INSERT INTO anomalies
            (id, device_id, anomaly_time, temperature, vibration, pressure, embedding, match_distance, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(
            sql,
            (
                rec["id"],
                rec["device_id"],
                rec.get("anomaly_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                rec.get("temperature"),
                rec.get("vibration"),
                rec.get("pressure"),
                rec["embedding"],   # <-- pass list directly
                match_distance,
                source
            ),
        )
    conn.commit()

def bootstrap_rule_hit(temperature: float, vibration: float, pressure: float) -> bool:
    # Simple, transparent rules to seed the anomalies library
    if temperature is not None and temperature > 100:
        return True
    if vibration is not None and vibration > 1.85:
        return True
    # add more domain rules if you have them
    return False

def nearest_anomaly_in_tidb(conn, embedding, days_window=30):
    """Find the nearest anomaly using cosine distance."""
    if isinstance(embedding, str):
        embedding = json.loads(embedding)

    embedding = [float(x) for x in embedding]
    embedding_json = json.dumps(embedding)

    sql = f"""
        SELECT
            id,
            device_id,
            anomaly_time,
            temperature,
            vibration,
            pressure,
            VEC_COSINE_DISTANCE(embedding, CAST(%s AS VECTOR({VECTOR_DIM}))) AS match_distance
        FROM anomalies
        WHERE anomaly_time >= NOW() - INTERVAL %s DAY
        ORDER BY match_distance
        LIMIT 1
    """

    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(sql, (embedding_json, days_window))
        return cur.fetchone()

# --------------------------
# SLACK NOTIFY
# --------------------------
def notify_slack(deviceId, timestamp, temperature, vibration, reason):
    webhook_url = "https://hooks.slack.com/services/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"

    payload = {
        "text": f"⚠️ Device {deviceId} Alert!",
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*⚠️ Fault Alert Detected*"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Device ID:*\n{deviceId}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{timestamp}"},
                    {"type": "mrkdwn", "text": f"*Temperature:*\n{temperature} °C"},
                    {"type": "mrkdwn", "text": f"*Vibration:*\n{vibration} g"},
                    {"type": "mrkdwn", "text": f"*Detection Method:*\n{reason}"}
                ]
            }
        ]
    }

    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Slack error: {response.status_code}, {response.text}")    

def check_and_insert_using_tidb_vector(rec, days_window=DEFAULT_DAYS_WINDOW):
    if "embedding" not in rec:
        raise ValueError("Record must contain 'embedding' key with a list of floats")

    conn = get_conn()
    try:
        nearest = nearest_anomaly_in_tidb(conn, rec["embedding"], days_window=days_window)

        if nearest:
            distance = nearest["match_distance"]
            distance_str = f"{distance:.4f}" if distance is not None else "None"

            print(f"[DEBUG] Nearest anomaly for {rec['device_id']} "
                  f"at {nearest['anomaly_time']} → distance={distance_str}")

            if distance is not None and distance <= DISTANCE_THRESHOLD:
                print(f"[INFO] Distance {distance_str} <= threshold {DISTANCE_THRESHOLD}. Inserting.")
                insert_anomaly(conn, rec, match_distance=distance, source="vector")

                # 🔔 Send Slack notification
                notify_slack(
                    rec["device_id"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    rec.get("temperature"),
                    rec.get("vibration"),
                    f"Vector distance {distance_str} <= {DISTANCE_THRESHOLD}"
                )
            else:
                print(f"[INFO] Distance {distance_str} > threshold {DISTANCE_THRESHOLD}. Inserting anyway.")
                insert_anomaly(conn, rec, match_distance=distance, source="vector")

                notify_slack(
                    rec["device_id"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    rec.get("temperature"),
                    rec.get("vibration"),
                    f"Vector distance {distance_str} > {DISTANCE_THRESHOLD}"
                )
        else:
            # 🚨 No anomalies in DB yet → Only insert if bootstrap rules fire
            if bootstrap_rule_hit(rec.get("temperature"), rec.get("vibration"), rec.get("pressure")):
                print("[INFO] First bootstrap anomaly detected. Inserting.")
                insert_anomaly(conn, rec, match_distance=None, source="bootstrap-rule")

                notify_slack(
                    rec["device_id"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    rec.get("temperature"),
                    rec.get("vibration"),
                    "Bootstrap rule hit"
                )
            else:
                print("[INFO] No anomalies yet, but bootstrap rules not hit → Skipping insert.")
    finally:
        conn.close()



# --------------------------
# EXAMPLE USAGE
# --------------------------
if __name__ == "__main__":
    new_record = {
        "id": "r7",
        "device_id": "device_7",
        "temperature": 98.5,
        "vibration": 0.92,
        "pressure": 120.4,
        "embedding": [0.001, -0.002, 0.003] + [0.0]*(VECTOR_DIM-3)
    }

    #check_and_insert_using_tidb_vector(new_record)
