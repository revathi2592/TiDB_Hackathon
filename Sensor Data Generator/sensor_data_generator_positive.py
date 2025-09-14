import random
import uuid
from datetime import datetime, timedelta
import json
import pymysql
import numpy as np
from embeddingGenerator_v1 import create_embedding
from insertAnomaly import check_and_insert_using_tidb_vector

DEVICE_IDS = [f"device_{i}" for i in range(1, 6)]

TIDB_HOST = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com"
TIDB_PORT = 4000
TIDB_USER = "3QfRdCFvT88wpCx.root"
TIDB_PASSWORD = "W0o0czmbYdCNxx3H"
TIDB_DATABASE = "test"

# -------------------------
# Normalize embedding vector
# -------------------------
def normalize_embedding(embedding):
    arr = np.array(embedding, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return (arr / norm).tolist()

def classify_status(temperature, vibration, pressure):
    status_levels = []

    if temperature < 60 or temperature > 95:
        status_levels.append("FAIL")
    elif 85.1 <= temperature <= 95:
        status_levels.append("WARNING")
    else:
        status_levels.append("OK")

    if vibration > 1.5:
        status_levels.append("FAIL")
    elif 1.01 <= vibration <= 1.5:
        status_levels.append("WARNING")
    else:
        status_levels.append("OK")

    if pressure < 1.0 or pressure > 2.5:
        status_levels.append("FAIL")
    elif 2.01 <= pressure <= 2.5:
        status_levels.append("WARNING")
    else:
        status_levels.append("OK")

    if "FAIL" in status_levels:
        return "FAIL"
    elif "WARNING" in status_levels:
        return "WARNING"
    else:
        return "OK"

def generate_sensor_reading(device_id, timestamp):
    temperature = round(random.uniform(60.0, 85.0), 2)
    vibration = round(random.uniform(0.5, 1.0), 2)
    pressure = round(random.uniform(1.0, 2.0), 2)

    status = classify_status(temperature, vibration, pressure)

    raw_embedding = create_embedding(
        f"Device ID:{device_id}|Timestamp:{timestamp}|Temperature:{temperature}|Vibration:{vibration}|Pressure:{pressure}|Status:{status}"
    )

    normalized_embedding = normalize_embedding(raw_embedding)

    reading = {
        "id": str(uuid.uuid4()),
        "device_id": device_id,
        "timestamp": timestamp.isoformat(),
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "status": status,
        "embedding": normalized_embedding
    }
    return reading

def generate_past_2_hours_data(interval_seconds=10,
                               output_file=r"C:\Users\rmani@deloitte.com\Documents\TiDB\sampleOutputData\sensor_data.json"):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=1)
    current_time = start_time

    conn = pymysql.connect(
        host=TIDB_HOST,
        port=TIDB_PORT,
        user=TIDB_USER,
        password=TIDB_PASSWORD,
        database=TIDB_DATABASE,
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        ssl_ca=r"C:\Users\rmani@deloitte.com\Documents\TiDB\isrgrootx1.pem"
    )
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO sensor_data1 (id, device_id, temperature, vibration, pressure, status, reading_time, embedding)
        VALUES (%s, %s, %s, %s, %s, %s,%s,%s)
    """
    try:
        with open(output_file, "w") as f:
            while current_time <= end_time:
                for device in DEVICE_IDS:
                    reading = generate_sensor_reading(device, current_time)
                    cursor.execute(
                        insert_sql,
                        (
                            reading["id"],
                            reading["device_id"],
                            reading["temperature"],
                            reading["vibration"],
                            reading["pressure"],
                            reading["status"],
                            reading["timestamp"],
                            json.dumps(reading["embedding"])
                        )
                    )
                    conn.commit()

                    # Pass normalized embedding to anomaly check
                    new_record = {
                        "id": reading["id"], 
                        "device_id": reading["device_id"],
                        "reading_time": reading["timestamp"],
                        "temperature": reading["temperature"],
                        "vibration": reading["vibration"],
                        "pressure": reading["pressure"],
                        "embedding": json.dumps(reading["embedding"])
                    }
                    check_and_insert_using_tidb_vector(new_record, days_window=30)

                    f.write(json.dumps(reading) + "\n")
                current_time += timedelta(seconds=interval_seconds)

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    generate_past_2_hours_data()
