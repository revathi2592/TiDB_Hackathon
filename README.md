## Replicating the Project

To replicate this project:  

- After setting the TiDB connection details and Slackbot token in the environment file, deploy `slackBot.py` on any cloud provider (AWS, GCP, Azure).  
- Use the scripts inside **Sensor Data Generator** to simulate real-time data ingestion in the TiDB table.  
- `insertAnomaly.py` will calculate the similarity between new records and existing anomalies, and notify the Slack channel.  

---

## Components and Tools Used

| Component            | Tool/Service Used                          |
|-----------------------|---------------------------------------------|
| Data Simulation       | Python Script                              |
| Database & Vector Search | TiDB with HNSW Indexing                 |
| Embedding Model       | Text Embedding (1536D)                     |
| Anomaly Detection     | Bootstrap Rule + Cosine Similarity         |
| Alerts & Interaction  | Slack + Slack Bot                          |
| Query Understanding   | LLMs for Classification & Embedding        |
| Deployment            | Docker                                     |
| Hosting               | Cloud Platform (e.g., AWS, GCP, Azure)     |
