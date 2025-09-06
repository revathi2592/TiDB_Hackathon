from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

#SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")  # store securely in env var
SLACK_BOT_TOKEN ="xoxb-9437322388582-9455478622388-QTsG2cjfSnHpDI93X5POJGmN"

@app.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.get_json()

    # 1. Handle Slack's URL verification challenge
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})

    # 2. Handle actual events
    if data.get("type") == "event_callback":
        event = data.get("event", {})

        # Capture message event
        if event.get("type") == "message" and "bot_id" not in event:
            user = event.get("user")
            text = event.get("text")
            channel = event.get("channel")

            print(f"Message from {user} in {channel}: {text}")

            # Echo the message back to Slack (for demo)
            reply_text = f"You said: {text}"
            send_message_to_slack(channel, reply_text)

    return "", 200


def send_message_to_slack(channel, text):
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    payload = {"channel": channel, "text": text}
    requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
