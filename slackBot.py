import slack_sdk    as slack
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, Response
from slackeventsapi import SlackEventAdapter

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter("74c46a189125278236c8b5a090d8ab02", '/slack/events', app)

client = slack.WebClient(token="xoxb-9437322388582-9455478622388-RU87KWoTmc2QjMnb3kvXFqgv")
BOT_ID = client.api_call("auth.test")['user_id']
@ slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    if BOT_ID != user_id:
        client.chat_postMessage(
                channel=channel_id,  text=text)
if __name__ == "__main__":

    app.run(debug=True, port=8080)

