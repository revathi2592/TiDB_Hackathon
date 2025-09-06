from flask import Flask, request, jsonify
 
app = Flask(__name__)


@app.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.get_json()
    print("======================================")
    print(data)
 
    # Respond to Slack's URL verification challenge
    if data.get('type') == 'url_verification':
        return jsonify({'challenge': data.get('challenge')})
 
    # Handle other event types here
    return '', 200
 
if __name__ == '__main__':

    app.run(port=3000)

