from flask import Flask, request, jsonify
from api_ai import update_index, generate_answer

app = Flask(__name__)


@app.route('/update_index', methods=['POST'])
def item_rec():
    payload = request.json
    ans = update_index(payload["index_id"], payload["type"])
    return jsonify(ans)


@app.route('/generate_answer_channel', methods=['POST'])
def generate_answer_channel():
    payload = request.json
    ans = generate_answer(payload["msg"], payload["index_id"])
    return jsonify(ans)




@app.route('/status', methods=["GET"])
def status():
    return jsonify({"message": "ok"})

app.run(host='0.0.0.0', port=8020, debug=False )
