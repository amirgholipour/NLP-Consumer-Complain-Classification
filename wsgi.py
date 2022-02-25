import json
from flask import Flask, jsonify, request
from prediction import predict
application = Flask(__name__)


@application.route('/')
@application.route('/status')
def status():
    return jsonify({'status': 'ok'})


@application.route('/predictions', methods=['POST'])
def NlpClassification():
    data = request.data or '{}'
    # print (data)
    body = json.loads(data)
    # print (body)
    return predict(body)
