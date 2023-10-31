from flask import Flask, request, jsonify
import requests  # To send image data to the ML model server



def main():

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files["file"]
    # Add logic to send this file to ML Model Server
    response = requests.post("http://localhost:MODEL_PORT/predict", files={"file": file})
    tags = response.json()
    return jsonify({"tags": tags})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
