from flask import Flask, request, jsonify
import os

from flask_cors import CORS


app = Flask(__name__)
CORS(app)

#
@app.route("/system", methods=["POST"])
def system_control():
    data = request.json
    command = data.get("command", "").lower()

    try:
        if "chrome" in command:
            os.system("start chrome")

        elif "vs code" in command:
            os.system("code")

        elif "terminal" in command:
            os.system("start cmd")

        elif "shutdown" in command:
            os.system("shutdown /s /t 5")

        else:
            return jsonify({"message": "Command not implemented yet"})

        return jsonify({"message": "System command executed successfully"})

    except Exception as e:
        return jsonify({"message": str(e)})

if __name__ == "__main__":
    app.run(port=5000)
