from flask import Flask, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

eye_checker_process = None

@app.route('/execute', methods=['POST'])
def execute_eye_checker():
    try:
        # Run the eyechecker.py script and capture the output
        result = subprocess.run(
            ["python", "eyechecker.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the script ran successfully
        if result.returncode == 0:
            return jsonify({"status": "success", "output": result.stdout}), 200
        else:
            return jsonify({"status": "error", "output": result.stderr}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start', methods=['POST'])
def start_eye_checker():
    global eye_checker_process
    if eye_checker_process is None or eye_checker_process.poll() is not None:
        eye_checker_process = subprocess.Popen(["python", "eyechecker.py", "start"])
        return jsonify({"status": "started"}), 200
    return jsonify({"status": "already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_eye_checker():
    global eye_checker_process
    if eye_checker_process and eye_checker_process.poll() is None:
        eye_checker_process.terminate()
        eye_checker_process = None
        return jsonify({"status": "stopped"}), 200
    return jsonify({"status": "not running"}), 400

if __name__ == '__main__':
    app.run(debug=True)
