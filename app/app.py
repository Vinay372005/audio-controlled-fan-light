from flask import Flask, render_template, request
from predict import predict_audio
from fsm import ClapFSM
import os

app = Flask(__name__)
fsm = ClapFSM()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    state = fsm.state

    if request.method == "POST":
        file = request.files["audio"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        result = predict_audio(path)
        state = fsm.update(result)

    return render_template("index.html", result=result, state=state)

if __name__ == "__main__":
    app.run(debug=True)
