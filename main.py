from flask import Flask, request
from utils.model import Summarizer

model = Summarizer()

app = Flask(__name__)

@app.route("/")
def test():
    return {'test': "Server Online!"}


@app.route("/summary",methods=["POST"])
def summary():
    text = request.form.get('text')
    summary = model.infer(text)
    return {"summary":summary}

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5001,debug=True)
