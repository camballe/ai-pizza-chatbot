from flask import Flask, render_template, request, jsonify
from chatbot.model import ChatbotModel

app = Flask(__name__)
chatbot = ChatbotModel()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    intents_list = chatbot.predict_class(user_input)
    response = chatbot.get_response(intents_list)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
