from flask import Flask, render_template, request, jsonify
from chatbot.model import ChatbotModel

app = Flask(__name__)
chatbot = ChatbotModel(openai_api_key="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")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    intents_list = chatbot.predict_class(user_input)
    response = chatbot.get_response(intents_list, user_input)
    return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset_conversation():
    """
    Reset the chatbot's conversation state, including context and history.
    """
    chatbot.reset_conversation()
    return jsonify({"message": "Conversation reset successfully."})


if __name__ == "__main__":
    app.run(debug=True)
