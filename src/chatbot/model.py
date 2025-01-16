import json
import pickle
import random
import openai
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

class ChatbotModel:
    def __init__(self, openai_api_key):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open("chatbot/intents.json", encoding="utf-8").read())
        self.words = pickle.load(open("chatbot/words.pkl", "rb"))
        self.classes = pickle.load(open("chatbot/classes.pkl", "rb"))
        self.contexts = pickle.load(open("chatbot/contexts.pkl", "rb"))
        self.model = load_model("chatbot/chatbot_model.h5")
        self.current_context = ""
        self.conversation_history = []
        openai.api_key = openai_api_key

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) 
                        for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def context_vector(self):
        """Create a vector representing the current context"""
        return np.array([1 if self.current_context == ctx else 0 
                        for ctx in self.contexts])

    def predict_class(self, sentence):
        # Get bag of words and context vector
        bow = self.bag_of_words(sentence)
        context = self.context_vector()
        
        # Predict using both inputs
        res = self.model.predict([
            np.array([bow]),
            np.array([context])
        ])[0]
        
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({
                "intent": self.classes[r[0]], 
                "probability": str(r[1])
            })
        return return_list

    def get_response(self, intents_list, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        
        if not intents_list:
            response = self.get_llm_response(user_input)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        tag = intents_list[0]["intent"]
        list_of_intents = self.intents["intents"]
        
        # Find matching intent
        matched_intent = None
        for intent in list_of_intents:
            if intent["tag"] == tag:
                matched_intent = intent
                break
        
        if matched_intent:
            # Check if this intent requires specific context
            if "requires_context" in matched_intent:
                required_context = matched_intent["requires_context"]
                if required_context != self.current_context:
                    response = self.get_llm_response(
                        f"User is trying to {tag} but current context is {self.current_context}"
                    )
                    self.conversation_history.append({"role": "assistant", "content": response})
                    return response
            
            # Get random response from intent
            response = random.choice(matched_intent["responses"])
            
            # Update context if this intent sets new context
            if "context_set" in matched_intent:
                self.current_context = matched_intent["context_set"]
            
            # Clear context if specified
            if "context_clear" in matched_intent and matched_intent["context_clear"]:
                self.current_context = ""
            
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Fallback to LLM
        response = self.get_llm_response(user_input)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def get_llm_response(self, user_input):
        prompt = (
            "You are a helpful pizza-ordering assistant. "
            "Current conversation context: " + self.current_context + "\n"
            "Respond naturally to the user while maintaining appropriate context.\n"
            "Previous conversation:\n" + 
            "\n".join([f"{msg['role']}: {msg['content']}" 
                      for msg in self.conversation_history[-5:]])
            + f"\nUser: {user_input}\nBot:"
        )
        
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return "I'm sorry, I'm having trouble processing your request right now."

    def reset_conversation(self):
        """Reset the conversation state"""
        self.current_context = ""
        self.conversation_history = []