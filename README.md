yay -Syu python311
rm src/chatbot/chatbot_model.h5 src/chatbot/classes.pkl src/chatbot/words.pkl
deactivate  # if in any existing venv
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
cd src
pip install -r requirements.txt
python chatbot/training.py
python app.py