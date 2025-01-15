Run the following commands to start the chatbot:

```
deactivate # if inside a venv
rm -rf venv # if venv exists in current dir
python3.11 -m venv venv
source venv/bin/activate
cd src
pip install -r requirements.txt
python chatbot/training.py
python app.py
```
