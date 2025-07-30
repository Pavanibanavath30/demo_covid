import google.generativeai as genai
genai.configure(api_key="AIzaSyD6xqLaCrWtU7K9Kzj5VI5wheA49EDtuY4")
model=genai.GenerativeModel(model_name="gemini-2.0-flash")
chat=model.start_chat(history=[])
#prmt="Where is the nearest hospital near mysammaguda?"
while True:
    prmt=input()
    if(prmt=="exit"):
        break
    res=chat.send_message(prmt)
    print(res.text)