from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth

load_dotenv()

def send_whatsapp(message_text, to_number):
    sid = os.getenv("TWILIO_SID")
    token = os.getenv("TWILIO_TOKEN")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

    data = {
        "From": os.getenv("TWILIO_WHATSAPP_FROM"),
        "To": f"whatsapp:{to_number}",
        "Body": message_text
    }

    response = requests.post(
        url,
        data=data,
        auth=HTTPBasicAuth(sid, token)
    )

    return response.status_code == 201, response.text


# Test Message
success, result = send_whatsapp(
    "hello arul",
    "+916384742246"
)

print("Success:", success)
print("Result:", result)