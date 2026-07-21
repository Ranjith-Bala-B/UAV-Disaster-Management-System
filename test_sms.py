import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from dotenv import load_dotenv
load_dotenv()

SID   = os.getenv("TWILIO_SID")
TOKEN = os.getenv("TWILIO_TOKEN")
FROM  = os.getenv("TWILIO_FROM")
TO    = "+916384742246"

print("Testing Twilio SMS...")
try:
    c = Client(SID, TOKEN)
    m = c.messages.create(body="UAV DISASTER ALERT TEST - System working!", from_=FROM, to=TO)
    print("SUCCESS! SID:", m.sid, "| Status:", m.status)
except TwilioRestException as e:
    print("TWILIO ERROR:", e.code, "|", e.msg)
except Exception as e:
    print("EXCEPTION:", str(e))
