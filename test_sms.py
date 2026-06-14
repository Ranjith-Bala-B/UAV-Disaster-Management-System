from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

SID   = "ACfee9b9cf6cec670dfef0822f1dc53f87"
TOKEN = "e86c414d638b1ce96aa816db5e18a51e"
FROM  = "+18143880207"
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
