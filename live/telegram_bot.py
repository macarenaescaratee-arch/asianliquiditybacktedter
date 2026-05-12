text = update["message"].get("text", "")
import os
import requests
import time

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

URL = f"https://api.telegram.org/bot{TOKEN}"

last_update_id = None

def send_message(text):
    requests.post(
        f"{URL}/sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text": text
        }
    )

send_message("🔥 BOT TELEGRAM CONECTADO")

while True:
    try:
        response = requests.get(
            f"{URL}/getUpdates",
            params={
                "offset": last_update_id,
                "timeout": 30
            }
        ).json()

        for update in response["result"]:

            last_update_id = update["update_id"] + 1

            if "message" not in update:
                continue

            text = update["message"].get("text", "").strip().lower()

            if text == "/status":
                send_message("✅ Bot activo y online")

            elif text == "/si":
                send_message("🔥 Operaremos sesión Asia")

            elif text == "/no":
                send_message("😴 Bot descansando esta noche")

    except Exception as e:
        print(e)

    time.sleep(2)

