# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import base64
import requests
from io import BytesIO
import banana_dev as banana


model_payload = {
    "youtube_url":"",
    "wav_url":"",
    "mp3_url":"https://tmpfiles.org/dl/987390/hobbies.mp3",
    "language": "en",
    "num_speakers": 2
}

res = requests.post("http://localhost:8000/",json=model_payload)

print(res.text)

