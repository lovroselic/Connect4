# make_ckpt_b64.py
import base64, zlib
from pathlib import Path

pt_path = Path("PPO_Models/PPO_11.pt") 
raw = pt_path.read_bytes()
blob = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
print(blob)
