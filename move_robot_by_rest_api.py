#!/usr/bin/env python3
import time
import threading
import serial
from flask import Flask, jsonify, request, abort

# ------------------ Configuration ------------------
ARDUINO_PORT = "/dev/ttyUSB0"
ARDUINO_BAUD = 115200

# Arduino commands (must match Arduino firmware)
CMD_STOP     = "STOP\n"

CMD_FORWARD  = "FORWARD\n"
CMD_BACKWARD = "BACKWARD\n"
CMD_LEFT     = "LEFT\n"
CMD_RIGHT    = "RIGHT\n"
CMD_FL       = "FL\n"
CMD_FR       = "FR\n"
CMD_BL       = "BL\n"
CMD_BR       = "BR\n"

CMD_ROT_L    = "LR\n"
CMD_ROT_R    = "RR\n"

CMD_SPEED_LOW  = "LOW\n"
CMD_SPEED_MED  = "MEDIUM\n"
CMD_SPEED_HIGH = "HIGH\n"
# ---------------------------------------------------

app = Flask(__name__)

ser = None
ser_lock = threading.Lock()
last_cmd = ""


def send_cmd(cmd: str):
    """Send a command to Arduino only if it differs from the last one."""
    global last_cmd, ser

    if cmd == last_cmd:
        return {"sent": False, "cmd": cmd.strip(), "reason": "same_as_last"}

    with ser_lock:
        ser.write(cmd.encode("utf-8"))
        ser.flush()

    last_cmd = cmd
    print(f"-> Arduino: {cmd.strip()}")
    return {"sent": True, "cmd": cmd.strip()}


# -------- Health --------
@app.get("/health")
def health():
    return jsonify({"ok": True})


# -------- Movement --------
@app.post("/stop")
def stop():
    return jsonify(send_cmd(CMD_STOP))


@app.post("/move/forward")
def move_forward():
    return jsonify(send_cmd(CMD_FORWARD))


@app.post("/move/backward")
def move_backward():
    return jsonify(send_cmd(CMD_BACKWARD))


@app.post("/move/left")
def move_left():
    return jsonify(send_cmd(CMD_LEFT))


@app.post("/move/right")
def move_right():
    return jsonify(send_cmd(CMD_RIGHT))


@app.post("/move/fl")
def move_fl():
    return jsonify(send_cmd(CMD_FL))


@app.post("/move/fr")
def move_fr():
    return jsonify(send_cmd(CMD_FR))


@app.post("/move/bl")
def move_bl():
    return jsonify(send_cmd(CMD_BL))


@app.post("/move/br")
def move_br():
    return jsonify(send_cmd(CMD_BR))


# -------- Rotation --------
@app.post("/rotate/left")
def rotate_left():
    return jsonify(send_cmd(CMD_ROT_L))


@app.post("/rotate/right")
def rotate_right():
    return jsonify(send_cmd(CMD_ROT_R))


# -------- Speed --------
@app.post("/speed/low")
def speed_low():
    return jsonify(send_cmd(CMD_SPEED_LOW))


@app.post("/speed/medium")
def speed_medium():
    return jsonify(send_cmd(CMD_SPEED_MED))


@app.post("/speed/high")
def speed_high():
    return jsonify(send_cmd(CMD_SPEED_HIGH))


# -------- Optional generic command --------
ALLOWED = {
    "STOP": CMD_STOP,
    "FORWARD": CMD_FORWARD,
    "BACKWARD": CMD_BACKWARD,
    "LEFT": CMD_LEFT,
    "RIGHT": CMD_RIGHT,
    "FL": CMD_FL,
    "FR": CMD_FR,
    "BL": CMD_BL,
    "BR": CMD_BR,
    "LR": CMD_ROT_L,
    "RR": CMD_ROT_R,
    "LOW": CMD_SPEED_LOW,
    "MEDIUM": CMD_SPEED_MED,
    "HIGH": CMD_SPEED_HIGH,
}


@app.post("/cmd")
def cmd():
    """
    POST /cmd
    JSON body:
      { "cmd": "FORWARD" }
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("cmd") or "").upper().strip()

    if name not in ALLOWED:
        abort(400, description=f"cmd must be one of {sorted(ALLOWED.keys())}")

    return jsonify(send_cmd(ALLOWED[name]))


# -------- Startup --------
def connect_arduino():
    global ser
    ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
    time.sleep(2.0)  # allow Arduino reset
    send_cmd(CMD_SPEED_MED)
    send_cmd(CMD_STOP)
    print("Arduino connected")


if __name__ == "__main__":
    connect_arduino()
    app.run(host="0.0.0.0", port=5000, debug=False)
