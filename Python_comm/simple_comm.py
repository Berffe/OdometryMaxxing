import time
import numpy as np

from gz.transport13 import Node

try:
    from gz.msgs10.image_pb2 import Image
except ImportError:
    from gz.msgs.image_pb2 import Image


TOPIC = "/bee_x500/camera/image"
# If this does not work, replace TOPIC with the exact one from:
# gz topic -l | grep -E "camera|image|bee|landing"


node = Node()
frame_count = 0


def on_image(msg: Image):
    global frame_count

    frame_count += 1

    width = msg.width
    height = msg.height

    data = np.frombuffer(msg.data, dtype=np.uint8)

    # Basic RGB/RGBA handling
    if len(data) == width * height * 3:
        frame = data.reshape((height, width, 3))
    elif len(data) == width * height * 4:
        frame = data.reshape((height, width, 4))
    else:
        print(f"Received image, but unexpected size: {len(data)} bytes")
        print(f"width={width}, height={height}, step={msg.step}")
        return

    if frame_count % 30 == 0:
        print(f"Received frame {frame_count}: shape={frame.shape}")


if not node.subscribe(Image, TOPIC, on_image):
    print(f"Failed to subscribe to {TOPIC}")
    raise SystemExit(1)

print(f"Listening to camera topic: {TOPIC}")

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopped.")