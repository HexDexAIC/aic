import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np, cv2
rclpy.init()
n = Node("snap")
got = []
def cb(m):
    got.append(m)
sub = n.create_subscription(Image, "/center_camera/image", cb, 1)
import time
t0 = time.time()
while not got and time.time() - t0 < 8:
    rclpy.spin_once(n, timeout_sec=0.5)
if got:
    m = got[0]
    arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
    bgr = arr[:, :, ::-1] if m.encoding == "rgb8" else arr
    cv2.imwrite("/tmp/live.jpg", bgr)
    print("saved", m.width, m.height)
else:
    print("no img")
