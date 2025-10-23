import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# Modeli yükle
model_path = os.path.join("models", "allegro_hand_demo.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Kontrol parametreleri
kp, kd = 20.0, 0.5
close_pos = np.array([0.5, 0.5, 0.5, 0.5])
open_pos = np.zeros(4)

def pd_control(target):
    error = target - data.qpos[:4]
    derror = -data.qvel[:4]
    data.ctrl[:] = kp * error + kd * derror

# Görselleştirme
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    phase = "close"
    while viewer.is_running():
        t = time.time() - start

        if t < 2.0:
            pd_control(close_pos)
        elif t < 4.0:
            pd_control(close_pos)
        elif t < 6.0:
            pd_control(open_pos)
        else:
            start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()
