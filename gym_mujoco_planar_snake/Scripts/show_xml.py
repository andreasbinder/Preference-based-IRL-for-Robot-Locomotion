from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from pathlib import Path


#model = load_model_from_path("tosser.xml")
path = Path()

model = load_model_from_path('/home/andreas/LRZ_Sync+Share/MasterThesis-master/gym_mujoco_planar_snake/gym_mujoco_planar_snake/envs/assets/planar_snake_cars_servo.xml')
sim = MjSim(model)

viewer = MjViewer(sim)


sim_state = sim.get_state()

while True:
    sim.set_state(sim_state)

    for i in range(1000):
        if i < 150:
            sim.data.ctrl[:] = 0.0
        else:
            sim.data.ctrl[:] = -1.0
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break