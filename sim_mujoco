import time
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R
import eas
import ctrnn 

transient = 10000
duration = 30000    
maxForce = 500             

nnsize = 5
sensor_inputs = 2
motor_outputs = 2

dt = 0.01
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0
nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)

m = mujoco.MjModel.from_xml_path('melty_sim_mujoco.xml')
d = mujoco.MjData(m)
print(d)
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()
genotype = np.load("bestgenotype.npy")

nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
nn.initializeState(np.zeros(nnsize))


def quat2euler(quat_mujoco):
    #mujocoy quat is constant,x,y,z,
    #scipy quaut is x,y,z,constant
    # quat_scipy = np.array([quat_mujoco[3],quat_mujoco[0],quat_mujoco[1],quat_mujoco[2]])

    r = R.from_matrix(quat_mujoco.reshape(3,3))
    euler = r.as_euler('xyz', degrees=True)

    return euler

def controller(m, d):
    #put the controller here. This function is called inside the simulation.
    # pass
    # print("controller")
    # print(d.xpos[1])
    forward = -90
    direction = np.round(quat2euler(d.xmat[1]), decimals=4)[2]
    nn.step(dt,(forward/180,direction/180))
    d.ctrl[0] = 20 * nn.out()[0] + 200
    d.ctrl[1] = 20 * nn.out()[1] - 200
    # print(d.ctrl[0])
    # print(direction)
    # # print(d.xmat[1])
    # if (direction > 87 and direction < 93):
    #     d.ctrl[0] = 210
    #     d.ctrl[1] = -190
    # else:
    #     d.ctrl[0] = 200
    #     d.ctrl[1] = -200

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)


mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(m, maxgeom=10000)
context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)

cam.azimuth = 90 ; cam.elevation = -45 ; cam.distance =  100
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])


mujoco.set_mjcb_control(controller)
# i = 0
while not glfw.window_should_close(window):
    time_prev = d.time

    

    while (d.time - time_prev < 1.0/60.0):
        # print(d.time)
        # i += 1
        # print(i)
        mujoco.mj_step(m, d)
        # mujoco.mj_forward(m,d)
    # if (i>duration + transient):
    #     break

    if (d.time>=duration):
        # mujoco.mj_resetData(m, d)
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mujoco.mjv_updateScene(m, d, opt, None, cam,
                       mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
