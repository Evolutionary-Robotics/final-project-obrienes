import time
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R
import eas
import ctrnn 

transient = 5
duration = 30000    
maxForce = 500             

nnsize = 5
sensor_inputs = 1
motor_outputs = 2

dt = 0.001
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 2.0
BiasRange = 2.0
nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)

m = mujoco.MjModel.from_xml_path('melty_sim_mujoco.xml')
d = mujoco.MjData(m)
print(d)
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()
genotype = np.load("bestgenotype339.npy")
timestep = m.opt.timestep

nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
nn.initializeState(np.zeros(nnsize))


def quat2euler(quat_mujoco):
    #mujocoy quat is constant,x,y,z,
    #scipy quaut is x,y,z,constant
    # quat_scipy = np.array([quat_mujoco[3],quat_mujoco[0],quat_mujoco[1],quat_mujoco[2]])
    try:
        r = R.from_matrix(quat_mujoco.reshape(3,3))
        euler = r.as_euler('xyz', degrees=True)
    except:
        euler = [0,0,0]

    return euler

def controller(m, d):
    #put the controller here. This function is called inside the simulation.
    # pass
    # print("controller")
    # print(d.xpos[1])
    # forward = 90
    # direction = np.round(quat2euler(d.xmat[1]), decimals=4)[2]
    if (d.time < transient*timestep):
        d.qvel[4] = 100
        d.qvel[5] = 100
        # print("wait")
    else:
        d.qvel[4] = np.clip((50 * (nn.out()[0]*2 - 1)) + 100,-200,200)
        d.qvel[5] = np.clip((50 * (nn.out()[1]*2 - 1)) + 100,-200,200)
    # print(d.time)

    # nn.step(dt,np.array([((forward - direction + 180 + 360) % 360 - 180)/180]))
    # d.ctrl[0] = 20 * nn.out()[0] + 200
    # d.ctrl[1] = 20 * nn.out()[1] - 200
    # print(d.ctrl[0])
    # print(direction)
    # # print(d.xmat[1])
    # if (abs(((forward - direction + 180 + 360) % 360 - 180))<10):
    #     # print("yes")
    #     d.ctrl[0] = 300
    #     d.ctrl[1] = -100
    # else:
    #     # print("no")
    #     d.ctrl[0] = 200
    #     d.ctrl[1] = -200

    # d.ctrl[0] = 500
    # d.ctrl[1] = -500

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
# nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
# nn.initializeState(np.zeros(nnsize))
# i = 0
while not glfw.window_should_close(window):
    time_prev = d.time

    forward = 90
    f_vec = np.array([np.cos(forward*np.pi/180),np.sin(forward*np.pi/180)])
    while (d.time - time_prev < 1.0/60.0):
        # print(d.time)
        # i += 1
        # print(i)
        direction = np.round(quat2euler(d.xmat[1]), decimals=4)[2]
        nn.step(dt,np.array([2*(1-(abs((forward - direction + 180 + 360) % 360 - 180)/180))-1]))
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
