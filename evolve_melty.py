import eas
import ctrnn 
import numpy as np
import matplotlib.pyplot as plt
import time
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R

m = mujoco.MjModel.from_xml_path('melty_sim_mujoco.xml')
# m.nconmax = 2000 
# m.narena = 1000000
d = mujoco.MjData(m)
# print(d)
# cam = mujoco.MjvCamera()                        # Abstract camera
# opt = mujoco.MjvOption()

timestep = m.opt.timestep
# print(timestep)
transient = int(1/timestep)
duration = int(20/timestep)
# print(transient)
maxForce = 500             

nnsize = 5
sensor_inputs = 1
motor_outputs = 2

dt = timestep
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 2.0
BiasRange = 2.0

stepstaken = 0


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
    # print(direction)
    # print(d.xmat[1])
    if (d.time < transient*timestep):
        d.qvel[4] = 100
        d.qvel[5] = 100
        # print("wait")
    else:
        d.qvel[4] = np.clip((50 * (nn.out()[0]*2 - 1)) + 100,-200,200)
        d.qvel[5] = np.clip((50 * (nn.out()[1]*2 - 1)) + 100,-200,200)
    # stepstaken += 1
    # print(stepstaken)
    # print(d.ctrl[0])

nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)
mujoco.set_mjcb_control(controller)


# <velocity name="left-velocity" joint="left-wheel" kv="10000"/>
# 		<velocity name="right-velocity" joint="right-wheel" kv="10000"/>




def fitnessFunction(genotype):
    # Reset joints / Reset the coordinates the body 
    mujoco.mj_resetData(m, d)

    # Set and initialize the neural network
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))
    # stepstaken = 0

    # Simulate both NN and Body for a little while 
    # without connecting them, so that transients pass 
    # for both of them
    forward = np.random.randint(-180, 180)
    f_vec = np.array([np.cos(forward*np.pi/180),np.sin(forward*np.pi/180)])
    for i in range(transient):
        nn.step(dt,np.zeros(sensor_inputs))
        mujoco.mj_step(m, d)
        # mujoco.mj_forward(m,d)
    # print(d.time)

    # Test period
    # # Get starting position (after the transient)
    posx_start = d.xpos[1][0]
    posy_start = d.xpos[1][1]
    posx_current = d.xpos[1][0]
    posy_current = d.xpos[1][1]

    distance_traveled = 0.0 # distance traveled in the x-y plane at each step, to be maximized

    # Simulate both NN and Body
    for i in range(duration):
        direction = np.round(quat2euler(d.xmat[1]), decimals=4)[2]
        # print(direction/180)
        # print(forward/180 )
        nn.step(dt,np.array([2*(1-(abs((forward - direction + 180 + 360) % 360 - 180)/180))-1]))
        mujoco.mj_step(m, d)
        posx_past = posx_current
        posy_past = posy_current
        posx_current = d.xpos[1][0]
        posy_current = d.xpos[1][1]
        # np.dot(np.array([d.xpos[1][0],d.xpos[1][1]]), f_vec)
        # np.linalg.norm(f_vec)
        moved = np.dot(np.array([posx_current-posx_past,posy_current-posy_past]), f_vec) * f_vec
        # print(f"{np.arctan2(moved[0],moved[1])*180/np.pi}, {np.arctan2(f_vec[0],f_vec[1])*180/np.pi}")
        if np.round(np.arctan2(moved[0],moved[1])*180/np.pi) == np.round(forward):
            distance_traveled += 5*np.dot(moved,moved)
        distance_traveled += np.sqrt((posx_current - posx_past)**2 + (posy_current - posy_past)**2)
        if abs(posx_current) < 2 and abs(posy_current) < 2:
            distance_traveled = distance_traveled / 2

    # Get final position 
    posx_end = d.xpos[1][0]
    posy_end = d.xpos[1][1]
    distance_final = 0
    moved = np.dot(np.array([posx_end-posx_start,posy_end-posy_start]), f_vec) * f_vec
    # print(f"{np.arctan2(moved[0],moved[1])*180/np.pi}, {np.arctan2(f_vec[0],f_vec[1])*180/np.pi}")
    if np.round(np.arctan2(moved[0],moved[1])*180/np.pi) == np.round(forward):
        distance_final = 10*np.dot(moved,moved)
    return distance_traveled + distance_final

# EA Params
popsize = 4
genesize = nnsize*nnsize + 2*nnsize + sensor_inputs*nnsize + motor_outputs*nnsize 
recombProb = 0.5
mutatProb = 0.2
demeSize = 4
generations = 10000

# Evolve and visualize fitness over generations
ga = eas.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()

# Get best evolved network
af,bf,bi = ga.fitStats()    

# Save 
np.save("bestgenotype.npy",bi)


