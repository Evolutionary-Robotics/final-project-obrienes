import eas
import ctrnn

import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data
import time   

import numpy as np
import matplotlib.pyplot as plt
import math

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf")

pyrosim.Prepare_To_Simulate(robotId)

transient = 1000
duration = 50000 
t = np.linspace(0,1,num=duration)

nnsize = 5
sensor_inputs = 5
motor_outputs = p.getNumJoints(robotId)
joint_names = []
for i in range(motor_outputs): 
    joint_names.append(p.getJointInfo(robotId, i)[1].decode("utf-8"))
maxForce = 500

dt = 0.01
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0


visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.1,
    length=100, 
    rgbaColor=[1, 0, 0, 1], 
    specularColor=[1, 0, 0] 
)
beaconId = p.createMultiBody(
    baseMass=0, 
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0,0,0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)

def reset_robot(robotId, base_pos=[0,0,1], base_orn=[0,0,0,1]):
    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robotId, base_pos, base_orn)

    # zero out velocity
    p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

    # Reset all joint angles and velocities
    num_joints = p.getNumJoints(robotId)
    for j in range(num_joints):
        p.resetJointState(robotId, j, targetValue=0.0, targetVelocity=0.0)
    for j in range(13):
        p.changeDynamics(robotId, j, lateralFriction=2, mass = 3)
        # p.changeDynamics(robotId, j, lateralFriction=1, mass = 1)

    

def fitnessFunction(genotype):
    reset_robot(robotId,base_pos=[0,0,3.5])

    nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))

    for i in range(transient):
        nn.step(dt,np.zeros(sensor_inputs))
        p.stepSimulation()

    output = np.zeros((duration,nnsize))
    motorout = np.zeros((duration,motor_outputs))
    legsensor_hist = np.zeros((duration,sensor_inputs))

    linkState = p.getLinkState(robotId,0)
    posx_start = linkState[0][0]
    posy_start = linkState[0][1]

    # Test period
    # Get starting position (after the transient)
    linkState = p.getLinkState(robotId,0)
    posx_start = linkState[0][0]
    posy_start = linkState[0][1]
    posx_current = linkState[0][0]
    posy_current = linkState[0][1]
    posz_current = linkState[0][2]   

    distance_traveled = 0.0 # distance traveled in the x-y plane at each step, to be maximized
    distance_jumped = 0.0 # amount of movement up and down, to be minimized
    hitHead = 1
    cameraDistance = 8.0  
    cameraYaw = math.degrees(math.atan2(posy_current,posx_current)) + 90
    cameraPitch = -30.0   
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, linkState[0])

    time.sleep(2)
    for i in range(duration):
        legsensor0 = pyrosim.Get_Touch_Sensor_Value_For_Link("0")
        legsensor1 = pyrosim.Get_Touch_Sensor_Value_For_Link("12")
        legsensor2 = pyrosim.Get_Touch_Sensor_Value_For_Link("9")
        legsensor3 = pyrosim.Get_Touch_Sensor_Value_For_Link("6")
        headsensor = pyrosim.Get_Touch_Sensor_Value_For_Link("3")
        for ttt in range(10):
            nn.step(dt,[legsensor0,legsensor1,legsensor2,legsensor3,headsensor])
        legsensor_hist[i] = [legsensor0,legsensor1,legsensor2,legsensor3,headsensor]
        output[i] = nn.Output
        motorout[i] = nn.out()
        motoroutput = nn.out()
        p.stepSimulation()

        
        for j in range(motor_outputs):
            pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                        jointName=joint_names[j], 
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = (motoroutput[j]*2-1)*np.pi/4, # NEW
                                        maxForce = maxForce,
                                        maxVelocity = .5
                                        )
        
        
        time.sleep(1/1000) # NEW 
        posx_past = posx_current
        posy_past = posy_current
        posz_past = posz_current   
        linkState = p.getBasePositionAndOrientation(robotId)
        posx_current = linkState[0][0]
        posy_current = linkState[0][1]
        posz_current = linkState[0][2]    
        distance_traveled += np.sqrt((posx_current - posx_past)**2 + (posy_current - posy_past)**2)
        if posz_current > 8:
            distance_jumped += np.sqrt((posz_current - posz_past)**2)
        hitHead += (headsensor+1)*0.25

        cameraYaw = math.degrees(math.atan2(posy_current,posx_current)) + 90  
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, linkState[0])
        
        

    linkState = p.getBasePositionAndOrientation(robotId)
    posx_end = linkState[0][0]
    posy_end = linkState[0][1]

    distance_final = np.sqrt((posx_start - posx_end)**2 + (posy_start - posy_end)**2)

    print(distance_final, distance_traveled, distance_jumped)

    return distance_final + distance_traveled - distance_jumped, output, motorout, legsensor_hist

# Load
# fitnessFunction(np.load("generation0.npy"))
best = np.load("bestgenotype_Final.npy")
# best = np.load("bestgenotype_Slide.npy")
fit, output, motorout, sensor = fitnessFunction(best)
print(fit)

p.disconnect() 

# # Plot activity
# plt.plot(output,alpha=0.5)
# plt.plot(motorout.T[0],'k')
# plt.plot(motorout.T[1],'k:')
# #plt.plot(sensor/2 + 0.5,'r',alpha=.2)
# plt.xlabel("Time")
# plt.ylabel("Output")
# plt.title("Neural activity")
# plt.show()