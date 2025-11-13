import pyrosim.pyrosim as pyrosim

def World(n,xnum,ynum):
    pyrosim.Start_SDF("boxes.sdf")
    for x in range(xnum):
        for y in range(ynum):
            for i in range(n):
                s = 1- i/n
                pyrosim.Send_Cube(name="Box", pos=[2*x,2*y,0.5 + i] , size=[s,s,s])

    pyrosim.End()


def Robot():
    pyrosim.Start_URDF("robot.urdf")
    pyrosim.Send_Cube(name="0", pos=[-0.5,0,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="1_0", parent="1", child="0",type="revolute", position=[-1,0,-1], axis=[0,1,0])
    pyrosim.Send_Cube(name="1", pos=[-0.5,0,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="2_1", parent="2", child="1",type="revolute", position=[-1,0,-1], axis=[0,1,0])
    pyrosim.Send_Cube(name="2", pos=[-0.5,0,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="3_2", parent="3", child="2",type="revolute", position=[0,0,0], axis=[0,1,0])
    pyrosim.Send_Cube(name="3", pos=[0.5,0,0.5] , size=[1,1,1])

    pyrosim.Send_Joint(name="3_4", parent="3", child="4",type="revolute", position=[1,0,0], axis=[0,1,0])
    pyrosim.Send_Cube(name="4", pos=[0.5,0,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="4_5", parent="4", child="5",type="revolute", position=[1,0,-1], axis=[0,1,0])
    pyrosim.Send_Cube(name="5", pos=[0.5,0,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="5_6", parent="5", child="6",type="revolute", position=[1,0,-1], axis=[0,1,0])
    pyrosim.Send_Cube(name="6", pos=[0.5,0,-0.5] , size=[1,1,1])

    pyrosim.Send_Joint(name="3_7", parent="3", child="7",type="revolute", position=[0,0.5,0], axis=[1,0,0])
    pyrosim.Send_Cube(name="7", pos=[0.5,0.5,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="7_8", parent="7", child="8",type="revolute", position=[0,1,-1], axis=[1,0,0])
    pyrosim.Send_Cube(name="8", pos=[0.5,0.5,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="8_9", parent="8", child="9",type="revolute", position=[0,1,-1], axis=[1,0,0])
    pyrosim.Send_Cube(name="9", pos=[0.5,0.5,-0.5] , size=[1,1,1])
   
    pyrosim.Send_Joint(name="3_10", parent="3", child="10",type="revolute", position=[0,-0.5,0], axis=[1,0,0])
    pyrosim.Send_Cube(name="10", pos=[0.5,-0.5,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="10_11", parent="10", child="11",type="revolute", position=[0,-1,-1], axis=[1,0,0])
    pyrosim.Send_Cube(name="11", pos=[0.5,-0.5,-0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="11_12", parent="11", child="12",type="revolute", position=[0,-1,-1], axis=[1,0,0])
    pyrosim.Send_Cube(name="12", pos=[0.5,-0.5,-0.5] , size=[1,1,1])



    pyrosim.End()


def melty():
    pyrosim.Start_URDF("melty.urdf")
    pyrosim.Send_Cylinder(name="0",pos=[0,0,0.5],size=[0.5,2])
    pyrosim.Send_Joint(name="0_1", parent="0", child="1",type="revolute", position=[1.5,0,0.5], axis=[1,0,0])
    pyrosim.Send_Cylinder(name="1",pos=[0,0,0],size=[0.5,0.5],orientation=[0,1.57,0])
    pyrosim.Send_Joint(name="0_2", parent="0", child="2",type="revolute", position=[-1.5,0,0.5], axis=[1,0,0])
    pyrosim.Send_Cylinder(name="2",pos=[0,0,0],size=[0.5,0.5],orientation=[0,1.57,0])
    pyrosim.End()


# Robot()
melty()
# World(10,5,5)

