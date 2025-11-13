from pyrosim.commonFunctions import Save_Whitespace

class ORIGIN_URDF: 

    def __init__(self,pos,orientation):

        self.depth  = 3

        posString = str(pos[0]) + " " + str(pos[1]) + " " + str(pos[2])

        oriString = str(orientation[0]) + " " + str(orientation[1]) + " " + str(orientation[2])

        self.string = '<origin xyz="' + posString + '" rpy="' + oriString + '"/>'

    def Save(self,f):

        Save_Whitespace(self.depth,f)

        f.write( self.string + '\n' )
