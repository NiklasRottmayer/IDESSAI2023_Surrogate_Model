# Author:   Niklas Rottmayer
# Date:     11.04.2023

# This file contains the class *CRotationGenerator*
# It produces random rotations based on given attributes.

# Individual generators for rotations are defined as subclasses of 'CRotationGenerator'.
# They require the '__init__' and 'generate_rotation' functions to be implemented.

from abc import abstractmethod
import numpy as np
from numpy import arccos, arcsin, sqrt, log, exp
from numpy import random as rd
from math import pi
from scipy.spatial.transform import Rotation


class CRotationGenerator:
    def __init__(self,
                 distribution='Fixed',
                 parameters=np.array([0,0,0],dtype=float),
                 seed=None):
        # Attributes
        self.Distribution = distribution
        self.Parameters   = parameters
        if (self.Distribution == 'von Mises-Fisher') and (self.Parameters[2] <= 0):
            raise ValueError('The third parameter has to be positive and non-zero for the von Mises-Fisher distribution')
        elif (self.Distribution == 'Schladitz') and (self.Parameters[2] <= 0):
            raise ValueError('The third parameter has to be positive and non-zero for the Schladitz distribution')

        # Attribute - private
        self._Seed = seed

    @staticmethod
    def create_generator(distribution='Fixed',
                         parameters=np.array([0,0,0],dtype=float),
                         seed=None):
        if distribution == 'Fixed':
            return CFixedRotation(parameters,seed)
        elif distribution == 'von Mises-Fisher':
            return CvonMisesFisherRotation(parameters,seed)
        elif distribution == 'Schladitz':
            return CSchladitzRotation(parameters,seed)
        elif distribution == 'Uniform':
            return CUniformRotation(parameters,seed)
        else:
            raise ValueError('Distribution is invalid.')

    # This method returns rotations given as Euler angles. These are easier to store but require
    # more calculations due to transformations.
    @abstractmethod
    def generate_rotation(self, n = 1):
        pass
# New Code

class CFixedRotation(CRotationGenerator):
    def __init__(self,parameters,seed):
        super().__init__('Fixed',parameters,seed)

    def generate_rotation(self,n=1):
        if self._Seed:
            rd.seed(self._Seed)
        return np.tile( Rotation.from_euler('ZXZ',self.Parameters).as_quat().reshape(1,-1), (n,1))


class CvonMisesFisherRotation(CRotationGenerator):
    def __init__(self,parameters,seed):
        super().__init__('von Mises-Fisher',parameters,seed)

    def generate_rotation(self, n=1):
        if self._Seed:
            rd.seed(self._Seed)
        EulerAngles = np.empty([n, 3])
        # Rotation of z-axis towards mu
        rotmu = Rotation.from_euler('ZXZ',[self.Parameters[0],self.Parameters[1],0])
        for i in range(n):
            # Sampling rotation from von Mises-Fisher distribution with mu = z-axis
            direction_Euler = Rotation.from_euler('ZXZ',np.append(self.von_Mises_Fisher_Euler(self.Parameters[2]),0))

            # Concatenation of rotations
            EulerAngles[i,:] = (rotmu*direction_Euler).as_euler('ZXZ')
            EulerAngles[i,2] = 2*pi*rd.random()
        return Rotation.from_euler('ZXZ',EulerAngles).as_quat()

    @staticmethod
    def von_Mises_Fisher_Euler(kappa):
        # von Mises-Fisher distribution sampling - mean direction is the z-axis
        # kappa - scalar value of the "concentration"
        # mu - mean direction specified by 2 Euler angles
        # Output: direction specified by 2 Euler angles
        lmb = exp(-2.0*kappa)
        theta = 2.0*arcsin(sqrt(-log(rd.random()*(1.0-lmb)+lmb)/(2.0*kappa)))
        phi = 2.0*pi*rd.random()
        return np.array([phi,theta],dtype=float)

class CSchladitzRotation(CRotationGenerator):
    def __init__(self,parameters,seed):
        super().__init__('Schladitz',parameters,seed)

    def generate_rotation(self, n=1):
        if self._Seed:
            rd.seed(self._Seed)
        EulerAngles = np.empty([n, 3])
        # Rotation of z-axis towards mu
        rotmu = Rotation.from_euler('ZXZ',[self.Parameters[0],self.Parameters[1],0])
        for i in range(n):
            # Sampling rotation from von Mises-Fisher distribution with mu = z-axis
            direction_Euler = Rotation.from_euler('ZXZ',np.append(self.Schladitz_Euler(self.Parameters[2]),0))

            # Concatenation of rotations
            EulerAngles[i,:] = (rotmu*direction_Euler).as_euler('ZXZ')
            EulerAngles[i,2] = 2*pi*rd.random()
        return Rotation.from_euler('ZXZ',EulerAngles).as_quat()

    @staticmethod
    def Schladitz_Euler(beta):
        # Schladitz distribution sampling - mean direction is the z-axis
        # Input & Output are Euler angles. Only 2 are needed!
        xi = 2 * rd.random() - 1
        if abs(beta - 1) > 10 ** (-8):
            xi /= sqrt(xi ** 2 - xi ** 2 * beta ** 2 + beta ** 2)
        theta = arccos(xi)
        phi = 2 * pi * rd.random()
        return np.array([phi,theta],dtype=float)

class CUniformRotation(CRotationGenerator):
    def __init__(self,parameters,seed):
        super().__init__('Uniform',parameters,seed)

    def generate_rotation(self, n=1):
        if self._Seed:
            rd.seed(self._Seed)
        return Rotation.random(n).as_quat()


