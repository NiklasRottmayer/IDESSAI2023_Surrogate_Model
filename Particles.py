# Author:   Niklas Rottmayer
# Date:     11.04.2023

# This file contains the class *CParticle*
# It has multiple methods required for sampling and rendering particles into an image

# The parameters of each particle represent the most commonly used quantities:
# For spheres, ellipsoids, and the radius of cylinders -> radii / half-widths
# For cubes, cuboids and the height of cylinders -> full length
# This is properly treated by all methods.

# Individual types of particles are defined as subclasses. They require the functions
# 'calculate_mean_volume', 'calculate_half_expansion', 'generate_parameters' and 'draw'.
# - calculate_mean_volume outputs the mean volume of a particle
# - calculate_half_expansion outputs the half-length (radius, height/2, ...) of a specific particle with given parameters
# - generate_parameters outputs samples of parameters from the given distribution
# - draw outputs an image section when given a centered grid (x,y,z) of the particle

from abc import abstractmethod
import numpy as np
import numpy.random as rd
from math import pi
from scipy.spatial.transform import Rotation  # Import of rotation expressions

class CParticle:
    def __init__(self,
                 particle_shape='Sphere',
                 particle_distribution='Constant',
                 particle_parameters=np.array([[10, 10], [10, 10], [10, 10]],dtype=float),
                 edge_treatment='Periodic'):
        self.Particle_Shape = particle_shape
        self.Particle_Distribution = particle_distribution
        assert ((self.Particle_Distribution == 'Constant') or (self.Particle_Distribution == 'Uniform')), \
                'Particle_Distribution must be \'Constant\' or \'Uniform\'.'
        self.Edge_Treatment = edge_treatment
        assert ((self.Edge_Treatment == 'Periodic') or (self.Edge_Treatment == 'Plus Sampling')), \
                'Edge_Treatment must be \'Periodic\' or \'Plus Sampling\'.'

        # Ensure correct definition of particle parameters in line with following functions
        if particle_distribution == 'Constant':
            self.Particle_Parameters = np.array([[particle_parameters[0,0],particle_parameters[0,0]],
                                                 [particle_parameters[1,0],particle_parameters[1,0]],
                                                 [particle_parameters[2,0],particle_parameters[2,0]]])
        elif particle_distribution == 'Uniform':
            self.Particle_Parameters = np.sort(particle_parameters)
        else:
            raise ValueError('Particle distribution must be one of \'Constant\',\'Uniform\'')

    def __str__(self):
        return 'Class: CParticle'

    @staticmethod
    def create_particle(particle_shape='Sphere',
                        particle_distribution='Constant',
                        particle_parameters=np.array([[10, 10], [10, 10], [10, 10]],dtype=float),
                        edge_treatment='Periodic'):
        if particle_shape == 'Sphere':
            return CSphere(particle_distribution,
                           particle_parameters,
                           edge_treatment)
        elif particle_shape == 'Cube':
            return CCube(particle_distribution,
                         particle_parameters,
                         edge_treatment)
        elif particle_shape == 'Cylinder':
            return CCylinder(particle_distribution,
                             particle_parameters,
                             edge_treatment)
        elif particle_shape == 'Ellipsoid':
            return CEllipsoid(particle_distribution,
                              particle_parameters,
                              edge_treatment)
        elif particle_shape == 'Cuboid':
            return CCuboid(particle_distribution,
                           particle_parameters,
                           edge_treatment)
        else:
            raise ValueError('Particle shape is invalid.')


    # Draw a particle
    # Method 1: Use half expansion from center to determine maximum cube to be modified
    # (Method 2: Utilize rotation to calculate the exact bounding box)
    # Inputs:
    # img - image to draw into
    # center - the center points of the particle
    # parameters - an array of lengths 3 containing size parameters
    # rotation - a rotation given by Euler angles, i.e. an array of length 3
    def draw_particle(self,img,center,parameters,rotation=np.array([0,0,0,1.0],dtype=float),eps=10**(-12)):
        image_size = np.array(img.shape)
        if len(image_size) < 3:
            raise ValueError('The size of the input image is less than 3. ')

        # Step 1: Calculate the bounding box to draw in
        r = self.calculate_half_expansion(parameters=parameters)  # half-length of particle
        if self.Edge_Treatment == 'Plus Sampling':
            minind = np.maximum(np.ceil(center-r),0).astype(int)
            maxind = np.minimum(np.floor(center+r),image_size-1).astype(int)
        elif self.Edge_Treatment == 'Periodic':
            minind = np.ceil(center-r).astype(int)
            maxind = np.floor(center+r).astype(int)
        else:
            raise ValueError('Edge_Treatment must be either "Plus Sampling" or "Periodic".')

        # Generating a meshgrid - this step may be faster by directly subtracting center
        x,y,z = np.ogrid[minind[0]:maxind[0]+1,minind[1]:maxind[1]+1,minind[2]:maxind[2]+1]
        # Rotate the particle by rotating x,y,z accordingly - Possibly requires RotMat' instead of RotMat!!!
        RotMat = Rotation.from_quat(rotation).as_matrix().transpose()
        xout = RotMat[0,0]*(x-center[0]) + RotMat[0,1]*(y-center[1]) + RotMat[0,2]*(z-center[2])
        yout = RotMat[1,0]*(x-center[0]) + RotMat[1,1]*(y-center[1]) + RotMat[1,2]*(z-center[2])
        zout = RotMat[2,0]*(x-center[0]) + RotMat[2,1]*(y-center[1]) + RotMat[2,2]*(z-center[2])
        # Place particle into image
        # Python automatically wraps negative indices in a 'periodic' fashion - but positives need to be cared for
        indices = np.ix_([(k % image_size[0]) if k > image_size[0]-1 else k for k in np.arange(minind[0],maxind[0]+1)],
                         [(k % image_size[1]) if k > image_size[1]-1 else k for k in np.arange(minind[1],maxind[1]+1)],
                         [(k % image_size[2]) if k > image_size[2]-1 else k for k in np.arange(minind[2],maxind[2]+1)])

        img[indices] = np.logical_or(img[indices],self.draw(xout,yout,zout,parameters,eps=10**(-12)))
        return img

    @abstractmethod
    def calculate_mean_volume(self):
        pass

    @abstractmethod
    def calculate_half_expansion(self,parameters):
        pass

    @abstractmethod
    def generate_parameters(self, n):
        pass

    @abstractmethod
    def draw(self,x,y,z,parameters,eps=10**(-12)):
        pass

# End of parent class definition
# ---------------------------------------------------------------------------------------------------------------------

# Definition of class CSphere:
class CSphere(CParticle):
    def __init__(self,
                 particle_distribution,
                 particle_parameters,
                 edge_treatment):
        super().__init__('Sphere',particle_distribution,particle_parameters,edge_treatment)
        self.Particle_Parameters[1:3,:] = 0
        if self.Particle_Parameters[0,0] <= 0 or self.Particle_Parameters[0,1] <= 0:
            raise ValueError('The entered parameters for particle size are invalid. Please enter values larger than 0.')

    def __str__(self):
        return 'Class: CSphere(CParticle)'

    def calculate_mean_volume(self):
        return 1/3*pi*(self.Particle_Parameters[0,1]**2 + self.Particle_Parameters[0,0]**2) \
                     *(self.Particle_Parameters[0,1] + self.Particle_Parameters[0,0])

    def calculate_half_expansion(self,parameters):
        return parameters[0]

    def generate_parameters(self, n):
        assert int(n)==n and n > 0, 'n must be a positive integer.'
        return np.tile(rd.rand(n,1) * (self.Particle_Parameters[0,1] - self.Particle_Parameters[0,0]) \
                       + self.Particle_Parameters[0,0],(1,3))

    def draw(self,x,y,z,parameters,eps=10**(-12)):
        return (x**2 + y**2 + z**2)/parameters[0]**2 <= (1 + eps)**2

# ---------------------------------------------------------------------------------------------------------------------
class CCube(CParticle):
    def __init__(self,
                 particle_distribution,
                 particle_parameters,
                 edge_treatment):
        super().__init__('Cube',particle_distribution,particle_parameters,edge_treatment)
        self.Particle_Parameters[1:3,:] = 0
        if self.Particle_Parameters[0,0] <= 0 or self.Particle_Parameters[0,1] <= 0:
            raise ValueError('The entered parameters for particle size are invalid. Please enter values larger than 0.')

    def __str__(self):
        return 'Class: CCube(CParticle)'

    def calculate_mean_volume(self):
        return 1/4*(self.Particle_Parameters[0,1]**2 + self.Particle_Parameters[0,0]**2) \
                  *(self.Particle_Parameters[0,1] + self.Particle_Parameters[0,0])

    def calculate_half_expansion(self,parameters):
        return np.sqrt(3)*parameters[0]/2

    def generate_parameters(self, n):
        assert int(n)==n and n >= 0, 'n must be a non-negative integer.'
        return np.tile(rd.rand(n,1) * (self.Particle_Parameters[0,1] - self.Particle_Parameters[0,0]) \
                       + self.Particle_Parameters[0,0],(1,3))

    def draw(self,x,y,z,parameters,eps=10**(-12)):
        return 2*np.maximum(np.maximum(abs(x),abs(y)),abs(z))/parameters[0] <= (1 + eps)

# ---------------------------------------------------------------------------------------------------------------------
class CCylinder(CParticle):
    def __init__(self,
                 particle_distribution,
                 particle_parameters,
                 edge_treatment):
        super().__init__('Cylinder',particle_distribution,particle_parameters,edge_treatment)
        self.Particle_Parameters[2:3,:] = 0
        if self.Particle_Parameters[0,0] <= 0 or self.Particle_Parameters[0,1] <= 0 or \
            self.Particle_Parameters[1,0] <= 0 or self.Particle_Parameters[1,1] <= 0:
            raise ValueError('The entered parameters for particle size are invalid. Please enter values larger than 0.')

    def __str__(self):
        return 'Class: CCylinder(CParticle)'

    # Hier befindet sich vermutlich ein Fehler!
    def calculate_mean_volume(self):
        return 1/6*pi*(self.Particle_Parameters[0,1]**2
                     + self.Particle_Parameters[0,1]*self.Particle_Parameters[0,0]
                     + self.Particle_Parameters[0,0]**2) \
                     * (self.Particle_Parameters[1,1] + self.Particle_Parameters[1,0])

    def calculate_half_expansion(self,parameters):
        return np.sqrt(parameters[0]**2 + parameters[1]**2/4)

    def generate_parameters(self, n):
        assert int(n)==n and n > 0, 'n must be a positive integer.'
        return np.concatenate((rd.rand(n,1) * (self.Particle_Parameters[0,1] - self.Particle_Parameters[0,0])
                               + self.Particle_Parameters[0,0],
                               rd.rand(n,1) * (self.Particle_Parameters[1,1] - self.Particle_Parameters[1,0])
                               + self.Particle_Parameters[1,0],
                               np.zeros((n,1))), axis=1)

    def draw(self,x,y,z,parameters,eps=10**(-12)):
        return np.logical_and((x**2 + y**2)/parameters[0]**2 <= (1 + eps)**2,
                                      2*abs(z)/parameters[1] <= (1 + eps))

# ---------------------------------------------------------------------------------------------------------------------
class CEllipsoid(CParticle):
    def __init__(self,
                 particle_distribution,
                 particle_parameters,
                 edge_treatment):
        super().__init__('Ellipsoid',particle_distribution,particle_parameters,edge_treatment)
        if self.Particle_Parameters[0,0] <= 0 or self.Particle_Parameters[0,1] <= 0 or\
            self.Particle_Parameters[1,0] <= 0 or self.Particle_Parameters[1,1] <= 0 or\
            self.Particle_Parameters[2,0] <= 0 or self.Particle_Parameters[2,1] <= 0:
            raise ValueError('The entered parameters for particle size are invalid. Please enter values larger than 0.')

    def __str__(self):
        return 'Class: CEllipsoid(CParticle)'

    def calculate_mean_volume(self):
        return 1/6*pi*(self.Particle_Parameters[0,1] + self.Particle_Parameters[0,0]) \
                     *(self.Particle_Parameters[1,1] + self.Particle_Parameters[1,1]) \
                     *(self.Particle_Parameters[2,1] + self.Particle_Parameters[2,1])

    def calculate_half_expansion(self,parameters):
        return np.max(parameters)

    def generate_parameters(self, n):
        assert int(n)==n and n > 0, 'n must be a positive integer.'
        return rd.rand(n,3)*(self.Particle_Parameters[:,1] - self.Particle_Parameters[:,0]) \
                           + self.Particle_Parameters[:,0]

    def draw(self,x,y,z,parameters,eps=10**(-12)):
        return ((x/parameters[0])**2 + (y/parameters[1])**2 + (z/parameters[2])**2) <= (1+eps)**2

# ---------------------------------------------------------------------------------------------------------------------
class CCuboid(CParticle):
    def __init__(self,
                 particle_distribution,
                 particle_parameters,
                 edge_treatment):
        super().__init__('Cuboid',particle_distribution,particle_parameters,edge_treatment)
        if self.Particle_Parameters[0,0] <= 0 or self.Particle_Parameters[0,1] <= 0 or\
            self.Particle_Parameters[1,0] <= 0 or self.Particle_Parameters[1,1] <= 0 or\
            self.Particle_Parameters[2,0] <= 0 or self.Particle_Parameters[2,1] <= 0:
            raise ValueError('The entered parameters for particle size are invalid. Please enter values larger than 0.')

    def __str__(self):
        return 'Class: CCuboid(CParticle)'

    def calculate_mean_volume(self):
        return 1/8*(self.Particle_Parameters[0,1] + self.Particle_Parameters[0,0]) \
                  *(self.Particle_Parameters[1,1] + self.Particle_Parameters[1,1]) \
                  *(self.Particle_Parameters[2,1] + self.Particle_Parameters[2,1])

    def calculate_half_expansion(self,parameters):
        return np.sqrt(parameters[0]**2 + parameters[1]**2 + parameters[2]**2)/2

    def generate_parameters(self, n):
        assert int(n)==n and n > 0, 'n must be a positive integer.'
        return rd.rand(n,3)*(self.Particle_Parameters[:,1] - self.Particle_Parameters[:,0]) \
                           + self.Particle_Parameters[:,0]

    def draw(self,x,y,z,parameters,eps=10**(-12)):
        return np.maximum(np.maximum(2*abs(x)/parameters[0],
                                     2*abs(y)/parameters[1]),
                                     2*abs(z)/parameters[2]) <= (1 + eps)