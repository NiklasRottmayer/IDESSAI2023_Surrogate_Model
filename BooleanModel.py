# Author:   Niklas Rottmayer
# Date:     25.07.2023
# Version:  1.4

# Patch notes:
# Version 1.4 - 01.09.2023: Error Correction
# - Storing the configuration files has been corrected -> previously large arrays were not stored in their entirety!
# Version 1.3 - 22.08.2023: Added functionality
# - added error check for parameter selection in Particles.py
# - added save_ITWM_Euler_configuration which works only for cylinders (currently)
# - added support for cylinders in save_ITWM_configuration
# - error correction in save_ITWM_configuration for spheres
# Version 1.2 - 13.08.2023: Incompatible with configuration files of Version 1.1.
# - added load_configuration function
# - adjusted save_configuration function by renaming orientation to orientation (quaternion) when
#   saving the sampled particle rotations
# - added function save_ITWM_configuration with the support for spheres. Does not support other particles yet.
# Version 1.1 - 02.08.2023:
# - updated comments
# - adjusted function save_image to ensure folder_name to end with '/'
# - adjusted function save_configuration to ensure folder_name to end with '/'
# - added function num2str which converts a number to a minimal format (rounds + removes trailing zeros)
# - adjusted generate_automatic_name to use num2str instead of str
# Version 1.0 - 25.07.2023:
# - added function save_image to CBooleanModel
# - added function save_configuration to CBooleanModel
#   -> currently with orientations ZXZ-Euler angles, quaternions and orientation axis + rotation angle
#      The last one is the axis of orientation of the particle and the angle is the internal rotation
#      applied after orienting the particle into the specific direction.
# - adjusted the expansion functions of cuboids, cubes and cylinder
#   -> corrected, e.g. half expansion of cubes is sqrt(3)*a^2 (distance center to corner)

# This file contains the main class *CBooleanModel*
# It contains the general configuration and is used to generate realizations
# with the specified setup.

# Comment: The methods save_configuration and save_image need to properly implemented
#          Furthermore, the calculation of 'field features' remains to be done, if necessary.

import numpy as np
from numpy import log
from numpy import random as rd
from DirectionalDistributions import CRotationGenerator
from Particles import CParticle
from tifffile import imwrite, imread
import configparser
from scipy.spatial.transform import Rotation
import vtk
from vtk.util import numpy_support


# Helper function - necessity to be outside of class definitions
# Calculation of the intensity
# Dependencies: Volume density, particle shape, particle paramters
def calculate_intensity(Volume_Density,Mean_Particle_Volume,eps = 10**(-12)):
    return -log(1-Volume_Density)/max(Mean_Particle_Volume,eps)


class CBooleanModel:
    def __init__(self,
                 volume_density = 0.6,
                 image_size = np.array([512, 512, 512],dtype=int),
                 particle_shape = 'Sphere',
                 particle_distribution = 'Constant',
                 particle_parameters = np.array([[10,10], [10,10], [10,10]],dtype=float),
                 orientation = 'Fixed',
                 orientation_parameters = np.array([0, 0, 0],dtype=float),
                 edge_treatment = 'Periodic',
                 inversion = False):

        # Current Version
        self._Version = 1.4

        # Attributes / Options - public
        self.Volume_Density           = volume_density
        self.Image_Size               = np.array(image_size,dtype=int)
        self.Particle_Shape           = particle_shape
        # self.Particle_Parameters      = particle_parameters  # This is set below
        self.Particle_Distribution    = particle_distribution
        self.Orientation              = orientation
        self.Orientation_Parameters   = orientation_parameters
        self.Edge_Treatment           = edge_treatment
        self.Inversion                = inversion

        self.Image = None  # Stores a rendered image of the generated realization - public

        # Attributes - private
        self._Generated = False  # Tracker if geometry has been generated
        self._Rendered = False # Tracker if the geometry has been rendered
        self._Sampled_Particle_Number = None
        self._Sampled_Centers = None
        self._Sampled_Parameters = None
        self._Sampled_Rotations = None
        self._Name = None

        self._Configuration_Classes = np.array([0,1,1,2,1,2,3,5,1,3,2,5,2,5,5,8,1,2,3,5,3,5,7,9,4,6,6,10,6,10,13,16,
                                      1,3,2,5,4,6,6,10,3,7,5,9,6,13,10,16,2,5,5,8,6,10,13,16,6,13,10,16,11,15,15,19,
                                      1,3,4,6,2,5,6,10,3,7,6,13,5,9,10,16,2,5,6,10,5,8,13,16,6,13,11,15,10,16,15,19,
                                      3,7,6,13,6,13,11,15,7,12,13,14,13,14,15,18,5,9,10,16,10,16,15,19,13,14,15,18,15,18,17,20,
                                      1,4,3,6,3,6,7,13,2,6,5,10,5,10,9,16,3,6,7,13,7,13,12,14,6,11,13,15,13,15,14,18,
                                      2,6,5,10,6,11,13,15,5,13,8,16,10,15,16,19,5,10,9,16,13,15,14,18,10,15,16,19,15,17,18,20,
                                      2,6,6,11,5,10,13,15,5,13,10,15,8,16,16,19,5,10,13,15,9,16,14,18,10,15,15,17,16,19,18,20,
                                      5,13,10,15,10,15,15,17,9,14,16,18,16,18,19,20,8,16,16,19,16,19,18,20,16,18,19,20,19,20,20,21],dtype=int)
        self._Voxel_Size = [1,1,1]  # Comment: Currently fixed! Extension possible.
        self._Pixel_Configurations = np.empty(shape = [1,314],dtype=int)
        self._Particle_Class = CParticle.create_particle(particle_shape=self.Particle_Shape,
                                         particle_distribution=self.Particle_Distribution,
                                         particle_parameters=particle_parameters,
                                         edge_treatment=self.Edge_Treatment)
        self._Directions_Class = CRotationGenerator.create_generator(distribution=self.Orientation,
                                                                     parameters=self.Orientation_Parameters)
        self._Intensity = calculate_intensity(self.Volume_Density,self._Particle_Class.calculate_mean_volume())
        # After setting Particle_Class, set parameters accordingly!
        self.Particle_Parameters = self._Particle_Class.Particle_Parameters

        self._Name = self.generate_automatic_name()

        # To-Do: Sanity Checks!!!
        # Volume Density 0<Vv<1
        # Image Size > 0
        # Particle Shape: Cube, Cuboid, Cylinder, Ellipsoid, Sphere
        # Particle Parameters >0, 3x2 Matrix
        # Orientation: Fixed, von Mises-Fisher, Schladitz, Uniform
        # Orientation Parameters <- to be determined!!!
        # Edge Treatment: Periodic or Plus Sampling
        # Inversion: True or False

    # Sampling of a Poisson point process:
    # Output:
    # - centers     a numpy array of size n x 3 of points in R^3
    # - n           number of sampled points
    def Poisson_point_process(self):
        if self.Edge_Treatment == 'Periodic':
            # Generate values in [0,size-1] (length = size-1)
            n = rd.poisson(self._Intensity*np.prod(self.Image_Size-1))
            centers = rd.rand(n,3)*(self.Image_Size-1)
            return centers,n
        elif self.Edge_Treatment == 'Plus Sampling':
            # Generate values in [-r,size-1+r] (length = size-1+2r)
            r = self._Particle_Class.calculate_half_expansion(parameters=np.max(self.Particle_Parameters,axis=1))
            n = rd.poisson(self._Intensity*np.prod(self.Image_Size+2*r-1))
            centers = rd.rand(n,3)*(self.Image_Size+2*r-1) - r
            return centers,n
        else:
            raise ValueError('Edge treatment must be one of \'Periodic\',\'Plus Sampling\'!')


    # Generate: This method generates the geometry but does NOT render it. It performs all random aspects
    #           such that a realization is completely determined. Rendering has its own method!
    # Sets internal properties:
    # _Sampled_Centers          - center points of particles
    # _Sampled_Particle_Number  - number of particles
    # _Sampled_Parameters       - parameters of particles
    # _Sampled_Rotations        - Rotations as Quaternions
    # _Generated                - boolean variable to encode if a sample was generated
    # _Rendered                 - boolean variable to encode if a sample was rendered
    def generate(self,verbose=True):

        # Step 1: Generate center points
        self._Sampled_Centers,self._Sampled_Particle_Number = self.Poisson_point_process()
        # Step 2: Generate parameter array
        self._Sampled_Parameters = self._Particle_Class.generate_parameters(self._Sampled_Particle_Number)
        # Step 3: Generate rotations
        self._Sampled_Rotations = self._Directions_Class.generate_rotation(self._Sampled_Particle_Number)
        # Update tracker data
        self._Generated = True
        self._Rendered = False
        self.Image = None
        if verbose:
            print(f'Generation complete. {self._Sampled_Particle_Number} objects have been created.')
        return

    # Render: This method renders an existing realization to an image. It performs the drawing of objects
    #         which is the most time consuming step in this pipeline.
    def render(self,verbose=True):
        if self._Generated == False:
            raise ValueError('No realization was generated yet. Use the method \'Generate\' before rendering.')
        if self.Image is None:
            self.Image = np.zeros(self.Image_Size,dtype=bool)

        for i in range(self._Sampled_Particle_Number):
            self._Particle_Class.draw_particle(self.Image,self._Sampled_Centers[i,:],self._Sampled_Parameters[i,:],self._Sampled_Rotations[i,:])
        if verbose:
            print(f'Rendering complete. The image can be accessed with \'self.Image\'.')
        self._Rendered = True
        return

    def save_configuration(self,foldername='',number=None):
        if not self._Generated:
            raise ValueError('No realization has been generated yet. Please call the function generate before save_configuration.')
        if foldername and not foldername.endswith('/'):
            foldername += '/'

        config = configparser.ConfigParser()
        config['Information'] = {'version': str(self._Version)}

        config['Settings'] = {'volume density': str(self.Volume_Density),
                             'image size': self.Image_Size,
                             'particle shape': self.Particle_Shape,
                             'particle parameters': self.Particle_Parameters,
                             'particle distribution': self.Particle_Distribution,
                             'orientation': self.Orientation,
                             'orientation parameters': self.Orientation_Parameters,
                             'edge treatment': self.Edge_Treatment,
                             'inverted': self.Inversion}

        # Temporary code for Euler representation in config file
        if self._Sampled_Particle_Number > 0:
            Euler_Angles = Rotation.from_quat(self._Sampled_Rotations).as_euler('ZXZ')
            Direction_Angle = np.array([np.append(Rotation.from_quat(self._Sampled_Rotations[i,:]).as_matrix()[:,2],Euler_Angles[i,2]) \
                                for i in range(self._Sampled_Particle_Number)])
        else:
            Euler_Angles = []
            Direction_Angle =[]

        config['Geometry'] = {'number of particles': self._Sampled_Particle_Number,
                              'center points': np.array2string(self._Sampled_Centers,threshold=np.inf),
                              'parameters': np.array2string(self._Sampled_Parameters,threshold=np.inf),
                              'orientation (quaternion)': np.array2string(self._Sampled_Rotations,threshold=np.inf),
                              'zxz euler orientation': np.array2string(Euler_Angles,threshold=np.inf),
                              'direction angle orientation': np.array2string(Direction_Angle,threshold=np.inf)}

        if number:
            if not int(number) == number:
                raise ValueError('number must be of integer value')
            with open(foldername + self._Name + '_Num' + str(number) + '.txt', 'w') as configfile:
                config.write(configfile)
        else:
            with open(foldername + self._Name + '.txt', 'w') as configfile:
                config.write(configfile)
        return


    def save_ITWM_configuration(self, foldername='', number=None):
        if not self._Generated:
            raise ValueError('No realization has been generated yet. Please call the function generate before save_configuration.')
        if foldername and not foldername.endswith('/'):
            foldername += '/'

        if number or number == 0:
            if not int(number) == number:
                raise ValueError('number must be of integer value')
            file = open(foldername + self._Name + '_Num' + str(number) + '_ITWM.txt','w')

        else:
            file = open(foldername + self._Name + '_ITWM.txt','w')

        file.write(f'# FIBSEMSIMMODEL v02\n')
        file.write(f'# version ' + num2str(self._Version) + '\n')
        file.write(f'#\n')
        file.write(f'# Settings\n')
        file.write(f'# volume density = ' + num2str(self.Volume_Density) + '\n')
        file.write(f'# image size = ' + np.array2string(self.Image_Size) + '\n')
        file.write(f'# particle shape = ' + self.Particle_Shape + '\n')
        file.write(f'# particle parameters = ' + np.array2string(self.Particle_Parameters).replace('\n',';') + '\n')
        file.write(f'# particle distribution = ' + self.Particle_Distribution + '\n')
        file.write(f'# orientation = ' + self.Orientation + '\n')
        file.write(f'# orientation parameters = ' + np.array2string(self.Orientation_Parameters) + '\n')
        file.write(f'# edge treatment = ' + self.Edge_Treatment + '\n')
        file.write(f'# inverted = ' + str(self.Inversion) + '\n')
        file.write(f'#\n')
        file.write(f'# Number of Spheres, Cylinders, Cubes\n')
        file.write(f'# NUM SPHERES CYLINDERS CUBES\n')

        if self.Particle_Shape == 'Sphere':
            file.write(num2str(self._Sampled_Particle_Number) + '\n0\n0\n')
            file.write(f'# ' + 'SPHERES' + '\n')
            for i in range(self._Sampled_Particle_Number):
                file.write(
                    f'{i}\t{self._Sampled_Centers[i,0]}\t{self._Sampled_Centers[i,1]}\t{self._Sampled_Centers[i,2]}' +
                    f'\t{self._Sampled_Parameters[i,0]}\n')
            file.write(f'# ' + 'CYLINDERS' + '\n')
            file.write(f'# ' + 'CUBES' + '\n')
        elif self.Particle_Shape == 'Cylinder':
            file.write('0\n' + num2str(self._Sampled_Particle_Number) + '\n0\n')
            file.write(f'# ' + 'SPHERES' + '\n')
            file.write(f'# ' + 'CYLINDERS' + '\n')
            for i in range(self._Sampled_Particle_Number):
                dir = Rotation.from_quat(self._Sampled_Rotations[i,:]).as_matrix()[:,2]
                file.write(
                    f'{i}\t{self._Sampled_Centers[i,0]}\t{self._Sampled_Centers[i,1]}\t{self._Sampled_Centers[i,2]}\t' +
                    f'{dir[0]}\t{dir[1]}\t{dir[2]}\t' +
                    f'{self._Sampled_Parameters[i,0]}\t{self._Sampled_Parameters[i,1]}\n')
            file.write(f'# ' + 'CUBES' + '\n')
        elif self.Particle_Shape == 'Cube':
            file.close()
            raise ValueError('Cubes are not yet supported for this function.')
        else:
            file.close()
            raise ValueError('The function save_ITWM_configuration only supports spheres, cylinders and cubes.')

        file.write(f'# EOF')
        file.close()
        return

    def save_ITWM_Euler_configuration(self,foldername='', number=None):
        if self.Particle_Shape != 'Cylinder':
            raise ValueError('Currently this function only support cylinders!')

        if not self._Generated:
            raise ValueError('No realization has been generated yet. Please call the function generate before save_configuration.')
        if foldername and not foldername.endswith('/'):
            foldername += '/'

        if number:
            if not int(number) == number:
                raise ValueError('number must be of integer value')
            file = open(foldername + self._Name + '_Num' + str(number) + '_ITWM-Euler.txt','w')

        else:
            file = open(foldername + self._Name + '_ITWM-Euler.txt','w')

        file.write(f'# FIBSEMSIMMODEL v02\n')
        file.write(f'# version ' + num2str(self._Version) + '\n')
        file.write(f'#\n')
        file.write(f'# Settings\n')
        file.write(f'# volume density = ' + num2str(self.Volume_Density) + '\n')
        file.write(f'# image size = ' + np.array2string(self.Image_Size,threshold=np.inf) + '\n')
        file.write(f'# particle shape = ' + self.Particle_Shape + '\n')
        file.write(f'# particle parameters = ' + np.array2string(self.Particle_Parameters).replace('\n',';') + '\n')
        file.write(f'# particle distribution = ' + self.Particle_Distribution + '\n')
        file.write(f'# orientation = ' + self.Orientation + '\n')
        file.write(f'# orientation parameters = ' + np.array2string(self.Orientation_Parameters) + '\n')
        file.write(f'# edge treatment = ' + self.Edge_Treatment + '\n')
        file.write(f'# inverted = ' + str(self.Inversion) + '\n')
        file.write(f'#\n')
        file.write(f'# Number of Spheres, Cylinders, Cubes\n')
        file.write(f'# NUM SPHERES CYLINDERS CUBES\n')
        if self.Particle_Shape == 'Cylinder':
            file.write('0\n' + num2str(self._Sampled_Particle_Number) + '\n0\n')
            file.write(f'# ' + 'SPHERES' + '\n')
            file.write(f'# ' + 'CYLINDERS' + '\n')
            for i in range(self._Sampled_Particle_Number):
                Angles = Rotation.from_quat(self._Sampled_Rotations[i,:]).as_euler('ZXZ')
                file.write(
                    f'{i}\t{self._Sampled_Centers[i,0]}\t{self._Sampled_Centers[i,1]}\t{self._Sampled_Centers[i,2]}\t' +
                    f'{Angles[0]}\t{Angles[1]}\t{Angles[2]}\t' +
                    f'{self._Sampled_Parameters[i,0]}\t{self._Sampled_Parameters[i,1]}\n')
            file.write(f'# ' + 'CUBES' + '\n')
        file.write(f'# EOF')
        file.close()

    def save_image(self,foldername='',number=None):
        if not self._Rendered:
            raise ValueError('No realization has been generated yet. Please call the function generate before save_image.')
        if foldername and not foldername.endswith('/'):
            foldername += '/'

        if number:
            if not int(number) == number:
                raise ValueError('number must be of integer value')
            imwrite(foldername + self._Name + '_Num' + str(number) + '.tif',self.Image,bitspersample = 1,
                    photometric='MINISBLACK',compression=None)
        else:
            imwrite(foldername + self._Name + '.tif',self.Image,bitspersample = 1,
                    photometric='MINISBLACK',compression=None)
        return

    def generate_automatic_name(self):
        name = self.Particle_Shape + '_Vv' + str(self.Volume_Density) + '_'
        if self.Particle_Distribution == 'Constant':
            parameter_array = self.Particle_Parameters[:,0].reshape((3,1))
        elif self.Particle_Distribution == 'Uniform':
            parameter_array = self.Particle_Parameters

        if self.Particle_Shape == 'Sphere':
            name += 'r' + '-'.join([num2str(num) for num in parameter_array[0]])
        elif self.Particle_Shape == 'Cube':
            name += 'l' + '-'.join([num2str(num) for num in parameter_array[0]])
        elif self.Particle_Shape == 'Cylinder':
            name += 'r' + '-'.join([num2str(num) for num in parameter_array[0]]) + \
                   '_h' + '-'.join([num2str(num) for num in parameter_array[1]])
        elif self.Particle_Shape == 'Ellipsoid':
            name += 'rx' + '-'.join([num2str(num) for num in parameter_array[0]]) + \
                   '_ry' + '-'.join([num2str(num) for num in parameter_array[1]]) + \
                   '_rz' + '-'.join([num2str(num) for num in parameter_array[2]])
        elif self.Particle_Shape == 'Cuboid':
            name += 'lx' + '-'.join([num2str(num) for num in parameter_array[0]]) + \
                   '_ly' + '-'.join([num2str(num) for num in parameter_array[1]]) + \
                   '_lz' + '-'.join([num2str(num) for num in parameter_array[2]])

        name += '_' + self.Orientation.replace(' ','').replace('-','') + '_' + self.Edge_Treatment.replace(' ','')
        return name.replace('.','p')

    # This function loads a configuration file and sets parameters accordingly
    # Input:
    # - file_name   The full name and directory of the configuration file to load
    def load_configuration(self,file_name):
        config = configparser.ConfigParser()
        config.read(file_name)

        if not 'Information' in config or not 'Settings' in config or not 'Geometry' in config:
            raise ValueError('The given file does not contain the necessary sections.')
        if float(config['Information']['version']) != self._Version:
            raise Warning('The version of the configuration file and the project are different. Compatibility may not be guaranteed.')
        self.Volume_Density = float(config['Settings']['volume density'])
        self.Image_Size = np.fromstring(config['Settings']['image size'][1:-1],dtype = int, count= 3,sep = ' ')
        self.Particle_Shape = config['Settings']['particle shape']
        self.Particle_Parameters = np.fromstring(config['Settings']['particle parameters'].replace('[','').replace(']',''),
                                                 sep =' ').reshape(3,2)
        self.Particle_Distribution = config['Settings']['particle distribution']
        self.Edge_Treatment = config['Settings']['edge treatment']
        self.Inversion = bool(config['Settings']['inverted'])

        self._Sampled_Particle_Number = int(config['Geometry']['number of particles'])
        self._Sampled_Centers = np.fromstring(config['Geometry']['center points'].replace('[','').replace(']',''),
                                              sep=' ').reshape(self._Sampled_Particle_Number, 3)
        self._Sampled_Parameters = np.fromstring(config['Geometry']['parameters'].replace('[','').replace(']',''),
                                                 sep=' ').reshape(self._Sampled_Particle_Number, 3)
        self._Sampled_Rotations = np.fromstring(config['Geometry']['orientation (quaternion)'].replace('[','').replace(']',''),
                                                sep=' ').reshape(self._Sampled_Particle_Number, 4)
        self._Generated = True
        self._Rendered = False
        self.Image = None
        return

# Appropriate conversion of number to string
# Only non-zero decimals are displayed
def num2str(number,k = 2):
    if float(number).is_integer():
        return str(int(number))
    else:
        return ('{:.'+str(k)+'f}').format(round(number,k)).rstrip('0').rstrip('.')


# Rendering function to a 3D Viewer
def Render3DImage(img):
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(img.shape[::-1])
    image_data.SetSpacing(1, 1, 1)
    image_data.SetOrigin(0, 0, 0)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    vtk_array = numpy_support.numpy_to_vtk(np.swapaxes(img,0,2).ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image_data.GetPointData().SetScalars(vtk_array)

    # Create a Marching Cubes algorithm to extract the iso-surface
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image_data)
    mc.ComputeNormalsOn()
    mc.SetValue(0, 0.5)

    # Create a mapper and actor for the extracted iso-surface
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(mc.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.2, 0.3)
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Add the actor to the renderer and start the interactor
    renderer.AddActor(actor)
    renderer.ResetCamera()

    # Create an orientation marker widget
    axes_actor = vtk.vtkAxesActor()
    axes_actor.SetTotalLength(2, 2, 2)
    axes_actor.SetShaftTypeToCylinder()
    axes_actor.SetCylinderRadius(0.01)
    axes_actor.SetAxisLabels(1)
    axes_actor.SetXAxisLabelText("x")
    axes_actor.SetYAxisLabelText("y")
    axes_actor.SetZAxisLabelText("z")
    axes = vtk.vtkOrientationMarkerWidget()
    axes.SetOrientationMarker(axes_actor)
    axes.SetInteractor(interactor)
    axes.EnabledOn()
    axes.InteractiveOn()
    axes.SetViewport(0.0, 0.0, 0.2, 0.2)
    axes.SetEnabled(1)
    axes.InteractiveOn()
    axes.SetInteractive(1)
    axes.SetOutlineColor(0.9300, 0.5700, 0.1300)

    orientation_marker = axes.GetOrientationMarker()
    #orientation_marker.SetOrientation(45, 0, 0) # no idea what this does
    orientation_marker.SetPosition(0.5, 0.5, 0.0)
    orientation_marker.PickableOff()

    render_window.Render()
    interactor.Start()
    return