#-------------------------------------------------------------------
#
#                 interface_1.0/classes.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
# This module contains the simulation class that reads input file
# that contains the essential simulation parameters for plasma 
# interface. Notice that all the quantities are calculated in real
# units so that these can be fed into LAMMPS for simulations with
# real unit.
# 
# Module prerequisites:
# The versions below are the ones that I use when developing the 
# code. Earlier/later version may or may not work. 
# python: 3.8.5
#
# Here are brief descriptions for the functions/classes in this
# module.
#
# species:
# class for species (type of particles) information
# 
# simulation:
# class for the simulation parameters and useful parameters for 
# analysis.
#
#-------------------------------------------------------------------
#
#-------------------------------------------------------------------
#
# 2021.06.29 Created                                 TPZ
# 2021.07.02 Modified                                TPZ            
#            (Updated script up to the initialisation of the 
#            simulation)   
# 
#-------------------------------------------------------------------

from const import hbar,e,me,kB,mp,e0,e2,EF23prefac, eV
from math import sqrt, pi, exp, floor

# InitSpecies class:
# A class specifically designed for storing initial configuration of species
# InitSpecies class contains:
# mass: mass in mp
# charge: charge number
# numDen: particle number density in A^-3
class InitSpecies:
  def __init__(self, mass, charge, numDen):
    self.mass     = mass
    self.charge   = charge
    self.numDen   = numDen

# SimSpecies class:
# A class for each individual type of particle in each grid (notice that the 
# same species in two different grids are two different types of particles 
# when using Debye screening)
# requires to feed in InitSpecies class object
# other than attributes in InitSpecies, it has:
# attributes:
# num: total number of particles in this type
# TypeID: the type ID used in the MD simulation
# force: additional force fo the type
# methods:
# SetForce, numDenUpdate, SetN, SetTypeID requires one argument which is the
# updated value of Force, numDen, N or TypeID
class SimSpecies(InitSpecies):
  def __init__(self, InputInitSpecies):
    if not(isinstance(InputInitSpecies,InitSpecies)):
      raise Exception("error: input is not an InitSpecies class")
    self.mass = InputInitSpecies.mass
    self.charge = InputInitSpecies.charge
    self.numDen = InputInitSpecies.numDen
    self.num = None
    self.TypeID = None 
    self.force = None
  def SetForce(self, force):
    self.force = force
  def numDenUpdate(self, numDen):
    self.numDen = numDen
  def SetN(self, N):
    if type(N) is not int:
      raise Exception("Error: input must be integer")
    self.num = N
  def SetTypeID(self, TypeID):
    if type(TypeID) is not int:
      raise Exception("Error: input must be integer")
    self.TypeID = TypeID

# SimGrid class:
# A class designed for calculating properties in simulation grids
# requires to feed in: 
# SpeciesList: a list of SimSpecies object
# Ly, Lz: length of the simulation box in y and z directions in A
# Ti, Te: ion and electron temperatures in K
# dx: grid size in x direction
# the class contains:
# attributes: 
# eDen: electron density in A^-3
# Ly, Lz, Ti, Te, dx: same as those in the input arguments
# dV: volume of the grid in A^3
# kappa: inverse screening length in A^-1
# numDenSum: the sum of number densities of all species in A^-3
# eNum: total electron number in the grid 
# omega_p: aggregate plasma frequency in  fs^-1
# Efield: electric field in the grid in V/A
# aWSi, aWSe: Wigner-Seitz radii of ions and electrons in A
class SimGrid:
  def __init__(self, SpeciesList,Ly,Lz,Ti,Te,dx):
    self.SpeciesList = SpeciesList
    self.NSpecies    = len(self.SpeciesList) 
    self.eDen        = None
    self.eDenCalc()
    self.Ly, self.Lz = Ly, Lz
    self.Ti          = Ti 
    self.Te          = Te
    self.dx          = dx  
    self.dV          = dx*Ly*Lz 
    self.kappa       = None
    self.kappaCalc()
    self.numDenSum   = None
    self.numDenSumCalc()
    self.eNum        = None 
    self.eNumCalc()
    self.omega_p     = None
    self.omega_pCalc()
    self.Efield      = None
    self.aWSi        = None 
    self.aWSe        = None 
    self.aWSiCalc()
    self.aWSeCalc()
  def aWSiCalc(self):
    """
    calculate Wigner-Seitz radius of ions
    """
    self.aWSi = (3/(4*pi*self.numDenSum))**(1/3)
  def aWSeCalc(self):
    """
    calculate Wigner-Seitz radius of electrons
    """
    self.aWSe = (3/(4*pi*self.eDen))**(1/3)
  def eDenCalc(self):
    """
    calculate electron density according to number densities of ions and their charge numbers
    """
    self.eDen = sum([species.numDen*species.charge for species in self.SpeciesList])
  def eNumCalc(self):
    """
    calculate electron number
    """
    self.eNum = int(self.eDen*self.dV)
  def numUpdate(self, numArray):
    """
    update unumber counts of all species with an array of numbers
    """
    for iSpecies in range(self.NSpecies):
      self.SpeciesList[iSpecies].SetN(numArray[iSpecies])
  def numDenCalc(self):
    """
    calculate number density
    """
    for iSpecies in range(self.NSpecies):
      self.SpeciesList[iSpecies].numDenUpdate(self.SpeciesList[iSpecies].num/self.dV)
  def SetL(self, Lyin, Lzin):
    self.Ly, self.Lz = Lyin, Lzin
  def SetTi(self, Tiin):
    self.Ti = Tiin
  def setTe(self, Tein):
    self.Te = Tein
  def Setdx(self, dxin):
    self.dx = dxin
  def SetEfield(self, Ein):
    self.Efield = Ein
  def kappaCalc(self):
    """
    function for obtaining the TF screening length in 1/A
    """
    EF23 = EF23prefac*self.eDen**(2/3)*1E20
    # kappa_TF = 1/lambda_TF
    self.kappa = 1E-10*e*sqrt(1E30*self.eDen/(e0*sqrt(kB*kB*self.Te*self.Te + EF23*EF23)))
  def omega_pCalc(self):
    """
    obtain aggregate plasma frequency for a simulation grid, in Shaffer et al. 2017 
    omega_p = sqrt(n*<Z>^2*e^2/<m>*epsilon_0), <> denotes number averages, the frequency is in 1/fs
    requires to update numDen and numDenSum
    """
    ZAvg = self.numAvg([self.SpeciesList[iSpecies].charge for iSpecies in range(self.NSpecies)])
    mAvg = self.numAvg([self.SpeciesList[iSpecies].mass for iSpecies in range(self.NSpecies)])
    self.omega_p = 1E-15*sqrt(self.numDenSum*1E30*ZAvg*ZAvg*e2/(mAvg*mp*e0))
  def numDenSumCalc(self):
    """
    calculate the sum of number densities of ions
    requires updating numDen first to get an updated numDenSum
    """
    self.numDenSum = sum([self.SpeciesList[iSpecies].numDen for iSpecies in range(self.NSpecies)])
  def numAvg(self,AList):
    """
    determine the number average of A, given as a list with length NSpecies
    requires to update numDen and then numDenSum to make numAvg() is averaging with the up-to-date number densities
    """
    return sum([AList[iSpecies]*self.SpeciesList[iSpecies].numDen/self.numDenSum for iSpecies in range(self.NSpecies)])

# class of the entire simulation, comprises of (1) reading the input script file
# and (2) initialise parameters that are necessary to be used in LAMMPS. 
class simulation:
  # open the file
  def __init__(self, InputFile):
    with open(InputFile, "r") as f:
      # remove "\n", empty lines and comment lines
      lines = f.read().split("\n")
      lines = [jline for jline in lines if jline != '']
      lines = [jline for jline in lines if jline[0] != '#']
      # line count tool: whenever a line is read, make lineCount += 1, so the 
      # next line can be read
      lineCount = 0
      lineUpdate = lambda x: x+1
      
      # output file directories
      # directory of the input script
      self.dir = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      # equlibrium log file name
      self.EqmLogName = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      # production log file name
      self.ProdLogName = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      # stem of the dump file
      self.DumpStemName = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      
      # get species info
      # NSpecies for number of species
      self.NSpecies = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      # build a (2,NSpecies) array to have the information of two mixtures
      self.InitMixture = [[],[]]
      for iSpecies in range(self.NSpecies):
        word = lines[lineCount].split(); lineCount = lineUpdate(lineCount)
        self.InitMixture[0].append(InitSpecies(float(word[0].strip()), float(word[1].strip()) , float(word[2].strip())))
        self.InitMixture[1].append(InitSpecies(float(word[0].strip()), float(word[1].strip()) , float(word[3].strip())))
      
      # simulation box size info
      word = lines[lineCount].split(); lineCount = lineUpdate(lineCount)
      self.Lx, self.Ly, self.Lz = float(word[0].strip()), float(word[1].strip()), float(word[2].strip())
      self.Lx2, self.Ly2, self.Lz2 = self.Lx/2, self.Ly/2, self.Lz/2
      self.NGrid = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.dx = self.Lx/self.NGrid
      self.dV = self.dx*self.Ly*self.Lz
      self.SimGridList = list(range(self.NGrid))
      # the distribution width scale
      self.aWidth = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      word = lines[lineCount].split(); lineCount = lineUpdate(lineCount)
      # temperature input in the input file is in eV, but the LAMMPS input requires K
      self.Ti, self.Te = float(word[0].strip())*eV, float(word[1].strip())*eV
      
      # assemble simulation grids
      
      # get an array of Fermi-Dirac distribution values
      gDistArray = [self.gDist(self.grid2pos(ix)) for ix in range(self.NGrid)]

      # initialise arrays of grids
      self.SimulationBox = []
      for iGrid in range(self.NGrid):
        # initialise the list, which is made of a list of SimSpecies object made from InitSpecies
        SpeciesList = [SimSpecies(iSpecies) for iSpecies in self.InitMixture[0]]
        for iSpecies in range(self.NSpecies):
          # assign Type ID
          SpeciesList[iSpecies].SetTypeID(iSpecies+1)
          # calculate particle numbers
          SpeciesList[iSpecies].SetN(int(self.dV*SpeciesList[iSpecies].numDen))
        # assemble the simulation grid
        self.SimulationBox.append(SimGrid(SpeciesList,self.Ly,self.Lz,self.Ti,self.Te,self.dx))
        # update the number density of species; calculated from mixing the preset mixtures with the prescribed distribution function
        self.SimulationBox[iGrid].numUpdate([int(self.SimulationBox[iGrid].dV*self.InitMixture[0][iSpecies].numDen*gDistArray[iGrid] + self.SimulationBox[iGrid].dV*self.InitMixture[1][iSpecies].numDen*(1-gDistArray[iGrid])) for iSpecies in range(self.NSpecies)])
        # calculate number density
        self.SimulationBox[iGrid].numDenCalc()
        # calculate kappa, meanwhile update electron density
        self.SimulationBox[iGrid].eDenCalc()
        self.SimulationBox[iGrid].kappaCalc()
        # calculate omega_p with updated number density
        self.SimulationBox[iGrid].omega_pCalc()
      
      # time scale: calculate omega_p for all grids, the aggregate plasma frequency, follows the expression in Shaffer et al. 2017 
      # then take the maximum value 
      # omega_p = sqrt(n*<Z>^2*e^2/<m>*epsilon_0)
      self.omega_pmax = max([iGrid.omega_p for iGrid in self.SimulationBox])
      # plasma time 
      self.tp = 1./self.omega_pmax 
      # timestep number measured in fs
      self.tStep = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      # check the relations between plasma time and time step
      self.tStepCheck()
      # time and step number for equilibrium stage
      self.tEqm = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount) 
      self.NEqm = int(self.tEqm/self.tStep)
      # time and step number for production stage
      self.tProd = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.NProd = int(self.tProd/self.tStep)
      # time and step number for taking dumps
      self.tDump = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)  
      self.NDump = int(self.tDump/self.tStep)
      # forcefield type
      self.forcefield = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)

      # finding maximum aWS for ions and electrons
      self.aWSmaxi = (3/(4*pi*min([iGrid.numDenSum for iGrid in self.SimulationBox])))**(1/3)
      self.aWSmaxe = (3/(4*pi*min([iGrid.eDen for iGrid in self.SimulationBox])))**(1/3)
      # potential paramteres for Debye forcefield:
      if self.forcefield == 'Debye':
        # time and step numbers for updating kappa once
        self.tkappaUpdate = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        self.NkappaUpdate = int(self.tkappaUpdate/self.tStep)
        # number of cycles of updating kappa
        self.kappaUpdateNum = floor(self.NProd/self.NkappaUpdate)
        # residual steps after the last kappa update
        self.residualStep = self.NProd - self.kappaUpdateNum*self.NkappaUpdate
        # global cutoff measured in 1/kappa
        cutoffGlobalIn = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        # calculate the global cutoff with the function
        self.cutoffGlobal = self.cutoffGlobalCalc(cutoffGlobalIn)
        for iGrid in range(self.NGrid):
          for iSpecies in range(self.NSpecies):
            # reassign Type ID for Debye style since for different grids we require to assign the same species different types
            self.SimulationBox[iGrid].SpeciesList[iSpecies].SetTypeID(iSpecies*self.NGrid + iGrid+1)
        # skip the next few lines about parameters of other force fields
        for i in range(5):
          lineCount = lineUpdate(lineCount)
      # potential paramteres for eFF forcefield:
      elif self.forcefield == 'eFF':
        # skip the parameters for the previous force fields
        for i in range(2):
          lineCount = lineUpdate(lineCount)
        # global cutoff in A
        cutoffGlobalIn = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        self.cutoffGlobal = cutoffGlobalIn
        # skip the rest parameters for other force fields
        for i in range(4):
          lineCount = lineUpdate(lineCount)
      elif self.forcefield == 'Coul':
        # skip the parameters for the previous force fields
        for i in range(3):
          lineCount = lineUpdate(lineCount)
        # global cutoff measured in aWSmaxi
        cutoffGlobalIn = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        # cutoffGlobalIn*Largest Wigner-Seitz radius of the ion mixtures within all the simulation grid
        self.cutoffGlobal = cutoffGlobalIn*self.aWSmaxi
        # PPPM grid numbers
        word = lines[lineCount].split(); lineCount = lineUpdate(lineCount)
        self.PPPMNGridx, self.PPPMNGridy, self.PPPMNGridz = int(word[0].strip()), int(word[1].strip()), int(word[2].strip())
        # the cutoff for the repulsive core, measured in aWSmaxi
        cutoffCoreIn = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        self.cutoffCore = cutoffCoreIn*self.aWSmaxe
        # length scale of the repulsive core
        self.screenLengthCore = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      
      # neighbor list parameters
      self.neigh_one = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.neigh_page = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)

  # helper functions
  def FDDist(self,x):
    """
    function for obtaining the Fermi-Dirac distribution, f = 1/(exp(x/a)+1)
    """
    return 1/(exp(x/self.aWidth)+1)
  
  def gDist(self,x):
    """
    distribution that is designed for making the F-D distribution periodic
    """
    return self.FDDist(x) - self.FDDist(x+self.Lx2) + 1 - self.FDDist(x-self.Lx2)
  
  def pos2grid(self,x):
    """
    function that returns grid index when a position is provided
    """
    if abs(x) > self.Lx2 :
      raise Exception("error: x-coordinate outside of the simulation box")
    return floor((x + self.Lx2)/self.dx)
  
  def grid2pos(self,ix):
    """
    function that returns position when a grid number is provided
    """
    if ix > self.NGrid-1 or ix < 0:
      raise Exception("error: grid number outside of the box number range")
    return (ix+1/2)*self.dx-self.Lx2   

  def cutoffGlobalCalc(self, cutoffGlobalIn):
    """
    calculate cutoffGlobalIn/kappa for all grids, then obtain the max, which is 
    candidate global cutoff, and compare with Ly/2 and Lz/2. If the global 
    cutoff is larger than these values, then report error and terminate simulation
    as the candidate global cutoff is too large and when PBC is implemented,
    particle interact with itself
    """
    cutoffGlobalMax = max([cutoffGlobalIn/iGrid.kappa for iGrid in self.SimulationBox])
    if cutoffGlobalMax > min([self.Ly2, self.Lz2]):
      raise Exception("error: proposed Global cutoff is larger than 1/2 of the shortest side length, which are "+str(cutoffGlobalMax)+" m and "+str(min([self.Ly2, self.Lz2]+" m")))
    return cutoffGlobalMax

  def tStepCheck(self):
    """
    check if time step is smaller than the aggregate plasma time
    """
    if self.tp < self.tStep:
      raise Exception("error: proposed time step is larger than the smallest plasma time, which are "+str(self.tp)+" s and "+str(self.tStep)+" s")
    elif 0.1*self.tp < self.tStep:
      print("warning: time step is larger than the smallest plasma time, which are "+str(self.tp)+" s and "+str(self.tStep)+" s")
    else:
      pass