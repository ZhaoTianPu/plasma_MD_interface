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
# interface.
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
from math import sqrt, pi, exp

# InitSpecies class:
# A class specifically designed for storing initial configuration of species
# InitSpecies class contains:
# mass: mass
# charge: charge number
# numDen: particle number density
class InitSpecies:
  def __init__(self, mass, charge, numDen):
    self.mass     = mass
    self.charge   = charge
    self.numDen   = numDen

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
class SimGrid:
  def __init__(self, SpeciesList,Ly,Lz,T,dx):
    self.SpeciesList = SpeciesList
    self.NSpecies    = len(self.SpeciesList) 
    self.eDen        = None
    self.eDenCalc()
    self.Ly, self.Lz = Ly, Lz
    self.T           = T 
    self.dx          = dx  
    self.dV          = dx*Ly*Lz 
    self.kappa       = None
    self.kappaCalc()
    self.numDenSum   = None
    self.numDenSumCalc()
    self.omega_p     = None
    self.omega_pCalc()
  def eDenCalc(self):
    self.eDen = sum([species.numDen*species.charge for species in self.SpeciesList])
  def numUpdate(self, numArray):
    for iSpecies in range(self.NSpecies):
      self.SpeciesList[iSpecies].SetN(numArray[iSpecies])
  def numDenCalc(self):
    for iSpecies in range(self.NSpecies):
      self.SpeciesList[iSpecies].numDenUpdate(self.SpeciesList[iSpecies].num/self.dV)
  def SetL(self, Lyin, Lzin):
    self.Ly, self.Lz = Lyin, Lzin
  def SetT(self, Tin):
    self.T = Tin
  def Setdx(self, dxin):
    self.dx = dxin
  def kappaCalc(self):
    """
    function for obtaining the TF screening length in 1/A
    """
    EF23 = EF23prefac*self.eDen**(2/3)*1E20
    # kappa_TF = 1/lambda_TF
    self.kappa = 1E-10*e*sqrt(1E30*self.eDen/(e0*sqrt(kB*kB*self.T*self.T + EF23*EF23)))
  def omega_pCalc(self):
    """
    obtain aggregate plasma frequency for a simulation grid, in Shaffer et al. 2017 
    omega_p = sqrt(n*<Z>^2*e^2/<m>*epsilon_0), <> denotes number averages in 1/fs
    """
    ZAvg = self.numAvg([self.SpeciesList[iSpecies].charge for iSpecies in range(self.NSpecies)])
    mAvg = self.numAvg([self.SpeciesList[iSpecies].mass for iSpecies in range(self.NSpecies)])
    self.omega_p = 1E-15*sqrt(self.numDenSum*ZAvg*ZAvg*e2/(mAvg*mp*e0))
  def numDenSumCalc(self):
    self.numDenSum = sum([self.SpeciesList[iSpecies].numDen for iSpecies in range(self.NSpecies)])
  def numAvg(self,AList):
    """
    determine the number average of A, given as a list with length NSpecies
    """
    return sum([AList[iSpecies]*self.SpeciesList[iSpecies].numDen/self.numDenSum for iSpecies in range(self.NSpecies)])

class simulation:
  def __init__(self, InputFile):
    with open(InputFile, "r") as f:
      lines = f.read().split("\n")
      lines = [jline for jline in lines if jline != '']
      lines = [jline for jline in lines if jline[0] != '#']
      lineCount = 0
      lineUpdate = lambda x: x+1
      
      # output file directory
      self.dir = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      self.EqmLogName = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      self.ProdLogName = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      self.DumpStemName = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)
      
      # species info
      self.NSpecies = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
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
      self.aWidth = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.T = float(lines[lineCount].strip())*eV; lineCount = lineUpdate(lineCount)
      
      # assemble simulation grids
      
      # get an array of Fermi-Dirac distribution values
      FDDistArray = [self.FDDist(self.grid2pos(ix)) for ix in range(self.NGrid)]

      # initialise arrays of grids
      self.SimulationBox = []
      for iGrid in range(self.NGrid):
        # initialise the list, which is made of a list of SimSpecies object made from InitSpecies
        SpeciesList = [SimSpecies(iSpecies) for iSpecies in self.InitMixture[0]]
        for iSpecies in range(self.NSpecies):
          # assign Type ID
          SpeciesList[iSpecies].SetTypeID(iSpecies*self.NGrid + iGrid+1)
          # calculate particle numbers
          SpeciesList[iSpecies].SetN(int(self.dV*SpeciesList[iSpecies].numDen))
        # assemble the simulation grid
        self.SimulationBox.append(SimGrid(SpeciesList,self.Ly,self.Lz,self.T,self.dx))
        # update the number density of species
        self.SimulationBox[iGrid].numUpdate([int(self.SimulationBox[iGrid].dV*self.InitMixture[0][iSpecies].numDen*FDDistArray[iGrid] + self.SimulationBox[iGrid].dV*self.InitMixture[1][iSpecies].numDen*(1-FDDistArray[iGrid])) for iSpecies in range(self.NSpecies)])
        self.SimulationBox[iGrid].numDenCalc()
        # calculate kappa, meanwhile update electron density
        self.SimulationBox[iGrid].eDenCalc()
        self.SimulationBox[iGrid].kappaCalc()
        # calculate omega_p with updated number density
        self.SimulationBox[iGrid].omega_pCalc()
      
      # time scale: calculate omega_p for all grids, the aggregate plasma frequency, follows the expression in Shaffer et al. 2017 
      # then take the maximum value 
      # omega_p = sqrt(n*<Z>^2*e^2/<m>*epsilon_0)
      self.omega_p = max([iGrid.omega_p for iGrid in self.SimulationBox])
      self.tp = 1./self.omega_p 
      self.tStep = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      # check the relations between plasma time and time step
      self.tStepCheck()
      self.tEqm = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount) 
      self.NEqm = int(self.tEqm/self.tStep)
      self.tProd = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.NProd = int(self.tProd/self.tStep)
      self.tDump = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)  
      self.NDump = int(self.tDump/self.tStep)
      self.forcefield = lines[lineCount].strip(); lineCount = lineUpdate(lineCount)

      # potential paramteres:
      if self.forcefield == 'Debye':
        self.tkappaUpdate = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        # global cutoff
        cutoffGlobalIn = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        self.cutoffGlobal = self.cutoffGlobalCalc(cutoffGlobalIn)
      elif self.forcefield == 'eFF':
        lineCount = lineUpdate(lineCount)
        lineCount = lineUpdate(lineCount)
        cutoffGlobalIn = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
        self.cutoffGlobal = cutoffGlobalIn

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
    return self.FDDist(x,self.aWidth) - self.FDDist(x+self.Lx2,self.a) + 1 - self.FDDist(x-self.Lx2,self.a)
  
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