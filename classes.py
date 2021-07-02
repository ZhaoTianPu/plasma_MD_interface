#-------------------------------------------------------------------
#
#                 interface_1.0/classes.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
# This module contains the simulation class that reads input file
# that contains the essential simulation parameters for either one
# component fluid or binary mixture, interacting via an inverse 
# polynomial potential or long-range Coulombic potential. It also
# calculates several quantities for analysis. All the quantities in
# the simulations are in LJ dimensionless units, as specified in [2], 
# and all the expressions for derived variables can also be found in [1].
# 
# Module prerequisites:
# The versions below are the ones that I use when developing the 
# code. Earlier/later version may or may not work. 
# python: 3.8.5
# numpy: 1.19
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
# [1]: unit_conversions.pdf
# [2]: https://lammps.sandia.gov/doc/units.html
# [3]: Shaffer et al. PRE 95 013206 (2017)
# [4]: https://lammps.sandia.gov/doc/fix_ave_correlate.html
#
#-------------------------------------------------------------------
#
# 2021.06.29 Created                                 TPZ
# 2021.07.02 Modified                                TPZ            
#            (Updated script up to the initialisation of the 
#            simulation box)   
# 
#-------------------------------------------------------------------

import numpy as np
from const import hbar,e,me,kB,mp,e0,e2,EF23prefac

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
  def SetN(self, N):
    if type(N) is not int:
      raise Exception("Error: input must be integer")
    self.num = N
  def SetTypeID(self, TypeID):
    if type(TypeID) is not int:
      raise Exception("Error: input must be integer")

# SimGrid class:
# A class designed for calculating properties in simulation grids
class SimGrid:
  def __init__(self, speciesList):
    self.speciesList = speciesList
    self.NSpecies    = len(self.speciesList) 
    self.eDen        = self.eDenCalc()
    self.Ly          = None
    self.Lz          = None
    self.kappa       = None
    self.T           = None
    self.dx          = None 
  def eDenCalc(self):
    self.eDen = sum([species.numDen*species.charge for species in self.speciesList])
  def numDenUpdate(self, numDenArray):
    for iSpecies in range(NSpecies)
      self.speciesList[iSpecies].numDen = numDenArray[iSpecies]
  def SetL(self, Lyin, Lzin):
    self.Ly, self.Lz = Lyin, Lzin
  def SetT(self, Tin):
    self.T = Tin
  def Setdx(self, dxin):
    self.dx = dxin
  def kappaCalc(self):
    """
    function for obtaining the TF screening length
    """
    self.eDenCalc()
    EF23 = EF23prefac*self.eDen**(2/3)
    # kappa_TF = 1/lambda_TF
    self.kappa = e*sqrt(self.eDen/(e0*sqrt(kB*kB*self.T*self.T + EF23*EF23)))

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
      
      # species info
      self.NSpecies = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.InitMixture = [[],[]]
      for iSpecies in range(NSpecies):
        word = lines[lineCount].split(); lineCount = lineUpdate(lineCount)
        InitMixture[0].append(InitSpecies(float(word[0].strip()), float(word[1].strip()) , float(word[2].strip())))
        InitMixture[1].append(InitSpecies(float(word[0].strip()), float(word[1].strip()) , float(word[3].strip())))
      
      # simulation box size info
      word = lines[lineCount].split(); lineCount = lineUpdate(lineCount)
      self.Lx, self.Ly, self.Lz = float(word[0].strip()), float(word[1].strip()), float(word[2].strip())
      self.Lx2, self.Ly2, self.Lz2 = self.Lx/2, self.Ly/2, self.Lz/2
      self.NGrid = int(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.dx = Lx/NGrid
      self.dV = self.dx*self.Ly*self.Lz
      self.SimGridList = list(range(self.NGrid))
      self.aWidth = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      self.T = float(lines[lineCount].strip()); lineCount = lineUpdate(lineCount)
      
      # assemble simulation grids
      
      # get an array of Fermi-Dirac distribution values
      FDDistArray = [self.FDDist(self.grid2pos(ix)) for ix in range(self.NGrid)]

      # initialise arrays of grids
      self.SimulationBox = []
      for iGrid in range(self.NGrid):
        AtomNum = int(\
        (SpeciesInfo[iSpecies].numDen[0]*FDDistArray[iGrid]+SpeciesInfo[iSpecies].numDen[1]*(1-FDDistArray[iGrid]))*dV \
         )
        speciesList = [SimSpecies(self.InitMixture[0][iSpecies]) for iSpecies in range(self.NSpecies)]
        for iSpecies in range(self.NSpecies):
          speciesList[iSpecies].SetnumDen(InitMixture[0][iSpecies].numDen*FDDistArray[iGrid] + InitMixture[1][iSpecies].numDen*(1-FDDistArray[iGrid]))
          speciesList[iSpecies].SetTypeID(iSpecies*NGrid + iGrid+1)
          speciesList[iSpecies].SetN(int(self.dV*speciesList[iSpecies].numDen))
        self.SimulationBox.add(SimGrid(speciesList))
        self.SimulationBox[iGrid].numDenUpdate([InitMixture[0][iSpecies].numDen*FDDistArray[iGrid] + InitMixture[1][iSpecies].numDen*(1-FDDistArray[iGrid]) for iGrid in range(NGrid)])
        self.SimulationBox[iGrid].SetL(self.Ly,self.Lz)
        self.SimulationBox[iGrid].SetT(self.T)
        self.SimulationBox[iGrid].Setdx(self.dx)
        self.SimulationBox[iGrid].kappaCalc()

      # to-dos
      # self.omega_p
      # self.cutoff_global
      # self.tEqm
      # self.tProd
      # self.dumpInterval
      # self.updateInterval

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
    if ix > NGrid-1 or ix < 0:
      raise Exception("error: grid number outside of the box number range")
    return (ix+1/2)*self.dx-self.Lx2   



#--------------------------------------------------------------------
  #     # simulation times
  #     self.tStep     = float(lines[42])
  #     self.tEqm      = float(lines[45])
  #     word = lines[57].split()
  #     self.NReps,   self.tInt,      self.tLength,   self.tFreq,     self.NEvery = \
  #     int(word[0]), float(word[1]), float(word[2]), float(word[3]), int(word[4])
      
  #     # convert time to steps
  #     self.NEqm    = self.ttoN(self.tEqm)
  #     self.NInt    = self.ttoN(self.tInt)
  #     self.NLength = self.ttoN(self.tLength)
  #     self.NFreq   = self.ttoN(self.tFreq)
  #     if not (self.NLength % self.NEvery == 0):
  #       raise Exception("NLength need to be multiples of NEvery")
  #     if not (self.NFreq % self.NEvery == 0):
  #       raise Exception("NFreq need to be multiples of NEvery")
      
  #     # file names
  #     word = lines[68].split()
  #     self.EqmLogStem, self.ProdLogStem, self.FileStem = \
  #     word[0].strip(), word[1].strip() , word[2].strip()

  #     # cutoff
  #     self.cutoff    = float(lines[75])

  #     # pair style specific arguments
  #     if self.PairStyle == "Coul":
  #       word = lines[84].split()
  #       if word[0] == "Gamma0":
  #         self.Gamma0,    self.NGrids  =\
  #         float(word[1]), int(word[2])
  #         self.Gamma0toGamma()
  #         self.CalcPlasmaParameters()
  #       elif word[0] == "Gamma":
  #         self.Gamma,     self.NGrids  =\
  #         float(word[1]), int(word[2])
  #         self.GammatoGamma0()
  #         self.CalcPlasmaParameters()
  #       else:
  #         raise Exception("GammaMode need to be either Gamma or Gamma0")
  #     elif self.PairStyle == "poly":
  #       self.numden = float(lines[88])
  #       word = lines[93].split()
  #       self.polyCoeff = [float(iCoeff) for iCoeff in word]
  #     else:
  #       raise Exception("PairStyle need to be Coul or poly, other styles are not yet supported")


  
  # # convert time into timestep
  # def ttoN(self, tval):
  #   Nval = int(tval/self.tStep)
  #   return Nval
  
  # # get the mole and mass fraction of each species
  # def GetComposition(self):
  #   for iSpecies in self.species:
  #     iSpecies.AssignMolFrac(iSpecies.number / self.NTotal)
  #     iSpecies.AssignMassFrac(iSpecies.number*iSpecies.mass / self.MTotal)
  
  # # number average of a quantity A for each species, A need to be a vector with length NSpecies
  # def NumAvg(self, A):
  #   if self.NSpecies != np.size(A):
  #     raise Exception("The value to be averaged must have the same length as number of species")
  #   avgA = np.dot([iSpecies.molfrac for iSpecies in self.species], A)
  #   return avgA
  
  # # calculate various parameters of plasma, and overall density 
  # def CalcPlasmaParameters(self):
  #   # number-averaged charge number
  #   avgZ     = self.NumAvg([iSpecies.charge for iSpecies in self.species])
  #   # number-averaged mass
  #   avgMass  = self.NumAvg([iSpecies.mass for iSpecies in self.species])
  #   # Wigner-Seitz radius
  #   aWS           = 1./self.Gamma0
  #   self.aWS      = aWS
  #   # mixture plasma frequency, eq. (36) in [3]
  #   self.omega_p  = (3.                          / ((aWS*aWS*aWS) * avgMass      )) ** (1./2.) * avgZ
  #   # plasma frequency of the lighter species
  #   self.omega_p1 = (3. * self.species[0].molfrac/ ((aWS*aWS*aWS) * self.species[0].mass)) ** (1./2.) * self.species[0].charge
  #   # total number density
  #   self.numden   = 3./(4.*np.pi*aWS*aWS*aWS)
  #   # total mass density
  #   self.rho      = avgMass*self.numden
  
  # # Gamma_0 as defined by eq. (3) in [3], from Gamma bar defined by eq. (2) in [3]
  # def Gamma0toGamma(self):
  #   avgZ = self.NumAvg([iSpecies.charge for iSpecies in self.species])
  #   self.Gamma  = self.Gamma0 * (avgZ ** (1./3.) * self.NumAvg([iSpecies.charge ** (5./3.) for iSpecies in self.species]))
  # # Gamma bar as defined by eq. (2) in [3], from Gamma_0 defined by eq. (3) in [3]
  # def GammatoGamma0(self):
  #   avgZ = self.NumAvg([iSpecies.charge for iSpecies in self.species])
  #   self.Gamma0 = self.Gamma  / (avgZ ** (1./3.) * self.NumAvg([iSpecies.charge ** (5./3.) for iSpecies in self.species]))

  #       