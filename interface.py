#-------------------------------------------------------------------
#
#                    interface_1.0/interface.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
#
# Glossaries:
# 
# Module dependencies:
#
#-------------------------------------------------------------------
#
#-------------------------------------------------------------------
#
# 2021.04.18 Created                                 TPZ
# 
#-------------------------------------------------------------------

import os
# supress multithreading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from lammps import PyLammps
import numpy as np
from random import randint
from mpi4py import MPI
from const import hbar,e,me,kB,mp,e0,e2,EF23prefac
from math import pi, floor, exp
from classes import species, grid




def FDDist(x,a):
  """
  function for obtaining the Fermi-Dirac distribution, f = 1/(exp(x/a)+1)
  """
  return 1/(exp(x/a)+1)

def gDist(x,a,Lx2):
  """
  distribution that is designed for making the F-D distribution periodic
  """
  return FDDist(x,a) - FDDist(x+Lx2,a) + 1 - FDDist(x-Lx2,a)

def pos2grid(x,dx,Lx2):
  """
  function that returns grid index when a position is provided
  """
  return floor((x + Lx2)/dx)

def grid2pos(ix,dx,Lx2):
  """
  function that returns position when a grid number is provided
  """
  return (ix+1/2)*dx-Lx2

def interface(sim, neigh_one = 5000, neigh_page = 50000):
  """
  
  """
  # variables
  Lx
  Ly
  Lz
  Lx2 = Lx/2
  Ly2 = Ly/2
  Lz2 = Lz/2
  omega_p
  tp
  NSpecies
  NRange
  aWidth
  NGrid
  InitSpeciesInfo[1][iSpecies].numDen
  InitSpeciesInfo[2][iSpecies].numDen
  SpeciesInfo[iSpecies].mass
  SpeciesInfo[iSpecies].charge
  Grid
  T
  cutoff_global
  ne[iGrid]
  EqmLogName
  ProdLogName


  


  # supress multithreading
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"

  # initiate PyLammps
  L = PyLammps()

  L.log(EqmLogName) 

  L.variable("NEvery", "equal", NEvery)
  
  # random number generator for random number seeds by using lambda variable, 
  # so that two calls won't produce the same sequence of random numbers 
  RNG = lambda: randint(1, 100000)

  # SI units
  L.units("si")
  # dimensions and BCs - periodic for y and z, but fixed for x
  L.dimension(3)
  L.boundary("f p p")

  # calculate timestep which is measured in LJ units
  tp = 1. / omega_p
  # time step
  dt = tStep * tp
  # step size of x 
  dx = Lx/(NGrid - 1)
  # volume of grid
  dV = dx*Ly*Lz

  #-------------------------------------------------------------------
  # create box
  L.region("box block", -Lx2, Lx2, -Ly2, Ly2, -Lz2, Lz2)

  # create simulation box
  L.create_box(NSpecies, "box") 
  
  # create regions, make solid walls
  for iGrid in range(NGrid):
    L.region("Region"+"_"+str(iGrid), "block", -Lx2+iGrid*dx, -Lx2+(iGrid+1)*dx, -Ly2, Ly2, -Lz2, Lz2)
    L.fix("Wall"+"_"+str(iGrid), "all", "wall/reflect", "xlo", -Lx2+iGrid*dx, "xhi", -Lx2+(iGrid+1)*dx, "units", "box")

  # make an array that stores the value of distribution function
  FDDistArray = [FDDist(grid2pos(ix,dx,Lx2),aWidth) for ix in range(NGrid)]

  # create (NSpecies,NGrid) number of random numbers
  RandCreate = []
  # only ask the processor w/ rank 0 to generate random numbers
  if MPI.COMM_WORLD.rank == 0:
    for iSpecies in range(NSpecies):
      # generates NSpecies number of random seed numbers for each grid
      RandCreate.append([RNG() for iGrid in range(NGrid)])
  # broadcast (distribute) the generated random numbers to every processor
  RandCreate = MPI.COMM_WORLD.bcast(RandCreate, root=0)
  
  # create an array of size (NSpecies,NGrid) to store particle numbers in each grid for each type
  AtomNum = [\
    [\
      int(\
        (SpeciesInfo[iSpecies].numDen[0]*FDDistArray[iGrid]+SpeciesInfo[iSpecies].numDen[1]*(1-FDDistArray[iGrid]))*dV \
         ) \
      for iGrid in range(NGrid) \
    ] for iSpecies in range(NSpecies) \
            ]

  # create an array of size (NSpecies,NGrid) to store particle type
  TypeNumber = [[ iSpecies*NGrid + iGrid+1 for iGrid in range(NGrid)] for iSpecies in range(NSpecies)]

  # create and set atoms, and their masses and charges
  for iSpecies in range(NSpecies):
    for iGrid in range(NGrid):
      L.create_atoms(TypeNumber[iSpecies][iGrid], "random", AtomNum[iSpecies][iGrid], RandCreate[iSpecies][iGrid], "Region"+"_"+str(iGrid))
      L.mass(iSpecies+1, SpeciesInfo[iSpecies].mass) 
      L.set("type", iSpecies+1, "charge", SpeciesInfo[iSpecies].charge)   
  
  # set the timestep
  L.timestep(dt) 
  
  # check neighbor parameters
  L.neigh_modify("delay", 0, "every", 1)
  
  # interaction style - Debye variable Kappa
  L.pair_style("coul/debye/vk",cutoff_global)
  # pair coeff for same type particles
  for iGrid in range(NGrid):
    for iSpecies in range(NSpecies):
      L.pair_coeff(TypeNumber[iSpecies][iGrid], TypeNumber[iSpecies][iGrid], TFScreen(simGrid[iGrid][iSpecies].eDen, T))

  #-------------------------------------------------------------------
  # Grouping based on species
  for iSpecies in range(NSpecies):
    L.group("Species"+str(iSpecies+1), "id", str(sum([sum(TypeNumber[jSpecies]) for jSpecies in range(iSpecies)]))+":"+str(sum(TypeNumber[iSpecies])))
  
  # generate a random number for setting velocity
  RandV = 0
  # only let the rank 0 processor to generate
  if MPI.COMM_WORLD.rank == 0:
    RandV = RNG()
  # broadcast to all the processors
  RandV = MPI.COMM_WORLD.bcast(RandV,root=0)
  L.velocity("all create", T, RandV)
  
  # Integrator set to be verlet
  L.run_style("verlet")
  # Nose-Hoover thermostat, the temperature damping parameter 
  # is suggested in [7]
  L.fix("Nose_Hoover all nvt temp", T, T, 100.0*dt) 

  #-------------------------------------------------------------------
  # Minimizing potential energy to prevent extremely high potential 
  # energy and makes the system reach equilibrium faster
  L.minimize("1.0e-6 1.0e-8 1000 10000")
  #-------------------------------------------------------------------
  # Equilibration run
  # log for equilibrium run
  
  L.reset_timestep(0)
  L.thermo(1000)
  # Equilibriation time
  L.run(NEqm)

  #-------------------------------------------------------------------
  # Production run
  # unfix NVT
  L.unfix("Nose_Hoover")
  # fix NVE, energy is conserved, not using NVT because T requires 
  # additional heat bath
  L.fix("NVEfix all nve") 
  L.reset_timestep(0)

  L.log(ProdLogName) 

  

