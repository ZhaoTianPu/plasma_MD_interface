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
# 2021.04.18 Created                                             TPZ
# 2021.07.02 Modified                                            TPZ
#            (transfer functions in the simulation class and other
#            classes; simulation code to be reviewed and updated as
#            the strategy of code development now focuses on using
#            classes to include simulation information)
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
from classes import InitSpecies, SimSpecies, SimGrid, simulation

def interface(sim, neigh_one = 5000, neigh_page = 50000):
  """
  
  """
  # supress multithreading
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"

  if not isinstance(sim,simulation):
    raise Exception("error: the input is not a simulation class")
  # random number generator for random number seeds by using lambda variable, 
  # so that two calls won't produce the same sequence of random numbers 
  RNG = lambda: randint(1, 100000)
  
  # initiate PyLammps
  L = PyLammps()

  L.log(sim.EqmLogName) 

  # SI units
  L.units("si")
  # dimensions and BCs - periodic for y and z, but fixed for x
  L.dimension(3)
  L.boundary("f p p")

  #-------------------------------------------------------------------
  # create box
  L.region("box block", -sim.Lx2, sim.Lx2, -sim.Ly2, sim.Ly2, -sim.Lz2, sim.Lz2)

  # create simulation box
  L.create_box(sim.NSpecies*sim.NGrid, "box") 
  
  # create regions, make solid walls
  for iGrid in range(sim.NGrid):
    L.region("Region"+"_"+str(iGrid), "block", -sim.Lx2+iGrid*sim.dx, -sim.Lx2+(iGrid+1)*sim.dx, -sim.Ly2, sim.Ly2, -sim.Lz2, sim.Lz2)
    L.fix("Wall"+"_"+str(iGrid), "all", "wall/reflect", "xlo", -sim.Lx2+iGrid*sim.dx, "xhi", -sim.Lx2+(iGrid+1)*sim.dx, "units", "box")

  # create (NSpecies,NGrid) number of random numbers
  RandCreate = []
  # only ask the processor w/ rank 0 to generate random numbers
  if MPI.COMM_WORLD.rank == 0:
    for iSpecies in range(sim.NSpecies):
      # generates NSpecies number of random seed numbers for each grid
      RandCreate.append([RNG() for iGrid in range(sim.NGrid)])
  # broadcast (distribute) the generated random numbers to every processor
  RandCreate = MPI.COMM_WORLD.bcast(RandCreate, root=0)
  
  # create an array of size (NSpecies,NGrid) to store particle numbers in each grid for each type
  # AtomNum = [\
  #   [\
  #     int(\
  #       (SpeciesInfo[iSpecies].numDen[0]*FDDistArray[iGrid]+SpeciesInfo[iSpecies].numDen[1]*(1-FDDistArray[iGrid]))*dV \
  #        ) \
  #     for iGrid in range(NGrid) \
  #   ] for iSpecies in range(NSpecies) \
  #           ]

  # create an array of size (NSpecies,NGrid) to store particle type
  # TypeNumber = [[ iSpecies*NGrid + iGrid+1 for iGrid in range(NGrid)] for iSpecies in range(NSpecies)]

  # create and set atoms, and their masses and charges
  for iSpecies in range(NSpecies):
    for iGrid in range(NGrid):
      L.create_atoms(sim.SimulationBox[iGrid].speciesList[iSpecies].TypeID, "random", AtomNum[iSpecies][iGrid], RandCreate[iSpecies][iGrid], "Region"+"_"+str(iGrid))
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

  

