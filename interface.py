#-------------------------------------------------------------------
#
#                    interface_1.0/interface.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
# in real units
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
from time import sleep

def interface(sim):
  """
  
  """
  # supress multithreading
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"

  # create directory with only processor at rank 0
  if MPI.COMM_WORLD.rank == 0:
    if not os.path.isdir(sim.dir):
      os.mkdir(sim.dir)

  sleep(3)
  os.chdir(sim.dir)

  if not isinstance(sim,simulation):
    raise Exception("error: the input is not a simulation class")
  # random number generator for random number seeds by using lambda variable, 
  # so that two calls won't produce the same sequence of random numbers 
  RNG = lambda: randint(1, 100000)
  
  # initiate PyLammps
  L = PyLammps()

  # L.log(sim.EqmLogName) 

  # real units
  L.units("real")
  # dimensions and BCs - periodic for y and z, but fixed for x
  L.dimension(3)
  L.boundary("f p p")

  if sim.forcefield == "eFF":
    L.atom_style("electron")
  elif sim.forcefield == "Debye":
    L.atom_style("charge")

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

  # create and set atoms, and their masses and charges
  for iSpecies in range(sim.NSpecies):
    for iGrid in range(sim.NGrid):
      L.create_atoms(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, "random", sim.SimulationBox[iGrid].SpeciesList[iSpecies].num, RandCreate[iSpecies][iGrid], "Region"+"_"+str(iGrid))
      L.mass(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].mass) 
      L.set("type", sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, "charge", sim.SimulationBox[iGrid].SpeciesList[iSpecies].charge)   
  
  # set the timestep
  L.timestep(sim.tStep) 
  
  # check neighbor parameters
  L.neigh_modify("delay", 0, "every", 1, "one", sim.neigh_one, "page", sim.neigh_page)
  
  # interaction style
  # for Debye with variable Kappa
  if sim.forcefield == "Debye":
    L.pair_style("coul/debye/vk",sim.cutoffGlobal)
    # pair coeff for same type particles
    for iGrid in range(sim.NGrid):
      for iSpecies in range(sim.NSpecies):
        L.pair_coeff(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].kappa, sim.T)
  elif sim.forcefield == "eFF":
    L.pair_style("eff/cut", sim.cutoffGlobal)
    L.pair_coeff("* *")
    L.compute("effTemp all temp/eff")
    L.thermo_modify("temp effTemp")
    L.comm_modify("vel yes")


  #-------------------------------------------------------------------
  # Grouping based on species
  for iSpecies in range(sim.NSpecies):
    L.group("Species"+str(iSpecies+1), "type", str(sim.SimulationBox[0].SpeciesList[iSpecies].TypeID)+":"+str(sim.SimulationBox[-1].SpeciesList[iSpecies].TypeID))
  
  # generate a random number for setting velocity
  RandV = 0
  # only let the rank 0 processor to generate
  if MPI.COMM_WORLD.rank == 0:
    RandV = RNG()
  # broadcast to all the processors
  RandV = MPI.COMM_WORLD.bcast(RandV,root=0)
  if sim.forcefield == "eFF":
    L.velocity("all create", sim.T, RandV, "rot yes mom yes dist gaussian")
  elif sim.forcefield == "Debye":
    L.velocity("all create", sim.T, RandV, "dist gaussian")  
  # Integrator set to be verlet
  L.run_style("verlet")
  # Nose-Hoover thermostat, the temperature damping parameter is suggested by the official document
  if sim.forcefield == "eFF":
    L.compute("effTemp all temp/eff")
    L.thermo_modify("temp effTemp")
    L.fix("Nose_Hoover all nvt/eff temp", sim.T, sim.T, 100.0*sim.tStep) 
  elif sim.forcefield == "Debye":
    L.fix("Nose_Hoover all nvt temp", sim.T, sim.T, 100.0*sim.tStep) 

  L.thermo(10)
  #-------------------------------------------------------------------
  # Minimizing potential energy to prevent extremely high potential 
  # energy and makes the system reach equilibrium faster
  if sim.forcefield == "Debye":
    L.minimize("0.0 1.0e-4 1000 10000")
  #-------------------------------------------------------------------
  # Equilibration run
  # log for equilibrium run
  
  L.reset_timestep(0)
  # Equilibriation time
  L.run(sim.NEqm)

  #-------------------------------------------------------------------
  # Production run
  L.log(sim.ProdLogName) 
  # unfix NVT
  L.unfix("Nose_Hoover")
  # fix NVE, energy is conserved, not using NVT because T requires 
  # additional heat bath
  if sim.forcefield == "eFF":
    L.fix("NVEfix all nve/eff") 
  elif sim.forcefield == "Debye":
    L.fix("NVEfix all nve") 
  L.reset_timestep(0)
  L.run(sim.NProd)
