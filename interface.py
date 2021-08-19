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
from const import hbar,e,me,kB,mp,e0,e2,EF23prefac,Rkcal,qqr2e
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
  L.dimension(3)
  L.boundary("p p p")

  if sim.forcefield == "eFF":
    L.atom_style("electron")
  else:
    L.atom_style("charge")

  #-------------------------------------------------------------------
  # create box
  L.region("box block", -sim.Lx2, sim.Lx2, -sim.Ly2, sim.Ly2, -sim.Lz2, sim.Lz2)

  # create simulation box
  if sim.forcefield == "Debye":
    L.create_box(sim.NSpecies*sim.NGrid, "box") 
  elif sim.forcefield == "Coul":
    L.create_box(sim.NSpecies+1, "box")
  elif sim.forcefield == "eFF":
    L.create_box(sim.NSpecies+1, "box")

  # create regions, make solid walls
  for iGrid in range(sim.NGrid):
    L.region("Region"+"_"+str(iGrid), "block", -sim.Lx2+iGrid*sim.dx, -sim.Lx2+(iGrid+1)*sim.dx, -sim.Ly2, sim.Ly2, -sim.Lz2, sim.Lz2)
    L.fix("Wall"+"_"+str(iGrid), "all", "wall/lj126", "xlo",-sim.Lx2+iGrid*sim.dx, Rkcal*min(sim.Ti,sim.Te), sim.aWSmaxi, sim.aWSmaxi, "xhi", -sim.Lx2+(iGrid+1)*sim.dx, Rkcal**min(sim.Ti,sim.Te), sim.aWSmaxi, sim.aWSmaxi, "units", "box", "pbc", "yes")

  # create (NSpecies+1,NGrid) number of random numbers
  RandCreate = []
  # only ask the processor w/ rank 0 to generate random numbers
  if MPI.COMM_WORLD.rank == 0:
    for iSpecies in range(sim.NSpecies):
      # generates NSpecies number of random seed numbers for each grid
      RandCreate.append([RNG() for iGrid in range(sim.NGrid)])
  # broadcast (distribute) the generated random numbers to every processor
  RandCreate = MPI.COMM_WORLD.bcast(RandCreate, root=0)

  # create and set atoms, and their masses and charges
  for iGrid in range(sim.NGrid):
    for iSpecies in range(sim.NSpecies):
      L.create_atoms(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, "random", sim.SimulationBox[iGrid].SpeciesList[iSpecies].num, RandCreate[iSpecies][iGrid], "Region"+"_"+str(iGrid))
  
  if sim.forcefield == "Debye":
    for iGrid in range(sim.NGrid):
      for iSpecies in range(sim.NSpecies):
        L.mass(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].mass) 
        L.set("type", sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, "charge", sim.SimulationBox[iGrid].SpeciesList[iSpecies].charge)  
  elif sim.forcefield == "Coul":
    for iSpecies in range(sim.NSpecies):
      L.mass(sim.SimulationBox[0].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].mass) 
      L.set("type", sim.SimulationBox[0].SpeciesList[iSpecies].TypeID, "charge", sim.SimulationBox[iGrid].SpeciesList[iSpecies].charge)  
  elif sim.forcefield == "eFF":
    for iSpecies in range(sim.NSpecies):
      L.mass(sim.SimulationBox[0].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].mass) 
      L.set("type", sim.SimulationBox[0].SpeciesList[iSpecies].TypeID, "charge", sim.SimulationBox[iGrid].SpeciesList[iSpecies].charge)  

  # create electrons
  if sim.forcefield == "eFF":
    for iGrid in range(sim.NGrid):
      L.create_atoms(sim.NSpecies+1, "random", sim.SimulationBox[iGrid].eNum, RandCreate[NSpecies][iGrid], "Region"+"_"+str(iGrid))
    L.mass(sim.NSpecies+1, sim.emass) 
    L.set("type", sim.NSpecies+1, "charge", -1)  
  elif sim.forcefield == "Coul":
    for iGrid in range(sim.NGrid):
      L.create_atoms(sim.NSpecies+1, "random", sim.SimulationBox[iGrid].eNum, RandCreate[NSpecies][iGrid], "Region"+"_"+str(iGrid))
    L.mass(sim.NSpecies+1, sim.emass) 
    L.set("type", sim.NSpecies+1, "charge", -1)  
    
  print("particles are set up")

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
        L.pair_coeff(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].kappa)
  elif sim.forcefield == "eFF":
    L.pair_style("eff/cut", sim.cutoffGlobal)
    L.pair_coeff("* *")
    L.compute("effTemp all temp/eff")
    L.thermo_modify("temp effTemp")
    L.comm_modify("vel yes")
  elif sim.forcefield == "Coul":
    L.pair_style("hybrid", "coul/long", sim.cutoffGlobal, "yukawa", 1/sim.screenLengthCore, sim.cutoffCore)
    for iSpecies in sim.SimulationBox[0].SpeciesList:
      L.pair_coeff(sim.NSpecies+1, iSpecies.TypeID, "coul/long","yukawa", iSpecies.charge*qqr2e)
      L.pair_coeff(iSpecies.TypeID, sim.NSpecies+1, "coul/long","yukawa", iSpecies.charge*qqr2e) 
    L.pair_coeff("1*"+str(sim.NSpecies), "1*"+str(sim.NSpecies), "coul/long")
    L.pair_coeff(sim.NSpecies+1, sim.NSpecies+1, "coul/long")

    L.kspace_style("pppm", 1.0E-5)
    # NGrids need to be integer value of 2, 3 or 5
    L.kspace_modify("mesh", sim.PPPMNGridx, sim.PPPMNGridy, sim.PPPMNGridz) 
    # Change tabinner to be smaller than default so that the real space
    # potential calculation can be benefitted by using the table; the 
    # default inner cutoff for using spline table is too large, larger
    # than the cutoff between direct/inverse cutoff of coul/long potential
    L.pair_modify("tabinner", 0.1*sim.aWSmaxi)
    # neighbor list need to be increased accordingly if the density (Gamma)
    # is too high, the "page" value need to be at least 10x the "one" value

  print("pair style config is finished")
  #-------------------------------------------------------------------
  # Grouping based on species
  # electron/ion groups
  ionGroup = ""
  for iSpecies in range(sim.NSpecies):
    L.group("Species_"+str(iSpecies+1), "type", str(sim.SimulationBox[0].SpeciesList[iSpecies].TypeID)+":"+str(sim.SimulationBox[-1].SpeciesList[iSpecies].TypeID))
    ionGroup += "Species_"+str(iSpecies+1)+" "
  L.group("ion", "union", ionGroup)
  if sim.forcefield == "eFF":
    L.group("electron", "type", sim.NSpecies*sim.NGrid+1)
  elif sim.forcefield == "Coul":
    L.group("electron", "type", sim.NSpecies+1)
  # for Debye: 
  # elif sim.forcefield == "Debye":
  #   for iGrid in range(sim.NGrid):
  #     # set type group for each type
  #     for iSpecies in sim.SimulationBox[iGrid].SpeciesList:
  #       L.group("Type_"+str(iSpecies.TypeID), "type", str(iSpecies.TypeID))
  #     # set groups based on Regions
  #     L.group("RegionGroup_"+str(iGrid), "region", "Region"+"_"+str(iGrid) )
  
  print("particle grouping is finished")

  # generate a random number for setting velocity
  RandV = []
  # only let the rank 0 processor to generate
  if MPI.COMM_WORLD.rank == 0:
    RandV = [RNG(), RNG()]
  # broadcast to all the processors
  RandV = MPI.COMM_WORLD.bcast(RandV,root=0)
  if sim.forcefield == "eFF":
    L.velocity("ion", "create", sim.Ti, RandV[0], "rot yes dist gaussian")
    L.velocity("electron", "create", sim.Te, RandV[1], "rot yes dist gaussian")
  elif sim.forcefield == "Debye":
    L.velocity("all create", sim.Ti, RandV[0], "dist gaussian")  
  elif sim.forcefield == "Coul":
    L.velocity("ion", "create", sim.Ti, RandV[0], "dist gaussian")  
    L.velocity("electron", "create", sim.Te, RandV[1], "dist gaussian")  
  # Integrator set to be verlet
  L.run_style("verlet")
  # Nose-Hoover thermostat, the temperature damping parameter is suggested by the official document
  if sim.forcefield == "eFF":
    L.fix("Nose_Hoover_i", "ion", "nvt/eff", "temp", sim.Ti, sim.Ti, 100.0*sim.tStep) 
    L.fix("Nose_Hoover_e", "electron", "nvt/eff", "temp", sim.Te, sim.Te, 100.0*sim.tStep) 
  elif sim.forcefield == "Debye":
    L.fix("Nose_Hoover all nvt temp", sim.Ti, sim.Ti, 100.0*sim.tStep) 
  elif sim.forcefield == "Coul":
    L.fix("Nose_Hoover_i", "ion", "nvt", "temp", sim.Ti, sim.Ti, 100.0*sim.tStep) 
    L.fix("Nose_Hoover_e", "electron", "nvt", "temp", sim.Te, sim.Te, 100.0*sim.tStep) 
  
  print("force field config is finished")

  L.thermo(1000)
  #-------------------------------------------------------------------
  # Minimizing potential energy to prevent extremely high potential 
  # energy and makes the system reach equilibrium faster
  L.minimize("1.0E-4 1.0e-4 1000 10000")
  #-------------------------------------------------------------------
  # Equilibration run
  # log for equilibrium run
  
  print("run equilibration stage")

  L.reset_timestep(0)
  # Equilibriation time
  L.run(sim.NEqm)

  print("equilibration stage is finished")
  print("run production stage")
  #-------------------------------------------------------------------
  # Production run
  L.log(sim.ProdLogName) 
  # unfix NVT
  if sim.forcefield == "Debye":
    L.unfix("Nose_Hoover")
  else:
    L.unfix("Nose_Hoover_i")
    L.unfix("Nose_Hoover_e")
  # fix NVE, energy is conserved, not using NVT because T requires 
  # additional heat bath
  if sim.forcefield == "eFF":
    L.fix("NVEfix all nve/eff") 
  else:
    L.fix("NVEfix all nve") 
  # relax the reflective walls
  for iGrid in range(sim.NGrid):
    L.unfix("Wall"+"_"+str(iGrid))
  
  # set and calculate force
  Fprefac1 = -Rkcal*sim.Te/(2*sim.dx)
  for iGrid in range(sim.NGrid-1):
    Fprefac2 = Fprefac1*(sim.SimulationBox[iGrid+1].eDen - sim.SimulationBox[iGrid-1].eDen)/sim.SimulationBox[iGrid+1].eDen
    for iSpecies in sim.SimulationBox[iGrid].SpeciesList:
      iSpecies.SetForce(iSpecies.charge*Fprefac2)

  Fprefac2 = Fprefac1*(sim.SimulationBox[0].eDen - sim.SimulationBox[NGrid-2].eDen)/sim.SimulationBox[NGrid-1].eDen  
  for iSpecies in sim.SimulationBox[NGrid-1].SpeciesList:
    iSpecies.SetForce(iSpecies.charge*Fprefac2)
  
  for iGrid in range(sim.NGrid):
    for iSpecies in range(sim.NSpecies):
      L.fix("Force_Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "addforce", sim.SimulationBox[iGrid].SpeciesList[iSpecies].force, 0.0, 0.0)
  
  # run simulations
  L.reset_timestep(0)
  
  # for each cycle
  for iCycle in range(self.DumpNum):
    # run the simulation for NDump steps
    L.run(sim.NDump)
    # reassign types for species
    for iGrid in range(sim.NGrid):
      # temporary group for atoms in a region
      L.group("RegionGroup_"+str(iGrid), "region", "Region"+"_"+str(iGrid))
      # temporary storage of count number of each type
      num_Type_temp = []
      for iSpecies in range(sim.NSpecies):
        # temporary group of a species in a region
        L.group("TypeGroup_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "intersect", "Species_"+str(iSpecies+1), "RegionGroup_"+str(iGrid))
        # redefine types for these species
        L.set("group", "TypeGroup_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "type", str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID))
        # remove the temporary group for a species in a region
        L.group("TypeGroup_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "delete")
        # count numbers for these species
        L.variable("num_Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "equal", "count("+"TypeGroup_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID)+")")
        # add number count of each type into the array
        num_Type_temp.append(L.variables["num_Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID)])
        # delete the variable
        L.variable("num_Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "delete")
      sim.SimulationBox[iGrid].numUpdate(num_Type_temp)
      # calculate number density
      sim.SimulationBox[iGrid].numDenCalc()
      # calculate electron density
      sim.SimulationBox[iGrid].eDenCalc()
      # update screening parameters
      sim.SimulationBox[iGrid].kappaCalc()
      for iSpecies in range(sim.NSpecies):
        L.pair_coeff(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].kappa)
      # remove the temporary group for atoms in a region
      L.group("RegionGroup_"+str(iGrid), "delete")

    # update force applied
    for iGrid in range(sim.NGrid-1):
      Fprefac2 = Fprefac1*(sim.SimulationBox[iGrid+1].eDen - sim.SimulationBox[iGrid-1].eDen)/sim.SimulationBox[iGrid+1].eDen
      for iSpecies in sim.SimulationBox[iGrid].SpeciesList:
        iSpecies.SetForce(iSpecies.charge*Fprefac2)
  
    Fprefac2 = Fprefac1*(sim.SimulationBox[0].eDen - sim.SimulationBox[NGrid-2].eDen)/sim.SimulationBox[NGrid-1].eDen  
    for iSpecies in sim.SimulationBox[NGrid-1].SpeciesList:
      iSpecies.SetForce(iSpecies.charge*Fprefac2)
    
    for iGrid in range(sim.NGrid):
      for iSpecies in range(sim.NSpecies):
        # unfix previous forces
        L.unfix("Force_Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID))
        # fix new forces
        L.fix("Force_Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "Type_"+str(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID), "addforce", sim.SimulationBox[iGrid].SpeciesList[iSpecies].force, 0.0, 0.0)
  
  L.run(sim.residualStep)
  
  print("production stage is finished")