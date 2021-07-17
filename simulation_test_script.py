# to be used in login node

import os
from lammps import PyLammps
import numpy as np
from random import randint
from const import hbar,e,me,kB,mp,e0,e2,EF23prefac
from math import pi, floor, exp
from classes import InitSpecies, SimSpecies, SimGrid, simulation
from time import sleep
from classes import simulation
import sys

sim = simulation("input_benchmark_1000step_DT_N_1E5_T_50eV_node_1_cpu_48.txt")
if not isinstance(sim,simulation):
  raise Exception("error: the input is not a simulation class")
RNG = lambda: randint(1, 100000)
L = PyLammps()
L.units("real")
L.dimension(3)
L.boundary("f p p")

if sim.forcefield == "eFF":
  L.atom_style("electron")
elif sim.forcefield == "Debye":
  L.atom_style("charge")

L.region("box block", -sim.Lx2, sim.Lx2, -sim.Ly2, sim.Ly2, -sim.Lz2, sim.Lz2)
L.create_box(sim.NSpecies*sim.NGrid, "box") 
for iGrid in range(sim.NGrid):
  L.region("Region"+"_"+str(iGrid), "block", -sim.Lx2+iGrid*sim.dx, -sim.Lx2+(iGrid+1)*sim.dx, -sim.Ly2, sim.Ly2, -sim.Lz2, sim.Lz2)
  L.fix("Wall"+"_"+str(iGrid), "all", "wall/reflect", "xlo", -sim.Lx2+iGrid*sim.dx, "xhi", -sim.Lx2+(iGrid+1)*sim.dx, "units", "box")

RandCreate = []
for iSpecies in range(sim.NSpecies):
  RandCreate.append([RNG() for iGrid in range(sim.NGrid)])

for iSpecies in range(sim.NSpecies):
  for iGrid in range(sim.NGrid):
    L.create_atoms(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, "random", sim.SimulationBox[iGrid].SpeciesList[iSpecies].num, RandCreate[iSpecies][iGrid], "Region"+"_"+str(iGrid))
    L.mass(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].mass) 
    L.set("type", sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, "charge", sim.SimulationBox[iGrid].SpeciesList[iSpecies].charge)

L.timestep(sim.tStep) 
L.neigh_modify("delay", 0, "every", 1, "one", sim.neigh_one, "page", sim.neigh_page)
  
if sim.forcefield == "Debye":
  L.pair_style("coul/debye/vk",sim.cutoffGlobal)
  for iGrid in range(sim.NGrid):
    for iSpecies in range(sim.NSpecies):
      L.pair_coeff(sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].SpeciesList[iSpecies].TypeID, sim.SimulationBox[iGrid].kappa, sim.T)
elif sim.forcefield == "eFF":
  L.pair_style("eff/cut", sim.cutoffGlobal)
  L.pair_coeff("* *")
  L.comm_modify("vel yes")

for iSpecies in range(sim.NSpecies):
  L.group("Species"+str(iSpecies+1), "type", str(sim.SimulationBox[0].SpeciesList[iSpecies].TypeID)+":"+str(sim.SimulationBox[-1].SpeciesList[iSpecies].TypeID))

RandV = 0
RandV = RNG()
L.velocity("all create", sim.T, RandV)
  
L.run_style("verlet")
L.fix("Nose_Hoover all nvt temp", sim.T, sim.T, 100.0*sim.tStep) 
L.thermo(10)

L.minimize("1.0e-6 1.0e-8 1000 10000")
  
L.reset_timestep(0)
L.run(sim.NEqm)

L.log(sim.ProdLogName) 
L.unfix("Nose_Hoover")
L.fix("NVEfix all nve") 
L.reset_timestep(0)
L.run(sim.NProd)
