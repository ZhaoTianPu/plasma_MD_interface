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


def TFScreen(ne, T):
  """
  function for obtaining the TF screening length
  """
  EF23 = EF23prefac*ne**(2/3)
  # kappa_TF = 1/lambda_TF
  kappa_TF = e*sqrt(ne/(e0*sqrt(kB*kB*T*T + EF23*EF23)))
  return kappa_TF

def FDdist(x,a):
  """
  function for obtaining the Fermi-Dirac distribution, f = 1/(exp(x/a)+1)
  """
  return 1/(exp(x/a)+1)

def gdist(x,a,Lx2):
  """
  distribution that is designed for making the F-D distribution periodic
  """
  return FDdist(x,a) - FDdist(x+Lx2,a) + 1 - FDdist(x-Lx2,a)

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
  Lx
  Ly
  Lz
  Lx2 = Lx/2
  Ly2 = Ly/2
  Lz2 = Lz/2
  omega_p1
  tp
  NType
  NRange
  aWidth
  NGrid
  AtomInfo[iType].numDen[0]
  AtomInfo[iType].mass
  AtomInfo[iType].charge
  T

  


  # supress multithreading
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"

  # extract values from simulation class

  # file stem name for Dii and D12
  
  # 
  # initiate PyLammps
  L = PyLammps()

  L.log(EqmLogStem+"_"+str(iTraj)+".txt") 

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
  tp = 1. / omega_p1
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
  L.create_box(NType, "box") 
  
  # create regions, make solid walls
  for iGrid in range(NGrid):
    L.region("Region"+"_"+str(iGrid), "block", -Lx2+iGrid*dx, -Lx2+(iGrid+1)*dx, -Ly2, Ly2, -Lz2, Lz2)
    L.fix("Wall"+"_"+str(iGrid), "all", "wall/reflect", "xlo", -Lx2+iGrid*dx, "xhi", -Lx2+(iGrid+1)*dx, "units", "box")

  # make an array that stores the value of distribution function
  FDdistArray = [FDdist(grid2pos(ix,dx,Lx2),aWidth) for ix in range(NGrid)]

  # create NType*NGrid number of random numbers
  RandCreate = []
  # only ask the processor w/ rank 0 to generate random numbers
  if MPI.COMM_WORLD.rank == 0:
    for iGrid in range(NGrid):
      # generates NType number of random seed numbers for each grid
      RandCreate.append([RNG() for iType in range(NType)])
  # broadcast (distribute) the generated random numbers to every processor
  RandCreate = MPI.COMM_WORLD.bcast(RandCreate, root=0)

  AtomNum = [\
    [\
      int(\
        (AtomInfo[iType].numDen[0]*FDdistArray[iGrid]+AtomInfo[iType].numDen[1]*(1-FDdistArray[iGrid]))*dV \
         ) \
      for iType in NType \
    ] for iGrid in NGrid \
            ]

  # create and set atoms 
  for iGrid in range(NGrid):
    for iType in range(NType):
      L.create_atoms(iType+1, "random", AtomNum[iGrid][iType], RandCreate[iGrid][iType], "Region"+"_"+str(iGrid))
  
  # set particle mass and charge
  for iType in range(NType):
    L.mass(iType+1, AtomInfo[iType].mass) 
    L.set("type", iType+1, "charge", AtomInfo[iType].charge)   
  
  # set the timestep
  L.timestep(dt) 
  
  # check neighbor parameters
  L.neigh_modify("delay", 0, "every", 1)
  
  # interaction style
  L.pair_style("yukawa","")

  #-------------------------------------------------------------------
  # Grouping based on types
  for iType in range(NType):
    L.group("Type"+str(iType+1), "type", iType+1)
  
  # generate a random number for setting velocity
  RandV = 0
  # only let the rank 0 processor to generate
  if MPI.COMM_WORLD.rank == 0:
    RandV = RNG()
  # broadcast to all the processors
  RandV = MPI.COMM_WORLD.bcast(RandV,root=0)
  L.velocity("all create 1.0", RandV)
  
  # Integrator set to be verlet
  L.run_style("verlet")
  # Nose-Hoover thermostat, the temperature damping parameter 
  # is suggested in [7]
  L.fix("Nose_Hoover all nvt temp", 1.0, 1.0, 100.0*dt) 

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

  L.log(ProdLogStem+"_"+str(iTraj)+".txt") 

  # Labels in string texts for computes, fixes and variables that are 
  # labelled by x, y, and z
  DimLabel = ["x", "y", "z"]
  # particle current, for binary species only
  if NType == 2:
    # Define variables for mole fraction of two species 
    for iType in range(NType):
      L.variable("MolFrac"+"_"+str(iType+1), "equal", AtomInfo[iType].molfrac)
    # Define variables for particle currents 
    for iDim in range(3):
      L.variable("ParticleCurrent"+DimLabel[iDim], "equal",\
      "v_MolFrac_1*v_MolFrac_2*"+ \
      "(vcm(Type1,"+DimLabel[iDim]+")-"+ \
       "vcm(Type2,"+DimLabel[iDim]+"))") 
      
    # Set up fix ave/correlate for particle currents.
    L.fix("VACF"+"_"+"12", "all", "ave/correlate", NEvery, int(NLength/NEvery) + 1, NFreq,
    "v_ParticleCurrentx", "v_ParticleCurrenty", "v_ParticleCurrentz", 
    "type auto/lower", "ave running", "file", D12FileStem+".txt")

  # create two lists that stores the start and the end of each VACF calculations
  # the time origins are in the StartList, the ends are in the EndList
  # NReps is the number of time origins, NInt is the timesteps between consecutive 
  # origins, NLength is the timestep length of each VACF
  StartList = np.arange(0, NReps*NInt, NInt)
  EndList = np.arange(NLength, NLength+NReps*NInt, NInt)
  
  # How VACF and the integral are essentially computed:
  # Initiate compute VACF, store them by using `fix vector` and integrate by 
  # using `trap()` [3]
  # Then store the VACFs and the integral by `fix ave/time` command (see [8])
  # When the VACF calculation comes to the end, unfix/uncompute all the fixes and 
  # computes that are used for VACF calculations

  # initialize the VACF start and end counter 
  iStart = 0
  iEnd = 0
  while iEnd != NReps: 
    while iStart != NReps:
      # if the begin time of the next new VACF is earlier than the end time of the
      # oldest VACF that is running, start the new VACF
      if StartList[iStart] < EndList[iEnd]:
        for iType in range(NType):
          # define compute for VACF     
          L.compute("VACF"+"_"+str(iType+1)+"_"+str(iStart), "Type"+str(iType+1), "vacf")
          for iDim in range(1,4):
            # define fix vector for stacking VACFs
            L.fix(     "VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "all", "vector", NEvery,
            "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"["+str(iDim)+"]")
            # integrate with trapezium rule
            L.variable("Diff"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "equal",
            "${NEvery}*dt*trap("+"f_VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart)+")")
          # write out the VACFs by using fix ave/time
          L.fix("VACFout"+"_"+str(iType+1)+"_"+str(iStart), "all", "ave/time", 1, 1, NEvery, 
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[1]", "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[2]",
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[3]", 
          "v_Diffx"+"_"+str(iType+1)+"_"+str(iStart), "v_Diffy"+"_"+str(iType+1)+"_"+str(iStart), 
          "v_Diffz"+"_"+str(iType+1)+"_"+str(iStart), 
          "file", DiiFileStem+"Type"+str(iType+1)+"_"+str(iStart)+".txt", "mode scalar")
        # if this is the last VACF to be started, run the simulation until the next VACF end comes
        if iStart == (NReps - 1):
          L.run(EndList[iEnd]-StartList[iStart])
        # if not, run the simulation until either the next running VACF comes to the end or the next new
        # VACF need to be started
        else:
          L.run(min(EndList[iEnd], StartList[iStart+1])-StartList[iStart])
        # increast the count of new VACF started by 1
        iStart += 1
      
      # if the begin time of the next new VACF is later than the end time of the
      # oldest VACF that is running, terminate the oldest VACF
      elif StartList[iStart] > EndList[iEnd]:
        # unfix and uncompute
        for iType in range(NType):
          for iDim in range(1,4):
            # unfix stacking VACF
            L.unfix("VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iEnd))
          # unfix the output of VACF and the integral
          L.unfix("VACFout"+"_"+str(iType+1)+"_"+str(iEnd))
          # uncompute for VACF
          L.uncompute("VACF"+"_"+str(iType+1)+"_"+str(iEnd))
        # run the simulation until either the next running VACF comes to the end or the next new
        # VACF need to be started
        L.run(min(StartList[iStart], EndList[iEnd+1]) - EndList[iEnd])
        # increast the count of VACF terminated by 1
        iEnd += 1
      
      # if the begin time of the next new VACF and the end time of the oldest VACF that
      # is running happens at the same time, terminate the old and start the new
      else:
        for iType in range(NType):
          # fix and compute
          # define compute for VACF     
          L.compute("VACF"+"_"+str(iType+1)+"_"+str(iStart), "Type"+str(iType+1), "vacf")
          for iDim in range(1,4):
            # define fix vector for stacking new VACFs
            L.fix(     "VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "all", "vector", NEvery,
            "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"["+str(iDim)+"]")
            # integrate the new VACF with trapezium rule
            L.variable("Diff"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "equal",
            "${NEvery}*dt*trap("+"f_VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart)+")")
            # unfix stacking VACF
            L.unfix("VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iEnd))
          
          # write out the new VACF and integral
          L.fix("VACFout"+"_"+str(iType+1)+"_"+str(iStart), "all", "ave/time", 1, 1, NEvery, 
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[1]", "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[2]",
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[3]", 
          "v_Diffx"+"_"+str(iType+1)+"_"+str(iStart), "v_Diffy"+"_"+str(iType+1)+"_"+str(iStart), 
          "v_Diffz"+"_"+str(iType+1)+"_"+str(iStart), 
          "file", DiiFileStem+"Type"+str(iType+1)+"_"+str(iStart)+".txt", "mode scalar")
          
          # unfix output of VACF
          L.unfix("VACFout"+"_"+str(iType+1)+"_"+str(iEnd))
          # uncompute VACF
          L.uncompute("VACF"+"_"+str(iType+1)+"_"+str(iEnd))
        
        # if this is the last VACF to be started, run the simulation until the next VACF end comes
        if iStart == (NReps - 1):
          L.run(EndList[iEnd+1] - EndList[iEnd])
        # if not, run the simulation until either the next running VACF comes to the end or the next new
        # VACF need to be started
        else:
          L.run(min(EndList[iEnd+1], StartList[iStart+1]) - EndList[iEnd])
        # increase both counts by 1
        iStart += 1
        iEnd += 1
    
    # after all the VACFs are started, run the rest of the simulation and terminate those remaining runs
    # in order
    for iType in range(NType):
      for iDim in range(1,4):
        L.unfix("VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iEnd))
      L.unfix("VACFout"+"_"+str(iType+1)+"_"+str(iEnd))
      L.uncompute("VACF"+"_"+str(iType+1)+"_"+str(iEnd))

    if iEnd != (NReps - 1):
      L.run(EndList[iEnd+1] - EndList[iEnd])
    iEnd += 1


