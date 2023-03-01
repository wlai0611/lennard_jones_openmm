from sys import stdout
#import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import pandas
from openmm import *
from openmm.app import *
from openmm.unit import *

# Physical parameters
temperature = 293.15 * kelvin
pressure = 1 * bar
mass = 39.948 * amu
sigma = 3.419 * angstrom
epsilon = 117.8 * kelvin * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
print(epsilon)

# Simulation parameters
box_size = 150 * angstrom  # initial value only
natom = 199
cutoff = 3 * sigma

# Define the OpenMM system with the argon atoms.
system = System()
box_matrix = box_size * np.identity(3)
system.setDefaultPeriodicBoxVectors(*box_matrix)
for iatom in range(natom):
    system.addParticle(mass)

# Define a relatively boring topology object.
topology = Topology()
topology.setPeriodicBoxVectors(box_matrix)
chain = topology.addChain()
residue = topology.addResidue("argon", chain)
element_Ar = Element.getByAtomicNumber(18)
for iatom in range(natom):
    topology.addAtom("Ar", element_Ar, residue)

# Define the force field as a "force" object to be added to the system.
force = openmm.NonbondedForce()
force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
for iatom in range(natom):
    force.addParticle(0.0, sigma, epsilon)
force.setCutoffDistance(cutoff)
force.setUseSwitchingFunction(True)
force.setSwitchingDistance(0.8 * cutoff)
force.setUseDispersionCorrection(True)
force_index = system.addForce(force)

# Define the ensemble to be simulated in.
integrator = LangevinIntegrator(temperature, 1 / picosecond, 2 * femtoseconds)
system.addForce(MonteCarloBarostat(pressure, temperature))

platform = Platform.getPlatformByName('CUDA')

# Define a simulation object.
simulation = Simulation(topology, system, integrator, platform)

h5_reporter = mdtraj.reporters.HDF5Reporter(file='traj.h5',reportInterval=1000,velocities=True)

# Initialization steps before MD.
# - Asign random positions
simulation.context.setPositions(np.random.uniform(0, box_size / angstrom, (natom, 3)) * angstrom)
# - Minimize the energy
simulation.minimizeEnergy()
# - Initialize velocities with random values at 300K.
simulation.context.setVelocitiesToTemperature(300)

# Remove existing reporters, in case this cell is executed more than once.
simulation.reporters = []

# Write the initial geometry as a PDB file.
positions = simulation.context.getState(getPositions=True).getPositions()
with open("ljinit.pdb", "w") as f:
    PDBFile.writeFile(simulation.topology, positions, f)

# Write a frame to the DCD trajectory every 100 steps.
simulation.reporters.append(DCDReporter("ljtraj.dcd", 100))
simulation.reporters.append(h5_reporter)
# Write scalar properties to a CSV file every 10 steps.
simulation.reporters.append(
    StateDataReporter(
        "ljscalars.csv",
        10,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
    )
)

# Write scalar properties to screen every 1000 steps.
simulation.reporters.append(
    StateDataReporter(stdout, 1000, step=True, temperature=True, volume=True)
)

# Actually run the molecular dynamics simulation.
simulation.step(30000)
