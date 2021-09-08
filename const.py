from math import pi 

# all from NIST CODATA
# hbar (in J s)
hbar = 1.054571817E-34
# elementary charge (in C)
e    = 1.602176634E-19
# electron mass (in kg)
me   = 9.1093837015E-31
# electon volt to kelvin
eV   = 1.160451812E4
# Boltzmann const (in J K-1)
kB   = 1.380649E-23 
# ideal gas const (in kCal K-1 mol-1)
Rkcal= 1.98720425864083E-3 
# J K-1 mol-1
RJ = 8.314462618
# proton mass (in kg)
mp   = 1.67262192369E-27
# vacuum permittivity (in F m-1)
e0   = 8.8541878128E-12 
# derived variables
# (2/3)*hbar^2*(3*pi^2)^(2/3)/(2*me)
EF23prefac = hbar*hbar*(3*pi*pi)**(2/3)/(3*me)
# e^2
e2   = e*e
# e^2/epsilon_0 in real unit
qqr2e = 332.06371