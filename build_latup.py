import argparse
import numpy as np
import numpy.random
import math
import logging

logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("-nx", type=int, default=2)
parser.add_argument("-ny", type=int, default=4)
parser.add_argument("-nz", type=int, default=2)
parser.add_argument("-scale", type=float, default=3.5)
parser.add_argument("-fmt", type=str, choices=["xyz", "lmp"], default='xyz')
parser.add_argument("-nano", action='store_true', help="nanoparticle instead of bulk")
parser.add_argument("-nanoz", action='store_true', help="nanoparticle but periodic in z")
parser.add_argument("-noiz", type=float, default=0.0, help="add some noiz to lattice positions")
parser.add_argument("-ndefect", type=int, default=0, help="at most, perform ndefect attempts at defect insertions")
#parser.add_argument("")
args = parser.parse_args()

logging.debug("n defects = {}".format(args.ndefect))


if args.nano and args.nanoz:
    raise ValueError("Cannot pass nano and nanoz options at the same time!")
# if args.nano:
#     raise NotImplementedError("Nano not yet a valid option")


deg = np.pi / 180

SE_CELL=[]
CD_CELL=[]

# Make a square box

args.nx = max( args.nx, int(args.ny / 2 / np.cos(30 * deg)) + 1 )
args.ny = int( args.nx * 2 * np.cos(30 * deg) ) + 1

# Set up default directions - scale things by tetx and tetz
zero = np.array([0,0,0])
tetx = np.cos(19.5 * deg)
tetz = np.sin(19.5 * deg)
axy1 = np.array([-tetx, 0, 0])
axy2 = np.array([-np.cos( 120*deg) * tetx, -np.sin( 120*deg) * tetx, 0])
axy3 = np.array([-np.cos(-120*deg) * tetx, -np.sin(-120*deg) * tetx, 0])
az   = np.array([0, 0, -tetz])


s0   = np.array([0, 0, 0])
c1   = s0 + axy1 + az
c2   = s0 + axy2 + az
c3   = s0 + axy3 + az

c0   = np.array([0, 0, 1])
s1   = c0 + axy1 - az
s2   = c0 + axy2 - az
s3   = c0 + axy3 - az

# L = np.sqrt( (tetx * np.cos(30*deg))**2 +  (tetx * (1+np.sin(30*deg)))**2 )
L = 2 * tetx * np.cos(30*deg)
A = np.array([0, L, 0])
B = np.array([L * np.sin(60*deg), L * np.cos(60*deg), 0])
C = A - B
Z = np.array([0, 0, 2 + 2 * tetz])

CD_CELL.append(c0)
CD_CELL.append(c1)
# CD_CELL.append(c2)
# CD_CELL.append(c3)
SE_CELL.append(s0)
SE_CELL.append(s1)
# SE_CELL.append(s2)
# SE_CELL.append(s3)

CD_LATT = []
SE_LATT = []

Nx = args.nx
Ny = args.ny
Nz = args.nz

# CD LATTICE
for REPx in xrange(Nx):
    for v in CD_CELL:
        CD_LATT.append(v     + REPx * (B-C))
    for v in CD_CELL:
        CD_LATT.append(v + B + REPx * (B-C))
CD_LATT = np.array(CD_LATT)

N0 = CD_LATT.shape[0]
CD_LATT_copy = np.copy(CD_LATT)
CD_LATT = np.zeros([N0 * Ny, 3])
for REPy in xrange(Ny):
    CD_LATT[N0 * REPy : N0 * (REPy+1), :] = CD_LATT_copy[:,:] + A * REPy

N0 = CD_LATT.shape[0]
CD_LATT_copy = np.copy(CD_LATT)
CD_LATT = np.zeros([N0 * Nz, 3])
for REPz in xrange(Nz):
    CD_LATT[N0 * REPz : N0 * (REPz+1), :] = CD_LATT_copy[:,:] + Z * REPz

# SE LATTICE
for REPx in xrange(Nx):
    for v in SE_CELL:
        SE_LATT.append(v     + REPx * (B-C))
    for v in SE_CELL:
        SE_LATT.append(v + B + REPx * (B-C))
SE_LATT = np.array(SE_LATT)

N0 = SE_LATT.shape[0]
SE_LATT_copy = np.copy(SE_LATT)
SE_LATT = np.zeros([N0 * Ny, 3])
for REPy in xrange(Ny):
    SE_LATT[N0 * REPy : N0 * (REPy+1), :] = SE_LATT_copy[:,:] + A * REPy

N0 = SE_LATT.shape[0]
SE_LATT_copy = np.copy(SE_LATT)
SE_LATT = np.zeros([N0 * Nz, 3])
for REPz in xrange(Nz):
    SE_LATT[N0 * REPz : N0 * (REPz+1), :] = SE_LATT_copy[:,:] + Z * REPz

scale = args.scale
CD_LATT = CD_LATT * scale
SE_LATT = SE_LATT * scale

# print CD_LATT.shape
# print SE_LATT.shape
x_min = np.min(np.vstack([CD_LATT[:,0], SE_LATT[:,0]]))
y_min = np.min(np.vstack([CD_LATT[:,1], SE_LATT[:,1]]))
z_min = np.min(np.vstack([CD_LATT[:,2], SE_LATT[:,2]]))
CD_LATT = CD_LATT[:,:] + np.array([-x_min, -y_min, -z_min])[np.newaxis, :]
SE_LATT = SE_LATT[:,:] + np.array([-x_min, -y_min, -z_min])[np.newaxis, :]

CD_LATT[CD_LATT < 1E-6 ] = 0.0
SE_LATT[SE_LATT < 1E-6 ] = 0.0

if args.nanoz or args.nano:
    Nx_box = Nx * np.cos(30 * deg)
    CD_x = CD_LATT[:,0] - L * Nx_box
    CD_y = CD_LATT[:,1] - L * Ny / 2.0
    mask =                       CD_y <= (  L * (Nx_box - 0.5) - CD_x ) * np.tan(60 * deg)
    mask = np.logical_and( mask, CD_y >= ( -L * (Nx_box - 0.5) + CD_x ) * np.tan(60 * deg) )
    mask = np.logical_and( mask, CD_y <= (  L * (Nx_box - 0.5) + CD_x ) * np.tan(60 * deg) )
    mask = np.logical_and( mask, CD_y >= ( -L * (Nx_box - 0.5) - CD_x ) * np.tan(60 * deg) )
    mask = np.logical_and( mask, CD_y <=    L / 2.0 * (Ny - 1.5) )
    mask = np.logical_and( mask, CD_y >=   -L / 2.0 * (Ny - 1.5) )
    CD_LATT = CD_LATT[mask, :]

    SE_x = SE_LATT[:,0] - L * Nx_box
    SE_y = SE_LATT[:,1] - L * Ny / 2.0
    mask =                       SE_y <= (  L * (Nx_box - 0.5) - SE_x ) * np.tan(60 * deg)
    mask = np.logical_and( mask, SE_y >= ( -L * (Nx_box - 0.5) + SE_x ) * np.tan(60 * deg) )
    mask = np.logical_and( mask, SE_y <= (  L * (Nx_box - 0.5) + SE_x ) * np.tan(60 * deg) )
    mask = np.logical_and( mask, SE_y >= ( -L * (Nx_box - 0.5) - SE_x ) * np.tan(60 * deg) )
    mask = np.logical_and( mask, SE_y <=    L / 2.0 * (Ny - 1.5) )
    mask = np.logical_and( mask, SE_y >=   -L / 2.0 * (Ny - 1.5) )
    SE_LATT = SE_LATT[mask, :] 

CD_shift = numpy.random.normal(scale=args.noiz, size=CD_LATT.shape)
CD_LATT  = CD_LATT + CD_shift
SE_shift = numpy.random.normal(scale=args.noiz, size=SE_LATT.shape)
SE_LATT  = SE_LATT + SE_shift

## Choose the defect positions with a seed so that defects are chosen
## repeatably for xyz and lammps options
numpy.random.seed(90210)
n_defect = numpy.random.randint(CD_LATT.shape[0], size=args.ndefect)

if args.fmt == "xyz":
    print "{}".format(CD_LATT.shape[0] + SE_LATT.shape[0])
    print "Comment Line"
    
    for i in xrange(SE_LATT.shape[0]):
        if i in n_defect:
                logging.debug("Inserting xyz defect...")
                print "3   {}   {}   {}".format(CD_LATT[i,0], CD_LATT[i,1], CD_LATT[i,2])
        else:
                print "1   {}   {}   {}".format(CD_LATT[i,0], CD_LATT[i,1], CD_LATT[i,2])
    for j in xrange(CD_LATT.shape[0]):
        print "2   {}   {}   {}".format(SE_LATT[j,0], SE_LATT[j,1], SE_LATT[j,2])

if args.fmt == "lmp":
    Xl = (B-C)[0]
    Yl = A[1]
    Zl = Z[2]
    print "LAMMPS coords for wurtzite LJ solid with scale = {}".format(scale)
    print ""
    print "{}    atoms".format(CD_LATT.shape[0] + SE_LATT.shape[0])
    print "3     atom types".format(CD_LATT.shape[0] + SE_LATT.shape[0])
    print ""
    if args.nanoz:
        # limit the box in the x- and y-directions, but keep infinite in z   
        Xb = Xl * Nx * scale
        Yb = Yl * Ny * scale
        print "{} {}  xlo xhi".format(-.5 * Xb, 1.5 * Xb)
        print "{} {}  ylo yhi".format(-.5 * Yb, 1.5 * Yb)
        print "0 {}  zlo zhi".format(Zl * Nz * scale)
    if args.nano:
        Xb = Xl * Nx * scale
        Yb = Yl * Ny * scale
        Zb = Zl * Nz * scale
        print "{} {}  xlo xhi".format(-.5 * Xb, 1.5 * Xb)
        print "{} {}  ylo yhi".format(-.5 * Yb, 1.5 * Yb)
        print "{} {}  zlo zhi".format(-.5 * Zb, 1.5 * Zb)
    else:
        print "0 {}  xlo xhi".format(Xl * Nx * scale)
        print "0 {}  ylo yhi".format(Yl * Ny * scale)
        print "0 {}  zlo zhi".format(Zl * Nz * scale)
    print ""
    print "Masses"
    print ""
    # print "1 1.0"
    print "1 112.4"
    # print "2 1.0"
    print "2 78.96"
    print "3 65.38"
    print ""
    print "Atoms"
    print ""
   
    for i in xrange(SE_LATT.shape[0]):
        if i in n_defect:
                logging.debug("Inserting lammps defect...")
                print "{}   3   1.18  {}   {}   {}".format(i + 1, CD_LATT[i,0], CD_LATT[i,1], CD_LATT[i,2])
        else:
                print "{}   1   1.18  {}   {}   {}".format(i + 1, CD_LATT[i,0], CD_LATT[i,1], CD_LATT[i,2])


    for j in xrange(CD_LATT.shape[0]):
        print "{}   2  -1.18  {}   {}   {}".format(j + i + 1, SE_LATT[j,0], SE_LATT[j,1], SE_LATT[j,2])
        

