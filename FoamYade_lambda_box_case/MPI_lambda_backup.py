import os
from yade import mpy as mp
from yade import export 
import math
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from yade import pack



comm = mp.comm_slave



"""
print("\n=== Available communicators in mpy:  ")
for a in dir(mp):
    if "comm" in a.lower():
        print(a, "=>", getattr(mp, a))
#print("======================================\n")
 """
#print("YADE: Current working directory:", os.getcwd())



frame_number = 0

parallelYade=True 
numProcOF=2

Lambda=1

O.periodic = False # if true, define O.cell.setBox
#O.cell.setBox(0.01000005, 0.0100005, 0.0100005)   #comment out if O.periodic = False
numspheres = 10
young = 5e6
density = 1000
NSTEPS = 1000#int(2e5)

O.materials.append(FrictMat(young=young, poisson=0.5, frictionAngle=radians(15), density=density, label='spheremat'))
O.materials.append(FrictMat(young=young * 100, poisson=0.5, frictionAngle=0, density=0, label='wallmat'))



minval = 0
maxvalX = 0.01
maxvalY = 0.01
maxvalZ = 0.02

v0 = Vector3(minval, minval, minval)
v1 = Vector3(maxvalX, minval, minval)
v2 = Vector3(maxvalX, maxvalY, minval)
v3 = Vector3(minval, maxvalY, minval)

v4 = Vector3(minval, minval, maxvalZ)
v5 = Vector3(maxvalX, minval, maxvalZ)
v6 = Vector3(maxvalX, maxvalY, maxvalZ)
v7 = Vector3(minval, maxvalY, maxvalZ)

yminus = 0.25 * (v0 + v1 + v4 + v5)
yplus  = 0.25 * (v2 + v3 + v6 + v7)


mn, mx = Vector3(5e-5, 5e-5, 5e-5), Vector3(0.0095, 0.0095, 0.0195) # sphere cloud

sp = pack.SpherePack()
sp.makeCloud(mn, mx, rMean=0.00075, rRelFuzz=0, num=numspheres)
O.bodies.append([sphere(center, rad, material='spheremat') for center, rad in sp])
O.bodies.append(box(center=yminus, extents=(maxvalX/2, 0, maxvalZ/2), fixed=True))
O.bodies.append(box(center=yplus, extents=(maxvalX/2, 0, maxvalZ/2), fixed=True))


#Build spring list
#-----------------
maxSpringsPerSphere = 6
springsList = []   #  contains ID, sphere1, sphere2, and restLength

sphereIDs = [b.id for b in O.bodies if isinstance(b.shape, Sphere)]
positions = {b.id: b.state.pos for b in O.bodies if isinstance(b.shape, Sphere)}


#  KDTree for fast nearest neighbor search
from scipy.spatial import cKDTree
coords = [positions[i] for i in sphereIDs]
tree = cKDTree(coords)

# Each sphere connects to its 6 nearest neighbors (excluding itself)
springID = 0
connected_pairs = set()
for i, sid in enumerate(sphereIDs):
    dists, idxs = tree.query(coords[i], k=maxSpringsPerSphere + 1)  # +1 includes itself
    for j in range(1, len(idxs)):  # skip itself
        sid2 = sphereIDs[idxs[j]]
        pair = tuple(sorted((sid, sid2)))
        if pair in connected_pairs:
            continue
        connected_pairs.add(pair)
        length = dists[j]
        springsList.append({
            "springID": springID,
            "sphere1": sid,
            "sphere2": sid2,
            "length": length
        })
        springID += 1

# save springsList.txt
os.makedirs("springs", exist_ok=True)
with open("springs/springsList.txt", "w") as f:
    f.write("springID sphere1 sphere2 length\n")
    for s in springsList:
        f.write(f"{s['springID']} {s['sphere1']} {s['sphere2']} {s['length']:.6e}\n")

print(f" Created {len(springsList)} springs connecting {len(sphereIDs)} spheres.")

# -------------------


fluidCoupling = FoamCoupling()
fluidCoupling.couplingModeParallel = parallelYade
fluidCoupling.isGaussianInterp = True


sphereIDs = [b.id for b in O.bodies if type(b.shape) == Sphere]


fluidCoupling.SetOpenFoamSolver("lambdaFoamYade", numProcOF)

#fluidCoupling.SetOpenFoamSolver("dummyFoamYade", numProcOF)



# tell the coupling engine which IDs to couple
fluidCoupling.setIdList(sphereIDs)
fluidCoupling.setNumParticles(len(sphereIDs))



def printStep():
	print("step = ", O.iter)


def savePos():
    export.text(f"spheres/spheres_{O.iter:05d}.txt")
    
export.text("spheres/spheres_0.txt")
#export.VTKExporter("spheres/vtk-0").exportSpheres()


def updateSprings():
    # each rank  updates for the springs it can see
    local_updates = []
    for s in springsList:
        sid1, sid2 = s["sphere1"], s["sphere2"]

        b1 = O.bodies[sid1] if sid1 < len(O.bodies) else None
        b2 = O.bodies[sid2] if sid2 < len(O.bodies) else None
        if b1 is None or b2 is None:
            continue

        p1, p2 = b1.state.pos, b2.state.pos
        vec = p2 - p1
        dist = vec.norm()
        if dist == 0:
            continue

        target = s["length"] * Lambda
        delta = 0.5 * (target - dist)
        direction = vec / dist
        newP1 = p1 - direction * delta
        newP2 = p2 + direction * delta

        local_updates.append((sid1, newP1))
        local_updates.append((sid2, newP2))

    #send all updates to rank 0
    all_updates = mp.comm.gather(local_updates, root=0)

    
    if mp.rank == 0:
        merged = {}
        for sub in all_updates:
            for sid, pos in sub:
                merged[sid] = pos
        keys = list(merged.keys())
        vals = [merged[k] for k in keys]
    else:
        keys, vals = None, None

    # send final positions to every rank
    keys = mp.comm.bcast(keys, root=0)
    vals = mp.comm.bcast(vals, root=0)

    # ach rank sets what it owns locally (no mp.setBodyPosition)
    for sid, pos in zip(keys, vals):
        if sid < len(O.bodies) and O.bodies[sid] is not None:
            O.bodies[sid].state.pos = pos
            O.bodies[sid].state.vel = Vector3(0,0,0)  # neutralize integrator


def updateLambda():
    global Lambda
    if Lambda > 0.1:        # stop when it reaches 0.x
        Lambda -= 0.001
    if mp.rank == 0 and O.iter % 100 == 0:
        print(f"Lambda = {Lambda:.4f}")



# Export springs as VTK
# ----------------------------------------
def export_springs():
    global frame_number
    os.makedirs("springs", exist_ok=True)

    # each rank builds its local list of line endpoints
    local_points = []
    for s in springsList:
        i, j = s["sphere1"], s["sphere2"]

        # skip if either sphere not present on this rank
        b1 = O.bodies[i] if i < len(O.bodies) else None
        b2 = O.bodies[j] if j < len(O.bodies) else None
        if b1 is None or b2 is None:
            continue

        local_points.extend([b1.state.pos, b2.state.pos])

    # gather all local lists to the master 
    all_points = mp.comm.gather(local_points, root=0)

    
    if mp.rank == 0:
        # flatten the gathered list
        points = [p for sublist in all_points for p in sublist]

        file_path = f"springs/springs_{frame_number:04d}.vtk"
        with open(file_path, 'w') as f:
            f.write("# vtk DataFile Version 3.0\nSpring connections\nASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {len(points)} float\n")
            for pt in points:
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

            num_springs = len(points) // 2
            f.write(f"\nLINES {num_springs} {num_springs * 3}\n")
            for k in range(0, len(points), 2):
                f.write(f"2 {k} {k+1}\n")

            f.write(f"\nCELL_DATA {num_springs}\n")
            f.write("SCALARS springType int 1\nLOOKUP_TABLE default\n")
            for _ in range(num_springs):
                f.write("0\n")

        frame_number += 1
        print(f"Exported {num_springs} springs to {file_path}")


"""
if mp.rank == 0:
    print("[YADE] FoamCoupling type:", type(fluidCoupling))
    print("[YADE] Has inCommProcs?", hasattr(fluidCoupling, "inCommProcs"))
    print("[YADE] dir(fluidCoupling):", dir(fluidCoupling))
"""    
    

def gatherHydroFT():
    local = []
    ids = fluidCoupling.getIdList()

    for i in ids:
        if i < len(O.bodies) and O.bodies[i] is not None:
            
            f = O.forces.f(i)
            t = O.forces.t(i)

            # NEW: read lambdaDot from YADE state :D it works!
            lam = O.bodies[i].state.lambdaDot

            local.append((i, (f[0], f[1], f[2],
                               t[0], t[1], t[2],
                               lam)))

    all_data = mp.comm.gather(local, root=0)

    if mp.rank == 0:
        merged = {}
        for lst in all_data:
            for i, vals in lst:
                merged[i] = vals

        # Print first 50
        for k in sorted(merged)[:50]:
            Fx, Fy, Fz, Tx, Ty, Tz, lam = merged[k]
            print(f"[hydro+lambdaDot] id {k}: F=({Fx:.3e},{Fy:.3e},{Fz:.3e}) "
                  f"T=({Tx:.3e},{Ty:.3e},{Tz:.3e}) lambdaDot={lam:.7e}")


def printlambdaDotNew():      
	for b in O.bodies:
	    if isinstance(b.shape, utils.Sphere):
	    	print(f"particle ID={b.id}  lambdaDot={b.state.lambdaDot}")

	    	#if b: print(f"particle ID=" {b.id}, " lambdaDot="{b.state.lambdaDot},)
    
#---------------
#YADE ENGINES 
#---------------
O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()], label='collider', allowBiggerThanPeriod=True),
        InteractionLoop(
                [Ig2_Sphere_Sphere_ScGeom(), Ig2_Box_Sphere_ScGeom()], [Ip2_FrictMat_FrictMat_FrictPhys()], [Law2_ScGeom_FrictPhys_CundallStrack()],
                label='InteractionLoop'
        ),
        GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.7, defaultDt=5e-6, timeStepUpdateInterval=200, parallelMode=True, label="ts"),
        fluidCoupling,  #to be called after timestepper
        NewtonIntegrator(damping=0.99, label='newton', gravity=(0, 0.0, 0)),
        
        #PyRunner(command='updateLambda()', iterPeriod=1, firstIterRun=1), # it works ! 
        PyRunner(command='updateSprings()', iterPeriod=1, firstIterRun=2),  # it works!
   	
        PyRunner(command="export_springs()", iterPeriod=10), # i works!
        PyRunner(command='printlambdaDotNew()', iterPeriod=1), #it works! prints a list of sphereID and lambdaDot 
        #VTKRecorder(fileName='spheres/vtk-', recorders=['spheres'], parallelMode=True, virtPeriod=1e-3),
        VTKRecorder(fileName='spheres/vtk-', recorders=['spheres'], parallelMode=True, iterPeriod=10),
        PyRunner(iterPeriod=1, command='gatherHydroFT()'), # it works! prints a list of hydrodynamic forces + lambdaDot
        #PyRunner(command='savePos()', iterPeriod=5000) 
        
         
        ]
        
#---------------       
        
collider.verletDist = 0.00075
mp.YADE_TIMING = False
mp.FLUID_COUPLING = True
mp.VERBOSE_OUTPUT = False
mp.USE_CPP_INTERS = False
mp.ERASE_REMOTE_MASTER = True
mp.REALLOC_FREQUENCY = 12
mp.fluidBodies = sphereIDs
mp.DOMAIN_DECOMPOSITION = True
mp.mpirun(NSTEPS,np=numProcOF)
mp.mprint("RUN FINISH")      

        

exit()



