import bpy
from math import floor, copysign
import numpy as np
scn = bpy.context.scene


def set_rigidbody_world(col,steps = 60, iter = 10, start=1, end=250):
    if scn.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
        scn.rigidbody_world.enabled = True
    scn.rigidbody_world.collection = col
    scn.rigidbody_world.steps_per_second = steps
    scn.rigidbody_world.solver_iterations = iter
    scn.frame_start = start
    scn.frame_end = end
    scn.rigidbody_world.point_cache.frame_start = start
    scn.rigidbody_world.point_cache.frame_end = end
    ccache()
    return
    
def ccache():
    bpy.ops.ptcache.free_bake_all()
    return

def set_rigidbody_object(ob,type = 'ACTIVE',mass = 1, friction = 0.5, bounce = 0, shape = 'CONVEX_HULL', sens = None):
    rb = ob.rigid_body
    if rb is not None:
        rb.type = type
        rb.mass = mass
        rb.collision_shape = shape
        rb.friction = friction
        rb.restitution = bounce
        if sens is not None:
            rb.use_margin = True
            rb.collision_margin = sens
    return        
        
def extract_locations(ob):
    start = scn.frame_start
    end = scn.frame_end
    locs = []
    for n in range(start,end+1):
        bpy.context.scene.frame_set(n)
        locs.append(ob.matrix_world.to_translation())
    return locs

    
def extract_euler(ob):
    start = scn.frame_start
    end = scn.frame_end
    eul = []
    for n in range(start,end+1):
        bpy.context.scene.frame_set(n)
        locs.append(ob.matrix_world.to_euler())
    return eul

def find_collisions_wall(locs,co,side=0,rad=0):
    '''Encuentra colisiones *cambios en velocidad mayores que umbral
    para una serie de ubicaciones y las devuelve en fraciones de frame
    junto con la velocidad normal
    '''
    collisions = []
    vels = [t - s for s, t in zip(locs, locs[1:])]
    for n in range(len(vels)-1):
        if vels[n][co]*vels[n+1][co] < 0:
            dir = copysign(1,vels[n][co])
            if (side==0 or side*dir >0) :
                #interpola para hallar el tiempo de colision
                t = n+0.5+vels[n][co]/(vels[n][co]-vels[n+1][co])
                # interpola para hallar las coordenadas a ese tiempo
                pos = locs[floor(t)]+(t%1)*vels[n]
                pos[co] = pos[co] + dir*rad
                collisions.append([t,pos[0],pos[1],pos[2],vels[n][co]])
    return collisions        

# planetary motion
def getAcc( pos, mass, G, softening ):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]
    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    # pack together the acceleration components
    a = np.hstack((ax,ay,az))
    return a      

def getEnergy( pos, vel, mass, G ):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    # Potential Energy:
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]
    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    # matrix that stores 1/r for all particle pairwise particle separations 
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    return KE, PE;
