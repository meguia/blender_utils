import bpy
from math import floor, copysign
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
        