import bpy
import numpy as np
import blender_methods as bm
import material_utils as mu
from scipy.integrate import solve_ivp
from scipy.fft import fft
import importlib as imp


def solve(func,t,x0,method='DOP853',args=None):
    dt = np.abs(t[1]-t[0])
    sol = solve_ivp(func, t[[0,-1]], x0, method=method, t_eval=t, args=args,max_step=dt,dense_output=True)
    if sol.status < 0:
        print(sol.message)
    return sol.y.T

def lorenz(t, x, S, P, B):
    return [
        S*(x[1]-x[0]),
        x[0]*(P-x[2])-x[1],
        x[0]*x[1]-B*x[2],
    ] 



def particleSetter(scene, depsgraph):
    particle_systems = object.evaluated_get(depsgraph).particle_systems
    particles = particle_systems[0].particles
    cFrame = scene.frame_current
    sFrame = scene.frame_start    
    #at start-frame, clear the particle cache
    if cFrame == sFrame:
        psSeed = object.particle_systems[0].seed 
        object.particle_systems[0].seed  = psSeed
    flatList = []
    for n,p in enumerate(particles):
        pt = object['data_part'][n][cFrame]
        flatList.extend([pt[0],pt[1],pt[2]])
    # additionally set the location of all particle locations to flatList
    particles.foreach_set("location", flatList)

# INTEGRACION DEL SISTEMA
cmap_path = 'C:/Users/Camilo/blender_utils/maps/ziegler.png'
out_path = 'C:/tmp/takens/'
tmax = 1
dt = 0.001
t = np.arange(0, tmax, dt)
# flujo
x0 = -5.1
y0 = -5.1
z0 = 0.1
nx = 10
ny = 10
nz = 10
dx = 1
dy = 1
dz = 1
xini_array = [[x0+n*dx, y0+m*dy, z0+l*dz] for n in range(nx) for m in range(ny) for l in range(nz)]
# Chaos
S = 10
P = 28 
B = 8/3.0
dtframe=10
print(xini_array[0])
frameini=0
pts = []
for xini in xini_array:
    print(xini)
    s = solve(lorenz, t, xini, method='RK45',args=(S,P,B),) 
    x = s[frameini::dtframe,0]
    y = s[frameini::dtframe,1]
    z = s[frameini::dtframe,2]
    pts.append([[x[n],y[n],z[n]] for n in range(len(x))])


[npart,nfr,_] = np.array(pts).shape

object = bpy.data.objects["Plane"]
particle = bpy.data.objects["Icosphere"]
if object.particle_systems.active is None:
    ps = object.modifiers.new("ps",'PARTICLE_SYSTEM')
    pset = ps.particle_system.settings
else:
    pset = object.particle_systems.active.settings    
pset.count = npart
pset.frame_start = 1
pset.frame_end = 1
pset.lifetime = nfr
pset.render_type = 'OBJECT'
object.show_instancer_for_render = False
pset.instance_object = particle

object['data_part'] = pts
print(pts[0][0])
print(len(pts))
bpy.app.handlers.frame_change_post.clear() 
bpy.app.handlers.frame_change_post.append(particleSetter)
