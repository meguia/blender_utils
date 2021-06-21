import numpy as np
import bpy
from mathutils import Vector

def mesh_for_board_planks(name,xs,ys,zs,hlist=[]):
        
    M=len(xs)
    N=len(zs)-1
    holes = np.zeros((N,M-1),dtype=int)
    hrow=np.ones((1,M-1),dtype=int)
    hcol=np.ones((N,1),dtype=int)
    for h in hlist:
        holes[h] = 1
        
    holesv = np.diff(np.vstack((hrow,holes,hrow)),axis=0)
    holesh=np.diff(np.hstack((hcol,holes,hcol)))
    Fs = [0,1,2*M+1,2*M]
    Bs = [0,-1,2*M-1,2*M]
    Rs = [0,M,3*M,2*M]
    Ls = [0,-M,M,2*M]
    Ds = [0,-1,M-1,M]
    Us = [0,1,M+1,M]
    ve = []
    for n in range(N+1):
        ve.extend([Vector((x,-ys[n],zs[n])) for x in xs])
        ve.extend([Vector((x,ys[n],zs[n])) for x in xs])

    fa = []  
    for p in range(N):
        for q in range(M-1):
            r = 2*M*p+q
            if holes[p,q]==0:
                fa.append(tuple([r+s for s in Fs]))
                fa.append(tuple([r+M+1+s for s in Bs]))
            if holesv[p,q]<0:
                fa.append(tuple([r+1+s for s in Ds]))
            if holesv[p,q]>0:
                fa.append(tuple([r+s for s in Us]))
        for q in range(M):
            r = 2*M*p+q
            if holesh[p,q]>0:        
                fa.append(tuple([r+s for s in Rs]))
            if holesh[p,q]<0:           
                fa.append(tuple([r+M+s for s in Ls]))

    for q in range(M-1):
        r = 2*M*N+q
        if holesv[N,q]>0:
            fa.append(tuple([r+s for s in Us]))

    me = bpy.data.meshes.new(name)
    me.from_pydata(ve, [], fa)
    return me


# test 
#xs = [-3, -2, -1.2, 0.8, 1.6, 3]
#zs = [-0.3, 0, 0.2, 0.21, 2.1, 2.5, 3]
#ys = [0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.08]
#hlist = [(1,1),(2,1),(3,1),(1,3),(2,3),(3,3)]
Nx = 30
Ny = 12
xs = np.cumsum(0.1+np.random.random(Nx+1))
ys = 0.1+0.1*np.random.random(Ny+1)
zs = np.cumsum(0.2+0.2*np.random.random(Ny+1))
nholes = 70
hx = np.random.randint(0,Nx,size=nholes)
hy = np.random.randint(0,Ny,size=nholes)
hlist = [tuple([hy[n],hx[n]]) for n in range(nholes)]
me = mesh_for_board_planks('pepe',xs,ys,zs,hlist)
ob = bpy.data.objects.new(me.name,me)
bpy.data.collections[0].objects.link(ob)