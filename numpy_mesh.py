import numpy as np
import bpy
from mathutils import Vector
from math import pi
import bmesh

def assign_uv(bm,uvs,uv_idcs,uv,scale=1.0,move=(0,0)):
    
    for f in bm.faces:
        idx = f.index
        uv_tup = uv_idcs[idx]
        li = 0
        for g in f.loops:
            luv1 = g[uv]
            uv_idx = uv_tup[li]
            luv1.uv = ((uvs[uv_idx][0]+move[0])/scale,(uvs[uv_idx][1]+move[1])/scale)
            li = li+1
    return


def mesh_for_board_hbands(name,xs,ys,zs,holes):
    """ returns a mesh for a rectangular board made of horizontal bands of variable
    thickness (ys) and height *z), and vertical bands at xs. holes is a 
    len(ys)-1 times len(xs)-1 matriz with ones indicating the holes in the board
    ys and zs must arrays of the same length specifying the profile of the board
    (symmetric wrt z yz plane) in terms of thickness and height
    Some attributes as the dimensions of xs and zs are stored as custom properties of the mesh
    The holes array are also stored as a custom property (maybe not the most effcient way)
    """    
    M=len(xs)
    N=len(zs)-1
    #holes = np.zeros((N,M-1),dtype=int)
    hrow=np.ones((1,M-1),dtype=int)
    hcol=np.ones((N,1),dtype=int)
    #for h in hlist:
    #    holes[h] = 1   
    print(holes)
    print(holes.shape)
    holesv=np.diff(np.vstack((hrow,holes,hrow)),axis=0)
    holesh=np.diff(np.hstack((hcol,holes,hcol)))
    Fs = [0,1,2*M+1,2*M]
    Bs = [0,-1,2*M-1,2*M]
    Ls = [0,-M,M,2*M]
    Rs = [0,M,3*M,2*M]
    Ds = [0,-1,M-1,M]
    Us = [0,1,M+1,M]
    #vertices
    ve = []
    for n in range(N+1):
        ve.extend([Vector((x,-ys[n],zs[n])) for x in xs])
        ve.extend([Vector((x,ys[n],zs[n])) for x in xs])
    # faces (polygons)
    fa = []  
    # face orientation 0-5: FBDURL
    ot = []
    for p in range(N):
        for q in range(M-1):
            r = 2*M*p+q
            if holes[p,q]==0:
                fa.append(tuple([r+s for s in Fs]))
                ot.append(0)
                fa.append(tuple([r+M+1+s for s in Bs]))
                ot.append(1)
            if holesv[p,q]<0:
                fa.append(tuple([r+1+s for s in Ds]))
                ot.append(2)
            if holesv[p,q]>0:
                fa.append(tuple([r+s for s in Us]))
                ot.append(3)
        for q in range(M):
            r = 2*M*p+q
            if holesh[p,q]>0:        
                fa.append(tuple([r+s for s in Rs]))
                ot.append(4)
            if holesh[p,q]<0:           
                fa.append(tuple([r+M+s for s in Ls]))
                ot.append(5)
    for q in range(M-1):
        r = 2*M*N+q
        if holesv[N,q]>0:
            fa.append(tuple([r+s for s in Us]))
            ot.append(3)
    me = bpy.data.meshes.new(name)
    me.from_pydata(ve, [], fa)
    # create custom properties for storing geometrical data
    me['length']=xs[-1]-xs[0]
    me['width']=ys[0]
    me['height']=zs[-1]-zs[0]
    me['ot']=ot
    return me


def floor(name, mats = None, pos=[0,0,0],dims=[1,1,0.1],flip=0):
    """ convenience function returning a rectangular board  with material mats,
    dimensions dims = (length, width) , located at pos and laying in the XY plane  
    defaults thickness is 0.1 but it can be specified also in dims
    """
    dx=dims[0]/2
    dy=dims[1]/2
    if len(dims) < 3:
        dims.append(0.1)
    holem = np.zeros((1,1))   
    me = mesh_for_board_hbands(name,[-dx,dx],[dy,dy],[0,dims[2]],holem)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler[0] = flip*pi
    if mats is not None:
        ob.data.materials.append(mats)
    print(ob.name)
    return ob

def wall(name, pos=[0,0,0],rot=0, dims=[1,1,0.1],holes=[], bandmats=[],bands=[]):
    """ convenience function returning a vertical rectangular board with material mats,
    dimensions dims, located at pos and with z rotation rot
    It also can add bands with different thickness giveln by array of bands, each
    element of bands is a 2 element array with zprofile (array of heights) and yprofile
    (array of thickness). The bands must be non - overlapping. This is not checked!
    Holes is an array of holes. each element of holes is a 2 element array with 
    the min/max xcoordinates and min/max z coordinates. Also must be non-overlapping
    bandmats must be an array of materials of length 1+len(bands) for assigning to the
    wall + bands 
    """

    (length,height,thick)=dims 
    # oringe en length/2 (revisar)
    l0 = length/2
    z0 = [0,height]
    y0 = [thick,thick]
    x0 = [0,length]
    for band in bands:
        z0.extend(band[0])
        y0.extend([thick + b for b in band[1]])    
    for hole in holes:
        z0.extend(hole[1])
        y0.extend([thick,thick]) # holes have the thickness of the wall
        x0.extend(hole[0])
    zs, znv = np.unique(z0,return_inverse=True)
    y0 = np.array(y0)
    ys = [np.amax(y0[znv==n]) for n in range(len(zs))]    
    xs, xnv = np.unique(x0,return_inverse=True)        
    nby = len(ys)-1
    nbx = len(xs)-1
    holem = np.zeros((nby,nbx))
    for hole in holes:
        xi = np.nonzero(np.logical_and(xs[1:]<=hole[0][1],xs[:-1]>=hole[0][0]))
        zi = np.nonzero(np.logical_and(zs[1:]<=hole[1][1],zs[:-1]>=hole[1][0]))
        holem[zi,xi]=1
    me = mesh_for_board_hbands(name,(xs-l0).tolist(),ys,zs.tolist(),holem)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler[2] = rot
    # apply materials
    paint_list=[[0,height,0]]
    for n,mat in enumerate(bandmats):
        ob.data.materials.append(mat)
        if n>0:
            paint_list.append([bands[n-1][0][0],bands[n-1][0][-1],n])
    paint_regions(ob,2,paint_list)    
    print(ob.name)
    return ob

def paint_regions(ob,coord,paint_list):
    """ Paint object ob in N regions along coordinate coord (0 for x, 1 for y, 2 for z)
    with materials in ob.data.materials, using the recipe stored in paint list.
    paint list is a Nx3 array each row contain the min/max values of coord 
    (interval) and the material_index
    material_index must be integer between 0 and M-1, where M is len(list(ob.data.materials))
    """
    nmat = len(list(ob.data.materials)) 
    for f in ob.data.polygons:
        for p in paint_list:
            if (f.center[coord]>p[0] and f.center[coord]<p[1]):
                if (p[2]<nmat):
                    f.material_index = p[2]
                else:
                    break

# test 
#xs = [-3, -2, -1.2, 0.8, 1.6, 3]
#zs = [-0.3, 0, 0.2, 0.21, 2.1, 2.5, 3]
#ys = [0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.08]
##hlist = [(1,1),(2,1),(3,1),(1,3),(2,3),(3,3)]
#Nx = 30
#Ny = 12
#xs = np.cumsum(0.1+np.random.random(Nx+1))
#ys = 0.1+0.1*np.random.random(Ny+1)
#zs = np.cumsum(0.2+0.2*np.random.random(Ny+1))
#nholes = 70
#hx = np.random.randint(0,Nx,size=nholes)
#hy = np.random.randint(0,Ny,size=nholes)


#mat1 = bpy.data.materials['mat1']
#mat2 = bpy.data.materials['mat2']
#mat3 = bpy.data.materials['mat3']

#holes = [[[1.2,2],[0,2.1]],[[3.5,4.3],[0,2.1]],[[2.5,3],[0.7,2.1]]]
#dims = [6,0.1,3]
#bands = [[[0,0.2,0.21],[0.1,0.1,0]],[[2.4,2.41,3],[0,0.1,0.1]]]
#bandmats = [mat1,mat2,mat3]
#name = 'miparedzotapintada2'

#ob = wall(name, pos=[-6,8,0],rot=pi/2, dims=dims,holes=holes,bandmats=bandmats,bands=bands)


#ob = floor('sopiso', pos=[-3,3,0],dims=[3,2,0.15])


##hlist = [tuple([hy[n],hx[n]]) for n in range(nholes)]
###me = mesh_for_board_planks('pepe',xs,ys,zs,hlist)
###ob = bpy.data.objects.new(me.name,me)
##ob = floor('sopiso', pos=[-3,3,0],dims=[3,2,0.15])
#bpy.data.collections[0].objects.link(ob)