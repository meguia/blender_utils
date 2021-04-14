import bpy
from mathutils import Vector, Euler, Color, Matrix
from math import pi, sin, cos, copysign, ceil, sqrt, radians

scn = bpy.context.scene
scnc = scn.collection
scno = scnc.objects
scnr = scn.render
scnw = scn.world
scne = scn.eevee

# ================================================================================
# GEOMETRIC METHODS

def rotate_list(l, n):
    """ Performs a list rotation n places
    """
    return l[n:] + l[:n]

def multwise(v1,v2):
    """ Performs elementwise multiplication between two vectors
    """
    return Vector(x * y for x, y in zip(v1, v2))

def vertex(x, y, z, scale = 1): 
    """ Return vertex coordinates fixed to the unit sphere 
    """ 
    length = sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x,y,z)]

def set_posangle(oblist,pos,angle):
    """ Set rotation euler to angle for object list oblist
    """
    for ob in oblist:
        ob.location = pos
        ob.rotation_euler = angle

def set_posrotar(ob,pos=None,rot=None,tar=None):
    """ Set location (pos) rotation euler (rot) or target (tar) for object
    """
    if (pos and rot) is not None:
        ob.location = pos
        ob.rotation_euler = rot
    if rot is None and (pos and tar) is not None:
        ob.location = pos
        direct = tar - pos 
        rot_quat = direct.to_track_quat('-Z','Y')
        ob.rotation_euler = rot_quat.to_euler()
    
def shift_origin(ob,shift_vector):
    """ Shift object ob origin vy coordinates shift_vector = [x,y,z]
    """
    shiftv = Vector(x / y for x, y in zip(shift_vector, ob.scale))
    for ve in ob.data.vertices:
        ve.co += shiftv    
        
def clamp(x, min, max):
    """ Clamp values of x between min and max
    """
    if x < min:
        return min
    elif x > max:
        return max
    return x

def linspace(x1,x2,n,fromzero=True):
    """ Return n equally spaced values between x1 and x2
    """
    if n == 1:
        yield x2
        return
    h = (x2 - x1) / (n - 1)
    if fromzero:
        for i in range(n):
            yield x1 + h * i
    else:
        for i in range(1,n):
            yield x1 + h * i
    return

def middle_point(ve, p1, p2, s = 1):
    """ Find a middle point and project to the unit sphere 
    """ 
    smaller_index = min(p1, p2)
    greater_index = max(p1, p2)
    #key = '{0}-{1}'.format(smaller_index, greater_index)
    #if key in middle_point_cache:
    #    return middle_point_cache[key]
    v1 = ve[p1]
    v2 = ve[p2]
    middle = [sum(i)/2 for i in zip(v1, v2)]
    ve.append(vertex(*middle,s))
    index = len(ve) - 1
    return index    

def test_dim(x, dim=0):
   """tests if x is a list and how many dimensions it has
   returns -1 if it is no list at all, 0 if list is empty 
   and otherwise the dimensions of it"""
   if isinstance(x, list):
      if x == []:
          return dim
      dim = dim + 1
      dim = test_dim(x[0], dim)
      return dim
   else:
      if dim == 0:
          return -1
      else:
          return dim
    
def frange(start, stop=None, step=None):
    """ implementation of range for float numbers
    """
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0
    while True:
        if (step > 0) and (start >= stop):
            break
        elif (step < 0) and (start <= stop):
            break
        yield  start
        start = start + step    

def in_box(pos, boxmin, boxmax):
    """ returns True only if Vector pos is within a box with extreme coordinates
    given by boxmin and boxmax
    """
    s1 = pos - boxmin
    s2 = pos - boxmax
    if (s1[0]*s2[0]<0) & (s1[1]*s2[1]<0) & (s1[2]*s2[2]<0):
        return True
    else: 
        return False

def get_center(ob):
    """ returns the center of the object using bounding boxes
    useful for setting origin to center of object
    """
    loc_bb = 0.125 * sum((Vector(b) for b in ob.bound_box), Vector())
    glob_bb = ob.matrix_world  @ loc_bb
    return glob_bb

# ================================================================================
# LINK AND COLLECTION METHODS 

def iscol(colname):
    """ Returns a collection named colname 
    """
    col = bpy.data.collections.get(colname) 
    if col is None:
        col = bpy.data.collections.new(colname)
    return col    
        
def link_col(col):
    """ Links collection to scene
    """
    if scnc.children.get(col.name) is None:
        scnc.children.link(col)
    return    
    
def list_link(oblist,col):
    """ Link list of objects oblist to colection col
    """
    colo = col.objects
    for ob in oblist:
        if colo.get(ob.name) is None:
            colo.link(ob)
        for ch in ob.children:    
            if colo.get(ch.name) is None:
                    colo.link(ch)  
                    
def list_unlink(oblist,col):
    """ Link list of objects oblist to colection col
    """
    colo = col.objects
    for ob in oblist:
        if colo.get(ob.name) is None:
            colo.unlink(ob)
        for ch in ob.children:    
            if colo.get(ch.name) is None:
                    colo.unlink(ch)                      
        
def link_all(ob,col):
    """ Links recursively childrens up to 2nd generation
    from object ob to collection col
    """
    colo = col.objects
    if colo.get(ob.name) is None:
        colo.link(ob)
    for pri in ob.children:
        if colo.get(pri.name) is None:
            colo.link(pri)
            for sec in pri.children:
                if colo.get(sec.name) is None:
                    colo.link(sec)               

def list_parent(name,oblist):
    """ Parents objects in oblist to parent name
    """
    par = bpy.data.objects.new(name,None)
    for ob in oblist:
        ob.parent = par
    return par    

# ================================================================================
# CONSTRAINT METHODS
def copy_loc(ob,target, influence=1.0):
    copyloc = ob.constraints.new('COPY_LOCATION')
    copyloc.target = target
    copyloc.influence = influence

# ================================================================================
# OPERATOR METHODS 

def apply_mod(ob,mod):
    """ Apply modifier mod to object ob
    """
    scno.link(ob)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = ob
    ob.select_set(state = True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.select_set(state = False)
    scno.unlink(ob)

def join(oblist):
    """ Joins object list to active selection
    """
    bpy.ops.object.select_all(action='DESELECT')
    for ob in oblist:
        scno.link(ob)
        ob.select_set(state = True)
    bpy.context.view_layer.objects.active = oblist[0]
    bpy.ops.object.join()
    oblist[0].select_set(state = False)
    scno.unlink(oblist[0])
    return oblist[0]

def simple_subdivide(ob,ncuts):
    """ Subdivide using operator only for linked objects
    """
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(state = True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=ncuts)
    bpy.ops.object.mode_set(mode="OBJECT")
    
    
def curve_to_beam(ob,thick,width):        
    scno.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(state = True)
    bpy.ops.object.convert(target='MESH')
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    trans1={"value":Vector((0,width,0)), "constraint_axis":(False, True, False)}
    trans2={"value":Vector((0,0,thick)), "constraint_axis":(False, False, True)}
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"mirror":False},TRANSFORM_OT_translate=trans1)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"mirror":False},TRANSFORM_OT_translate=trans2)
    bpy.ops.object.mode_set(mode = 'OBJECT')
    ob.select_set(state = False)
    scno.unlink(ob)
    
# =====================================================================
# CURVE METHODS

def pts_from_curve(cc):
    """ Returns point array from curve cc
    """
    pts = cc.data.splines[0].points
    return [pt.co.to_3d() for pt in pts]

def curve_from_pts(cname,ve,order=5,w=1):
    """ Returns NURBS curve of order from point list ve
    """
    cc = bpy.data.curves.new(name = cname, type='CURVE') 
    cc.dimensions = '3D'
    pl = cc.splines.new('NURBS')
    pl.points.add(len(ve)-1)
    for n in range(len(ve)):
        pl.points[n].co = list(ve[n]) + [w]
    pl.order_u = order
    pl.use_endpoint_u = True
    return cc

def cpath(cname,plist,step):
    """ Return curve from point list plist
    """
    vecs = [] 
    for n in range(len(step)):
        vecs += list(linspace(plist[n],plist[n+1],step[n],n==0))
    cc = curve_from_pts(cname,vecs,5,1)    
    ob = bpy.data.objects.new(cc.name,cc)          
    return ob

def spiral(cname,p1,p2,targc,nturn):
    """ Creates curve with name cname anf number of turns nturns
    between points p1 and p2 following curve targc
    """
    npts = len(targc)
    d1 = p1-targc[0]
    d2 = p2-targc[-1]
    a1 = d1.to_2d().angle_signed(Vector((1,0)))
    a2 = d2.to_2d().angle_signed(Vector((1,0)))
    a2 += nturn*2*pi
    angs = linspace(a1,a2,npts)
    rads = list(linspace(d1.to_2d().length,d2.to_2d().length,npts))
    zs = list(linspace(d1.z,d2.z,npts))
    vecs = []
    for n,ang in enumerate(angs):
        vecs.append(targc[n]+Vector((rads[n]*cos(ang),rads[n]*sin(ang),zs[n])))
    cc = curve_from_pts(cname,vecs,5,1)
    ob = bpy.data.objects.new(cc.name,cc)          
    return ob

def bezier_from_pts(cname,ve,ve_h):
    """ Returns BEZIER curve of order from point list ve
    handles are ALIGNED and right positioned at ve_h
    """
    cc = bpy.data.curves.new(name = cname, type='CURVE') 
    cc.dimensions = '3D'
    pl = cc.splines.new('BEZIER')
    pl.bezier_points.add(len(ve)-1)
    for n,pt in enumerate(pl.bezier_points):
        pt.co = list(ve[n])
        pt.handle_left_type = 'ALIGNED'
        pt.handle_right_type = 'ALIGNED'
        pt.handle_right = list(ve_h[n])
    return cc

    
# =====================================================================
# MESH METHODS

def mesh_for_circle(name,N,axis,r,offset):
    """ Returns mesh for a N-segment circle of radius r with normal pointing 
    along axis and with origin offset 
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = []
        for n in range(N):
            theta = 2*pi*n/N
            vv = rotate_list([0, r*cos(theta), r*sin(theta)],-axis)
            print(vv)
            ve.append(list(Vector(offset) + Vector(vv)))
        fa = [tuple(range(N))]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_plane(name,orient=1):
    """ Returns mesh for a default plane with name and orientation
    """
    mesh = bpy.data.meshes.get(name)
    ve = [ Vector((-1, -1, 0)), Vector((1, -1, 0)), Vector((-1, 1, 0)), Vector((1, 1, 0))]
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        if orient == 1:
            fa = [(0,2,3,1)]
        else:    
            fa = [(0,1,3,2)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_planeve(name,ve,orient=1):
    """ Returns mesh for plane definde by vertices ve with name and orientation
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        if orient == 1:
            fa = [(0,1,3,2)]
        else:    
            fa = [(0,2,3,1)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_cylinder(nfaces,name,r=1.0, h=1.0, zoffset=0):
    """ Returns mesh for a cylinder with name, n lateral faces,radius r, 
    hight h and origin offset zoffset
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        theta = [2*pi*n/nfaces for n in range(nfaces)]
        ve = [Vector((r*cos(t), r*sin(t), -zoffset)) for t in theta]
        ve.extend([v + Vector((0,0,h)) for v in ve])
        fa = []  
        for nh in range(2):
            for na in range(nfaces-1):
                fa.append([nh*nfaces+na, nh*nfaces+na+1, nh*nfaces+nfaces+na+1, nh*nfaces+nfaces+na])
            fa.append([nh*nfaces+nfaces-1, nh*nfaces, nh*nfaces+nfaces, nh*nfaces+nfaces+nfaces-1])  
        fa.append([nfaces-1-na for na in range(nfaces)])
        fa.append([(nheights-1)*nfaces+na for na in range(nfaces)])
        mesh.from_pydata(ve, [], fa)
    return mesh     


def mesh_for_tube(name, nfaces, rlist=1.0, hlist=1.0, top=True, bot=True, axis=2):
    """ Returns mesh for a tube mesh with name, n lateral faces, heights h 
    and vertices at heights hlist. For example a simple cylinder of height h
    mesh_for_cylinder(n,name,r, hlist=[0,h])
    the axis of the cylinder point in direction axis (0,1,2) = x,y,z
    """
    print(rlist)
    print(hlist)
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = []
        # Define vertices
        for nh,h in enumerate(hlist):
            for na in range(nfaces):
                theta = 2*pi*na/nfaces
                vv = [h, rlist[nh]*cos(theta), rlist[nh]*sin(theta)]            
                ve.append(rotate_list(vv,-axis))
        # Define faces
        fa = []  
        nheights = len(hlist)
        for nh in range(nheights-1):
            for na in range(nfaces-1):
                fa.append([nh*nfaces+na, nh*nfaces+na+1, nh*nfaces+nfaces+na+1, nh*nfaces+nfaces+na])
            fa.append([nh*nfaces+nfaces-1, nh*nfaces, nh*nfaces+nfaces, nh*nfaces+nfaces+nfaces-1])  
        if bot:
            fa.append([nfaces-1-na for na in range(nfaces)])
        if top:    
            fa.append([(nheights-1)*nfaces+na for na in range(nfaces)])
        mesh.from_pydata(ve, [], fa)
    return mesh        


def mesh_for_cube(name,zorigin=0):
    """ Returns mesh for a cube of side 1 (not 2!) with name, and 
    relative position of the origin zorigin along the vertical axis
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = [ Vector((-0.5, -0.5, -zorigin)), Vector((0.5, -0.5, -zorigin)),
           Vector((-0.5, 0.5, -zorigin)), Vector((0.5, 0.5, -zorigin))]
        ve.extend([v + Vector((0,0,1)) for v in ve])
        fa=[(0,2,3,1),(0,1,5,4),
            (1,3,7,5),(3,2,6,7),
            (2,0,4,6),(4,5,7,6)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_recboard(name,xs,ys,zs):
    """ Returns mesh for a rectangular board vith vertices
    located in xs[0] xs[1], yx[0] ys[1], and zs[0] zs[1]
    """     
    ve = [Vector((xs[0],ys[0],zs[0])),Vector((xs[1],ys[0],zs[0])),
      Vector((xs[0],ys[1],zs[0])),Vector((xs[1],ys[1],zs[0]))]
    if (zs[1]!=zs[0]):  
        me = mesh_for_board(name,ve,zs[1])
    else:
        me = mesh_for_planeve(name,ve,1)    
    return me  

def mesh_for_board(name,vertices,thick):
    """ Return mesh for board given four vertices and thickness
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = vertices
        ve.extend([v + Vector((0,0,thick)) for v in ve])
        #Inverted!
        fa = [(0,2,3,1),(1,3,7,5),(2,0,4,6),(4,5,7,6),
            (0,1,5,4),(3,2,6,7)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_recboard_with_hole(name,xs,ys,zs,xy,wh,internal=True):
    """ Returns mesh for a rectangular board vith vertices
    located in xs[0] xs[1], yx[0] ys[1], and zs[0] zs[1]
    and a hole located in xy[0], xy[1] of size wh[0] by wh[1]
    """
    ve = [Vector((xs[0],ys[0],zs[0])),Vector((xs[1],ys[0],zs[0])),
      Vector((xs[0],ys[1],zs[0])),Vector((xs[1],ys[1],zs[0]))]
    ve.extend([v + Vector((0,0,zs[1])) for v in ve])
    ve2 = [Vector((xs[0]+xy[0],ys[0],zs[0]+xy[1])),Vector((xs[0]+xy[0]+wh[0],ys[0],zs[0]+xy[1])),
    Vector((xs[0]+xy[0],ys[1],zs[0]+xy[1])),Vector((xs[0]+xy[0]+wh[0],ys[1],zs[0]+xy[1]))]
    ve2.extend([v + Vector((0,0,wh[1])) for v in ve2])
    ve.extend(ve2)
    if internal:
        fa = [(0,2,3,1),(1,3,7,5),(2,0,4,6),(4,5,7,6),
            (8,9,11,10),(11,9,13,15),(8,10,14,12),(12,14,15,13),
            (0,1,5,13,9,8),(4,0,8,12,13,5),(3,2,10,11,15,7),(2,6,7,15,14,10)] 
    else:        
        fa = [(0,2,3,1),(1,3,7,5),(2,0,4,6),(4,12,14,6),(13,5,7,15),
            (8,9,11,10),(11,9,13,15),(8,10,14,12),
            (0,1,5,13,9,8),(4,0,8,12),(3,2,10,11,15,7),(2,6,14,10)] 

    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(ve, [], fa)
    return mesh    

def mesh_for_box(name,xs,ys,zs,top=True,bottom=True):
    """Returns mesh for a rectangular box with vertices
    located in xs[0] xs[1], yx[0] ys[1], and zs[0] zs[1]
    and face on top (bottom) if true
    """
    ve = [Vector((xs[0],ys[0],zs[0])),Vector((xs[1],ys[0],zs[0])),
        Vector((xs[0],ys[1],zs[0])),Vector((xs[1],ys[1],zs[0])),
        Vector((xs[0],ys[0],zs[1])),Vector((xs[1],ys[0],zs[1])),
        Vector((xs[0],ys[1],zs[1])),Vector((xs[1],ys[1],zs[1])),]
    fa = [(1,3,7,5),(2,0,4,6),(0,1,5,4),(3,2,6,7)]
    if bottom:
        fa.append((0,2,3,1))
    if top:
        fa.append((4,5,7,6))
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(ve, [], fa)
    return mesh         

def mesh_for_lbeam(name,width,thick,length):
    """ Return mesh for square L beam of given width, length and thickness
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = [Vector((0,0,0)),Vector((0,0,width)),
            Vector((0,thick,width)),Vector((0,thick,thick)),
            Vector((0,width,thick)),Vector((0,width,0))]
        ve.extend([v + Vector((length,0,0)) for v in ve])    
        fa = [(0,1,2,3,4,5), (11,10,9,8,7,6),
            (0,6,7,1),(1,7,8,2),(2,8,9,3),
            (3,9,10,4),(4,10,11,5),(5,11,6,0)]
        mesh.from_pydata(ve, [], fa)
        mesh.update(calc_edges=True)
    return mesh

def mesh_for_tbeam(name,width,thick,length):
    """ Return mesh for T beam (stem = flange) of given width, length and thickness
    """
    mesh = bpy.data.meshes.get(name)
    a2 = width/2
    e2 = thick/2
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = [Vector((0,-a2,0)),Vector((0,-a2,thick)),
            Vector((0,-e2,thick)),Vector((0,-e2,width)),
            Vector((0,e2,width)),Vector((0,e2,thick)),
            Vector((0,a2,thick)),Vector((0,a2,0))]
        ve.extend([v + Vector((length,0,0)) for v in ve])    
        fa = [(0,1,2,3,4,5,6,7), (15,14,13,12,11,10,9,8),
            (0,8,9,1),(1,9,10,2),(2,10,11,3),(3,11,12,4),
            (4,12,13,5),(5,13,14,6),(6,14,15,7),(7,15,8,0)]
        mesh.from_pydata(ve, [], fa)
        mesh.update(calc_edges=True)
    return mesh

def mesh_for_polygon(name,vertlist):
    """ Returns mesh for polygon defined by the list of vertices vertlist
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = []
        for v in vertlist:
            ve.append(Vector((v[0],v[1],v[2])))
        nv = len(vertlist)   
        fa =  [tuple(n for n in range(0, nv))]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_icosphere(name, s = 1, subdiv = 1):
    """ Returns mesh for icosphere of size s and number of subdivisions subdiv
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        # Golden ratio 
        PHI = (1 + sqrt(5)) / 2
        ve = [ vertex(-1, PHI, 0, s), vertex( 1, PHI, 0, s), vertex(-1, -PHI, 0, s), vertex( 1, -PHI, 0, s), 
            vertex(0, -1, PHI, s), vertex(0, 1, PHI, s), vertex(0, -1, -PHI, s), vertex(0, 1, -PHI, s), 
            vertex( PHI, 0, -1, s), vertex( PHI, 0, 1, s), vertex(-PHI, 0, -1, s), vertex(-PHI, 0, 1, s), ]
        fa = [ [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], 
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], 
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], 
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1], ]
        for i in range(subdiv):
            faces_subdiv = []
            for tri in fa: 
                v1 = middle_point(ve,tri[0], tri[1], s) 
                v2 = middle_point(ve,tri[1], tri[2], s) 
                v3 = middle_point(ve,tri[2], tri[0], s) 
                faces_subdiv.append([tri[0], v1, v3]) 
                faces_subdiv.append([tri[1], v2, v1]) 
                faces_subdiv.append([tri[2], v3, v2]) 
                faces_subdiv.append([v1, v2, v3]) 
            fa = faces_subdiv    
        mesh.from_pydata(ve, [], fa)
    return mesh    
        
def mesh_for_frame(name,dims_hole,dims_frame):
    """ Return mesh for a frame enclosing a rectangular hole of dimension dims_hole
    dims_frame give the width and thickness of the frame
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        v1 = Vector((-dims_frame[1],dims_frame[0],dims_frame[0]+dims_frame[1]))
        (w1,w2,w3) = dims_hole[0]/2*Vector((1,1,1))+v1
        (h1,h2,h3) = dims_hole[1]/2*Vector((1,1,1))+v1
        (d1,d2) = [dims_hole[2]/2,dims_hole[2]/2+dims_frame[1]]
        ve = [Vector((w1,d2,h1)),Vector((w1,d2,-h1)),Vector((-w1,d2,-h1)),Vector((-w1,d2,h1)),
        Vector((w2,d2,h2)),Vector((w2,d2,-h2)),Vector((-w2,d2,-h2)),Vector((-w2,d2,h2)),
        Vector((w3,d1,h3)),Vector((w3,d1,-h3)),Vector((-w3,d1,-h3)),Vector((-w3,d1,h3))]
        ve.extend([multwise(v,Vector((1,-1,1))) for v in ve])
        fa = [(1,0,4,5),(2,1,5,6),(3,2,6,7),(0,3,7,4),(5,4,8,9),(6,5,9,10),(7,6,10,11),(4,7,11,8),
        (12,13,17,16),(13,14,18,17),(14,15,19,18),(15,12,16,19),(16,17,21,20),(17,18,22,21),
        (18,19,23,22),(19,16,20,23),(0,1,13,12),(1,2,14,13),(2,3,15,14),(3,0,12,15)]
        mesh.from_pydata(ve, [], fa)
        mesh.update(calc_edges=True)
    return mesh

def mesh_for_grid(name,Nx,Ny,dx,dy):
    """ Return mesh for a grid of Nx x Ny faces each with dimensions dx x dy
    or, Nx+1 by Ny+1 vertices located at range(Nx)*dx and range(Ny)*dy
    """
    Sx = Nx+1
    Sy = Ny+1
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve =  [Vector((nx*dx,ny*dy,0)) for ny in range(Sy) for nx in range(Sx)]
        fa = [[Sx*ny+nx,Sx*ny+nx+1,Sx*(ny+1)+nx+1,Sx*(ny+1)+nx] for ny in range(Ny) for nx in range(Nx)]
        mesh.from_pydata(ve, [], fa)
        mesh.update(calc_edges=True)
    return mesh        
                
        
def set_smooth(data):
    """ sets shading smooth for all faces
    """
    for f in data.polygons:
        f.use_smooth = True

def set_flat(data):
    """ sets shading flat for all faces
    """
    for f in data.polygons:
        f.use_smooth = False

def mesh_name_as_object(list_of_objects):
    """ Uses the names of the objects in list_of_objects to set the names of the mesh data
    """
    for ob in list_of_objects:
        if getattr(ob,'type','') in 'MESH':
            old_name = ob.data.name
            ob.data.name = ob.name

# ================================================================================
# OBJECT METHODS

def empty(name, type = 'PLAIN_AXES', size = 1, pos = [0,0,0]):
    """ returns empty object with given name, type and location pos
    """
    emp = bpy.data.objects.new(name,None)
    emp.empty_display_type = type
    emp.empty_display_size = size
    emp.location = pos
    return emp

def circle(name,N=32,axis=2,r=1.0,pos=[0,0,0],offset=[0,0,0]):
    """ returns a circle of radius r and normal along axis,
    at location pos and origin offset
    """
    me = mesh_for_circle(name,N,axis,r,offset)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    return ob

def rectangle(name, lx, ly, origin=[0,0], pos=[0,0,0]):
    """ returns a plane rectangle of dimensions lx by ly
    """
    (x0,y0) = origin
    ve = [ Vector((x0,y0,0)), Vector((x0+lx,y0,0)), Vector((x0,y0+ly,0)), Vector((x0+lx,y0+ly,0))]
    me = mesh_for_planeve(name,ve)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    return ob

def curve_bezier(name, ve=[[-1,0,0],[1,0,0]], theta=[-pi/4, pi/4], phi=None, pos=[0,0,0]):
    """ returns curve object of type 'BEZIER' with control points ve and 
    right_handles controled by angles theta y phi (None is XY plane)
    """
    ve_h = []
    for n,a in enumerate(theta):
        if phi is not None:
            slope = Vector((cos(phi[n])*cos(a),cos(phi[n])*sin(a),sin(phi[n])))
        else:     
            slope = Vector((cos(a),sin(a),0))
        ve_h.append(Vector(ve[n])+slope)
    cu = bezier_from_pts(name,ve,ve_h)
    ob = bpy.data.objects.new(cu.name,cu)
    ob.location = pos
    return ob

def cube(name, mats = None, pos = [0,0,0]): 
    """ returns cube object at location pos with materials mats
    """
    me = mesh_for_cube(name)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    if mats is not None:
        ob.data.materials.append(mats)
    return ob

def box(name, dims=[1,1,0.1], mats = None, pos = [0,0,0], zoffset = 0, top=True, bottom=True):
    """ returns a rectangular box of dimensions dims = [length(x), width(y), and height(z)]
    with origin at the base center and material mats
    """
    (length,width,height)=dims
    me = mesh_for_box(name,[-length/2,length/2],[-width/2,width/2],[zoffset,zoffset+height],top,bottom)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    if mats is not None:
        ob.data.materials.append(mats)
    return ob    

def cylinder(name, n=16, mats = None, r=1.0, h=1.0, pos=[0,0,0], rot=[0,0,0], zoffset=0): 
    """ returns a cylinder object with n lateral faces at location pos with materials mats
    origin offset zoffset
    """
    me = mesh_for_cylinder(name,n,r,h,zoffset)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler = rot
    if mats is not None:
        ob.data.materials.append(mats)
    return ob

def tube(name, n=16, mats = None, r=1.0, l=1.0, pos=[0,0,0], rot=[0,0,0], zoffset= 0, top=True, bot=True, axis=2): 
    """ returns a tube object with n lateral faces and vertical len(l) subdivisions of length(s) l
    and radius(ii) r.  l and r can be a single value (cylinder) or a list of lengths and radii 
    for subdivisions. top (bot) is the top (bottom) face and is added if True
    mats can be a single material or a list of len(l) materials assigned to each subdivision
    or len(l) + 2 for each subdivision + bottom + top
    """
    if type(l) is not list:
        h = [-zoffset, l-zoffset]
    else:
        h = [sum(l[:i])-zoffset for i in range(len(l)+1)]    
    if type(r) is not list:
        r = [r]*len(h) 
    else:
        if (len(r)<len(h)):
            r.extend([r[-1]]*(len(h)-len(r)))                
    me = mesh_for_tube(name,n,r,h,top,bot,axis)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler = rot
    if mats is not None:
        if (test_dim(mats)==1):
            for m in mats:
                ob.data.materials.append(m)
        else:    
            ob.data.materials.append(mats)
    return ob


def floor(name, mats = None, pos=[0,0,0],dims=[1,1,0.1],flip=0):
    """ convenience function returning a rectangular board  with material mats,
    dimensions dims = (length, width) , located at pos and laying in the XY plane  
    defaults thickness is 0.1 but it can be specified also in dims
    """
    dx=dims[0]/2
    dy=dims[1]/2
    if len(dims) < 3:
        dims.append(0.1)
    me = mesh_for_recboard(name,[-dx,dx],[-dy,dy],[0,dims[2]])
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler[0] = flip*pi
    if mats is not None:
        ob.data.materials.append(mats)
    print(ob.name)
    return ob

def wall(name, mats = None, pos=[0,0,0],rot=0, dims=[1,1,0.1], hole=None, basemat = None, basedim = None):
    """ convenience function returning a vertical rectangular board with material mats,
    dimensions dims, located at pos and with z rotation rot
    it also can add a base of basedim size with material basemat
    if hole is an array [x,y,w,h] adds a hole at position x,y relative to te lower left 
    corner of width w and height h
    """
    length=dims[0]
    thick=dims[2]
    height=dims[1] 
    if hole is None:
        me = mesh_for_recboard(name,[-length/2,length/2],[0,thick],[0,height])
    else:
        (x,y,w,h) = hole[:4]
        me = mesh_for_recboard_with_hole(name,[-length/2,length/2],[0,thick],[0,height],[x,y],[w,h])    
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler[2] = rot
    if mats is not None:
        ob.data.materials.append(mats)
    if basedim is not None:
        name = name + '_base'
        if hole is None:
            me = mesh_for_recboard(name,[-length/2,length/2],[-basedim[1],0],[0,basedim[0]])
        else:
            (x,y,w,h) = hole[:4]
            me = mesh_for_recboard_with_hole(name,[-length/2,length/2],[-basedim[1],0],[0,basedim[0]],
                [x,y],[w,basedim[0]-y],internal=False)    
        ba = bpy.data.objects.new(me.name,me)
        ba.location = pos
        ba.rotation_euler[2] = rot
        ba.data.materials.append(basemat)
        print(ob.name, ba.name)
        return ob, ba
    else:
        print(ob.name)
        return ob

def frame(name, mats=None, pos=[0,0,0], rot=0, dims_hole=[1,1,0.2], dims_frame=[0.1,0.01]):    
    """ specialized function returning a frame with material mats for a rectangular hole
    of dimensions dims_hole located at pos with z rotation rot. dims frame give the
    depth, width and thickness of the frame
    """
    me = mesh_for_frame(name,dims_hole,dims_frame)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler[2] = rot
    if mats is not None:
        ob.data.materials.append(mats)
    print(ob.name)    
    return ob    

def polygon(name, vertlist, mats = None, pos=[0,0,0],thick = 0.05):
    """ returns polygonal object determined by list of vertices vertlist with material mats
    located at pos and thickness thick
    """
    me = mesh_for_polygon(name,vertlist)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    s1 = ob.modifiers.new('S1','SOLIDIFY')
    s1.thickness = thick
    apply_mod(ob,s1)
    if mats is not None:
        ob.data.materials.append(mats)
    print(ob.name)
    return ob    

def icosphere(name, mats=None, pos=[0,0,0],r = 1,sub = 2, smooth = True):
    """ return icosphere object with radius r and materials mats, 
    located at pos and with number ob subdivisions subs
    """
    me = mesh_for_icosphere(name,r,sub)
    for face in me.polygons:
        face.use_smooth = smooth
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    if mats is not None:
        ob.data.materials.append(mats)
    print(ob.name)
    return ob


def grid(name, pos=[0,0,0], Nx=2, Ny =2, dx=1.0, dy=1.0):
    """ return a grid object with number of subdivisions Nx Ny and spacings dx dy 
    located at pos
    """
    me = mesh_for_grid(name,Nx,Ny,dx,dy)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    return ob


#################################################################################
# PAINT AND MATERIALS ASSIGNMENTS

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
                    # send error    

def replace_material(list_of_objects,name_match,material):
    """ Replace material in all objects in list list_of_objects with names matching string
    name_matr with material mat. Useful for duplicated materials with .001 .002, etc
    """
    for ob in list_of_objects:
        if getattr(ob,'type','') in 'MESH':
            for n in range(len(ob.data.materials)):
                if name_match in ob.data.materials[n].name:
                    ob.data.materials[n] = material


###################################################################################
# SIMPLE OPERATORS AND MODIFIER METHODS

def duplicate_ob(ob, name):
    """ duplicates object ob and returns new object with object name 'name'
    and mesh name idem
    """
    ob2 = ob.copy()
    ob2.data = ob.data.copy()
    ob2.name = name
    ob2.data.name = name
    return ob2

def apply_transforms(ob):
    ''' apply all transformation to object ob using matrix world
    '''
    ob.data.transform(ob.matrix_world)
    ob.matrix_world = Matrix()
    return

def hole(ob,hpos,hsize):
    """ makes a hole of size hsize at position hpos in object ob 
    using a boolean with a temp object, applying the mod
    and deleting the temp object afterwards
    """
    mtemp = mesh_for_cube('Temp',0.5)
    ob1 = bpy.data.objects.new(mtemp.name,mtemp)
    ob1.scale = (hsize[0],hsize[1],hsize[2])
    ob1.location = hpos
    h1 = ob.modifiers.new('H1','BOOLEAN')
    h1.object = ob1
    h1.operation = 'DIFFERENCE'
    # for Blender >= 2.91
    #h1.solver = 'FAST'
    apply_mod(ob,h1)
    bpy.data.objects.remove(ob1)
    return

def arraymod(ob,name='A1',count=2,off_relative=None,off_constant=None,obj2=None):
    """ returns an array modifier for object specifying count,
    relative or constant offset and eventually using an object object obj2
    """
    a1 = ob.modifiers.new(name,'ARRAY')
    a1.count = count
    if off_constant is not None:
        a1.use_relative_offset = False
        a1.use_constant_offset = True
        a1.constant_offset_displace = off_constant
    if off_relative is not None:    
        a1.use_relative_offset = True
        a1.relative_offset_displace = off_relative
    if obj2 is not None:
        a1.use_object_offset = True
        a1.offset_object = obj2
    return a1    

def tile_fill(name,dx,dy,Lx,Ly,offset=1):
    """
    Returns a rectangle object of dimensions dx by dy tiled 
    filling a rectangle of Lx by Ly with cuts in the end
    the relative offset of the array can be modified from the default
    """
    Nx = int(Lx//(dx*offset))
    Ny = int(Ly//(dy*offset))
    fx = Lx%(dx*offset)
    fy = Ly%(dy*offset)
    if Nx==0:
        fx=0
        Nx=1
        dx=Lx
    if Ny==0:
        fy=0
        Ny=1
        dy=Ly    
    print([Nx,Ny,fx,fy])
    tile = rectangle(name,dx, dy)
    tilex = rectangle(name+'x',fx*dx, dy)
    tiley = rectangle(name+'y',dx, fy*dy)
    tilexy = rectangle(name+'xy',fx*dx, fy*dy)
    a1 = tilex.modifiers.new('A1','ARRAY')
    a1.count = Ny
    a1.use_relative_offset = True
    a1.relative_offset_displace = Vector((0,offset,0))
    a1.end_cap = tilexy
    a2 = tile.modifiers.new('A2','ARRAY')
    a2.count = Ny
    a2.use_relative_offset = True
    a2.relative_offset_displace = Vector((0,offset,0))
    a2.end_cap = tiley
    a3 = tile.modifiers.new('A3','ARRAY')
    a3.count = Nx
    a3.use_relative_offset = True
    a3.relative_offset_displace = Vector((offset,0,0))
    a3.end_cap = tilex
    return tile

def embed_array(ob1,Nx,Ny,dx,dy,ob2,gap=0.0):
    """ makes an array of holes in ob1 using an array of ob2
    using a boolean with a temp object duplicated from ob2 and scaled
    by 1+gap, modified by the same array, applying the mod
    and deleting the temp object afterwards.
    The array is nx by ny with spacing dx dy and the initial 
    The initial position for ob2 must be assigned in advance
    """
    #apply_transforms(ob2)
    hole = duplicate_ob(ob2,'hole')
    hole.scale = [1+gap,1+gap,1+gap]
    #apply_transforms(hole)
    a1 = arraymod(hole,'A1',count=Ny,off_constant=[0,dy,0])
    a2 = arraymod(hole,'A2',count=Nx,off_constant=[dx,0,0])
    h1 = ob1.modifiers.new('H1','BOOLEAN')
    h1.object = hole
    h1.operation = 'DIFFERENCE'
    a3 = arraymod(ob2,'A3',count=Ny,off_constant=[0,dy,0])
    a4 = arraymod(ob2,'A4',count=Nx,off_constant=[dx,0,0])
    return
    

def deformmod(ob,name='D1',method='TWIST',axis='X',angle=10,limits=[0,1]):
    """ returns a simple deform modifier with method (twist,bend,taper,stretch)
    for object along axis with angle
    """
    d1 = ob.modifiers.new(name,'SIMPLE_DEFORM')
    d1.deform_axis = axis
    d1.deform_method = method
    d1.angle = radians(angle)
    d1.limits = limits
    return d1


def makecube(camrig, pos=(200,200,100)):
    """ convenience function returning a cube for equirectangular rendering 
    with cycles, the face ordering is: down front left rear right up
    """
    cube = []
    for n in range(6):
        caraname = camrig.children[n].name.lower()
        if bpy.data.objects.get(caraname) is not None:
            cara = bpy.data.objects.get(caraname)
        else:    
            plane = mesh_for_plane(camrig.children[n].name.lower())
            cara = bpy.data.objects.new(plane.name, plane)
            pname = cara.name.split('.')[0]
            if pname == 'up':
                cara.location = pos + Vector((0,0,1))
            elif pname == 'down':    
                cara.location = pos + Vector((0,0,-1))
            elif pname == 'front':
                cara.location = pos + Vector((0,1,0))
            elif pname == 'rear':    
                cara.location = pos + Vector((0,-1,0))
            elif pname == 'right':
                cara.location = pos + Vector((1,0,0))
            elif pname == 'left':    
                cara.location = pos + Vector((-1,0,0))
            cara.rotation_euler = camrig.children[n].rotation_euler
        cube.append(cara)
    return cube  

def apply_mat(ob,mat):
    """ apply material mat to object ob (first slot)
    """
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)
    return

###############################################################################################
# LIGHT METHODS

def new_spot(name='spot',rot=None,pos=None,tar=None,size=1,blend=0,color=(1,1,1),energy=10,spot_size=0.25,**kwargs):
    """ returns spot object with color energy, location pos, rotation rot or target tar, size and blend
    for EEVEE
    """
    sp = bpy.data.lights.new(name,'SPOT')
    sp.spot_size = size
    sp.spot_blend = blend
    sp.energy = energy
    sp.color = color
    # if render EEVEE
    #sp.use_shadow = True    
    # if render CYCLES
    sp.shadow_soft_size = spot_size
    ob_sp = bpy.data.objects.new(sp.name,sp)
    set_posrotar(ob_sp,pos,rot,tar)
    return ob_sp     

def new_area(name='area',rot=None,pos=None,tar=None,size=0.1,sizey=None,**kwargs):
    al = bpy.data.lights.new(name,'AREA')
    if sizey is not None:
        al.shape = 'RECTANGLE'
        al.size_y = sizey
    else:
        al.shape = 'SQUARE'   
    al.size = size
    ob_al = bpy.data.objects.new(al.name,al)
    set_posrotar(ob_al,pos,rot,tar)
    return ob_al

def new_point(name='point',pos=(0,0,0),color=Color((1,1,1)),power=10,r=0.25,contact=False):
    pt = bpy.data.lights.new(name,'POINT')
    pt.energy = power
    pt.color = color
    pt.use_shadow = True    
    pt.use_contact_shadow = contact
    ob_pt = bpy.data.objects.new(pt.name,pt)
    ob_pt.location = pos
    return ob_pt     
    
def new_sun(name='sun',pos=(0,0,0),rot=None,tar=None,power=1.0,color=Color((1,1,1))):
    sun = bpy.data.lights.new(name,'SUN')
    sun.energy = power
    sun.color = color
    sun.use_shadow = True
    ob_sun = bpy.data.objects.new(sun.name,sun)
    set_posrotar(ob_sun,pos,rot,tar)
    return ob_sun

def light_grid(ob,Nx,Ny,dx,dy):
    """
    creates an 'array' of instances from ob that can be a point light 
    (or any other objects that does not admit an array modifier) 
    with parameters nx ny (only for fixed z)
    and spacing dx dy, returns the grid
    """
    grid1 = grid('auxgrid',ob.location,Nx,Ny,dx,dy)
    ob.parent = grid1
    #ob.parent_type = 'VERTEX'  'OBJECT' default
    grid1.instance_type = 'VERTS'
    return grid1
   

def light_array(ob,Nx,Ny,dx,dy):
    """
    creates an array of copies from ob with the same light data
    and locations at the grid given by Nx x Ny with spacings dx dy
    at the same z coordinate
    Returns a list of lights including the original one
    """
    lights_list = [ob]
    for nx in range(Nx):
        for ny in range(Ny):
            if (nx+ny):
                ob2 = ob.copy()
                ob2.location = ob.location + Vector((nx*dx,ny*dy,0))
                ob2.name = ob.name + '_' + str(nx) + '_' + str(ny)
                lights_list.append(ob2)
    return(lights_list)         
    
    
# ================================================================================
# CAMERA METHODS

def new_camera(name='Cam',pos = (0,0,0), rotation = (pi/2,0,0), fl = 35, **kwargs):
    cam = bpy.data.cameras.new(name)    
    cam.lens = fl
    ob_cam = bpy.data.objects.new(cam.name,cam)
    set_posrotar(ob_cam,pos,rotation)
    return ob_cam

def new_equirectangular(name='Cam',pos = (0,0,0), rotation = (pi/2,0,0), **kwargs):
    """ for Cycles only
    """
    cam = bpy.data.cameras.new(name)    
    cam.type = 'PANO'
    cam.cycles.panorama_type = 'EQUIRECTANGULAR'
#    cam.stereo.convergence_mode = 'OFFAXIS'
#    cam.stereo.convergence_distance = convergence
#    cam.stereo.interocular_distance = convergence/30.0
#    cam.stereo.use_spherical_stereo = True
    ob_cam = bpy.data.objects.new(cam.name,cam)
    set_posrotar(ob_cam,pos,rotation)
    return ob_cam

def new_360camera(name='Cam',pos = None, rotation = None, convergence = 2.0, **kwargs):
    cam = bpy.data.cameras.new(name)    
    cam.type = 'PANO'
    cam.cycles.panorama_type = 'EQUIRECTANGULAR'
    cam.stereo.convergence_mode = 'OFFAXIS'
    cam.stereo.convergence_distance = convergence
    cam.stereo.interocular_distance = convergence/30.0
    cam.stereo.use_spherical_stereo = True
    ob_cam = bpy.data.objects.new(cam.name,cam)
    set_posrotar(ob_cam,pos,rotation)
    return ob_cam
    
def set_stereo3d():
    scnr.use_multiview = True
    scnr.views_format = 'STEREO_3D'
    scnr.image_settings.views_format = 'STEREO_3D'
    scnr.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'

def camera_track(cam,ob):
    camtrack = cam.constraints.new(type='TRACK_TO')
    camtrack.target = ob
    camtrack.up_axis = 'UP_Y'
    camtrack.track_axis = 'TRACK_NEGATIVE_Z'
    return 

def camera_trackxy(cam,ob,h=0):
    emp = bpy.data.objects.new('emp',None)
    emp.location = Vector((0,0,h))
    empl = emp.constraints.new('COPY_LOCATION')
    empl.target = ob
    empl.use_x = True
    empl.use_y = True
    empl.use_z = False
    camtrack = cam.constraints.new(type='TRACK_TO')
    camtrack.target = emp
    camtrack.up_axis = 'UP_Y'
    camtrack.track_axis = 'TRACK_NEGATIVE_Z'
    return emp
    
def new_camrig(name = 'Camrig', pos = (0,0,0), **kwargs):
    camrig = bpy.data.objects.new(name, None )
    camrig.location = pos
    posrel = Vector((0,0,0))
    c1 = cam4rig(camrig,'DOWN',posrel,Euler((0,0,0), 'XYZ'))
    c2 = cam4rig(camrig,'FRONT',posrel,Euler((pi/2,0,0), 'XYZ'))
    c3 = cam4rig(camrig,'LEFT',posrel,Euler((pi/2,0,pi/2), 'XYZ'))
    c4 = cam4rig(camrig,'REAR',posrel,Euler((pi/2,0,-pi), 'XYZ'))
    c5 = cam4rig(camrig,'RIGHT',posrel,Euler((pi/2,0,-pi/2), 'XYZ'))
    c6 = cam4rig(camrig,'UP',posrel,Euler((-pi,0,0), 'XYZ'))
    return camrig
    
def cam4rig(rig = None, name = None, pos = None, rot = None):    
    cam = bpy.data.cameras.new(name)
    cam.angle = pi/2
    cam.lens_unit = 'FOV'
    ob_cam = bpy.data.objects.new(cam.name,cam)
    ob_cam.location = pos
    ob_cam.rotation_euler = rot
    ob_cam.parent = rig
    return ob_cam
                
# =================================================================================
# ANIMATION METHODS

def animate_curve(p,nameanim,curve,tkeys,fkeys,skeys=None):
    """ returns action with name nameanim for data path curve of object p and
    keyframes fkeys at frames tkeys. Additionally slopes of handles (skeys)
    for keyframes can be set
    """ 
    dt = 10
    if test_dim(fkeys) == 2:
        nindex = len(fkeys[0])
    else:
        nindex = 1    
    if p.animation_data is None:
        p.animation_data_create()
    ac = bpy.data.actions.get(nameanim)
    if ac is None:
        ac = bpy.data.actions.new(name = nameanim)
    p.animation_data.action = ac
    for ni in range(nindex):
        fcu = ac.fcurves.new(data_path=curve, index=ni)
        fcu.keyframe_points.add(len(tkeys))
        for n in range(len(tkeys)):
            bt = fcu.keyframe_points[n]
            if type(fkeys[n]) is list:
                bt.co = Vector((tkeys[n],fkeys[n][ni]))
            else:    
                bt.co = Vector((tkeys[n],fkeys[n]))
            if skeys is None:    
                bt.interpolation = 'LINEAR'
            else:        
                bt.interpolation = 'BEZIER'
                if type(skeys[n]) is list:
                    bt.handle_left = bt.co + Vector((-dt, -skeys[n][ni]))
                    bt.handle_right = bt.co + Vector((dt, skeys[n][ni]))      
                else:    
                    bt.handle_left = bt.co + Vector((-dt, -skeys[n]))
                    bt.handle_right = bt.co + Vector((dt, skeys[n]))      
    return ac

def remove_curves(p,nameanim):
    ac = bpy.data.actions.get(nameanim)
    if ac is not None:
        for curve in ac.fcurves:
            ac.fcurves.remove(curve)
    return    


def animate_object_path(ob,plist,tlist,order=5,w=1,forw='FORWARD_X',up='UP_Z'):
    cc = curve_from_pts('pathanim',plist,order,w)
    co = bpy.data.objects.new(cc.name,cc)
    obc = ob.constraints.new(type='FOLLOW_PATH')
    obc.target = co
    obc.use_curve_follow = True
    obc.forward_axis = forw
    obc.up_axis = up
    keys=[[x,x] for x in tlist]
    set_path_keys(cc,keys)
    return

def animate_camera_path(cam,plist,flist,tlist,scaleframes):
    tl = [ceil((t - s)/scaleframes) for s, t in zip(tlist, tlist[1:])]
    camcurve = cpath('cam',plist,tl)
    tarcurve = cpath('tar',flist,tl)
    emp = bpy.data.objects.new('Foco', None)
    empath = emp.constraints.new(type='FOLLOW_PATH')
    empath.target = tarcurve
    campath = cam.constraints.new(type='FOLLOW_PATH')
    campath.target = camcurve
    keys=[[x,x] for x in tlist]
    set_path_keys(camcurve.data,keys)
    set_path_keys(tarcurve.data,keys)
    camera_track(cam,emp)
    return camcurve, tarcurve

def animate_camera_spiral(cam,p1,p2,t1,t2,nturn,npts,nframes):
    pa1 = cpath('path_empty',[t1,t2],[npts])
    pal = pts_from_curve(pa1)
    sp1 = spiral('path_cam',p1,p2,pal,nturn)
    emp = bpy.data.objects.new('Foco', None)
    empath = emp.constraints.new(type='FOLLOW_PATH')
    empath.target = pa1
    campath = cam.constraints.new(type='FOLLOW_PATH')
    campath.target = sp1
    keys=[[0,0],[nframes,nframes]]
    set_path_keys(pa1.data,keys)
    set_path_keys(sp1.data,keys)
    camera_track(cam,emp)
    return sp1,pa1,emp

def set_path_keys(pa,keys):
    pa.use_path = True
    pa.path_duration = keys[-1][1]
    for key in keys:
        pa.eval_time = key[0]
        pa.keyframe_insert(data_path="eval_time", frame=key[1])
    return        

def animation_bake(ob,start=1,end=250,step=1):
    bpy.context.view_layer.objects.active = ob
    ob.select_set(state = True)
    bpy.ops.nla.bake(frame_start=start, frame_end=end, step=step, visual_keying=True, clear_constraints=True,bake_types={'OBJECT'})
    return
    
# =================================================================================
# RENDER IMAGE AND ENVIRONMENT METHODS

def loadimage(path,fname):
    """ load an image in path, expect a path object (not string)
    """
    im = bpy.data.images.load(str(path / fname))
    return im

def set_cycles(gpu = True, experimental = False):
    if not scnr.engine == 'CYCLES':
        scnr.engine = 'CYCLES'
        if experimental:
            scn.cycles.feature_set = 'EXPERIMENTAL'
        else:
            scn.cycles.feature_set = 'SUPPORTED'
        if gpu:        
            scn.cycles.device = 'GPU'
        else:    
            scn.cycles.device = 'CPU'
    return

def set_eevee():
    if not scnr.engine == 'BLENDER_EEVEE':
        scnr.engine = 'BLENDER_EEVEE'
    return

def world_background(col):
    scnw.use_nodes = True
    Back = scnw.node_tree.nodes['Background']
    icolor = Back.inputs['Color']
    if icolor.is_linked:
        scnw.node_tree.links.remove(icolor.links[0])    
    icolor.default_value = col
    return 

def world_enviroment_image(image, projection='EQUIRECTANGULAR', strength = 1):
    scnw.use_nodes = True
    Back = scnw.node_tree.nodes['Background']
    Tex = scnw.node_tree.nodes.new('ShaderNodeTexEnvironment')
    Tex.image = image
    Back.inputs['Strength'].default_value = strength
    scnw.node_tree.links.new(Back.inputs['Color'], Tex.outputs['Color'])
    return
    
def set_resolution(x,y,p=100):
    scnr.resolution_x = x
    scnr.resolution_y = y    
    scnr.resolution_percentage = p
    return

def set_colorm(type,expos=0.5):
    if type == 'filmic':
        scn.view_settings.view_transform = 'Filmic'
        scn.view_settings.exposure = expos
        scn.view_settings.look = 'Filmic - Base Contrast'
        scn.sequencer_colorspace_settings.name = 'Filmic Log'
    return

def set_render_output(path, format = 'PNG', quality=90, encoding = None, codec=None):
    scnr.filepath = path
    scnr.image_settings.file_format = format
    if format == 'JPEG':
        scnr.image_settings.quality = quality
    if format == 'FFMPEG':
        scnr.ffmpeg.format = encoding
        scnr.ffmpeg.codec = codec
    return

def set_render_eevee(samples = 32, ssr = False, bloom = False, softs = True, gtao = False, refraction = False):
    set_eevee()
    scne.taa_render_samples = samples
    scne.use_ssr = ssr
    scne.use_ssr_refraction = refraction
    scne.use_ssr_halfres = False
    scne.use_bloom = bloom
    scne.use_soft_shadows = softs
    scne.use_shadow_high_bitdepth = True
    scne.use_gtao = gtao
    scne.shadow_cube_size = '1024'
    scne.shadow_cascade_size = '1024'
    return

def set_render_cycles(samples = 32, clmp = 3.0, blur = 1.0, caustics = False, denoise  = True, ao = False, ao_dist=10):
    set_cycles()
    scn.cycles.samples = samples
    scn.cycles.sample_clamp_indirect = clmp    
    scn.cycles.blur_glossy = blur
    scn.cycles.caustics_refractive = caustics
    scn.cycles.use_denoising = denoise
    scnw.light_settings.use_ambient_occlusion = ao
    scnw.light_settings.distance = ao_dist
    return

def render_cam(cam = None, anim = False, frame_ini=-1, frame_end=-1):
    pre_scene_frame_start = bpy.context.scene.frame_start
    pre_scene_frame_end = bpy.context.scene.frame_end 
    if frame_ini > -1:
        bpy.context.scene.frame_start = frame_ini
    if frame_end > -1:
        bpy.context.scene.frame_end = frame_end
    if cam is not None:
        scn.camera = cam
    print("render_cam: Rendering",cam.name,"from",frame_ini,"to",frame_end)
    if anim:    
        bpy.ops.render.render(animation = True, use_viewport = False)
    else:
        bpy.ops.render.render(write_still = True, use_viewport = True)
    bpy.context.scene.frame_start = pre_scene_frame_start
    bpy.context.scene.frame_end = pre_scene_frame_end
    return    

def bake_dynamics():
    bpy.ops.ptcache.bake_all(bake=True)
    return
