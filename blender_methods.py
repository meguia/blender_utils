import bpy
from mathutils import Vector, Euler, Color
from math import pi, sin, cos, copysign, ceil, sqrt

scn = bpy.context.scene
scnc = scn.collection
scno = scnc.objects
scnr = scn.render
scnw = scn.world
scne = scn.eevee

# ================================================================================
# INDICE EN ESPANOL DE LAS TRES PRIMERAS CATEGORIAS, EL RESTO DESCRIPTO EN LAS FUNCIONES
#
# GEOMETRIC METHODS
# vertex(x,y,z,scale=1) devuelve una lista de coordenadas x,y,z normalizadas a radio = scale
# set_posangle(oblist,pos,angulo) posiciona lista de objetos oblist en coordenadas pos y rotacion angle
# set_posrotar(ob,pos=None,rot=None,tar=None) idem anterior pero con target
# shift_origin(ob,shift) corre el origen
# spiral(cname,p1,p2,targc,nturn) espiral de nturn vueltas entre dos puntos siguiendo a targc 
# cpath(cname,plist,step) devuelve objeto curva entre lista de puntos con pasos step
# curve_from_pts(cname,ve,order,w) devuelve curva data nurbs orden order a partir de lista de puntos ve
# pts_from_curve(cc) la inversa, devuelve lista de puntos a partir de curva data
# clamp(x, min, max) clampea
# linspace(x1,x2,n) linspace
# OPERATOR METHODS
# list_link(oblist) linkea a la escena la lista de objetos oblist y su 1a generacion de child
# link_all(CS) linkea objeto y 2 generaciones de childs usada para el CS > prisma > caps
# list_parent(name,oblist) parenta la lista de objetos oblist a un empty
# apply_mod(ob,mod) aplica el modificador mod al objeto ob, tiene que linkear seleccionar y luego deslinkear
# join(oblist) une todos los objetos de oblist al primer elemento usa ops
# curve_to_beam(ob,thick,width) tranforma la curva ob a un mesh de espesor thick y ancho width usa ops
# MESH METHODS
# mesh_for_plane(name,ve,orient) devuelve mesh de plano con vertices ve y orientacion +/- 1
# mesh_for_cylinder(n,name,origin) devuelve mesh de cilindro de n nodos de altura origin
# mesh_for_cube(name,origin) devuelve mesh de cubo unidad de altura origin
# mesh_for_recboard(name,xs,ys,zs) devuelve mesh de placa rectangular entre xs[0] y xs[1], etc
# mesh_for_board(name,vertices,thick) devuelve mesh de placa con espesor thick de perfil vertices
# mesh_for_prisma(n,width,thick,alto) devuelve mesh de prisma U cuadrado de dimensiones dadas
# mesh_for_lbeam(name,width,thick,length) devuelve mesh de L beam en la direccion x de dimensiones dadas
# mesh_for_tbeam(name,width,thick,length) idem ant con T beam
# mesh_for_polygon(name,vertlist) devuelve mesh de poligono dado por lista de vertices

# ================================================================================
# GEOMETRIC METHODS

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
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod.name)
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
    
# =====================================================================
# MESH METHODS

def mesh_for_plane(name,orient=1):
    """ Returns mesh for a default plane with name and orientation
    """
    mesh = bpy.data.meshes.get(name)
    ve = [ Vector((-1, -1, 0)), Vector((1, -1, 0)), Vector((1, 1, 0)), Vector((-1, 1, 0))]
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        if orient == 1:
            fa = [(0,1,2,3)]
        else:    
            fa = [(3,2,1,0)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_planeve(name,ve,orient=1):
    """ Returns mesh for plane definde by vertices ve with name and orientation
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        if orient == 1:
            fa = [(0,1,2,3)]
        else:    
            fa = [(3,2,1,0)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_cylinder(n,name,r=1.0, h=1.0, zorigin=0.0):
    """ Returns mesh for a cylinder with name, n lateral faces,radius r, height h
    and relative position of the origin zorigin along the vertical axis
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = []
        fa = []
        for i in range(n):
            theta = 2*pi*i/n
            v0 = [ r*cos(theta), r*sin(theta), h*(1-zorigin)]
            v1 = [ r*cos(theta), r*sin(theta), -h*zorigin]
            ve.append(v0)
            ve.append(v1)
            i0 = i*2
            if i+1>=n:
                i2 = 0
            else:
                i2 = i0+2
            fa.append([i0, i0+1, i2+1, i2])
        fa.append( [ i*2 for i in range(n)])
        fa.append( [ (n-i)*2-1 for i in range(n)])
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
           Vector((0.5, 0.5, -zorigin)), Vector((-0.5, 0.5, -zorigin))]
        ve.extend([v + Vector((0,0,1)) for v in ve])
        fa=[(3,2,1,0),(4,5,6,7),
            (0,1,5,4),(1,2,6,5),
            (2,3,7,6),(3,0,4,7)]
        mesh.from_pydata(ve, [], fa)
    return mesh

def mesh_for_recboard(name,xs,ys,zs):
    """ Returns mesh for a rectangular board vith vertices
    located in xs[0] xs[1], yx[0] ys[1], and zs[0] zs[1]
    """     
    ve = [Vector((xs[0],ys[0],zs[0])),Vector((xs[1],ys[0],zs[0])),
      Vector((xs[1],ys[1],zs[0])),Vector((xs[0],ys[1],zs[0]))]
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
        ve.extend([v + Vector((0,0,-thick)) for v in ve])
        #Inverted!
        fa = [(3,2,1,0), (4,5,6,7),
            (0,1,5,4), (1,2,6,5),   
            (2,3,7,6), (3,0,4,7)]
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
        
def mesh_for_tube(n,m,name,r=1.0, h=1.0, zorigin=0.0):
    """ Returns mesh for tube segmented in m slices with  
    radii given by r and z coordinates given by h
    """
    mesh = bpy.data.meshes.get(name)
    if not hasattr(r, "__len__"):
        r = [r for _ in range(m)]
    if not hasattr(h, "__len__"):
        h = [j*h/m for j in range(m)]    
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        ve = []
        fa = []
        fa.append([n-i for i in range(1,n+1)])
        for j in range(m):
            for i in range(n):
                theta = 2*pi*i/n
                v0 = [ r[j]*cos(theta), r[j]*sin(theta), -h[-1]*zorigin + h[j]]
                ve.append(v0)
                if (j+1 < m): 
                    i0 = j*n+i
                    if i+1>=n:
                        i2 = j*n
                    else:
                        i2 = i0+1
                    fa.append([i0, i2, i2+n, i0+n])
        fa.append( [(m-1)*n+i for i in range(n)])
        mesh.from_pydata(ve, [], fa)
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


# ================================================================================
# OBJECT METHODS

def empty(name, type = 'PLAIN_AXES', size = 1, pos = [0,0,0]):
    """ returns empty object with ginve name, type and location pos
    """
    emp = bpy.data.objects.new(name,None)
    emp.empty_display_type = type
    emp.empty_display_size = size
    emp.location = pos
    return emp

def cube(name, mats = None, pos = [0,0,0]): 
    """ returns cube object at location pos with materials mats
    """
    me = mesh_for_cube(name)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    if mats is not None:
        ob.data.materials.append(mats)
    return ob

def cylinder(name, n=16, mats = None, r=1.0, h=1.0, zoffset = 0, pos = [0,0,0], rot = [0,0,0]): 
    """ returns a cylinder object at location pos with materials mats
    and origin offset zoffset, n is the number of faces
    """
    me = mesh_for_cylinder(n,name,r=r, h=h, zorigin=zoffset)
    ob = bpy.data.objects.new(me.name,me)
    ob.location = pos
    ob.rotation_euler = rot
    if mats is not None:
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

def wall(name, mats = None, pos=[0,0,0],rot=0, dims=[1,1,0.1]):
    """ convenience function returning a vertical rectangular board with material mats,
    dimensions dims, located at pos and with z rotation rot
    """
    dx=dims[0]/2
    dy=dims[2]/2
    dz=dims[1]/2
    me = mesh_for_recboard(name,[-dx,dx],[-dy,dy],[dz,2*dz])
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
    apply_mod(ob,h1)
    bpy.data.objects.remove(ob1)
    return

def arraymod(ob,count=2,off_relative=None,off_constant=None,obj2=None):
    """ returns an array modifier for object sepcifying count,
    relative or constant offset and eventually using an object object obj2
    """
    a1 = ob.modifiers.new('A1','ARRAY')
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

def new_spot(name='spot',rot=None,pos=None,tar=None,size=1,blend=0,color=(1,1,1),energy=10,**kwargs):
    """ returns spot object with color energy, location pos, rotation rot or target tar, size and blend
    """
    sp = bpy.data.lights.new(name,'SPOT')
    sp.spot_size = size
    sp.spot_blend = blend
    sp.energy = energy
    sp.color = color
    sp.use_shadow = True    
    sp.use_contact_shadow = True
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

def set_render_output(path, format = 'PNG'):
    scnr.filepath = path
    scnr.image_settings.file_format = format
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

def set_render_cycles(samples = 32, clmp = 3.0, blur = 1.0, caustics = False, denoise  = True):
    set_cycles()
    scn.cycles.samples = samples
    scn.cycles.sample_clamp_indirect = clmp    
    scn.cycles.blur_glossy = blur
    scn.cycles.caustics_refractive = caustics
    scn.cycles.use_denoising = denoise
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