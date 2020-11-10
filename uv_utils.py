import bmesh

def assign_uv(bm,uvs,uv_idcs,uv,scale):
    
    for f in bm.faces:
        idx = f.index
        uv_tup = uv_idcs[idx]
        li = 0
        for g in f.loops:
            luv1 = g[uv]
            uv_idx = uv_tup[li]
            luv1.uv = (x/scale for x in uvs[uv_idx])
            li = li+1
    return
        
def uv_board(mesh, dims, front=0, scale=None, withLM=True,  rot90 = False, name1='UVMap', name2='LM'):
    
    # remove all previous layers
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    mesh.uv_layers.new(name=name1)
    if withLM:
        mesh.uv_layers.new(name=name2)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_1 = bm.loops.layers.uv[name1]
    if withLM:
        uv_2 = bm.loops.layers.uv[name2]
    (l,h,e) = dims
    if scale is None:
        scale = h
    s = max(l+2*e,2*h+2*e)

    uvs = [ (e,e), (l+e,e), (e,0), (l+e,0), (e,h+e), (l+e,h+e), (e,h+2*e), (l+e,h+2*e),
            (0,e), (l+2*e,e), (0,h+e), (l+2*e,h+e), (e,2*h+2*e), (l+e,2*h+2*e)]
    if rot90:
        uvs =  list(map(lambda s: (s[1],s[0]), uvs))
            
    if front==1:
        uv_idcs = [(0,2,3,1),(0,1,5,4),(1,9,11,5),(13,12,6,7),(8,0,4,10),(4,5,7,6)]
    elif front==3:
        uv_idcs = [(3,1,0,2),(13,12,6,7),(8,0,4,10),(0,1,5,4),(1,9,11,5),(7,6,4,5)]
    elif front==5:
        uv_idcs = [(12,6,7,13),(2,3,0,1),(9,11,5,1),(7,6,4,5),(10,8,0,4),(0,1,5,4)]
    else:
        # zero default
        uv_idcs = [(0,1,5,4),(0,4,10,8),(4,5,7,6),(5,1,9,11),(1,0,2,3),(12,6,7,13)]

    assign_uv(bm,uvs,uv_idcs,uv_1,scale)
    if withLM:
        assign_uv(bm,uvs,uv_idcs,uv_2,s)
    bm.to_mesh(mesh)
    bm.free()
        
def uv_board_with_hole(mesh, dims, hole, scale=None, withLM=True,  rot90 = False, name1='UVMap', name2='LM'):

    # remove all previous layers
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    mesh.uv_layers.new(name=name1)
    if withLM:
        mesh.uv_layers.new(name=name2)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_1 = bm.loops.layers.uv[name1]
    if withLM:
        uv_2 = bm.loops.layers.uv[name2]
    (l,h,e) = dims
    (x,y,z,w) = hole # posicion alto y ancho de hole
    if scale is None:
        scale = h
    s = max(l+2*e,2*h+2*e)

    # loops originales de la mesh
    # (0,2,3,1), (0,14,12,10,5,4), (1,3,7,5), (3,9,11,13,6,7), (2,0,4,6), (4,5,7,6)
    # (13,12,14,15), (15,14,8,9), (5,10,8,14,0,1), (6,13,15,9,3,2), (10,11,9,8), (11,10,12,13)
    #
    uvs = [ (e,e), (l+e,e), (e,0), (l+e,0), (e,h+e), (l+e,h+e), (e,h+2*e), (l+e,h+2*e),
            (0,e), (l+2*e,e), (0,h+e), (l+2*e,h+e), (e,2*h+2*e), (l+e,2*h+2*e),
            (e+x,e+y), (2*e+x,2*e+y), (e+x+w,e+y), (x+w,2*e+y),  (e+x+w,e+y+z), (x+w,y+z), 
            (e+x,e+y+z), (2*e+x,y+z), (e+x,2*h+2*e-y), (e+x+w,2*h+2*e-y), (e+x+w,2*h+2*e-y-z), (e+x,2*h+2*e-y-z)] 

    if rot90:
        uvs =  list(map(lambda s: (s[1],s[0]), uvs))
    uv_idcs = [(0,2,3,1), (0,14,20,18,5,4), (1,9,11,5), (13,23,24,25,6,7), 
                (8,0,4,10), (4,5,7,6), (21,20,14,15), (15,14,16,17),
                (5,18,16,14,0,1), (6,25,22,23,13,12), (18,19,17,16), (19,18,20,21)]
                
                
                
    assign_uv(bm,uvs,uv_idcs,uv_1,scale)
    if withLM:
        assign_uv(bm,uvs,uv_idcs,uv_2,s)
    bm.to_mesh(mesh)
    bm.free()         

    
def uv_planks(mesh, scale=None, withLM=True,  rot90 = False, name1='UVMap', name2='LM'):
    
    # remove all previous layers
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])

    mesh.uv_layers.new(name=name1)
    if withLM:
        mesh.uv_layers.new(name=name2)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # calculates all lengths and heights
    length = []
    width = []
    for f in bm.faces:
        s = [l.edge.calc_length() for l in f.loops]
        a = f.calc_area()
        sm = max(s)
        length.append(sm)
        width.append(a/sm)
        
    uv_1 = bm.loops.layers.uv[name1]
    if withLM:
        uv_2 = bm.loops.layers.uv[name2]
    if scale is None:
        scale = sum(width)
    s = max(max(length),sum(width))

    uvs = []
    uv_idcs = []
    h = 0
    for n,l in enumerate(length):
        h2 = width[n]
        uvs.extend([(0,h),(l,h),(0,h+h2),(l,h+h2)])
        uv_idcs.append((4*n,4*n+1,4*n+3,4*n+2))
        h += h2
    if rot90:
        uvs =  list(map(lambda s: (s[1],s[0]), uvs))        
    assign_uv(bm,uvs,uv_idcs,uv_1,scale)
    if withLM:
        assign_uv(bm,uvs,uv_idcs,uv_2,s)
    bm.to_mesh(mesh)
    bm.free()         
