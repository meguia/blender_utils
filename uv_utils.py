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
    # mesh loops
    #(0,2,3,1),(1,3,7,5),(2,0,4,6),(4,5,7,6),(0,1,5,4),(3,2,6,7)
    uvs = [(e,e), (l+e,e), (e,0), (l+e,0), (e,h+e), (l+e,h+e), (e,h+2*e), (l+e,h+2*e),
            (0,e), (l+2*e,e), (0,h+e), (l+2*e,h+e), (e,2*h+2*e), (l+e,2*h+2*e)]
    if rot90:
        uvs =  list(map(lambda s: (s[1],s[0]), uvs))
            
    if front==1: #Top
        uv_idcs = [(4,0,1,5),(5,1,9,11),(0,4,10,8),(6,7,13,12),(4,5,7,6),(1,0,2,3)]
    elif front==2: #Down
        uv_idcs = [(12,6,7,13),(9,11,5,1),(10,8,0,4),(0,1,5,4),(2,3,1,0),(7,6,4,5)]
    else:
        # zero default lateral
        uv_idcs = [(0,2,3,1),(1,9,11,5),(8,0,4,10),(4,5,7,6),(0,1,5,4),(13,12,6,7)]

    assign_uv(bm,uvs,uv_idcs,uv_1,scale,(-e,-e))
    if withLM:
        assign_uv(bm,uvs,uv_idcs,uv_2,s)
    bm.to_mesh(mesh)
    bm.free()
        
def uv_board_with_hole(mesh, dims, hole, scale=None, withLM=True,  rot90 = False, name1='UVMap', name2='LM',internal=True):
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
    (x,y,w,z) = hole # posicion alto y ancho de hole
    if scale is None:
        scale = h
    s = max(l+2*e,2*h+2*e)

    # mesh loops internal
    #(0,2,3,1),(1,3,7,5),(2,0,4,6),(4,5,7,6),(8,9,11,10),(11,9,13,15),(8,10,14,12),(12,14,15,13),
    #(0,1,5,13,9,8),(4,0,8,12,13,5),(3,2,10,11,15,7),(2,6,7,15,14,10)
    # mesh loops top
    # 0,2,3,1),(1,3,7,5),(2,0,4,6),(4,12,14,6),(13,5,7,15),(8,9,11,10),(11,9,13,15),(8,10,14,12),
    # (0,1,5,13,9,8),(4,0,8,12),(3,2,10,11,15,7),(2,6,14,10)
    
    if internal:
        uvs = [ (e,e), (l+e,e), (e,0), (l+e,0), (e,h+e), (l+e,h+e), (e,h+2*e), (l+e,h+2*e),
            (0,e), (l+2*e,e), (0,h+e), (l+2*e,h+e), (e,2*h+2*e), (l+e,2*h+2*e),
            (e+x,e+y), (2*e+x,2*e+y), (e+x+w,e+y), (x+w,2*e+y),  (e+x+w,e+y+z), (x+w,y+z), 
            (e+x,e+y+z), (2*e+x,y+z), (e+x,2*h+2*e-y), (e+x+w,2*h+2*e-y), (e+x+w,2*h+2*e-y-z), (e+x,2*h+2*e-y-z)] 

        uv_idcs = [(0,2,3,1),(1,9,11,5),(8,0,4,10),(4,5,7,6),(14,16,17,15),(17,16,18,19),(14,15,21,20),
                (20,21,19,18),(0,1,5,18,16,14),(4,0,14,20,18,5),(13,12,22,23,24,7),(12,6,7,24,25,22)]
    else:
        uvs = [ (e,e), (l+e,e), (e,0), (l+e,0), (e,h+e), (l+e,h+e), (e,h+2*e), (l+e,h+2*e),
            (0,e), (l+2*e,e), (0,h+e), (l+2*e,h+e), (e,2*h+2*e), (l+e,2*h+2*e),
            (e+x,e+y), (e+x,h+e), (e+x+w,e+y), (e+x+w,h+e), (e+x+w,2*e+2*h-y), (e+x+w,2*e+h),
            (e+x,2*e+2*h-y), (e+x,2*e+h), (2*e+x,2*e+y), (x+w,2*e+y), (2*e+x,h+e),(x+w,h+e)]
            
        uv_idcs = [(0,2,3,1),(1,9,11,5),(8,0,4,10),(4,15,21,6),(17,5,7,19),(14,16,23,22),(23,16,17,25),
                (14,22,24,15),(0,1,5,17,16,14),(4,0,14,15),(13,12,20,18,19,7),(12,6,21,20)]            
    if rot90:
        uvs =  list(map(lambda s: (s[1],s[0]), uvs))
                               
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
