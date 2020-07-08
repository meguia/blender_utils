#CLEAR_UTIL.PY

import bpy

def clear(mesh=1,mat=1,cam=1,lamp=1,curve=1,empty=1):
    if mat:
        clear_mat()
    if mesh:
        clear_select_obj('MESH')
    if cam:
        clear_select_obj('CAMERA')
    if lamp:
        clear_select_obj('LIGHT')
    if curve:
        clear_select_obj('CURVE')
    if empty:
        clear_select_obj('EMPTY')
    clear_unused_obj()
    printable(list_obj())
    clear_unused_data()
    printable(list_mesh())
    printable(list_actions())
    printable(list_mat())
    
def printable(kmlist):        
    print(kmlist[0])
    for line in kmlist[1]:
        print(line)

def clear_select_obj(obtype):
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_by_type(type = obtype)
    bpy.ops.object.delete(use_global=False)

#dump

def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

#lists

def list_obj():
    oblist = []
    keys = ['#','Name', 'Type' ,'Users', 'Parent']
    for n, ob in enumerate(bpy.data.objects):
        oblist.append([n, ob.name, ob.type, ob.users, ob.parent])
    return keys, oblist      

def list_mesh():
    mlist = []
    keys = ['#','Name', 'Users', 'Verts', 'Edges', 'Faces']
    for n, m in enumerate(bpy.data.meshes):
        v = len(m.vertices.items())
        e = len(m.edges.items())
        f = len(m.polygons.items())
        mlist.append([n, m.name, m.users, v, e, f])
    return keys, mlist    

def list_mat():
    mlist = []
    keys = ['#','Name', 'Users']
    for n, m in enumerate(bpy.data.materials):
        mlist.append([n,m.name, m.users])
    return keys, mlist    

def list_actions():
    alist = []
    keys = ['#','Name', 'Users']
    for n, m in enumerate(bpy.data.actions):
        alist.append([n,m.name, m.users])
    return keys, alist    

#clear unused

def clear_unused_obj():
    for ob in bpy.data.objects:  
        ob.use_fake_user = False
        if ob.users == 0:
            bpy.data.objects.remove(ob)

def clear_mat():
    for m in bpy.data.materials:
        bpy.data.materials.remove(m)
    
def clear_unused_mat():
    for m in bpy.data.materials:
        if m.users == 0 :
            bpy.data.materials.remove(m)
        
def clear_unused_data():
    for m in bpy.data.meshes:
        if m.users == 0 :
            bpy.data.meshes.remove(m)        
    for c in bpy.data.cameras:
        if c.users == 0 :
            bpy.data.cameras.remove(c)        
    for l in bpy.data.lights:
        if l.users == 0 :
            bpy.data.lights.remove(l)
    for c in bpy.data.curves:
        if c.users == 0 :
            bpy.data.curves.remove(c)  
    for a in bpy.data.actions:
        if a.users == 0 :
            bpy.data.actions.remove(a)
    for o in bpy.data.collections:
        if o.users == 0 :
            bpy.data.collections.remove(o)              
            

# CLEAR OBJ

def clear_obj(ob):
    ob.use_fake_user = False
    if ob.users == 0:
        bpy.data.objects.remove(ob)        
        
# CLEAR COLLECTIONS        

def clear_col():
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)

def clear_act():
    for a in bpy.data.actions:
        bpy.data.actions.remove(a)

#CLEAR ALL

def clear_all():
    clear_col()
    clear_act()
    clear()
    


#def clear_all():
#    for m in bpy.data.materials:
#        m.user_clear()
#        bpy.data.materials.remove(m)
#    for m in bpy.data.meshes:
#        m.user_clear()
#        bpy.data.meshes.remove(m)
#    for o in bpy.data.objects:
#        o.user_clear()
#        bpy.data.objects.remove(o)    
#    for i in bpy.data.images:
#        i.user_clear()
#        bpy.data.images.remove(i)
#    clear_col()
#    clear_act()

        