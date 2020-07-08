import bpy
import blender_methods as bm
import cyclesmaterials as cm
from mathutils import Vector
import importlib as imp
imp.reload(bm)
imp.reload(cm)

#RENDER EQURECTANGULAR

#PRE RENDER EN EEVEEE
def pre_render_eeve(fpath,rigname = 'Camrig1', resolution=1920,frame_end=250,frame_ini=1,samples=32,bloom=False):
    bm.set_eevee()
    bm.set_resolution(resolution,resolution,100)
    bm.set_render_eevee(samples = samples, ssr = True, softs = True, refraction = True, bloom=bloom)
    rig1 = bpy.data.objects[rigname]
    for camo in rig1.children:
        cname = camo.name.split('.')[0]
        bm.set_render_output(str(fpath / cname))
        bm.render_cam(cam = camo, anim = True,frame_ini=frame_ini,frame_end=frame_end)
    return 

#RENDER EQUIRECTANGULAR EN CYCLES
def render_equirectangular_from_rig(fpath_in,fpath_out,rigname = 'Camrig1',resolution=[4096,2048],frame_end=250,frame_ini=1,samples=32):
    bm.set_cycles()
    rig1 = bpy.data.objects[rigname]
    cubopos = Vector((200,200,100))
    col_render = bm.iscol('RENDER')
    cubo = bm.makecube(rig1,cubopos)
    bm.list_link(cubo,col_render)
    came = bm.new_equirectangular('Equi',pos = cubopos)
    bm.link_all(came,col_render)
    bm.link_col(col_render)
    bm.set_resolution(resolution[0],resolution[1],100)
    bm.set_render_cycles(samples = samples)
    for n in range(frame_ini,frame_end):
        imagelist = []
        for camo in rig1.children:
            cname = camo.name.split('.')[0]
            fname = cname + str(n+1).zfill(4) + '.png'
            im = bm.loadimage(fpath_in,fname)   
            imagelist.append(im)
        matlist = cm.cubemat(imagelist,cubo,1.0)
        for ob,mat in zip(cubo,matlist):
            ob.active_material = mat
        bm.set_render_output(str(fpath_out / 'equi_') + str(n+1).zfill(4) + '.png')
        bm.render_cam(cam = came)
