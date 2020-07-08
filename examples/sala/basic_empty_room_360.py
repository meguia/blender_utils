# Version 1 simple usando Cycles
import sys
import bpy
from pathlib import Path
utildir = Path.home() / 'Dropbox/Blender280/blender_utils' # modificar esta ruta al directorio correspondiente
savedir = Path.home() / 'Dropbox/Blender280/Renders/' # directorio donde guarda los renders, modificar
thisdir = Path.home() / 'Dropbox/Blender280/room_lapso/' # directorio donde guarda los renders, modificar
sys.path.append(str(utildir))   

import blender_methods as bm
import cyclesmaterials as cm
import clear_util as cu
import importlib as imp
from mathutils import Vector
from math import radians
from random import randint
imp.reload(bm)
imp.reload(cm)

# borra todo lo anterior
cu.clear_all()
#CARGA LAS RUTAS PARA LAS IMAGENES USa Extreme PBR Lib
path_mats = Path('G:/Textures/Extreme_PBR_Combo/EXTREME_PBR_LIB')
path_mat_piso = path_mats / 'Fabric - Carpet/2K_Carpet12'
#path_mat_paredes = path_mats / 'Plaster - Wall/2K_grey_plaster'
# ruta y archivo para material fonac
# Ruta y nombre del material de las paredes
path_mat_paredes = thisdir / "sala-fonac.blend\\Material\\"
mat_name = "Fonac"

#path_mat_techo = 
matdict_piso = cm.make_imagedict(path_mat_piso)
#matdict_paredes = cm.make_imagedict(path_mat_paredes)
map_piso = cm.Mapping(scale=[0.5,0.5,1.0])
#map_paredes = cm.Mapping(scale=[0.2,0.2,0.2])
bpy.ops.wm.append(filename=mat_name, directory=str(path_mat_paredes))


# MATERIALES CYCLES 
def gray(val,alpha=1):
    return [val,val,val,alpha]
blanco = cm.simple_material('Blanco',gray(0.93),specular=0.2,rough=0.2)
matpiso = cm.texture_full_material('Piso',matdict_piso,mapping=map_piso)
#matparedes = cm.texture_full_material('Pared',matdict_paredes,projection='BOX',mapping=map_paredes)
matparedes = bpy.data.materials[mat_name]

# ajuste del mapping de la textura del material en las paredes
matparedes.node_tree.nodes["Mapping"].inputs[3].default_value = [1, 1, 0]

#PARAMETROS GEOMETRICOS DE LA SALA
ancho = 6.10
alto = 3.00
largo = 7.10
esp = 0.15 # espesor de las paredes
#PARAMETROS DE LA PUERTA
wpuerta = 1.20
hpuerta = 2.20
pospuerta = 3.0
# Crea tres colecciones separadas para sala luces y objetos y las linkea a la escena 
col_sala = bm.iscol('SALA')
col_luces = bm.iscol('LUCES')
col_obj = bm.iscol('OBJ')
bm.link_col(col_sala)
bm.link_col(col_obj)
bm.link_col(col_luces)

# SALA
pared1 = bm.wall('pared1',matparedes,pos=[0,-ancho/2,alto/2.0],rot=0, dims=[largo,alto,esp])
pared2 = bm.wall('pared2',matparedes,pos=[0,ancho/2,alto/2.0],rot=0, dims=[largo,alto,esp])
pared3 = bm.wall('pared3',matparedes,pos=[-largo/2,0,alto/2.0],rot=radians(90), dims=[ancho,alto,esp])
pared4 = bm.wall('pared4',matparedes,pos=[largo/2,0,alto/2.0],rot=radians(90), dims=[ancho,alto,esp])
piso = bm.floor('piso',matpiso,dims=[largo,ancho,esp])
techo = bm.floor('techo',blanco,pos=[0,0,alto],dims=[largo,ancho,esp])
bm.hole(pared1,hpos=[pospuerta,-ancho/2,hpuerta/2+esp],hsize=[wpuerta,3*esp,hpuerta])
sala = bm.list_parent('sala',[pared1,pared2,pared3,pared4,piso,techo])
bm.link_all(sala,col_sala)

#LUCES crea una diccionario con todos los parametros
Lp = {
    'name':'Spot1',
    'pos': Vector((0.85*largo/2,0,0.85*alto)),
    'rot': Vector((0,radians(45),0)),
    'energy':300,
    'size':radians(120.0),
    'blend':1
    }
spot1 = bm.new_spot(**Lp)    
# Para los otros spots modificamos los parametros
Lp['name']='Spot2'
Lp['pos']=Vector((-0.85*largo/2,0,0.85*alto))
Lp['rot']=Vector((0,radians(-45),0))
Lp['energy']=2000
Lp['size']=radians(90)
spot2 = bm.new_spot(**Lp)
bm.list_link([spot1,spot2],col_luces)


# Camara 360
bm.set_cycles()
eqcam =  bm.new_equirectangular(name='Cam360',pos = (0,0,1.5))
bm.link_all(eqcam,col_luces)

##RENDER CYCLES
# 4k para probar
bm.set_resolution(4000,2000)
bm.set_render_cycles(samples = 128, clmp = 0.5)
bm.set_render_output(str(savedir / 'room360.png'))
#bm.render_cam(cam = eqcam)
