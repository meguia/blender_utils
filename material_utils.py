#cyclematerials, also for eevee
import bpy
import os
from mathutils import Color,Vector

bw = bpy.data.worlds[0]

# CLASSES

class Material:

    def set_cycles(self):
        scn = bpy.context.scene
        if not scn.render.engine == 'CYCLES':
            scn.render.engine = 'CYCLES'
        
    def set_eevee(self):
        scn = bpy.context.scene
        if not scn.render.engine == 'BLENDER_EEVEE':
            scn.render.engine = 'BLENDER_EEVEE'    
            
#    def make_material(self, name):
#        self.mat = bpy.data.materials.new(name)
#        self.mat.use_nodes = True
#        self.nodes = self.mat.node_tree.nodes
        
    def link(self, from_node, from_slot_name, to_node, to_slot_name):
        input = to_node.inputs[to_slot_name]
        output = from_node.outputs[from_slot_name]
        self.mat.node_tree.links.new(input, output)
        
    def makeNode(self, type, name, xpos=0 , ypos=300):
        self.node = self.nodes.new(type)
        self.node.name = name
        self.node.location = xpos, ypos
        return self.node
    
    def dump_node(self, node):
        print (node.name)
        print ("Inputs:\n")
        for n in node.inputs: print ("\t", n)
        print ("Outputs:\n")
        for n in node.outputs: print ("\t", n) 
        
    def __init__(self,name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nodes = self.mat.node_tree.nodes
        
class Mapping:
    
    def __init__(self,scale=[1.0,1.0,1.0],rotation=[0.0,0.0,0.0],coord='Object',ob_coord=None):
        self.scale = scale
        self.rotation = rotation
        self.coord = coord
        self.ob_coord = ob_coord

########################
# UTILITIES

keynames = {
    'basecolor' : ['COL', 'diffuse', 'Color','color'] , 
    'metallic' : ['METAL', 'metal','Metallic','metallic'],
    'specular': ['REFL','specular','Specular'], 
    'roughness' : ['ROUGH','roughness','Roughness'] ,
    'gloss' : ['GLOSS','glossiness','Glossiness'],
    'normal' : ['NRM','normal','Normal'], 
    'height' : ['DISP','displace','bump','Height','height']
    }

def check_imagedict(path,keys):
    """
    Check if path contains all keynames in keys
    return an array of booleans     
    """ 
    keynames_sub = {key: keynames[key] for key in keys}
    flist = os.listdir(path)
    checked = []
    for k,v in keynames_sub.items():
          checked.append(any(vn in f for vn in v for f in flist))
    return checked      
                
def make_imagedict(path):
    """ search for image files in path with matching keynames 
    and returns a dictionary of image objects for making a full material
    """
    imagedict = {}
    flist = os.listdir(path)
    for k,v in keynames.items():
        for vn in v:
            for f in flist:
                if vn in f:
                    im = bpy.data.images.load(str(path / f))
                    imagedict[k] = im
                    break
    return imagedict
    
def op_smart_uv(ob):
    """ apply smart uv operator in object ob
    """
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    return

def add_mapping(m, mapping):
    """add a Mapping Texture coordinate input and a mapping vector node at the beggining 
    of the node tree and returns the mapping node for making links
    """
    xcoords = [node.location[0] for node in m.nodes]
    TexCoord = m.makeNode('ShaderNodeTexCoord','Input', xpos = min(xcoords)-600)
    Map = m.makeNode('ShaderNodeMapping','Mapping', xpos = min(xcoords)-400)
    if mapping.ob_coord is not None:
        TexCoord.object = mapping.ob_coord
        if mapping.coord == 'UV':
            op_smart_uv(mapping.ob_coord)    
    m.link(TexCoord,mapping.coord,Map,'Vector')
    Map.inputs[3].default_value = mapping.scale
    Map.inputs[2].default_value = mapping.rotation
    return Map
    
def add_image_texture(m,image,name,extension,projection,location=Vector((-300,300)),colorspace='sRGB'):
    """ retruns a new image texture node to material class m with parameters
    """
    Tex = m.makeNode('ShaderNodeTexImage',name.lower(),xpos=location[0],ypos=location[1])
    Tex.image = image
    Tex.extension = extension
    Tex.projection = projection
    Tex.image.colorspace_settings.name = colorspace
    return Tex

def gray(val,alpha=1):
    """ returns gray color val,val,val with alpha
    """
    return [val,val,val,alpha]

##############################
# MATERIALS


def simple_material(matName,basecolor,subcolor=None,specular=0,roughness=0,metallic=0,subsurf=0,emission=[0,0,0,1]):
    """ Simple Principled Material with name matName specifying 
    color, specular, roughness and metallic values, and Subsurface
    """    
    m = Material(matName)
    #m.make_material(matName)
    materialOutput = m.nodes['Material Output']
    PBSDF = m.nodes['Principled BSDF']
    PBSDF.inputs["Base Color"].default_value = basecolor
    PBSDF.inputs["Specular"].default_value = specular
    PBSDF.inputs["Roughness"].default_value = roughness
    PBSDF.inputs["Metallic"].default_value = metallic
    PBSDF.inputs["Emission"].default_value = emission
    if subcolor is not None:
        PBSDF.inputs["Subsurface"].default_value = subsurf
        PBSDF.inputs["Subsurface Color"].default_value = subcolor
    m.mat.diffuse_color = basecolor
    m.link(PBSDF,'BSDF',materialOutput,'Surface')
    return m.mat
    
def texture_simple_material(matName,image,extension='REPEAT',projection='FLAT',specular=0,roughness=0,metallic=0,
                            mapping=None):
    """ Simple Textured Principled Material with name matName from a single texture image 
    as base color, with specular, roughness and metallic values
    """
    m = Material(matName)
    #m.make_material(matName)
    materialOutput = m.nodes['Material Output']
    PBSDF = m.nodes['Principled BSDF']
    PBSDF.inputs["Specular"].default_value = specular
    PBSDF.inputs["Roughness"].default_value = roughness
    PBSDF.inputs["Metallic"].default_value = metallic
    Tex = add_image_texture(m,image,'Color',extension,projection)
    m.link(Tex,'Color',PBSDF,'Base Color')
    if mapping is not None:
        Map = add_mapping(m,mapping)
        m.link(Map,'Vector',Tex,'Vector')
    m.link(PBSDF,'BSDF',materialOutput,'Surface')    
    return m.mat    

def texture_simple_material_using_channels(matName,image,extension='REPEAT', projection='FLAT',RoughChan='G',
                                            BumpChan='R',specular=0,metallic=0,height = 0.5, mapping=None):
    """ Simple textured material from a single texture image as base color and using 
    channels Bumpchan and Roughchan as inputs for displacement and roughness
    """
    m = Material(matName)
    #m.make_material(matName)
    materialOutput = m.nodes['Material Output']
    PBSDF = m.nodes['Principled BSDF']
    PBSDF.inputs["Specular"].default_value = specular
    PBSDF.inputs["Metallic"].default_value = metallic
    RGB = m.makeNode('ShaderNodeSeparateRGB','Separate RGB', xpos = -430, ypos = 100)
    Bump = m.makeNode('ShaderNodeBump','Bump', xpos = -200, ypos = -130)
    HSV = m.makeNode('ShaderNodeHueSaturation','HSV', xpos = -200)
    Ramp = m.makeNode('ShaderNodeValToRGB','Color Ramp', xpos = -250, ypos = 100) 
    Bump.inputs["Strength"].default_value = height
    Tex = add_image_texture(m,image,'Color',extension,projection)
    m.link(Tex,'Color',HSV,'Color')
    m.link(Tex,'Color',RGB,'Image')
    m.link(RGB,BumpChan,Bump,'Height')
    m.link(RGB,RoughChan,Ramp,'Fac')
    m.link(Bump,'Normal',PBSDF,'Normal')
    m.link(Ramp,'Color',PBSDF,'Roughness')
    m.link(HSV,'Color',PBSDF,'Base Color')
    if mapping is not None:
        Map = add_mapping(m,mapping)
        m.link(Map,'Vector',Tex,'Vector')
    m.link(PBSDF,'BSDF',materialOutput,'Surface')
    return m.mat    

def texture_full_material(matName,imagedict,extension='REPEAT', projection='FLAT', metallic=0,normal = 0.2, 
                            height = 0.05, specular=0.8, roughness = 0.05, mapping = None, displacement = False):
    """ returns a full material using imagedict dictionary containing keys and images. 
    Image Texture Node is added only if key exists
    """
    m = Material(matName)
    ycor = 300
    xcor = -300
    Bump = None
    #m.make_material(matName)
    materialOutput = m.nodes['Material Output']
    PBSDF = m.nodes['Principled BSDF']
    Tex_color = add_image_texture(m,imagedict['basecolor'],'Base Color',extension,projection)
    m.link(Tex_color,'Color',PBSDF,'Base Color')
    Tex_color.hide = True
    if mapping is not None:
        Map = add_mapping(m,mapping)
        m.link(Map,'Vector',Tex_color,'Vector')
    else:
        Map = None    
    if 'metallic' in imagedict.keys():
        ycor = ycor -100
        loc = Vector((xcor,ycor))
        Tex_metal = add_image_texture(m,imagedict['metallic'],'Metallic',extension,projection,location=loc,colorspace='Non-Color')
        m.link(Tex_metal,'Color',PBSDF,'Metallic')
        Tex_metal.hide = True
        if mapping is not None:
            m.link(Map,'Vector',Tex_metal,'Vector')
    else:
        PBSDF.inputs["Metallic"].default_value = metallic
    if 'specular' in imagedict.keys():
        ycor = ycor -100
        loc = Vector((xcor,ycor))
        Tex_spec = add_image_texture(m,imagedict['specular'],'Specular',extension,projection,location=loc,colorspace='Non-Color')
        m.link(Tex_spec,'Color',PBSDF,'Specular')
        Tex_spec.hide = True
        if mapping is not None:
            m.link(Map,'Vector',Tex_spec,'Vector')
    else:
        PBSDF.inputs["Specular"].default_value = specular
    if 'roughness' in imagedict.keys():
        ycor = ycor -100
        loc = Vector((xcor,ycor))
        Tex_rough = add_image_texture(m,imagedict['roughness'],'Roughness',extension,projection,location=loc,colorspace='Non-Color')
        m.link(Tex_rough,'Color',PBSDF,'Roughness')
        Tex_rough.hide = True
        if mapping is not None:
            m.link(Map,'Vector',Tex_rough,'Vector')
    else:
        PBSDF.inputs["Roughness"].default_value = roughness
    if 'gloss' in imagedict.keys():
        ycor = ycor -100
        loc = Vector((xcor-200,ycor))
        Tex_gloss = add_image_texture(m,imagedict['gloss'],'Gloss',extension,projection,location=loc,colorspace='Non-Color')
        Invert = m.makeNode('ShaderNodeInvert','InvertGloss', xpos=xcor, ypos = ycor)
        m.link(Tex_gloss,'Color',Invert,'Color')
        m.link(Invert,'Color',PBSDF,'Roughness')
        Tex_gloss.hide = True
        if mapping is not None:
            m.link(Map,'Vector',Tex_gloss,'Vector')
    if 'height' in imagedict.keys():
        ycor = ycor -100
        xcor = xcor - 200
        loc = Vector((xcor,ycor))
        Tex_bump = add_image_texture(m,imagedict['height'],'Displacement',extension,projection,location=loc,colorspace='Non-Color')
        Tex_bump.hide = True   
        if mapping is not None:
            m.link(Map,'Vector',Tex_bump,'Vector') 
        if displacement:
            Displ =  m.makeNode('ShaderNodeDisplacement','Displacement', xpos = xcor+300, ypos = ycor)
            Displ.inputs['Scale'].default_value = height
            m.link(Tex_bump,'Color',Displ,'Height')
            m.link(Displ,'Displacement',materialOutput,'Displacement')
        else:    
            Bump =  m.makeNode('ShaderNodeBump','Bump', xpos = xcor+300, ypos = ycor)
            Bump.inputs['Strength'].default_value = height
            m.link(Tex_bump,'Color',Bump,'Height')
            m.link(Bump,'Normal',PBSDF,'Normal')      
    if 'normal' in imagedict.keys():
        ycor = ycor -100
        loc = Vector((xcor-200,ycor))
        Tex_normal = add_image_texture(m,imagedict['normal'],'Normal',extension,projection,location=loc,colorspace='Non-Color')
        Nmap =  m.makeNode('ShaderNodeNormalMap','NormalMap', xpos = xcor+100, ypos = ycor)
        Nmap.inputs['Strength'].default_value = normal
        m.link(Tex_normal,'Color',Nmap,'Color')
        if 'height' in imagedict.keys() and not displacement:
            m.link(Nmap,'Normal',Bump,'Normal')
        else:
            m.link(Nmap,'Normal',PBSDF,'Normal')
        Tex_normal.hide = True
        if mapping is not None:
            m.link(Map,'Vector',Tex_normal,'Vector')
    m.link(PBSDF,'BSDF',materialOutput,'Surface')        
    return m.mat    

def emission_material(matName,strength=1.0,baseColor=[1,1,1,1],image=None,ob_coord=None):
    m = Material(matName)
    #m.make_material(matName)
    materialOutput = m.nodes['Material Output']
    PBSDF = m.nodes['Principled BSDF']
    m.nodes.remove(PBSDF)
    emission = m.makeNode('ShaderNodeEmission','Emission')
    emission.inputs['Strength'].default_value = strength
    emission.inputs['Color'].default_value = baseColor
    m.link(emission,'Emission',materialOutput,'Surface')
    if image is not None:
        emission_image = m.makeNode('ShaderNodeTexImage',matName.lower())
        m.link(emission_image,'Color',emission,'Color')
        emission_image.image = image
        emission_image.extension = 'EXTEND'
    if ob_coord is not None:
        texture_coordinate = m.makeNode('ShaderNodeTexCoord',ob_coord.name)
        texture_coordinate.object = ob_coord
        m.link(texture_coordinate,'Generated',emission_image,'Vector')
    return m.mat


def texture_environment_material(matName,image,projection='EQUIRECTANGULAR',specular=0,rough=0,metal=0,
                                ob_coord=None,scale=Vector((1,1,1))):
    m = Material(matName)
    #m.make_material(matName)
    materialOutput = m.nodes['Material Output']
    PBSDF = m.nodes['Principled BSDF']
    PBSDF.inputs["Specular"].default_value = specular
    PBSDF.inputs["Roughness"].default_value = rough
    PBSDF.inputs["Metallic"].default_value = metal
    Tex = m.makeNode('ShaderNodeTexEnvironment',matName)
    if ob_coord is not None:
        TexCoord = m.makeNode('ShaderNodeTexCoord',ob_coord.name)
        TexCoord.object = ob_coord
        Map = m.makeNode('ShaderNodeMapping','Mapping')
        Map.scale = scale
        m.link(TexCoord,'Object',Map,'Vector')
        m.link(Map,'Vector',Tex,'Vector')
    m.link(Tex,'Color',PBSDF,'Base Color')
    Tex.image = image
    return m.mat    
    
def cubemat(imagelist,oblist,pow):
    matlist = []
    for n in range(6):
        mat = emission_material(imagelist[n].name.split('.')[0],imagelist[n],pow,oblist[n])
        matlist.append(mat)
    return matlist    
            
