import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path.home() / 'blender_utils')) 
import blender_methods as bm
import importlib as imp
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.fft import fft
import clear_utils as cu
import material_utils as mu
import importlib as imp

imp.reload(bm)


def solve(func,t,x0,method='DOP853',args=None):
    sol = solve_ivp(func, t[[0,-1]], x0, method=method, t_eval=t, args=args)
    if sol.status < 0:
        print(sol.message)
    return sol.y.T

def findperiod(t,x):
    peaks, _ = find_peaks(x)
    per = np.diff(t[peaks])
    return np.mean(per)

def plot2D(x,y,xlim='auto',ylim='auto',type='line',ticks='auto', wdot=False):
    ''' creates a static plot of array y vs array x in plane of xdim x ydim (in meters)
    the axis ranges can be set to 'auto' between min and max of array or to specific values xlim=[x1,x2]
    default type is 'line' suing segments but can also be 'scatter' (or 'bezier' in the future?)
    ticks can be 'auto' for min(xdim,ydim)/8 spacing, 'none', or set to specific values in a 2d list
    '''
    #creates plane xy for plot at origin 
    if xlim is 'auto':
        (xl,xh) = [min(x),max(x)]
        xlim = [1.1*xl-0.1*xh,1.1*xh-0.1*xl]
    if ylim is 'auto':
        (yl,yh) = [min(y),max(y)]
        ylim = [1.1*yl-0.1*yh,1.1*yh-0.1*yl]
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    pln = bm.rectangle('canvas', xrange, yrange, origin=[xlim[0],ylim[0]])
    l = min(xrange,yrange)
    dticks = l/8.0
    lwidth = l/1000 # line width 1/1000 of plot
    #creates bezier curve with data 
    pts = [[x[n],y[n],0] for n in range(len(x))]
    plt = bm.smooth_bezier('plot',pts,bevel=lwidth)
    plt.parent = pln
    # axis boxes
    xaxis = bm.box('xaxis', dims=[xrange,lwidth,lwidth], origin=[xlim[0],0,0])
    yaxis = bm.box('yaxis', dims=[lwidth,yrange,lwidth], origin=[0,ylim[0],0])
    xaxis.parent = pln
    yaxis.parent = pln
    # ticks boxes array
    if (wdot):
        # dot at initial condition
        dot = bm.cylinder('dot', r=lwidth*8, h=lwidth*2, pos=[x[0],y[0],0])
        dot.parent = pln
        return pln, dot
    else: 
        return pln
    
def plot2D_animated(x,y,t,xlim='auto',ylim='auto',type='line',ticks='auto'):
    ''' creates an animated plot of array y vs array x parametrized with array t in frames 
    all other parameters are the same as plot2D
    '''
    # create a dot following (x,y)
    pln, dot = plot2D(x,y,xlim,ylim,type,ticks,wdot=True)
    tkeys = t
    fkeys = [[x[n],y[n],0] for n in range(len(x))]
    bm.animate_curve(dot,'dotanim','location',tkeys,fkeys)        
    return pln

def solve_plot(syst,pars,xini,tmax,pv=[0,1],dt=0.005,dtframe=3):
    t = np.arange(0, tmax, dt)
    s = solve(syst, t, xini, args=pars, method='RK45') 
    print('system solved')
    # only two variables 
    x = s[::dtframe,pv[0]]
    y = s[::dtframe,pv[1]]
    frames = np.arange(len(x))
    pln = plot2D_animated(x,y,frames)
    return pln

def solve_plot_body(syst,pars,xini,tmax,pv=[0,1],dt=0.005,dtframe=3):
    t = np.arange(0, tmax, dt)
    s = solve(syst, t, xini, args=pars, method='RK45') 
    print('system solved')
    # only two variables 
    x = s[::dtframe,pv[0]]
    y = s[::dtframe,pv[1]]
    frames = np.arange(len(x))
    pln = plot2D_animated(x,y,frames)
    pln_width = max(x)-min(x)
    d = 0.1*pln_width
    pln.rotation_euler=[np.pi/2,0,0]
    body = bm.cube('body',zorigin=0.5)
    body.scale = [d,d,d]
    fkeys = [[x[n],-2*d,0] for n in range(len(x))]
    bm.animate_curve(body,'bodyanim','location',frames,fkeys)
    return pln,body
    
    
# Examples

def osc_harm(t, x, W,C):
    return [
        x[1],
        -W*x[0]-C*x[1],
    ]
    
def osc_bounce(t, x, W,G,C):
    if (x[0]>0):
        return [
            x[1],
            -G-C*x[1],
        ]
    else:
        return [
            x[1],
            -G-W*x[0]-C*x[1],
        ]

def van_der_pol(t, x, W, C):
    return [
        x[1],
        -W*x[0]-C*x[1]*(x[0]*x[0]-1),
    ]        
        
cu.clear_all()
black = mu.simple_material('black',[0,0,0,1])
white = mu.simple_material('white',[1,1,1,1])
bright_red = mu.simple_material('bright_red',[1,0,0,1],emission=[1,0,0,1])
red = mu.simple_material('red',[1,0.2,0.2,1])
#col_sys = bm.iscol('dynamical_system')
#cu.clear_collection(col_sys)

col_sys = bm.iscol('dynamical_system')
bm.link_col(col_sys)
W = 50
G = 0.2
C = 0.0
tmax = 6.79
pln,body = solve_plot_body(osc_bounce,(W,G,C),[1.0, 0],tmax,dt=0.0001,dtframe=100)
#pln,body = solve_plot_body(osc_harm,(W,C),[1.0, 0],tmax,dt=0.005,dtframe=20)
#pln,body = solve_plot_body(van_der_pol,(W,C),[0.1, 0.1],tmax,dt=0.005,dtframe=20)
pln.children[0].data.materials.append(bright_red)
pln.children[1].data.materials.append(red)
pln.children[2].data.materials.append(white)
pln.children[3].data.materials.append(white)
pln.data.materials.append(black)
body.data.materials.append(red)
bm.link_all(pln,col_sys)
bm.link_all(body,col_sys)

    

    
    