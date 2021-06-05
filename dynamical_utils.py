import numpy as np

import blender_methods as bm
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.fft import fft
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

def axes2D(xlim=[-1,1],ylim=[-1,1],ticks='auto'):
    #creates plane xy for plot at origin 
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    pln = bm.rectangle('canvas', xrange, yrange, origin=[xlim[0],ylim[0]])
    l = min(xrange,yrange)
    dticks = l/8.0
    lwidth = l/1000 # line width 1/1000 of plot
    # axis boxes
    xaxis = bm.box('xaxis', dims=[xrange,lwidth,lwidth], origin=[xlim[0],0,0])
    yaxis = bm.box('yaxis', dims=[lwidth,yrange,lwidth], origin=[0,ylim[0],0])
    xaxis.parent = pln
    yaxis.parent = pln
    # ticks boxes array
    return pln


def plot2D(x,y,xlim='auto',ylim='auto',type='line',ticks='auto', wdot=False):
    ''' creates a static plot of array y vs array x in plane of xdim x ydim (in meters)
    the axis ranges can be set to 'auto' between min and max of array or to specific values xlim=[x1,x2]
    default type is 'line' suing segments but can also be 'scatter' (or 'bezier' in the future?)
    ticks can be 'auto' for min(xdim,ydim)/8 spacing, 'none', or set to specific values in a 2d list
    '''
    #creates plane xy for plot at origin 
    if xlim == 'auto':
        (xl,xh) = [min(x),max(x)]
        xlim = [1.1*xl-0.1*xh,1.1*xh-0.1*xl]
    if ylim == 'auto':
        (yl,yh) = [min(y),max(y)]
        ylim = [1.1*yl-0.1*yh,1.1*yh-0.1*yl]
    pln = axes2D(xlim=xlim,ylim=ylim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    lwidth = l/1000 
    #creates bezier curve with data 
    pts = [[x[n],y[n],0] for n in range(len(x))]
    plt = bm.smooth_bezier('plot',pts,bevel=lwidth)
    plt.parent = pln
    return pln
    
def plot2D_animated(x,y,t,xlim='auto',ylim='auto',type='line',ticks='auto'):
    ''' creates an animated plot of array y vs array x parametrized with array t in frames 
    all other parameters are the same as plot2D
    '''
    #creates plane xy for plot at origin 
    if xlim == 'auto':
        (xl,xh) = [min(x),max(x)]
        xlim = [1.1*xl-0.1*xh,1.1*xh-0.1*xl]
    if ylim == 'auto':
        (yl,yh) = [min(y),max(y)]
        ylim = [1.1*yl-0.1*yh,1.1*yh-0.1*yl]
    pln = axes2D(xlim=xlim,ylim=ylim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    lwidth = l/1000 
    #creates bezier curve with data 
    pts = [[x[n],y[n],0] for n in range(len(x))]
    plt = bm.smooth_bezier('plot',pts,bevel=lwidth)
    bm.animate_curve(plt.data,'pltanim','bevel_factor_end',[0,len(x)],[0,1])
    plt.parent = pln    
    # create a dot following (x,y)
    dot = bm.cylinder('dot', r=lwidth*8, h=lwidth*2, pos=[x[0],y[0],0])
    dot.parent = pln
    fkeys = [[x[n],y[n],0] for n in range(len(x))]
    bm.animate_curve(dot,'dotanim','location',t,fkeys)        
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

def plot2D_flux(syst,pars,xini_array,tmax,pv=[0,1],dt=0.005,dtframe=3,xlim=[-1,1],ylim=[-1,1]):
    t = np.arange(0, tmax, dt)
    #creates plane xy for plot at origin 
    pln = axes2D(xlim=xlim,ylim=ylim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    lwidth = l/10000 
    # create a dot model
    dot0 = bm.cylinder('dot', r=lwidth*10, h=lwidth*5, pos=[0,0,0])    
    # loop over initial conditions]
    for m,xini in enumerate(xini_array):
        print(m)
        s = solve(syst, t, xini, args=pars, method='RK45') 
        x = s[::dtframe,pv[0]]
        y = s[::dtframe,pv[1]]
        frames = np.arange(len(x))
        pts = [[x[n],y[n],0] for n in range(len(x))]
        plt = bm.smooth_bezier('plot_'+ str(m),pts,bevel=lwidth)
        #animate curve
        bm.animate_curve(plt.data,'pltanim_'+str(m),'bevel_factor_end',[0,len(x)],[0,1])
        plt.parent = pln    
        # create a dot following (x,y)
        dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
        dot.parent = pln
        fkeys = [[x[n],y[n],0] for n in range(len(x))]
        frames = np.arange(len(x))
        bm.animate_curve(dot,'dotanim_'+str(m),'location',frames,fkeys)        
    return pln



def solve_plot_body(syst,pars,xini,tmax,body=None,pv=[0,1],dt=0.005,dtframe=3):
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
    if body is None:
        body = bm.cube('body',zorigin=0.5)
        body.scale = [d,d,d]
    fkeys = [[x[n],-2*d,0] for n in range(len(x))]
    bm.animate_curve(body,'bodyanim','location',frames,fkeys)
    return pln,body
    
    
# Examples dynamical systems

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
        


    

    
    