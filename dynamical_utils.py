import numpy as np

import blender_methods as bm
import material_utils as mu
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.fft import fft
import importlib as imp

imp.reload(bm)
imp.reload(mu)


def solve(func,t,x0,method='DOP853',args=None):
    sol = solve_ivp(func, t[[0,-1]], x0, method=method, t_eval=t, args=args)
    if sol.status < 0:
        print(sol.message)
    return sol.y.T

def findperiod(t,x):
    peaks, _ = find_peaks(x)
    per = np.diff(t[peaks])
    return np.mean(per)

def axes2D(name='canvas',xlim=[-1,1],ylim=[-1,1],ticks='auto'):
    #creates plane xy for plot at origin 
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    pln = bm.rectangle(name, xrange, yrange, origin=[xlim[0],ylim[0]])
    l = min(xrange,yrange)
    dticks = l/8.0
    lwidth = l/1000 # line width 1/1000 of plot
    # axis boxes
    xaxis = bm.box(name+'_xaxis', dims=[xrange,lwidth,lwidth], origin=[xlim[0],0,0])
    yaxis = bm.box(name+'_yaxis', dims=[lwidth,yrange,lwidth], origin=[0,ylim[0],0])
    xaxis.parent = pln
    yaxis.parent = pln
    # ticks boxes array
    return pln

def axes3D(name='axes',xlim=[-1,1],ylim=[-1,1],zlim=[-1,1],ticks='auto'):
    #creates plane xy for plot at origin 
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    zrange = ylim[1]-ylim[0]    
    l = min([xrange,yrange,zrange])
    dticks = l/8.0
    lwidth = l/1000 # line width 1/1000 of plot
    axs = bm.empty(name)
    # axis boxes
    xaxis = bm.box(name+'_xaxis', dims=[xrange,lwidth,lwidth], origin=[xlim[0],0,0])
    yaxis = bm.box(name+'_yaxis', dims=[lwidth,yrange,lwidth], origin=[0,ylim[0],0])
    zaxis = bm.box(name+'_zaxis', dims=[lwidth,lwidth,zrange], origin=[0,0,zlim[0]])
    xaxis.parent = axs
    yaxis.parent = axs
    zaxis.parent = axs
    # ticks boxes array
    return axs

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
    
def plot2D_animated(x,y,t,pltname='plotxy',xlim='auto',ylim='auto',type='line',ticks='auto'):
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
    pln = axes2D(name=pltname,xlim=xlim,ylim=ylim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    lwidth = l/5000 
    #creates bezier curve with data 
    pts = [[x[n],y[n],0] for n in range(len(x))]
    plt = bm.smooth_bezier(pltname+'_curve',pts,bevel=lwidth)
    bm.animate_curve(plt.data,pltname+'_pltanim','bevel_factor_end',[0,len(x)],[0,1])
    plt.parent = pln    
    # create a dot following (x,y)
    dot = bm.cylinder(pltname + '_dot', r=lwidth*10, h=lwidth*5, pos=[x[0],y[0],0])
    dot.parent = pln
    fkeys = [[x[n],y[n],0] for n in range(len(x))]
    bm.animate_curve(dot,pltname+'_dotanim','location',t,fkeys)        
    return pln



def solve_plot_2D(syst,pars,xini,tmax,pv=[0,1],dt=0.005,dtframe=3,xyt=False):
    t = np.arange(0, tmax, dt)
    s = solve(syst, t, xini, args=pars, method='RK45') 
    print('system solved')
    # only two variables 
    t = t[::dtframe]
    x = s[::dtframe,pv[0]]
    y = s[::dtframe,pv[1]]
    frames = np.arange(len(x))
    pln = plot2D_animated(x,y,frames)
    xrange = (max(x)-min(x))
    yrange = (max(y)-min(y))
    tscale = 0.75*xrange/tmax
    if xyt:
        plx = plot2D_animated(t*tscale,x/3,frames,'plotx')
        ply = plot2D_animated(t*tscale,y/3,frames,'ploty')
        plx.location = [-xrange*1.5,yrange/4,0]
        ply.location = [-xrange*1.5,-yrange/4,0]
        return (plx,ply,pln)
    else:    
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

def plot2D_flux_colors(syst,pars,xini_array,tmax,cmap_path,pv=[0,1],dt=0.005,dtframe=3,xlim=[-1,1],ylim=[-1,1]):
    t = np.arange(0, tmax, dt)
    #creates plane xy for plot at origin 
    pln = axes2D(xlim=xlim,ylim=ylim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    lwidth = l/10000 
    # create a dot model
    dot0 = bm.cylinder('dot', r=lwidth*10, h=lwidth*10, pos=[0,0,0])    
    # loop over initial conditions]
    for m,xini in enumerate(xini_array):
        print(m)
        # define material for orbit (dim) and point (bright)
        coord = [(xini[0]-xlim[0])/xrange,(xini[1]-ylim[0])/yrange]
        pmat = mu.colormap_material('pmat'+str(m),coord,cmap_path,emission=True,estrength=0.1)
        dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=100)
        s = solve(syst, t, xini, args=pars, method='RK45') 
        x = s[::dtframe,pv[0]]
        y = s[::dtframe,pv[1]]
        frames = np.arange(len(x))
        pts = [[x[n],y[n],0] for n in range(len(x))]
        plt = bm.smooth_bezier('plot_'+ str(m),pts,bevel=lwidth)
        #animate curve
        bm.animate_curve(plt.data,'pltanim_'+str(m),'bevel_factor_end',[0,len(x)],[0,1])
        plt.parent = pln    
        # assigns color
        plt.data.materials.append(pmat)
        # create a dot following (x,y)
        dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
        dot.parent = pln
        fkeys = [[x[n],y[n],0] for n in range(len(x))]
        frames = np.arange(len(x))
        bm.animate_curve(dot,'dotanim_'+str(m),'location',frames,fkeys)
        dot.data.materials.append(dmat)
        dot.material_slots[0].link = 'OBJECT'
        dot.material_slots[0].material = dmat
    return pln

def plot1D_flux_colors(syst,pars,xini_array,tmax,cmap_path,dt=0.005,dtframe=3,xlim=[-1,1],ylim=[-1,1]):
    t = np.arange(0, tmax, dt)
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    x0 = np.arange(xlim[0], xlim[1], xrange/20)
    #creates plane x f(x) for plot at origin 
    pln = axes2D(xlim=xlim,ylim=ylim,ticks='auto')
    lwidth = l/1000
    #plot function
    print(pars)
    y0 = syst(t,x0,*pars)
    pts = [[x0[n],y0[n],0] for n in range(len(x0))]
    func = bm.smooth_bezier('func',pts,bevel=lwidth)
    func.parent = pln
    pmat = mu.simple_material('pmat',[1,1,1,1],emission=[1,1,1,1],estrength=1)
    func.data.materials.append(pmat)
    # create a dot model
    dot0 = bm.cylinder('dot', r=lwidth*5, h=lwidth*20, pos=[0,0,0])    
    # loop over initial conditions]
    xmin = min(xini_array)
    xmax = max(xini_array)
    xrange = [x1 - x2 for (x1, x2) in zip(xmax, xmin)]
    for m,xini in enumerate(xini_array):
        print(m)
        # define material for orbit (dim) and point (bright)
        coord = [1-(xini[0]-xmin[0])/xrange[0],0]
        dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=10)
        s = solve(syst, t, xini, args=pars, method='RK45') 
        x = s[::dtframe]
        frames = np.arange(len(x))
        pts = [[x[n],0,0] for n in range(len(x))]
        # create a dot following (x)
        dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
        dot.parent = pln
        fkeys = [[x[n],0,0] for n in range(len(x))]
        frames = np.arange(len(x))
        bm.animate_curve(dot,'dotanim_'+str(m),'location',frames,fkeys)
        dot.data.materials.append(dmat)
        dot.material_slots[0].link = 'OBJECT'
        dot.material_slots[0].material = dmat
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

def plot3D_flux_colors(syst,pars,xini_array,tmax,cmap_path,pv=[0,1,2],dt=0.005,dtframe=3,xlim=[-1,1],ylim=[-1,1],zlim=[-1,1]):
    t = np.arange(0, tmax, dt)
    #creates 3D axes at origin
    axs = axes3D(xlim=xlim,ylim=ylim,zlim=zlim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]
    zrange = zlim[1]-zlim[0]
    l = min([xrange,yrange,zrange])
    lwidth = l/10000 
    # create a dot model
    dot0 = bm.icosphere('dot', r = lwidth*10, sub=1)
    # loop over initial conditions]
    xmin = min(xini_array)
    xmax = max(xini_array)
    xrange = [x1 - x2 for (x1, x2) in zip(xmax, xmin)]
    for m,xini in enumerate(xini_array):
        print(m)
        # define material for orbit (dim) and point (bright)
        coord = [(xini[0]-xmin[0])/xrange[0],(xini[1]-xmin[1])/xrange[1]]
        pmat = mu.colormap_material('pmat'+str(m),coord,cmap_path,emission=True,estrength=0.2)
        dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=10)
        s = solve(syst, t, xini, args=pars, method='RK45') 
        x = s[::dtframe,pv[0]]
        y = s[::dtframe,pv[1]]
        z = s[::dtframe,pv[2]]
        frames = np.arange(len(x))
        pts = [[x[n],y[n],z[n]] for n in range(len(x))]
        plt = bm.smooth_bezier('plot_'+ str(m),pts,bevel=lwidth)
        #animate curve
        bm.animate_curve(plt.data,'pltanim_'+str(m),'bevel_factor_end',[0,len(x)],[0,1])
        plt.parent = axs
        # assigns color
        plt.data.materials.append(pmat)
        # create a dot following (x,y)
        dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
        dot.parent = axs
        fkeys = [[x[n],y[n],z[n]] for n in range(len(x))]
        frames = np.arange(len(x))
        bm.animate_curve(dot,'dotanim_'+str(m),'location',frames,fkeys)
        dot.data.materials.append(dmat)
        dot.material_slots[0].link = 'OBJECT'
        dot.material_slots[0].material = dmat
    return axs

def plot3D_curves_colors(syst,pars,xini_array,tmax,cmap_path,pv=[0,1,2],dt=0.005,dtframe=3,xlim=[-1,1],ylim=[-1,1],zlim=[-1,1]):
    t = np.arange(0, tmax, dt)
    #creates 3D axes at origin
    axs = axes3D(xlim=xlim,ylim=ylim,zlim=zlim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]
    zrange = zlim[1]-zlim[0]
    l = min([xrange,yrange,zrange])
    lwidth = l/10000 
    # loop over initial conditions]
    xmin = min(xini_array)
    xmax = max(xini_array)
    xrange = [x1 - x2 for (x1, x2) in zip(xmax, xmin)]
    frameini = int(len(t)/10)
    for m,xini in enumerate(xini_array):
        print(m)
        # define material for orbit (dim) 
        coord = [(xini[0]-xmin[0])/xrange[0],(xini[1]-xmin[1])/xrange[1]]
        pmat = mu.colormap_material('pmat'+str(m),coord,cmap_path,emission=True,estrength=0.2)
        s = solve(syst, t, xini, args=pars, method='RK45') 
        x = s[frameini::dtframe,pv[0]]
        y = s[frameini::dtframe,pv[1]]
        z = s[frameini::dtframe,pv[2]]
        pts = [[x[n],y[n],z[n]] for n in range(len(x))]
        plt = bm.smooth_bezier('plot_'+ str(m),pts,bevel=lwidth)
        plt.parent = axs
        # assigns color
        plt.data.materials.append(pmat)
    return axs
    
# Examples dynamical systems

#1D
def logistic(t, x, R):
    return R*x*(1-x)

def logistic_outbreak(t, x, R, K):
    P = x*x/(1+x*x)
    return R*x*(1-x/K)-P


#2D

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
        

def takens(t, x, A, B):
    return [
        x[1],
        -A-B*x[0]-x[0]*(x[1]*(x[0]+1)+x[0]*(x[0]-1)),
    ]
    

# 3D

def lorenz(t, x, S, P, B):
    return [
        S*(x[1]-x[0]),
        x[0]*(P-x[2])-x[1],
        x[0]*x[1]-B*x[2],
    ]    
    
def nleipnik(t, x, A, B):
    return [
        -A*x[0]+x[1]+10*x[1]*x[2],
        -x[0]-0.4*x[1]+5.0*x[0]*x[2],
        B*x[2]-5*x[0]*x[1],
    ]

def halvorsen(t, x, A):
    return [
        -A*x[0]-4.0*(x[1]+x[2])-x[1]*x[1],
        -A*x[1]-4.0*(x[2]+x[0])-x[2]*x[2],
        -A*x[2]-4.0*(x[0]+x[1])-x[0]*x[0],
    ]   
    
def chenlee(t, x, A, B, C):
    return [
        A*x[0]-x[1]*x[2],
        B*x[1]+x[2]*x[0],
        C*x[2]+x[0]*x[1]/3,
    ]    