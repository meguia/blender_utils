import numpy as np
import blender_methods as bm
import material_utils as mu
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from scipy.fft import fft
import importlib as imp

imp.reload(bm)

def solve(func,t,x0,method='DOP853',args=None):
    dt = np.abs(t[1]-t[0])
    sol = solve_ivp(func, t[[0,-1]], x0, method=method, t_eval=t, args=args,max_step=dt,dense_output=True)
    if sol.status < 0:
        print(sol.message)
    return sol.y.T

def findperiod(t,x):
    peaks, _ = find_peaks(x)
    per = np.diff(t[peaks])
    return np.mean(per)

def length_xycurve(x,y):
    ''' find legth of planar curve given by arrays x,y
    '''
    s = 0
    for n in range(len(x)-1):
        dx = x[n+1]-x[n]
        dy = y[n+1]-y[n]
        ds = np.sqrt(dx*dx+dy*dy)
        s += ds
    return s    

def length_xyzcurve(x,y,z):
    ''' find legth of planar curve given by arrays x,y
    '''
    s = 0
    for n in range(len(x)-1):
        dx = x[n+1]-x[n]
        dy = y[n+1]-y[n]
        dz = z[n+1]-z[n]
        ds = np.sqrt(dx*dx+dy*dy+dz*dz)
        s += ds
    return s       

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

def axes1C(name='cycle',radius=1):
    #creates a 1D plot over the circle
    pln = bm.rectangle(name, 2.2*radius, 2.2*radius, origin=[-1.1*radius,-1.1*radius])
    dticks = radius/8.0
    lwidth = radius/1000 # line width 1/1000 of plot
    xaxis = bm.box(name+'_xaxis', dims=[radius*1.1,lwidth,lwidth], origin=[0,0,0])
    caxis = bm.annulus(name+'_cxis',N=128,r1=radius,r2=radius-lwidth,pos=[0,0,lwidth])    
    xaxis.parent=pln
    caxis.parent=pln    
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
    print(ylim)
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

def solve_plot1D_circle(syst,pars,xini,tmax,dt,sfp=[],ufp=[],dtframe=3,method='RK45'):
    ''' creates an animated plot of array x in the circle
    '''
    pltname='cycle'
    t = np.arange(0, tmax, dt)
    s = solve(syst, t, xini, args=pars, method='RK45') 
    print('system solved')
    t = t[::dtframe]
    x = s[::dtframe]
    frames = np.arange(len(x))
    rad = 1.0
    tscale = 2*rad/tmax
    pln = axes1C(name=pltname,radius=rad)
    plx = plot2D_animated(t*tscale,np.cos(x),frames,'plotx',ylim=[-rad*1.1,rad*1.1])
    plx.location = [0,-1.5*rad,0]
    plx.rotation_euler = [0,0,-np.pi/2]
    lwidth = rad/1000 
    for p in sfp:
        sp = bm.cylinder(pltname + '_sfp', r=lwidth*10, h=lwidth*5, pos=[rad*np.cos(p),rad*np.sin(p),0])
        sp.parent = pln
    for p in ufp:
        up = bm.box(pltname+'_ufp', dims=[lwidth*40,lwidth,lwidth], origin=[-lwidth*20,0,0],pos=[rad*np.cos(p),rad*np.sin(p),0],rot=[0,0,p]) 
        up.parent = pln
    # create a dot following circle x is teh angle in radians
    dot = bm.cylinder(pltname + '_dot', r=lwidth*10, h=lwidth*5, pos=[rad*np.cos(x[0]),rad*np.sin(x[0]),0])
    dot.parent = pln
    fkeys = [[rad*np.cos(x[n]),rad*np.sin(x[n]),0] for n in range(len(x))]
    bm.animate_curve(dot,pltname+'_dotanim','location',frames,fkeys)
    # and a dot hand
    hand =  bm.box(pltname+'_hand', dims=[rad,lwidth,lwidth])       
    hand.parent = pln
    fkeys = [[0,0,x[n]] for n in range(len(x))]
    bm.animate_curve(hand,pltname+'_handanim','rotation_euler',frames,fkeys)
    return pln,plx

def solve_plot2D_torus(syst,pars,xini,tmax,dt,dtframe=3,method='RK45'):
    ''' creates an animated plot of array x,y in the torus
    '''
    pltname='torus'
    t = np.arange(0, tmax, dt)
    s = solve(syst, t, xini, args=pars, method='RK45') 
    print('system solved')
    t = t[::dtframe]
    a1 = s[::dtframe,0]
    a2 = s[::dtframe,1]
    frames = np.arange(len(t))
    r1 = 1.0
    r2 = 0.2
    pln = bm.torus('torus',r1,r2,64,32)
    lwidth = r1/1000 
    # curve following torus
    x = [r1*np.cos(a1[n])+r2*np.cos(a1[n])*np.cos(a2[n]) for n in range(len(t))]
    y = [r1*np.sin(a1[n])+r2*np.sin(a1[n])*np.cos(a2[n]) for n in range(len(t))]
    z = [r2*np.sin(a2[n]) for n in range(len(t))]
    #creates bezier curve with data 
    pts = [[x[n],y[n],z[n]] for n in range(len(x))]
    plt = bm.smooth_bezier(pltname+'_curve',pts,bevel=lwidth)
    bm.animate_curve(plt.data,pltname+'_pltanim','bevel_factor_end',[0,len(x)],[0,1])
    plt.parent = pln    
    # create a dot following (x,y)
    dot = bm.icosphere(pltname + '_dot', r = lwidth*5, sub = 2)
    dot.parent = pln
    fkeys = [[x[n],y[n],z[n]] for n in range(len(x))]
    bm.animate_curve(dot,pltname+'_dotanim','location',frames,fkeys)      
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

def nullcline_regions(syst,pars,delta,xlim=[-1,1],ylim=[-1,1]):
    '''
    
    '''

def plot2D_flow(syst,pars,xini_array,tmax,pv=[0,1],dt=0.005,dtframe=3,xlim=[-1,1],ylim=[-1,1],arrow_scale=10,count=20):
    '''creates a static plot of the vector field as a flow with arrows indicating the time evolution
    along the orbits
    '''
    t = np.arange(0, tmax, dt)
    #creates plane xy for plot at origin 
    pln = axes2D(xlim=xlim,ylim=ylim,ticks='auto')
    xrange = xlim[1]-xlim[0]
    yrange = ylim[1]-ylim[0]    
    l = min(xrange,yrange)
    lwidth = l/10000 
    # create an arrow model
    al = lwidth*arrow_scale
    arr0 = bm.arrow('arrow', al, al*1.5, al, lwidth,axis='X')
    bm.cylinder('dot', r=lwidth*10, h=lwidth*5, pos=[0,0,0])    
    # Nullclines!
    for m,xini in enumerate(xini_array):
        print(m)
        s = solve(syst, t, xini, args=pars, method='RK45') 
        x = s[::dtframe,pv[0]]
        y = s[::dtframe,pv[1]]
        s = length_xycurve(x,y)
        ds = s/count
        pts = [[x[n],y[n],0] for n in range(len(x))]
        plt = bm.smooth_bezier('plot_'+ str(m),pts,bevel=lwidth)
        arr = bm.duplicate_ob(arr0,'arr_'+str(m))
        a1, c1 = bm.array_curve(arr,plt,'arrf_'+str(m),count=count,axis='POS_X',off_constant=[ds,0,0])
        #animate curve
        # arrows
        arr.parent = pln
        plt.parent = pln    
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
        dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=10)
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

def plot1D_flux_circle(syst,pars,xini_array,tmax,cmap_path,dt,sfp=[],ufp=[],dtframe=3,method='RK45'):
    ''' creates an animated plot of array x in the circle
    '''
    pltname='cycle'
    xrange = 2*np.pi
    rad = 1.0
    tscale = 2*rad/tmax
    pln = axes1C(name=pltname,radius=rad)
    lwidth = rad/1000 
    for p in sfp:
        sp = bm.cylinder(pltname + '_sfp', r=lwidth*10, h=lwidth*5, pos=[rad*np.cos(p),rad*np.sin(p),0])
        sp.parent = pln
    for p in ufp:
        up = bm.box(pltname+'_ufp', dims=[lwidth*40,lwidth,lwidth], origin=[-lwidth*20,0,0],pos=[rad*np.cos(p),rad*np.sin(p),0],rot=[0,0,p]) 
        up.parent = pln
    # create a dot following circle x is teh angle in radians
    dot0 = bm.cylinder('dot', r=lwidth*10, h=lwidth*5)
    for m,xini in enumerate(xini_array):
        t = np.arange(0, tmax, dt)
        print(m)
        # define material for orbit (dim) and point (bright)
        coord = [1-xini[0]/xrange,0]
        dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=10)
        pmat = mu.colormap_material('pmat'+str(m),coord,cmap_path,emission=True,estrength=0.1)
        s = solve(syst, t, xini, args=pars,method=method) 
        t = t[::dtframe]
        x = s[::dtframe]    
        frames = np.arange(len(x))
        if m==0:
            plx = axes2D(name='plotx',xlim=[-0.1*rad,2.1*rad],ylim=[-1.1*rad,1.1*rad],ticks='auto')
            plx.location = [0,-1.5*rad,0]
            plx.rotation_euler = [0,0,-np.pi/2]     
        pts = [[t[n]*tscale,np.cos(x[n]),0] for n in range(len(x))]           
        plt = bm.smooth_bezier('plot_'+ str(m),pts,bevel=lwidth)
        #animate curve
        bm.animate_curve(plt.data,'pltanim_'+str(m),'bevel_factor_end',[0,len(x)],[0,1])
        plt.parent = plx    
        # assigns color
        plt.data.materials.append(pmat)    
        dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
        dot.parent = pln
        fkeys = [[rad*np.cos(x[n]),rad*np.sin(x[n]),0] for n in range(len(x))]
        bm.animate_curve(dot,pltname+'_dotanim_'+str(m),'location',frames,fkeys)
        dot.data.materials.append(dmat)
        dot.material_slots[0].link = 'OBJECT'
        dot.material_slots[0].material = dmat
        # and a dot hand
        dot2 = bm.duplicate_linked_ob(dot0,'dot2_'+str(m))
        dot2.parent = plx
        fkeys = [[t[n]*tscale,rad*np.cos(x[n]),0] for n in range(len(x))]
        bm.animate_curve(dot2,pltname+'_dotanim2_'+str(m),'location',frames,fkeys)        
        dot2.data.materials.append(dmat)
        dot2.material_slots[0].link = 'OBJECT'
        dot2.material_slots[0].material = dmat
    return pln,plx

def plot1D_bifurcation(syst,par_list,xini_array,tmax,cmap_path,dt,dtframe,xlim,parlim,sp=[],up=[]):
    xrange = xlim[1]-xlim[0]
    parange = parlim[1]-parlim[0]
    pscale = xrange/parange
    t = np.arange(0, tmax, dt)
    parcor = np.linspace(parlim[0]*pscale,parlim[1]*pscale,len(par_list))
    #creates plane x f(x) for plot at origin 
    pln = axes2D(xlim=[parlim[0]*pscale,parlim[1]*pscale],ylim=xlim,ticks='auto')
    lwidth = xrange/5000
    #draw stable and unstable FP curves
    for c in sp:
        pts = [[c[n][0]*pscale,c[n][1],0] for n in range(len(c))]
        spc = bm.smooth_bezier('spc',pts,bevel=lwidth)
        spc.parent = pln
        pmat1 = mu.simple_material('pmat1',[1,1,1,1],emission=[0,0,1,1],estrength=1)
        spc.data.materials.append(pmat1)
    for c in up:
        pts = [[c[n][0]*pscale,c[n][1],0] for n in range(len(c))]
        upc = bm.smooth_bezier('upc',pts,bevel=lwidth)
        upc.parent = pln
        pmat2 = mu.simple_material('pmat2',[1,1,1,1],emission=[0.5,0,0,1],estrength=1)
        upc.data.materials.append(pmat2)    
    # create a dot model
    dot0 = bm.cylinder('dot', r=lwidth*10, h=lwidth*20, pos=[0,0,0])    
    # loop over initial conditions]
    xmin = min(xini_array)
    xmax = max(xini_array)
    xrange = [x1 - x2 for (x1, x2) in zip(xmax, xmin)]
    for npar,pars in enumerate(par_list):
        print(pars)
        xaxis = bm.box('xaxis_'+str(npar), dims=[lwidth,xrange[0],lwidth], origin=[parcor[npar],xlim[0],0])
        xaxis.parent = pln
        for m,xini in enumerate(xini_array):
            # define material for orbit (dim) and point (bright)
            coord = [1-(xini[0]-xmin[0])/xrange[0],0]
            dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=10)
            s = solve(syst, t, xini, args=pars, method='RK45') 
            x = s[::dtframe]
            frames = np.arange(len(x))
            dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
            dot.parent = pln
            fkeys = [[parcor[npar],x[n],0] for n in range(len(x))]
            frames = np.arange(len(x))
            bm.animate_curve(dot,'dotanim_'+str(m)+'_'+str(npar),'location',frames,fkeys)
            dot.data.materials.append(dmat)
            dot.material_slots[0].link = 'OBJECT'
            dot.material_slots[0].material = dmat
    return pln

def plot1D_bifurcation_codimension2(syst,par_list,xini_array,tmax,cmap_path,dt,dtframe,xlim,parlim,sp=[],up=[]):
    """par_list is a list of list of tuples, for example for codim 2 bifurcations [[(a1,b1,c),(a2,b1,c)],[(a1,b2,c),(a2,b2,c)]]
    sp and up are the stable and unstable manifolds are 3D lists: an array of manifolds written as an array of curves (arrays of points)
    the 1D phase space is oriented along the z axis
    parlim is [[amin,amax],[bmin,bmax]]
    """
    xrange = xlim[1]-xlim[0]
    pscale1 = xrange/(parlim[0][1]-parlim[0][0])
    pscale2 = xrange/(parlim[1][1]-parlim[1][0])
    t = np.arange(0, tmax, dt)
    parcor1 = np.linspace(parlim[0][0]*pscale1,parlim[0][1]*pscale1,len(par_list))
    parcor2 = np.linspace(parlim[1][0]*pscale2,parlim[0][1]*pscale2,len(par_list[0]))
    #creates plane x f(x) for plot at origin 
    pln = axes2D(xlim=[parlim[0][0]*pscale1,parlim[0][1]*pscale1],ylim=[parlim[1][0]*pscale2,parlim[1][1]*pscale2],ticks='auto')
    lwidth = xrange/5000
    #draw stable and unstable FP curves
    # loop sobre manifolds
    for man in sp:
        # loop sobre curvas
        for c in man: 
            if len(c) > 1:
                pts = [[c[n][0]*pscale1,c[n][1]*pscale2,c[n][2]] for n in range(len(c))]
                spc = bm.smooth_bezier('spc',pts,bevel=lwidth)
                spc.parent = pln
                pmat1 = mu.simple_material('pmat1',[1,1,1,1],emission=[0,0,1,1],estrength=2)
                spc.data.materials.append(pmat1)
    for man in up:
        for c in man:
            if len(c)>1:
                pts = [[c[n][0]*pscale1,c[n][1]*pscale2,c[n][2]] for n in range(len(c))]
                upc = bm.smooth_bezier('upc',pts,bevel=2*lwidth)
                upc.parent = pln
                pmat2 = mu.simple_material('pmat2',[1,1,1,1],emission=[0.5,0,0,1],estrength=1)
                upc.data.materials.append(pmat2)    
            # create a dot model
    dot0 = bm.icosphere('dot', r=lwidth*10, sub = 1) 
    # loop over initial conditions]
    xmin = min(xini_array)
    xmax = max(xini_array)
    xrange = [x1 - x2 for (x1, x2) in zip(xmax, xmin)]
    for npar1,subparlist in enumerate(par_list):
        for npar2,pars in enumerate(subparlist):
            print(pars)
            xaxis = bm.box('xaxis_'+str(npar1)+'_'+str(npar2), dims=[lwidth,lwidth,xrange[0]], origin=[parcor1[npar1],parcor2[npar2],xlim[0]])
            xaxis.parent = pln
            for m,xini in enumerate(xini_array):
                # define material for orbit (dim) and point (bright)
                coord = [1-(xini[0]-xmin[0])/xrange[0],0]
                dmat = mu.colormap_material('dmat'+str(m),coord,cmap_path,emission=True,estrength=10)
                s = solve(syst, t, xini, args=pars, method='RK45') 
                x = s[::dtframe]
                frames = np.arange(len(x))
                dot = bm.duplicate_linked_ob(dot0,'dot_'+str(m))
                dot.parent = pln
                fkeys = [[parcor1[npar1],parcor2[npar2],x[n]] for n in range(len(x))]
                frames = np.arange(len(x))
                bm.animate_curve(dot,'dotanim_'+str(m)+'_'+str(npar1)+'_'+str(npar2),'location',frames,fkeys)
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

#1D Bifurcations

def saddlenode(t, x, a):
    return x*x+a

def saddlenode_fp(a):
    if a > 0 :
        return [np.sqrt(a),-np.sqrt(a)]

def transcritical(t, x, a):
    return x*(a-x)

def transcritical_fp(a):
    if a>0:
        return [0, a]
    else:
        return [a, 0]
    
def pitchfork(t, x, a):
    return x*(a-x*x)

def pitchfork_fp(a):
    if a>0:
        return [np.sqrt(a), 0, -np.sqrt(a)]
    else:
        return [0]

def cusp(t,x,r,h):        
    return h + x*(r-x*x)


def cusp_fp(x,r,hrange):
    h = x*(x*x-r)
    if h<hrange[1] and h>hrange[0]:
        return h
    
        
    
    
# 1D
    
def logistic(t, x, R):
    return R*x*(1-x)

def logistic_outbreak(t, x, R, K):
    P = x*x/(1+x*x)
    return R*x*(1-x/K)-P

# circle
def adler(t, x, w, a):
    return w-a*np.cos(x)


# circle forced

def adler_forced(t, x, w, a, p, w1):
    return [
        w-a*np.cos(x[0])+p*np.cos(x[1]),
        w1,
    ]


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
        
def van_der_pol_hopf(t,x,W,C):
    return [
        x[1],
        -W*x[0]-x[1]*(x[0]*x[0]-C),
    ]
    
    
def lienard(t,x,C):
    return [
        C*(x[0]*(1-x[0]*x[0]/3)-x[1]),
        x[0]/C,
    ]    

def takens(t, x, A, B):
    return [
        x[1],
        -A-B*x[0]-x[0]*(x[1]*(x[0]+1)+x[0]*(x[0]-1)),
    ]

def duffing(t, x, B, C):
    return [
        x[1],
        -C*x[1]+x[0]*(B-x[0]*x[0]),
    ]

def bow(t, x, C, V):
    return [
        x[1],
        -x*friction(x[1]-V)-x[0],
    ]    

def bow_trans(t, x, C, V, Vt):
    return [
        x[1],
        -C*friction(x[1]-x[2])-x[0],
        -Vt*(x[2]-V),
    ]
    
def friction(x):
    return np.arctan(25*x)*np.exp(-2*np.abs(x))

# 2D forced

def duffing_forced(t, x, B, C, A, w):
    return [
        x[1],
        -C*x[1]+x[0]*(B-x[0]*x[0])+A*np.cos(x[2]),
        w, 
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