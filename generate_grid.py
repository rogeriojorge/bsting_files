import zoidberg as zb
import numpy as np
from boututils.datafile import DataFile
import boututils.calculus as calc
import matplotlib.pyplot as plt

def dommaschk(nx=68,ny=16,nz=128, C=None, xcentre=1.0, Btor=1.0, a=0.1, curvilinear=True, rectangular=False, fname='Dommaschk.fci.nc', curvilinear_inner_aligned=True, curvilinear_outer_aligned=True, npoints=421, show_maps=False, calc_curvature=True, smooth_curvature=False, return_iota=True, write_iota=False):

    if C is None:
        C = np.zeros((6,5,4))
        C[5,2,1] = .4 
        C[5,2,2] = .4 
    elif C == 'Coelho':
        C = np.zeros((6,10,4))
        C[5,2,1] = 1.5
        C[5,2,2] = 1.5
        C[5,4,1] = 10
        C[5,9,0] = -7.5E9
        C[5,9,3] = 7.5E9
    elif C== 'Coelho_noislands':
        C = np.zeros((6,5,4))
        C[5,2,1] = 1.5
        C[5,2,2] = 1.5
        C[5,4,1] = 10
        
    yperiod = 2*np.pi
    field = zb.field.DommaschkPotentials(C, R_0=xcentre, B_0=Btor)
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre+a/2.
    start_z = 0.

    if rectangular:
        print ("Making rectangular poloidal grid")
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(nx, nz, 1.0, 1.0, Rcentre=xcentre)
    elif curvilinear:
        print ("Making curvilinear poloidal grid")
        inner = zb.rzline.shaped_line(R0=xcentre, a=a/2., elong=0, triang=0.0, indent=0, n=npoints)
        outer = zb.rzline.shaped_line(R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints)

        if curvilinear_inner_aligned:
            print ("Aligning to inner flux surface...")
            inner_lines = get_lines(field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints)
        if curvilinear_outer_aligned:
            print ("Aligning to outer flux surface...")
            outer_lines = get_lines(field, xcentre+a, start_z, ycoords, yperiod=yperiod, npoints=npoints)
            
        print ("creating grid...")
        if curvilinear_inner_aligned:
            if curvilinear_outer_aligned:
                poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner, outer in zip(inner_lines, outer_lines) ]
            else:
                poloidal_grid = [ zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps) for inner in inner_lines ]
        else:
            poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz)
    
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps = zb.make_maps(grid, field)

    with zb.zoidberg.MapWriter(fname, metric2d=False) as mw:
        mw.add_grid_field(grid, field)
        mw.add_maps(maps)
        mw.add_dagp()

    if (curvilinear and calc_curvature):
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid)

    if (calc_curvature and smooth_curvature):
        smooth_metric(fname, write_to_file=True, return_values=False, smooth_metric=False)

    if (return_iota or write_iota):
        iota_bar = calc_iota(field, start_r, start_z)
        if (write_iota):
            f = DataFile(str(fname), write=True)
            f.write('iota_bar', iota_bar)
            f.close()
        else:
            print ("Iota_bar = ", iota_bar)

def get_lines(field, start_r, start_z, yslices, yperiod=2*np.pi, npoints=150, smoothing=False):
    rzcoord, ycoords = zb.fieldtracer.trace_poincare(field, start_r, start_z, yperiod, y_slices=yslices, revs=npoints)
    lines = []
    for i in range(ycoords.shape[0]):
        r = rzcoord[:,i,0,0]
        z = rzcoord[:,i,0,1]
        line = zb.rzline.line_from_points(r,z)
        line = line.equallySpaced()
        lines.append(line)
    return lines

def calc_curvilinear_curvature(fname, field, grid):
    f = DataFile(str(fname), write=True)
    B = f.read("B"); R = f.read("R"); Z = f.read("Z"); phi = f.read("phi")
    dx = grid.metric()["dx"]; dz = grid.metric()["dz"]
    g_11 = grid.metric()["g_xx"]; g_22 = grid.metric()["g_yy"]; g_33 = grid.metric()["g_zz"]
    g_13 = grid.metric()["g_xz"]
    GR = np.zeros(B.shape); GZ = np.zeros(B.shape); Gphi = np.zeros(B.shape)
    dRdz = np.zeros(B.shape); dZdz = np.zeros(B.shape); dRdx = np.zeros(B.shape); dZdx = np.zeros(B.shape)
    for y in np.arange(0,B.shape[1]):
        GR[:,y,:] = field.Bxfunc(R[:,y,:],Z[:,y,:],ycoords_func(y, B.shape[1]))/((B[:,y,:])**2)
        GZ[:,y,:] = field.Bzfunc(R[:,y,:],Z[:,y,:],ycoords_func(y, B.shape[1]))/((B[:,y,:])**2)
        Gphi[:,y,:] = field.Byfunc(R[:,y,:],Z[:,y,:],ycoords_func(y, B.shape[1]))/((B[:,y,:])**2)
        for x in np.arange(0,B.shape[0]):
            dRdz[x,y,:] = calc.deriv(R[x,y,:],periodic=False)/dz[x,y,:]
            dZdz[x,y,:] = calc.deriv(Z[x,y,:],periodic=False)/dz[x,y,:]
        for z in np.arange(0,B.shape[-1]):
            dRdx[:,y,z] = calc.deriv(R[:,y,z])/dx[:,y,z]
            dZdx[:,y,z] = calc.deriv(Z[:,y,z])/dx[:,y,z]
    dy = f.read("dy")
    J = R * (dZdz * dRdx - dZdx * dRdz )
    Gx = (GR*dZdz - GZ*dRdz)*(R/J)
    Gz = (GZ*dRdx - GR*dZdx)*(R/J)
    G_x = Gx*g_11 + Gz*g_13
    G_y = Gphi*g_22
    G_z = Gx*g_13 + Gz*g_33
    dG_zdy = np.zeros(B.shape); dG_ydz = np.zeros(B.shape); dG_xdz = np.zeros(B.shape)
    dG_zdx = np.zeros(B.shape); dG_ydx = np.zeros(B.shape); dG_xdy = np.zeros(B.shape)
    for y in np.arange(0,B.shape[1]):
        for x in np.arange(0,B.shape[0]):
            dG_ydz[x,y,:] = calc.deriv(G_y[x,y,:], periodic=False)/dz[x,y,:]
            dG_xdz[x,y,:] = calc.deriv(G_x[x,y,:], periodic=False)/dz[x,y,:]
        for z in np.arange(0,B.shape[-1]):
            dG_ydx[:,y,z] = calc.deriv(G_y[:,y,z])/dx[:,y,z]
            dG_zdx[:,y,z] = calc.deriv(G_z[:,y,z])/dx[:,y,z]
    for x in np.arange(0,B.shape[0]):
        for z in np.arange(0,B.shape[-1]):
            dG_zdy[x,:,z] = calc.deriv(G_z[x,:,z])/dy[x,:,z]
            dG_xdy[x,:,z] = calc.deriv(G_x[x,:,z])/dy[x,:,z]
    bxcvx = (dG_zdy - dG_ydz)/J
    bxcvy = (dG_xdz - dG_zdx)/J
    bxcvz = (dG_ydx - dG_xdy)/J
    bxcv = np.sqrt( g_11*(bxcvx**2) + g_22*(bxcvy**2) + g_33*(bxcvz**2) + 2*(bxcvz*bxcvx*g_13) )
    f.write('bxcvx', bxcvx); f.write('bxcvy', bxcvy); f.write('bxcvz', bxcvz); f.write('bxcv', bxcv)
    f.close()

def ycoords_func(y, ny):
    return (2*np.pi / ny) * y

def smooth_metric(fname, write_to_file=False, return_values=False, smooth_metric=True, order=7):
    from scipy.signal import savgol_filter
    f = DataFile(str(fname),write=True)
    bxcvx = f.read('bxcvx'); bxcvz = f.read('bxcvz'); bxcvy = f.read('bxcvy'); J = f.read('J')
    bxcvx_smooth = np.zeros(bxcvx.shape); bxcvy_smooth = np.zeros(bxcvy.shape); bxcvz_smooth = np.zeros(bxcvz.shape); J_smooth = np.zeros(J.shape)
    for y in np.arange(0,bxcvx.shape[1]):
        for x in np.arange(0,bxcvx.shape[0]):
            win = int(np.ceil(bxcvx.shape[-1]/2)//2*2+1)
            bxcvx_smooth[x,y,:] = savgol_filter(bxcvx[x,y,:],win,order)
            bxcvz_smooth[x,y,:] = savgol_filter(bxcvz[x,y,:],win,order)
            bxcvy_smooth[x,y,:] = savgol_filter(bxcvy[x,y,:],win,order)
            J_smooth[x,y,:] = savgol_filter(J[x,y,:],win,order)
    if(write_to_file):
        f.write('J',J_smooth)
    f.close()

def calc_iota(field, start_r, start_z):
    from scipy.signal import argrelextrema
    toroidal_angle = np.linspace(0.0, 400*np.pi, 10000, endpoint=False)
    result = zb.fieldtracer.FieldTracer.follow_field_lines(field,start_r,start_z,toroidal_angle)
    peaks = argrelextrema(result[:,0,0], np.greater, order=10)[0]
    iota_bar = 2*np.pi/(toroidal_angle[peaks[1]]-toroidal_angle[peaks[0]])
    return iota_bar

if __name__ == '__main__':
    # Using the default parameters from the function definition
    dommaschk(nx=68, ny=16, nz=128, fname='Dommaschk.fci.nc')
