from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import math

import numpy as np
from scipy.optimize import brentq

WaveType = Literal['p', 's']
WindowType = Literal['rect', 'cos', 'Han', 'Ham', 'Blk', 'tri']


def sind(a): return np.sin(np.deg2rad(a))
def cosd(a): return np.cos(np.deg2rad(a))
def tand(a): return np.tan(np.deg2rad(a))
def asind(x): return np.rad2deg(np.arcsin(x))


@dataclass(frozen=True)
class ArrayConfig:
    elements_x: int = 11
    elements_y: int = 11
    element_x_mm: float = 0.15
    element_y_mm: float = 0.15
    gap_x_mm: float = 0.05
    gap_y_mm: float = 0.05
    apodization_x: WindowType = 'rect'
    apodization_y: WindowType = 'rect'

    @property
    def pitch_x_mm(self): return self.element_x_mm + self.gap_x_mm
    @property
    def pitch_y_mm(self): return self.element_y_mm + self.gap_y_mm
    @property
    def aperture_x_mm(self): return self.element_x_mm + (self.elements_x - 1) * self.pitch_x_mm
    @property
    def aperture_y_mm(self): return self.element_y_mm + (self.elements_y - 1) * self.pitch_y_mm
    @property
    def active_elements(self): return self.elements_x * self.elements_y
    @property
    def fill_factor_x(self): return self.element_x_mm / self.pitch_x_mm
    @property
    def fill_factor_y(self): return self.element_y_mm / self.pitch_y_mm


@dataclass(frozen=True)
class MaterialConfig:
    frequency_mhz: float = 5.0
    density_1_g_cm3: float = 1.0
    cp1_m_s: float = 1480.0
    density_2_g_cm3: float = 7.9
    cp2_m_s: float = 5900.0
    cs2_m_s: float = 3200.0
    wave_type: WaveType = 'p'

    @property
    def c2_m_s(self): return self.cp2_m_s if self.wave_type == 'p' else self.cs2_m_s
    def wavelength_1_mm(self): return self.cp1_m_s / (1000 * self.frequency_mhz)
    def wavelength_2_mm(self): return self.c2_m_s / (1000 * self.frequency_mhz)


@dataclass(frozen=True)
class SteeringConfig:
    array_angle_deg: float = 10.217
    water_path_mm: float = 50.8
    theta_deg: float = 20.0
    phi_deg: float = 0.0
    focal_distance_mm: float = math.inf


@dataclass(frozen=True)
class GridConfig:
    plane: Literal['XZ', 'YZ', 'XY'] = 'XZ'
    horizontal_min_mm: float = -5.0
    horizontal_max_mm: float = 20.0
    vertical_min_mm: float = 1.0
    vertical_max_mm: float = 20.0
    horizontal_points: int = 30
    vertical_points: int = 30
    fixed_coordinate_mm: float = 0.0
    segments_x: int = 1
    segments_y: int = 1


@dataclass
class FieldResult:
    horizontal_axis_mm: np.ndarray
    vertical_axis_mm: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    magnitude: np.ndarray
    magnitude_db: np.ndarray
    delays_us: np.ndarray
    apodization: np.ndarray
    plane: str
    fixed_coordinate_mm: float
    horizontal_label: str
    vertical_label: str


def discrete_windows(M: int, kind: str) -> np.ndarray:
    if M < 1:
        raise ValueError('M must be >= 1')
    if M == 1:
        return np.ones(1)
    m = np.arange(1, M + 1, dtype=float)
    d = M - 1.0
    if kind == 'cos':
        return np.sin(np.pi * (m - 1) / d)
    if kind == 'Han':
        return np.sin(np.pi * (m - 1) / d) ** 2
    if kind == 'Ham':
        return 0.54 - 0.46 * np.cos(2 * np.pi * (m - 1) / d)
    if kind == 'Blk':
        return 0.42 - 0.5 * np.cos(2 * np.pi * (m - 1) / d) + 0.08 * np.cos(4 * np.pi * (m - 1) / d)
    if kind == 'tri':
        return 1 - np.abs(2 * (m - 1) / d - 1)
    if kind == 'rect':
        return np.ones(M)
    raise ValueError('Unknown apodization')


def interface2(x: float, cr: float, df: float, dp: float, dpf: float) -> float:
    return x / np.sqrt(x*x + dp*dp) - cr * (dpf - x) / np.sqrt((dpf - x)**2 + df*df)


def ferrari2(cr: float, DF: float, DT: float, DX: float) -> float:
    """Physical interface root. Uses the same Snell equation as ferrari2.m.

    The original Ferrari quartic is preserved conceptually, while the robust
    fzero fallback is used directly here through scipy.brentq. This solves the
    same exact equation, without a geometric approximation.
    """
    if DT <= 0 or DF <= 0:
        raise ValueError('DT and DF must be positive')
    if abs(DX) < 1e-15:
        return 0.0
    if abs(cr - 1.0) < 1e-6:
        return DX * DT / (DF + DT)
    a, b = (0.0, DX) if DX > 0 else (DX, 0.0)
    fa = interface2(a, cr, DF, DT, DX)
    fb = interface2(b, cr, DF, DT, DX)
    if abs(fa) < 1e-14: return a
    if abs(fb) < 1e-14: return b
    if fa * fb > 0:
        raise RuntimeError('Physical Snell root is not bracketed')
    return float(brentq(interface2, a, b, args=(cr, DF, DT, DX), xtol=1e-12, rtol=1e-12))


def element_centroids(array: ArrayConfig):
    nx = np.arange(1, array.elements_x + 1)
    ny = np.arange(1, array.elements_y + 1)
    ex = (2*nx - 1 - array.elements_x) * array.pitch_x_mm / 2
    ey = (2*ny - 1 - array.elements_y) * array.pitch_y_mm / 2
    return ex.astype(float), ey.astype(float)


def delay_laws3Dint(Mx, My, sx, sy, thetat, phi, theta2, DT0, DF, c1, c2):
    snell = c1 * sind(theta2) / c2
    if abs(snell) > 1:
        raise ValueError('No real incident ray for requested theta')
    cr = c1 / c2
    ex = (np.arange(1, Mx + 1) - 1 - (Mx - 1)/2) * sx
    ey = (np.arange(1, My + 1) - 1 - (My - 1)/2) * sy
    ang1 = float(asind(snell))
    if np.isinf(DF):
        ux = sind(ang1)*cosd(phi)*cosd(thetat) - cosd(ang1)*sind(thetat)
        uy = sind(ang1)*sind(phi)
        t = 1000 * (ux*ex[:, None] + uy*ey[None, :]) / c1
        return abs(float(np.min(t))) + t
    DQ = DT0*tand(ang1) + DF*tand(theta2)
    tx, ty = DQ*cosd(phi), DQ*sind(phi)
    Db = np.sqrt((tx - ex[:, None]*cosd(thetat))**2 + (ty - ey[None, :])**2)
    De = DT0 + ex*sind(thetat)
    xi = np.empty((Mx, My))
    for m in range(Mx):
        for n in range(My):
            xi[m, n] = ferrari2(cr, DF, float(De[m]), float(Db[m, n]))
    t = np.empty((Mx, My))
    for m in range(Mx):
        for n in range(My):
            t[m, n] = 1000*np.sqrt(xi[m,n]**2 + De[m]**2)/c1 + 1000*np.sqrt(DF**2 + (Db[m,n]-xi[m,n])**2)/c2
    return float(np.max(t)) - t


def T_fluid_solid(d1, cp1, d2, cp2, cs2, theta1):
    iang = np.deg2rad(theta1)
    sinp = (cp2/cp1)*np.sin(iang)
    sins = (cs2/cp1)*np.sin(iang)
    cosp = np.lib.scimath.sqrt(1 - sinp**2)
    coss = np.lib.scimath.sqrt(1 - sins**2)
    ci = np.sqrt(np.maximum(0.0, 1 - np.sin(iang)**2))
    denom = cosp + (d2/d1)*(cp2/cp1)*ci*(4*(cs2/cp2)**2*(sins*coss*sinp*cosp) + 1 - 4*sins**2*coss**2)
    tpp = (2*ci*(1 - 2*sins**2))/denom
    tps = -(4*cosp*sins*ci)/denom
    return tpp, tps


def _broadcast(x, y, z):
    return tuple(np.broadcast_arrays(np.asarray(x,float), np.asarray(y,float), np.asarray(z,float)))


def pts_3Dint(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z):
    x, y, z = _broadcast(x, y, z)
    De = Dt0 + (ex + xn)*sind(angt)
    Db = np.sqrt((x - (ex + xn)*cosd(angt))**2 + (y - (ey + yn))**2)
    out = np.empty_like(Db)
    for idx in np.ndindex(Db.shape):
        out[idx] = ferrari2(c1/c2, float(z[idx]), float(De), float(Db[idx]))
    return out


def ps_3Dint(lx, ly, f, mat: MaterialConfig, ex, ey, angt, Dt0, x, y, z, R=1, Q=1):
    c1, c2 = mat.cp1_m_s, mat.c2_m_s
    k1, k2 = 2000*np.pi*f/c1, 2000*np.pi*f/c2
    xc = -lx/2 + (lx/R)*(np.arange(R)+0.5)
    yc = -ly/2 + (ly/Q)*(np.arange(Q)+0.5)
    x, y, z = _broadcast(x, y, z)
    vx = np.zeros_like(x, dtype=complex); vy = vx.copy(); vz = vx.copy()
    dx, dy = lx/R, ly/Q
    for xn in xc:
        for yn in yc:
            Db = np.sqrt((x-(ex+xn)*cosd(angt))**2 + (y-(ey+yn))**2)
            Ds = Dt0 + (ex+xn)*sind(angt)
            xi = pts_3Dint(ex, ey, float(xn), float(yn), angt, Dt0, c1, c2, x, y, z)
            normal = np.isclose(Db, 0, atol=1e-14)
            ang1 = np.where(normal, 0.0, np.rad2deg(np.arctan2(xi, Ds)))
            ang2 = np.where(np.isclose(ang1,0,atol=1e-14), 0.0, np.rad2deg(np.arctan2(Db-xi, z)))
            r1 = np.sqrt(Ds**2 + xi**2); r2 = np.sqrt((Db-xi)**2 + z**2)
            safe_Db = np.where(normal, 1.0, Db)
            uxt = np.where(normal, -sind(angt), xi*(x-(ex+xn)*cosd(angt))*cosd(angt)/(safe_Db*r1) - Ds*sind(angt)/r1)
            uyt = np.where(normal, 0.0, xi*(y-(ey+yn))/(safe_Db*r1))
            dpx = np.where(normal, 0.0, (1-xi/safe_Db)*(x-(ex+xn)*cosd(angt))/r2)
            dpy = np.where(normal, 0.0, (1-xi/safe_Db)*(y-(ey+yn))/r2)
            dpz = np.where(normal, 1.0, z/r2)
            dsx0 = np.sqrt(dpy**2 + dpz**2); safe_dsx = np.where(np.isclose(dsx0,0), 1.0, dsx0)
            dsx = np.where(normal,1.0,dsx0); dsy = np.where(normal,0.0,-dpx*dpy/safe_dsx); dsz = np.where(normal,0.0,-dpx*dpz/safe_dsx)
            px, py, pz = (dpx,dpy,dpz) if mat.wave_type == 'p' else (dsx,dsy,dsz)
            tpp, tps = T_fluid_solid(mat.density_1_g_cm3, mat.cp1_m_s, mat.density_2_g_cm3, mat.cp2_m_s, mat.cs2_m_s, ang1)
            T = tpp if mat.wave_type == 'p' else tps
            dir_term = np.sinc((k1*uxt*dx/2)/np.pi) * np.sinc((k1*uyt*dy/2)/np.pi)
            ca2 = np.where(np.isclose(cosd(ang2),0), 1e-12, cosd(ang2))
            D1 = r1 + r2*(c2/c1)*(cosd(ang1)/ca2)**2
            D2 = r1 + r2*(c2/c1)
            common = T*dir_term*np.exp(1j*k1*r1 + 1j*k2*r2)/np.lib.scimath.sqrt(D1*D2)
            vx += px*common; vy += py*common; vz += pz*common
    ext = (-1j*k1*dx*dy)/(2*np.pi)
    return vx*ext, vy*ext, vz*ext


def build_grid(grid: GridConfig):
    h = np.linspace(grid.horizontal_min_mm, grid.horizontal_max_mm, grid.horizontal_points)
    v = np.linspace(grid.vertical_min_mm, grid.vertical_max_mm, grid.vertical_points)
    H, V = np.meshgrid(h, v)
    if grid.plane == 'XZ':
        return h, v, H, np.full_like(H, grid.fixed_coordinate_mm), V, ('x (mm)','z (mm)')
    if grid.plane == 'YZ':
        return h, v, np.full_like(H, grid.fixed_coordinate_mm), H, V, ('y (mm)','z (mm)')
    return h, v, H, V, np.full_like(H, grid.fixed_coordinate_mm), ('x (mm)','y (mm)')


def calculate_array_field(array: ArrayConfig, mat: MaterialConfig, steer: SteeringConfig, grid: GridConfig) -> FieldResult:
    h, v, x, y, z, labels = build_grid(grid)
    ex, ey = element_centroids(array)
    delays = delay_laws3Dint(array.elements_x, array.elements_y, array.pitch_x_mm, array.pitch_y_mm, steer.array_angle_deg, steer.phi_deg, steer.theta_deg, steer.water_path_mm, steer.focal_distance_mm, mat.cp1_m_s, mat.c2_m_s)
    phases = np.exp(1j*2*np.pi*mat.frequency_mhz*delays)
    apo = np.outer(discrete_windows(array.elements_x,array.apodization_x), discrete_windows(array.elements_y,array.apodization_y))
    vx = np.zeros_like(x,dtype=complex); vy=vx.copy(); vz=vx.copy()
    for i in range(array.elements_x):
        for j in range(array.elements_y):
            a,b,c = ps_3Dint(array.element_x_mm,array.element_y_mm,mat.frequency_mhz,mat,float(ex[i]),float(ey[j]),steer.array_angle_deg,steer.water_path_mm,x,y,z,grid.segments_x,grid.segments_y)
            coeff = apo[i,j]*phases[i,j]
            vx += coeff*a; vy += coeff*b; vz += coeff*c
    mag = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)
    m = float(np.nanmax(mag))
    if not np.isfinite(m) or m <= 0: raise RuntimeError('Invalid field maximum')
    mag /= m
    db = 20*np.log10(np.maximum(mag,1e-12))
    return FieldResult(h,v,vx,vy,vz,mag,db,delays,apo,grid.plane,grid.fixed_coordinate_mm,labels[0],labels[1])


def threshold_metrics(result: FieldResult, threshold_db: float) -> dict:
    mask = result.magnitude_db >= threshold_db
    if not np.any(mask):
        return {'threshold_db':threshold_db,'points':0,'area_mm2':0.0,'horizontal_span_mm':0.0,'vertical_span_mm':0.0}
    rows, cols = np.where(mask)
    hv = result.horizontal_axis_mm[cols]; vv = result.vertical_axis_mm[rows]
    dh = abs(result.horizontal_axis_mm[1]-result.horizontal_axis_mm[0]) if len(result.horizontal_axis_mm)>1 else 0
    dv = abs(result.vertical_axis_mm[1]-result.vertical_axis_mm[0]) if len(result.vertical_axis_mm)>1 else 0
    return {'threshold_db':threshold_db,'points':int(mask.sum()),'area_mm2':float(mask.sum()*dh*dv),'horizontal_span_mm':float(hv.max()-hv.min()),'vertical_span_mm':float(vv.max()-vv.min())}


def maximum_location(result: FieldResult) -> dict:
    r,c = np.unravel_index(np.argmax(result.magnitude), result.magnitude.shape)
    return {'horizontal_mm':float(result.horizontal_axis_mm[c]),'vertical_mm':float(result.vertical_axis_mm[r])}


def generic_delay_export(array: ArrayConfig, mat: MaterialConfig, steer: SteeringConfig, result: FieldResult):
    ex, ey = element_centroids(array)
    rows=[]; eid=0
    for i in range(array.elements_x):
        for j in range(array.elements_y):
            eid += 1
            rows.append({'sequence_id':1,'element_id':eid,'index_x':i+1,'index_y':j+1,'element_center_x_mm':float(ex[i]),'element_center_y_mm':float(ey[j]),'tx_delay_us':float(result.delays_us[i,j]),'rx_delay_us':float(result.delays_us[i,j]),'apodization_linear':float(result.apodization[i,j]),'active_tx':bool(result.apodization[i,j]>0),'active_rx':bool(result.apodization[i,j]>0),'frequency_mhz':mat.frequency_mhz,'theta_deg':steer.theta_deg,'phi_deg':steer.phi_deg,'focus_distance_mm':None if np.isinf(steer.focal_distance_mm) else steer.focal_distance_mm,'wave_type':mat.wave_type})
    return rows


def configuration_dict(array: ArrayConfig, mat: MaterialConfig, steer: SteeringConfig, grid: GridConfig):
    d={'schema':'byte-ndt.generic-array.v1','units':{'distance':'mm','frequency':'MHz','velocity':'m/s','delay':'microsecond','angle':'degree'},'array':asdict(array),'material':asdict(mat),'steering':asdict(steer),'grid':asdict(grid),'validation':{'matlab_python_numeric_validation':'pending','hardware_adapter':'not included in generic V1'}}
    if math.isinf(d['steering']['focal_distance_mm']): d['steering']['focal_distance_mm']=None
    return d
