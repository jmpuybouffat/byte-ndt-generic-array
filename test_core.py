import math
import numpy as np
from byte_ndt_physics import (
    ArrayConfig, MaterialConfig, SteeringConfig, GridConfig,
    discrete_windows, ferrari2, interface2, delay_laws3Dint,
    T_fluid_solid, calculate_array_field,
)

def test_windows():
    assert np.allclose(discrete_windows(5,'rect'),1.0)
    assert np.isclose(discrete_windows(5,'Han')[0],0.0)
    assert np.allclose(discrete_windows(1,'tri'),[1.0])

def test_identical_media_root():
    xi=ferrari2(1.0,20.0,10.0,15.0)
    assert np.isclose(xi,5.0)
    assert abs(interface2(xi,1.0,20.0,10.0,15.0))<1e-10

def test_zero_delay_law():
    td=delay_laws3Dint(5,5,1.0,1.0,0.0,0.0,0.0,10.0,math.inf,1480.0,5900.0)
    assert td.shape==(5,5)
    assert np.allclose(td,0.0)

def test_transmission_normal():
    tpp,tps=T_fluid_solid(1.0,1480.0,7.9,5900.0,3200.0,0.0)
    assert np.isfinite(tpp)
    assert np.isclose(tps,0.0)

def test_tiny_field():
    result=calculate_array_field(
        ArrayConfig(2,2,0.15,0.15,0.05,0.05,'rect','rect'),
        MaterialConfig(2.0,1.0,1480.0,7.9,5900.0,3200.0,'p'),
        SteeringConfig(0.0,10.0,0.0,0.0,math.inf),
        GridConfig('XZ',-1.0,1.0,2.0,4.0,5,5,0.0,1,1),
    )
    assert result.magnitude.shape==(5,5)
    assert np.isclose(np.max(result.magnitude),1.0)
    assert np.all(np.isfinite(result.magnitude_db))
