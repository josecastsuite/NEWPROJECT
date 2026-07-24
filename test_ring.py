import os, numpy as np, time
os.environ['QT_QPA_PLATFORM']='offscreen'
from PyQt6 import QtCore
from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.types import CastingParameters, BodyType
from ui.flow_animator import FlowAnimator

app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

class MockViewer:
    def remove_actor(self, a): pass
    def add_mesh(self, mesh, **kw):
        class A:
            class M:
                def __init__(self, ds): self.dataset=ds
            def __init__(self, ds): self.mapper=self.M(ds)
        return A(mesh)
    def render(self): pass

bodies = load_step('data/Deneme_Ring.STEP')
apply_unit_scale(bodies, 'mm')
# Assign body types based on image / naming convention
types = {
    0: BodyType.PART,
    1: BodyType.INGATE,
    2: BodyType.INGATE,
    3: BodyType.INGATE,
    4: BodyType.INGATE,
    5: BodyType.SPRUE,
    6: BodyType.SPRUE_THROAT,
    7: BodyType.RISER,
    8: BodyType.RISER,
}
for i, b in enumerate(bodies):
    b.body_type = types.get(i, BodyType.PART)
    print(i, b.name, b.body_type)

grid, body_index, origin, dx, bodies = build_voxel_grid(bodies, target_dim=160, gravity_vector=(0.0,-1.0,0.0))
params=CastingParameters(
    t_pour_c=1600.0, t_liquidus_c=1510.0, t_solidus_c=1410.0, t_mold_c=25.0,
    t_fill_s=25.0,
    rho_liquid_kg_m3=7850.0, viscosity_pa_s=0.005,
    gravity_vector=(0.0,-1.0,0.0),
    ingate_velocity_m_s=1.80,
    velocity_section_key='SPRUE_THROAT'
)
result = analyze(
    bodies, grid, body_index, origin, dx,
    alloy_key='42CrMo4', mold_key='sand',
    base_res=160, max_res=600, refine_local=False, sub_voxel=2,
    thermal_max_time_s=300, thermal_downsample=2, part_voxels_target=0,
    casting_params=params
)
print('flow result:', result.flow_result)
fr = result.flow_result
print('Q L/s', fr.Q_m3_s*1e3, 'fill_time_s', fr.fill_time_s, 'inlet_area_cm2', fr.inlet_area_m2*1e4)
print('node_velocities', fr.node_velocities)

anim = FlowAnimator(MockViewer())
t0=time.time()
anim.set_result(result)
print('precompute time', time.time()-t0)
print('frames', anim.frame_count(), 'n_fill', anim._n_fill, 'n_solid', anim._n_solid)
print('t_mold', anim._t_mold, 't_pour', anim._t_pour, 'clim', (anim._t_mold, anim._t_pour))

# Render a fill frame in the middle
cf = anim._n_fill // 2
anim._current_frame = cf
anim._update_scene()
print('fill frame', cf, 'time', anim._current_time)
