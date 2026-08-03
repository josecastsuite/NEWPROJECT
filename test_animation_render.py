"""Render one fill-animation frame with the original mesh overlay."""
import os
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt6 import QtCore
import pyvista as pv

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.types import CastingParameters, BodyType
from ui.flow_animator import FlowAnimator

app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

bodies = load_step('data/Deneme_Ring.STEP')
apply_unit_scale(bodies, 'mm')
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

grid, body_index, origin, dx, bodies = build_voxel_grid(
    bodies, target_dim=160, gravity_vector=(0.0, -1.0, 0.0)
)
params = CastingParameters(
    t_pour_c=1600.0, t_liquidus_c=1510.0, t_solidus_c=1410.0, t_mold_c=25.0,
    t_fill_s=25.0,
    rho_liquid_kg_m3=7850.0, viscosity_pa_s=0.005,
    gravity_vector=(0.0, -1.0, 0.0),
    ingate_velocity_m_s=1.80,
    velocity_section_key='SPRUE_THROAT',
    enable_gate_mesh=True,
)
result = analyze(
    bodies, grid, body_index, origin, dx,
    alloy_key='42CrMo4', mold_key='sand',
    base_res=160, max_res=600, refine_local=False, sub_voxel=2,
    thermal_max_time_s=300, thermal_downsample=2, part_voxels_target=0,
    casting_params=params,
)

pl = pv.Plotter(off_screen=True, window_size=(1024, 768))
for b in bodies:
    if len(b.faces) == 0:
        continue
    faces = np.c_[np.full(len(b.faces), 3, dtype=np.int64), b.faces].ravel()
    pmesh = pv.PolyData(b.vertices, faces)
    pl.add_mesh(pmesh, color='gray', opacity=0.35, show_edges=False, smooth_shading=True, style='wireframe')

anim = FlowAnimator(pl)
anim.set_result(result)

cf = anim._n_fill // 2
anim._current_frame = cf
anim._update_scene()

pl.reset_camera()
pl.view_isometric()

out_path = '/tmp/josecast_test/animation_frame_overlay.png'
pl.screenshot(out_path, return_img=False)
print('saved', out_path)
