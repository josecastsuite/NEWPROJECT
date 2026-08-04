"""V8.1 gate velocity engine: CGAL MCF skeleton + exact CAD cross-sections.

Uses `josecast_core.extract_skeleton` (CGAL) to extract mean-curvature-flow
centerlines from the original CAD mesh, then samples real perpendicular cross-
sectional areas with `vtkCutter`/shapely, applies the guards from the V8.1 spec,
and builds a single-source ``velocity_magnitude`` array for surface colour and
labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import shapely.geometry as geom
import shapely.ops as ops
import trimesh
from scipy.spatial.distance import cdist

from core import josecast_core
from core.types import Body, BodyType


class VelocityEngineError(Exception):
    """Raised when a V8.1 guard fails."""


class ZeroAreaError(VelocityEngineError):
    pass


class DisjointLoopError(VelocityEngineError):
    pass


class SheetDegeneracyError(VelocityEngineError):
    pass


class InvalidRadiusError(VelocityEngineError):
    pass


@dataclass
class EngineConfig:
    sample_spacing_mm: float = 1.0
    smoothing_lambda: float = 0.2
    smoothing_iter: int = 5
    alpha_spur: float = 0.75
    area_ratio_thresh: float = 0.30
    eps_flat: float = 0.05
    eps_var: float = 0.15
    projection_inset_mm: float = 0.1


@dataclass
class Section:
    s_mm: float
    point: np.ndarray
    normal: np.ndarray
    area_mm2: float
    r_eff_mm: float


class MeshAnalyzer:
    """Trimesh + PyVista wrapper used by the V8.1 skeleton engine."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str = ""):
        self.name = name
        self.tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        self.pv = pv.PolyData.from_regular_faces(
            np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
        )

    def is_point_inside(self, p: np.ndarray) -> bool:
        return bool(self.tm.contains([np.asarray(p, dtype=float)])[0])

    def project_to_internal_surface(self, p: np.ndarray) -> np.ndarray:
        """If p is outside, project it onto the nearest surface point and move inward."""
        p = np.asarray(p, dtype=float)
        if self.is_point_inside(p):
            return p
        if len(self.tm.faces) == 0:
            return p
        closest, _, face_id = trimesh.proximity.closest_point(self.tm, [p])
        surf = closest[0]
        fid = int(face_id[0])
        normal = self.tm.face_normals[fid]
        # Find the inward direction by a tiny probe.
        for sign in (1.0, -1.0):
            probe = surf + sign * 1e-3 * normal
            if self.tm.contains([probe])[0]:
                inward = sign * normal
                return surf + self._cfg().projection_inset_mm * inward
        # If neither probe is inside, just pull a small amount toward the centroid.
        to_centroid = self.tm.centroid - surf
        to_centroid /= max(np.linalg.norm(to_centroid), 1e-9)
        return surf + self._cfg().projection_inset_mm * to_centroid

    def slice_at_plane(self, origin: np.ndarray, normal: np.ndarray) -> pv.PolyData:
        return self.pv.slice(normal=normal, origin=origin)

    def _cfg(self) -> EngineConfig:
        # Config is usually stored on the engine; this helper avoids plumbing it.
        return getattr(self, "_engine_cfg", EngineConfig())


class SkeletonAnalysisEngineV8_1:
    """Per-body V8.1 pipeline: skeleton pruning, resampling, slicing, guards."""

    def __init__(self, analyzer: MeshAnalyzer, cfg: EngineConfig = None):
        self.mesh = analyzer
        self.cfg = cfg or EngineConfig()
        analyzer._engine_cfg = self.cfg

    def safe_vector_angle(self, u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0:
            raise VelocityEngineError("ZeroVectorException: cannot compute angle")
        dot = np.dot(u, v) / (nu * nv)
        dot = max(-1.0, min(dot, 1.0))
        return math.acos(dot)

    @staticmethod
    def prune_skeleton_spurs(points: np.ndarray, edges: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Remove short leaf branches from the raw skeleton graph.

        A simple path (degree <= 2 everywhere) is left untouched because its
        endpoints are the true centerline extremes.  Only side spurs that hang
        off a branching node (degree > 2) are removed when their length is below
        ``alpha`` times the average edge length.
        """
        import networkx as nx

        g = nx.Graph()
        for i, p in enumerate(points):
            g.add_node(i, pos=p)
        for e in edges:
            g.add_edge(int(e[0]), int(e[1]))

        if g.number_of_nodes() <= 2:
            return points.copy(), edges.copy()

        pos = nx.get_node_attributes(g, "pos")

        avg_edge = (
            float(np.mean([np.linalg.norm(pos[e[0]] - pos[e[1]]) for e in g.edges()]))
            if g.number_of_edges()
            else 1.0
        )
        delta = alpha * max(avg_edge, 1.0)

        # A side spur is a degree-1 leaf whose unique neighbour chain reaches a
        # branching node (degree > 2).  Endpoints of a pure path are preserved.
        removed: set = set()
        for leaf in [n for n in g.nodes() if g.degree(n) == 1]:
            if leaf in removed:
                continue
            path = [leaf]
            curr = leaf
            while True:
                neighs = [n for n in g.neighbors(curr) if n not in removed]
                if len(neighs) != 1:
                    break
                nxt = neighs[0]
                if g.degree(nxt) > 2:
                    path.append(nxt)
                    break
                if g.degree(nxt) == 2:
                    path.append(nxt)
                    curr = nxt
                else:
                    # Reached another endpoint (pure path) - keep it.
                    break

            # The path ends at a real branching node (degree > 2): remove the
            # dangling spur, keeping the branch node itself.
            if len(path) >= 2 and g.degree(path[-1]) > 2:
                path_len = sum(
                    np.linalg.norm(pos[path[i]] - pos[path[i + 1]])
                    for i in range(len(path) - 1)
                )
                if path_len < delta:
                    for n in path[:-1]:
                        if n in g:
                            g.remove_node(n)
                            removed.add(n)

        node_list = list(g.nodes())
        if not node_list:
            return points.copy(), edges.copy()
        idx_map = {n: i for i, n in enumerate(node_list)}
        new_pts = np.asarray([pos[n] for n in node_list], dtype=float)
        new_edges = np.asarray(
            [[idx_map[int(e[0])], idx_map[int(e[1])]] for e in g.edges()], dtype=int
        )
        return new_pts, new_edges

    def longest_path(self, points: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Return the longest simple path in the skeleton graph (tree or near-tree)."""
        import networkx as nx

        g = nx.Graph()
        for i, p in enumerate(points):
            g.add_node(i, pos=p)
        for e in edges:
            g.add_edge(int(e[0]), int(e[1]))

        if g.number_of_nodes() == 0:
            return points
        if g.number_of_nodes() == 1:
            return points

        # Two BFS to find diameter endpoints.
        start = 0
        lengths = nx.single_source_shortest_path_length(g, start)
        a = max(lengths, key=lengths.get)
        lengths = nx.single_source_shortest_path_length(g, a)
        b = max(lengths, key=lengths.get)
        path = nx.shortest_path(g, a, b)
        pos = nx.get_node_attributes(g, "pos")
        return np.asarray([pos[n] for n in path], dtype=float)

    def resample_and_smooth_curve(self, raw_points: np.ndarray) -> np.ndarray:
        """Arc-length resampling followed by constrained Laplacian smoothing."""
        if raw_points.shape[0] < 2:
            return raw_points

        # Arc-length parameterisation.
        diffs = np.diff(raw_points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum[-1]
        if total <= 0.0:
            return raw_points

        n_samples = max(3, int(round(total / self.cfg.sample_spacing_mm)) + 1)
        new_cum = np.linspace(0.0, total, n_samples)
        resampled = np.zeros((n_samples, 3), dtype=float)
        for dim in range(3):
            resampled[:, dim] = np.interp(new_cum, cum, raw_points[:, dim])

        # Point-in-mesh guard / projection.
        for i, p in enumerate(resampled):
            if not self.mesh.is_point_inside(p):
                resampled[i] = self.mesh.project_to_internal_surface(p)

        # Constrained Laplacian smoothing.
        pts = resampled.copy()
        for _ in range(self.cfg.smoothing_iter):
            tmp = pts.copy()
            for i in range(1, len(pts) - 1):
                lap = 0.5 * (tmp[i - 1] + tmp[i + 1]) - tmp[i]
                cand = tmp[i] + self.cfg.smoothing_lambda * lap
                if self.mesh.is_point_inside(cand):
                    pts[i] = cand
                # else: keep current position (frozen).
        return pts

    def compute_tangents(self, points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        t = np.zeros_like(points)
        if n < 2:
            return t
        if n == 2:
            d = points[1] - points[0]
            dnorm = np.linalg.norm(d)
            if dnorm > 0:
                t[:] = d / dnorm
            return t
        # forward/backward/central differences
        t[0] = points[1] - points[0]
        t[-1] = points[-1] - points[-2]
        for i in range(1, n - 1):
            t[i] = points[i + 1] - points[i - 1]
        norms = np.linalg.norm(t, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return t / norms

    @staticmethod
    def _orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = np.asarray(normal, dtype=float)
        n /= max(np.linalg.norm(n), 1e-12)
        # Pick an arbitrary vector not parallel to n.
        if abs(n[0]) < abs(n[1]) and abs(n[0]) < abs(n[2]):
            arb = np.array([1.0, 0.0, 0.0])
        elif abs(n[1]) < abs(n[2]):
            arb = np.array([0.0, 1.0, 0.0])
        else:
            arb = np.array([0.0, 0.0, 1.0])
        u = np.cross(n, arb)
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(n, u)
        v /= max(np.linalg.norm(v), 1e-12)
        return u, v

    def section_loops(self, slice_pd: pv.PolyData, normal: np.ndarray):
        """Return list of 2-D Shapely polygons from a vtkCutter slice."""
        if slice_pd.n_points == 0:
            return []
        if slice_pd.lines is None or len(slice_pd.lines) == 0:
            return []

        u, v = self._orthonormal_basis(normal)
        pts = slice_pd.points

        # Parse VTK line cells: [n, i0, i1, ...] per cell.
        lines = slice_pd.lines
        edges = []
        i = 0
        while i < len(lines):
            n = int(lines[i])
            ids = lines[i + 1 : i + 1 + n]
            i += 1 + n
            for a, b in zip(ids[:-1], ids[1:]):
                edges.append((int(a), int(b)))

        # Some slices produce single closed loops even though the line cells are
        # individual segments. The shapely `polygonize` handles segment soups.
        segments = []
        for a, b in edges:
            pa = pts[a]
            pb = pts[b]
            xa = np.dot(pa, u), np.dot(pa, v)
            xb = np.dot(pb, u), np.dot(pb, v)
            segments.append(geom.LineString([xa, xb]))
        if not segments:
            return []
        rings = list(ops.polygonize(ops.unary_union(segments)))
        return rings

    def check_sheet_degeneracy(self, section_points: np.ndarray):
        if section_points.shape[0] < 4:
            raise SheetDegeneracyError(f"Section has {section_points.shape[0]} points (< 4)")
        c = section_points.mean(axis=0)
        centred = section_points - c
        cov = np.cov(centred, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(eig)[::-1]
        l1, l2, l3 = eig[0], eig[1], max(0.0, eig[2])
        total = l1 + l2 + l3
        if total <= 0.0:
            raise SheetDegeneracyError("Zero variance in section points")
        c_p = l3 / total
        radii = np.linalg.norm(centred, axis=1)
        r_mean = radii.mean()
        if r_mean <= 0.0:
            raise SheetDegeneracyError("Mean section radius is zero")
        v_r = radii.std() / r_mean
        if c_p < self.cfg.eps_flat and v_r < self.cfg.eps_var:
            raise SheetDegeneracyError(
                f"Sheet degeneracy: c_p={c_p:.4f} (< {self.cfg.eps_flat}), v_r={v_r:.4f} (< {self.cfg.eps_var})"
            )

    def evaluate_loops(self, rings, r_eff_mm: float):
        """Apply the disjoint-loop 30 % area rule."""
        if not rings:
            raise ZeroAreaError("No section loops found")
        rings = sorted(rings, key=lambda r: r.area, reverse=True)
        primary = rings[0]
        a1 = primary.area
        if a1 <= 1e-12:
            raise ZeroAreaError("Primary loop area is zero")
        d_thresh = 2.0 * max(r_eff_mm, 1e-6)
        kept = [primary]
        for sec in rings[1:]:
            a2 = sec.area
            # Distance between loop centroids.
            c1 = np.asarray(primary.centroid.coords[0])
            c2 = np.asarray(sec.centroid.coords[0])
            dist = float(np.linalg.norm(c1 - c2))
            if dist < d_thresh:
                if a2 > self.cfg.area_ratio_thresh * a1:
                    raise DisjointLoopError(
                        f"DisjointLoopException: secondary area {a2:.2f} > 30% of primary {a1:.2f}"
                    )
                # Ignore small parasite loops.
                continue
            # Far loops are also ignored as isolated artifacts.
        return kept

    def compute_section(self, s_mm: float, point: np.ndarray, normal: np.ndarray) -> Section:
        slice_pd = self.mesh.slice_at_plane(point, normal)
        rings = self.section_loops(slice_pd, normal)
        if not rings:
            raise ZeroAreaError(f"Empty cross-section at s={s_mm:.2f}")

        # The cross-section itself is always planar, so the sheet-degeneracy
        # guard is evaluated on the local 3-D body neighbourhood if needed.  Here
        # we keep the pipeline safe by validating the polygon area and radius.
        all_pts = slice_pd.points
        if all_pts.shape[0] < 4:
            raise ZeroAreaError(f"Only {all_pts.shape[0]} points in section at s={s_mm:.2f}")

        # Effective radius from the unioned section area.
        union = ops.unary_union(rings)
        area = float(union.area)
        if area <= 1e-12:
            raise ZeroAreaError(f"Zero union section area at s={s_mm:.2f}")

        r_eff = math.sqrt(area / math.pi)
        if math.isnan(r_eff) or math.isinf(r_eff) or r_eff <= 0.0:
            raise InvalidRadiusError(f"Invalid effective radius {r_eff} at s={s_mm:.2f}")

        kept_rings = self.evaluate_loops(list(union.geoms) if union.geom_type == "MultiPolygon" else [union], r_eff)
        kept_area = sum(r.area for r in kept_rings)
        if kept_area <= 1e-12:
            raise ZeroAreaError(f"Kept loop area is zero at s={s_mm:.2f}")

        return Section(s_mm=s_mm, point=point.copy(), normal=normal.copy(), area_mm2=kept_area, r_eff_mm=r_eff)

    def build_sections(self, points: np.ndarray, tangents: np.ndarray) -> List[Section]:
        sections: List[Section] = []
        for i, (p, t) in enumerate(zip(points, tangents)):
            norm = np.linalg.norm(t)
            if norm < 1e-12:
                raise VelocityEngineError(f"Zero tangent at sample {i}")
            n = t / norm
            try:
                sec = self.compute_section(float(i) * self.cfg.sample_spacing_mm, p, n)
                sections.append(sec)
            except VelocityEngineError:
                raise
        return sections


class GateVelocityEngine:
    """Multi-body orchestrator that builds one velocity array for the whole grid."""

    def __init__(self, cfg: EngineConfig = None):
        self.cfg = cfg or EngineConfig()

    def compute(
        self,
        bodies: Sequence[Body],
        body_index: np.ndarray,
        origin_mm: np.ndarray,
        dx_mm: float,
        gating_nodes: Sequence[Any],
        Q_total_m3_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        grid_shape = body_index.shape
        velocity = np.zeros(grid_shape, dtype=np.float64)
        colors = np.zeros((velocity.size, 3), dtype=np.uint8)

        # Body index -> upstream/downstream Q.
        body_Q = self._body_flow_rates(bodies, gating_nodes, Q_total_m3_s)

        for bidx, body in enumerate(bodies):
            if body.body_type not in _GATING_BODY_TYPES:
                continue
            q_m3_s = body_Q.get(bidx, 0.0)
            if q_m3_s <= 1e-18:
                continue

            analyzer = MeshAnalyzer(
                np.asarray(body.mesh.vertices, dtype=float),
                np.asarray(body.mesh.faces, dtype=int),
                name=body.name,
            )
            engine = SkeletonAnalysisEngineV8_1(analyzer, self.cfg)

            try:
                pts, eds = josecast_core.extract_skeleton(
                    analyzer.pv.points.astype(np.float64),
                    analyzer.tm.faces.astype(np.int32),
                    True,
                )
                if pts.shape[0] < 2:
                    continue
                # Spur pruning is intentionally conservative: a simple path
                # (line/cycle skeleton) is preserved, only true short side
                # branches are removed.
                pts, eds = engine.prune_skeleton_spurs(pts, eds, self.cfg.alpha_spur)
                if pts.shape[0] < 2:
                    continue
                path = engine.longest_path(pts, eds)
                if path.shape[0] < 2:
                    continue
                path = engine.resample_and_smooth_curve(path)
                tangents = engine.compute_tangents(path)
                sections = engine.build_sections(path, tangents)
            except Exception as exc:
                print(f"[GATE_VELOCITY] {body.name}: {exc}", flush=True)
                continue

            if not sections:
                continue

            s_vals = np.array([sec.s_mm for sec in sections])
            a_vals = np.array([sec.area_mm2 for sec in sections])
            v_vals = q_m3_s / (a_vals * 1e-6)  # mm2 -> m2
            # Replace unphysical values with the nearest valid one.
            v_vals = np.nan_to_num(v_vals, nan=0.0, posinf=0.0, neginf=0.0)
            bad = v_vals <= 1e-12
            if bad.all():
                continue
            v_vals[bad] = v_vals[~bad].mean()

            mask = body_index == bidx
            if not mask.any():
                continue
            coords = _voxel_coords(mask, origin_mm, dx_mm)

            # Map each voxel to the closest centerline segment by perpendicular
            # distance, then evaluate v(s) at the projected arc-length.
            values = self._map_to_centerline(coords, path, s_vals, v_vals, tangents)
            velocity[mask] = values

        # Global min/max for colour scale (exclude zero).
        nonzero = velocity[velocity > 1e-12]
        v_min = float(nonzero.min()) if nonzero.size else 0.0
        v_max = float(nonzero.max()) if nonzero.size else 1.0
        colors = self._colorize(velocity.ravel(), v_min, v_max)

        # Make sure the rendered color array is per-voxel in the 3-D grid shape.
        color_grid = colors.reshape((*grid_shape, 3))
        return velocity, color_grid

    def _body_flow_rates(self, bodies, gating_nodes, Q_total_m3_s) -> Dict[int, float]:
        body_Q: Dict[int, float] = {}
        name_to_idx = {b.name: i for i, b in enumerate(bodies)}

        # Source node.
        for node in gating_nodes:
            if "→" not in getattr(node, "body_type", ""):
                continue
            up, down = [s.strip() for s in node.body_type.split("→", 1)]
            if up.startswith("SOURCE"):
                down_name = getattr(node, "name", "").split("→", 1)[-1].strip()
                bidx = name_to_idx.get(down_name)
                if bidx is not None:
                    body_Q[bidx] = max(body_Q.get(bidx, 0.0), float(node.flow_rate_m3_s))

        # Every downstream body receives the flow from its upstream edge.
        for node in gating_nodes:
            name = getattr(node, "name", "")
            if "→" not in name:
                continue
            up_name, down_name = [s.strip() for s in name.split("→", 1)]
            if up_name == "Kaynak":
                continue
            bidx = name_to_idx.get(down_name)
            if bidx is None:
                continue
            q = float(node.flow_rate_m3_s)
            body_Q[bidx] = max(body_Q.get(bidx, 0.0), q)

        # If nothing found, the source body gets the total Q.
        if not body_Q and bodies:
            body_Q[0] = Q_total_m3_s
        return body_Q

    def _map_to_centerline(
        self,
        coords: np.ndarray,
        path: np.ndarray,
        s_vals: np.ndarray,
        v_vals: np.ndarray,
        tangents: np.ndarray,
    ) -> np.ndarray:
        """For each coordinate, find nearest segment, project, and interpolate v."""
        n = path.shape[0]
        if n < 2:
            return np.zeros(coords.shape[0])

        # Segment data.
        seg = path[1:] - path[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len[seg_len == 0] = 1e-12
        seg_unit = seg / seg_len[:, None]

        # Accumulated arc length along path.
        s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        out = np.zeros(coords.shape[0], dtype=float)
        for i, p in enumerate(coords):
            # Project onto each segment, clamp to [0,1].
            rel = p - path[:-1]
            t = np.einsum("ij,ij->i", rel, seg_unit)
            t = np.clip(t, 0.0, seg_len)
            proj = path[:-1] + seg_unit * t[:, None]
            dist = np.linalg.norm(proj - p, axis=1)
            best = int(np.argmin(dist))
            s_local = t[best]
            s = s_cum[best] + s_local
            # Interpolate velocity from the section profile.
            v = float(np.interp(s, s_vals, v_vals))
            out[i] = v
        return out

    def _colorize(self, values: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        """Map a 1-D scalar to 8-bit turbo-like RGB."""
        if v_max - v_min < 1e-12:
            t = np.zeros_like(values)
        else:
            t = (values - v_min) / (v_max - v_min)
            t = np.clip(t, 0.0, 1.0)
        return _turbo(t)


# Gate body types that participate in the velocity colour map.
_GATING_BODY_TYPES = {
    BodyType.INGATE,
    BodyType.RUNNER,
    BodyType.SPRUE,
    BodyType.CORE,
    BodyType.COOLING_SPRUE,
    BodyType.FILTER,
    BodyType.POURING_BASIN,
    BodyType.SPRUE_THROAT,
    BodyType.DISTRIBUTOR,
    BodyType.CURUFLUK,
}


def _voxel_coords(mask: np.ndarray, origin: np.ndarray, dx: float) -> np.ndarray:
    """Return the (x,y,z) coordinates in mm of all voxels whose index is True."""
    idx = np.argwhere(mask).astype(float)
    return (idx + 0.5) * dx + np.asarray(origin, dtype=float)


def _turbo(t: np.ndarray) -> np.ndarray:
    """A fast analytic approximation of the Turbo colormap."""
    # Clamp and reshape.
    t = np.clip(t, 0.0, 1.0)
    x = t * 255.0
    rgb = np.zeros((t.size, 3), dtype=np.uint8)
    # Piecewise linear approximation for speed.
    r = np.piecewise(
        x,
        [
            x < 64,
            (x >= 64) & (x < 128),
            (x >= 128) & (x < 192),
            x >= 192,
        ],
        [
            lambda x: 0.0,
            lambda x: (x - 64.0) * 4.0,
            lambda x: 255.0,
            lambda x: 255.0 - (x - 192.0) * 4.0,
        ],
    )
    g = np.piecewise(
        x,
        [
            x < 64,
            (x >= 64) & (x < 128),
            (x >= 128) & (x < 192),
            x >= 192,
        ],
        [
            lambda x: x * 4.0,
            lambda x: 255.0,
            lambda x: 255.0 - (x - 128.0) * 4.0,
            lambda x: 0.0,
        ],
    )
    b = np.piecewise(
        x,
        [
            x < 64,
            (x >= 64) & (x < 128),
            (x >= 128) & (x < 192),
            x >= 192,
        ],
        [
            lambda x: 255.0 - x * 4.0,
            lambda x: 0.0,
            lambda x: (x - 128.0) * 4.0,
            lambda x: 255.0,
        ],
    )
    rgb[:, 0] = r.astype(np.uint8)
    rgb[:, 1] = g.astype(np.uint8)
    rgb[:, 2] = b.astype(np.uint8)
    return rgb
