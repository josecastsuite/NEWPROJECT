"""Turkish engineering report generator for JoseCast v8.0."""

import base64
import os
from datetime import datetime
from typing import Optional

import numpy as np

from core.materials import get_alloy
from core.types import AnalysisResult, SectionFlow


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_pore_size_summary(result: AnalysisResult) -> str:
    """Return an HTML table summarising estimated pore size distribution.

    Each porosity class now has its own display filter so macro and micro
    shrinkage are not forced through the same top-% noise threshold.
    """
    ps = result.pore_size_um
    if ps is None or ps.size == 0:
        return "<p>Gözenek boyutu hesabı mevcut değil.</p>"
    ps = np.asarray(ps, dtype=np.float64)

    alloy = get_alloy(result.alloy_key)
    macro_limit = float(alloy.macro_pore_limit_um)
    micro_limit = float(alloy.micro_pore_limit_um)

    macro_thr = float(getattr(result, "pore_size_macro_threshold_um", 0.0))
    micro_thr = float(getattr(result, "pore_size_micro_threshold_um", 0.0))
    fine_thr = float(getattr(result, "pore_size_fine_threshold_um", 0.0))
    # If a class has no voxels the stored threshold is 0; use the absolute class
    # lower bound so the row still reports the correct size range.
    if macro_thr <= 0.0:
        macro_thr = macro_limit
    if micro_thr <= 0.0:
        micro_thr = micro_limit
    if fine_thr <= 0.0:
        fine_thr = 0.0

    pm = result.pore_size_macro_mask & np.isfinite(ps) & (ps >= macro_thr)
    pmi = result.pore_size_micro_mask & np.isfinite(ps) & (ps >= micro_thr)
    pf = result.pore_size_fine_mask & np.isfinite(ps) & (ps >= fine_thr)
    macro_vox = int(pm.sum())
    micro_vox = int(pmi.sum())
    fine_vox = int(pf.sum())
    total = macro_vox + micro_vox + fine_vox
    if total == 0:
        return "<p>Tahmini gözenek tespit edilmedi.</p>"

    macro_vals = ps[pm]
    micro_vals = ps[pmi]
    fine_vals = ps[pf]

    def _max(a: np.ndarray) -> float:
        return float(np.max(a)) if len(a) else 0.0

    macro_pct = float(getattr(result, "pore_size_macro_percent", 60.0))
    micro_pct = float(getattr(result, "pore_size_micro_percent", 40.0))
    fine_pct = float(getattr(result, "pore_size_fine_percent", 20.0))

    return f"""<p>Filtre oranları farklı: Makro üst %{macro_pct:.0f} (eşik {macro_thr:.2f} µm), Mikro üst %{micro_pct:.0f} (eşik {micro_thr:.2f} µm), İnce üst %{fine_pct:.0f} (eşik {fine_thr:.2f} µm).</p>
    <table>
        <tr><th>Sınıf</th><th>Voxel sayısı</th><th>Oran</th><th>Max gözenek (µm)</th></tr>
        <tr><td>Makro (&gt;{macro_limit:.0f} µm)</td><td>{macro_vox}</td><td>{100.0*macro_vox/total:.1f}%</td><td>{_max(macro_vals):.1f}</td></tr>
        <tr><td>Mikro ({micro_limit:.0f}–{macro_limit:.0f} µm)</td><td>{micro_vox}</td><td>{100.0*micro_vox/total:.1f}%</td><td>{_max(micro_vals):.1f}</td></tr>
        <tr><td>İnce (&lt;{micro_limit:.0f} µm)</td><td>{fine_vox}</td><td>{100.0*fine_vox/total:.1f}%</td><td>{_max(fine_vals):.1f}</td></tr>
    </table>"""


def _format_hotspot_table(result: AnalysisResult) -> str:
    rows = []
    alloy = get_alloy(result.alloy_key)
    macro_limit = float(alloy.macro_pore_limit_um)
    micro_limit = float(alloy.micro_pore_limit_um)
    visible = [hs for hs in result.hotspots if not hs.solved]
    if visible:
        for i, hs in enumerate(visible, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            if hs.chill_ok:
                status = "ÇIKICI (chill) ile çözüldü"
            elif hs.feed_ok:
                status = "OK"
            else:
                status = "UZAK/DARALMA"
            niy = ", ".join(f"{k}={v:.3f}" for k, v in hs.niyama_variants.items())
            # Hotspot pore display should use the class absolute lower bound,
            # not the percentile-based display threshold.
            class_thr = {
                "macro": macro_limit,
                "micro": micro_limit,
                "fine": 0.0,
            }.get(hs.pore_size_class, 0.0)
            if hs.pore_size_class and hs.pore_size_um >= class_thr:
                pore = f"{hs.pore_size_um:.1f} µm ({hs.pore_size_class})"
            else:
                pore = "-"
            rows.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>({pos})</td>"
                f"<td>{hs.m_value_mm:.2f} ± {hs.m_uncertainty_mm:.2f}</td>"
                f"<td>{hs.t_section_mm:.2f}</td>"
                f"<td>{hs.dist_to_riser_mm:.1f}</td>"
                f"<td>{hs.max_feeding_distance_mm:.1f}</td>"
                f"<td>{hs.feeding_cost:.2f}</td>"
                f"<td>{hs.niyama_ensemble:.4f}</td>"
                f"<td>{hs.darcy_resistance:.2f}</td>"
                f"<td>{hs.curvature_mean:.2f}</td>"
                f"<td>{hs.shape_factor:.4f}</td>"
                f"<td>{'Evet' if hs.heuvers_ok else 'Hayır'}</td>"
                f"<td>{pore}</td>"
                f"<td>{status}</td>"
                f"</tr>"
            )
    else:
        rows.append('<tr><td colspan="14">Tespit edilmedi (çözülenler gizli).</td></tr>')
    return "".join(rows)


def _format_riser_table(result: AnalysisResult) -> str:
    rows = []
    if result.riser_results:
        for rr in result.riser_results:
            eff_m = max(rr.effective_m_value_mm, rr.m_value_mm)
            type_text = f" ({rr.feeder_type})" if rr.feeder_type else ""
            rows.append(
                f"<tr>"
                f"<td>{_html_escape(rr.name)}{type_text}</td>"
                f"<td>{rr.volume_cm3:.2f}</td>"
                f"<td>{rr.surface_area_cm2:.2f}</td>"
                f"<td>{rr.m_value_mm / 10.0:.2f} / etkin {eff_m / 10.0:.2f}</td>"
                f"<td>{rr.effective_m_required / 10.0:.2f}</td>"
                f"<td>{rr.required_volume_cm3:.2f}</td>"
                f"<td>{'Geçer' if rr.large_enough and rr.volume_ratio_ok else 'Geçersiz'}</td>"
                f"</tr>"
            )
    else:
        rows.append('<tr><td colspan="7">Besleyici body atanmamış.</td></tr>')
    return "".join(rows)


def _format_riser_proposal_table(result: AnalysisResult) -> str:
    rows = []
    if result.riser_proposals:
        for rp in result.riser_proposals:
            pos = ",".join(f"{v / 10.0:.1f}" for v in rp.placement_mm)
            if rp.infeasible:
                shape_text = "uyarı (sığmıyor)"
                dims = "Mini exotermik besleyici veya çıkıcı (chill) önerilir"
            elif rp.shape == "chill":
                shape_text = "çıkıcı (chill)"
                dims = f"çap={rp.diameter_mm / 10.0:.1f} cm, yükseklik={rp.height_mm / 10.0:.1f} cm, V={rp.volume_cm3:.2f} cm³"
            elif rp.exothermic:
                shape_text = "ekzotermik mini besleyici"
                dims = f"çap={rp.diameter_mm / 10.0:.1f} cm, yükseklik={rp.height_mm / 10.0:.1f} cm, V={rp.volume_cm3:.2f} cm³"
            else:
                shape_text = _html_escape(rp.shape)
                dims = f"çap={rp.diameter_mm / 10.0:.1f} cm, yükseklik={rp.height_mm / 10.0:.1f} cm, V={rp.volume_cm3:.2f} cm³, M={rp.m_required_mm / 10.0:.2f} cm"
            rows.append(
                f"<tr>"
                f"<td>{rp.target_hotspot_index + 1}</td>"
                f"<td>{shape_text}</td>"
                f"<td>({pos})</td>"
                f"<td>{dims}</td>"
                f"<td>{_html_escape(rp.reason)}</td>"
                f"</tr>"
            )
    else:
        rows.append('<tr><td colspan="5">Yeni besleyici/çıkıcı önerisi yok.</td></tr>')
    return "".join(rows)


def _section_flow_rows(gr) -> str:
    rows = []
    section_names = {
        "INGATE": "Meme",
        "RUNNER": "Yolluk",
        "DISTRIBUTOR": "Dağıtıcı",
        "CURUFLUK": "Curufluk",
        "SPRUE_THROAT": "Döküm ağzı boğazı",
        "SPRUE_BASE": "Döküm ağzı tabanı",
    }
    for key, sf in getattr(gr, "section_flows", {}).items():
        if sf.area_cm2 <= 0:
            continue
        name = section_names.get(key, key)
        if key == "INGATE" and getattr(gr, "effective_gate_section", "INGATE").startswith("RUNNER"):
            name = "Yolluk (meme yok)"
        target = ""
        if sf.target_v_min_m_s > 0 and sf.target_v_max_m_s > 0:
            target = (
                f" hedef v={sf.target_v_min_m_s:.1f}-{sf.target_v_max_m_s:.1f} m/s, "
                f"A={sf.target_area_min_cm2:.2f}-{sf.target_area_max_cm2:.2f} cm²"
            )
        rows.append(
            f"<tr><td>{name}: v / Re / Fr</td>"
            f"<td>{sf.velocity_m_s:.2f} m/s</td>"
            f"<td>Re={sf.reynolds:.0f}, Fr={sf.froude:.2f}, A={sf.area_cm2:.2f} cm², "
            f"türbülans={'Evet' if sf.turbulent else 'Hayır'}{target}</td></tr>"
        )
    return "".join(rows)


def _format_flow_result(flow) -> str:
    """Render the 3-D filling-flow node velocities as a table and bar chart."""
    if flow is None:
        return ""
    node_v = getattr(flow, "node_velocities", {}) or {}
    if not node_v:
        return ""
    # Filter only sections that are present in the model.
    items = [(k, float(v)) for k, v in node_v.items() if v > 1e-9]
    if not items:
        return ""
    max_v = max(v for _, v in items) or 1.0
    labels = {
        "SPRUE": "Döküm ağzı",
        "RUNNER": "Yolluk",
        "DISTRIBUTOR": "Dağıtıcı",
        "CURUFLUK": "Curufluk",
        "INGATE": "Meme",
        "FILTER": "Filtre",
        "RISER": "Besleyici",
    }
    rows = ""
    bars = ""
    for key, val in items:
        pct = 100.0 * val / max_v
        rows += f"<tr><td>{labels.get(key, key)}</td><td>{val:.3f}</td><td>{val*100:.1f} cm/s</td></tr>"
        bars += (
            f'<div style="margin:2px 0;"><span style="display:inline-block;width:90px;">{labels.get(key, key)}</span>'
            f'<span style="display:inline-block;background:#2a9d8f;height:16px;width:{pct:.1f}%;"></span>'
            f'<span style="margin-left:6px;">{val:.3f} m/s</span></div>'
        )
    return f"""
    <h3>3-B Darcy Akış Simülasyonu (v9.3)</h3>
    <p>Toplam debi Q = {flow.Q_m3_s*1e3:.3f} L/s | Giriş alanı = {flow.inlet_area_m2*1e4:.2f} cm² | Tahmini doldurma süresi = {flow.fill_time_s:.2f} s</p>
    <table>
        <tr><th>Kesit</th><th>Hız (m/s)</th><th>Hız (cm/s)</th></tr>
        {rows}
    </table>
    <div style="margin-top:10px;"><strong>Kesit hızları (m/s):</strong><br/>{bars}</div>
    """


def _format_gate_table(result: AnalysisResult) -> str:
    if result.gate_result is None:
        return "<p>Meme/yolluk/döküm ağzı body atanmamış.</p>"
    gr = result.gate_result
    selected_section = getattr(gr, "selected_section_key", "INGATE")
    selected_sf = getattr(gr, "section_flows", {}).get(selected_section)
    selected_area = selected_sf.area_cm2 if selected_sf else 0.0
    system_targets = {
        "basınçlı (pressurized)": {"sprue": (1.0, 1.2), "runner": (1.2, 1.5), "gate": (1.8, 2.5)},
        "basınçsız (unpressurized)": {"sprue": (1.5, 2.0), "runner": (0.8, 1.2), "gate": (0.4, 0.7)},
        "yarı basınçlı (semi-pressurized)": {"sprue": (1.2, 1.5), "runner": (0.6, 1.0), "gate": (0.9, 1.2)},
    }.get(gr.detected_gating_system, {})
    target_text = ""
    if system_targets:
        target_text = (
            f"sprue={system_targets['sprue'][0]:.1f}-{system_targets['sprue'][1]:.1f}, "
            f"runner={system_targets['runner'][0]:.1f}-{system_targets['runner'][1]:.1f}, "
            f"gate={system_targets['gate'][0]:.1f}-{system_targets['gate'][1]:.1f} m/s"
        )
    gate_flow = gr.section_flows.get("INGATE", SectionFlow())
    runner_flow = gr.section_flows.get("RUNNER", SectionFlow())

    def _target_range(sf: SectionFlow) -> str:
        if sf.target_area_min_cm2 > 0 and sf.target_area_max_cm2 > 0:
            return f"hedef {sf.target_area_min_cm2:.2f}-{sf.target_area_max_cm2:.2f} cm²"
        return "hedef -"

    target_ratio = 0.0
    if system_targets and "gate" in system_targets and "runner" in system_targets:
        v_gate_mid = (system_targets["gate"][0] + system_targets["gate"][1]) / 2.0
        v_runner_mid = (system_targets["runner"][0] + system_targets["runner"][1]) / 2.0
        if v_gate_mid > 0:
            target_ratio = v_runner_mid / v_gate_mid

    auto_fill = getattr(gr, "auto_fill_time_s", gr.recommended_fill_time_s)
    campbell_fill = getattr(gr, "campbell_fill_time_s", 0.0)
    flow = getattr(result, "flow_result", None) or getattr(gr, "flow_result", None)
    flow_html = _format_flow_result(flow)
    return f"""
    <p><strong>Ana motor:</strong> parça kütlesi → dolum süresi → metal yüksekliği → sprue hızı → As:Ar:Ag oranıyla hedef alanlar. CAD ölçümleri sadece karşılaştırmadır.</p>
    {flow_html}
    <table>
        <tr><th>Parametre</th><th>Değer</th><th>Durum / Gerekli</th></tr>
        <tr><td>Seçili giriş kesiti</td><td>{selected_section}</td><td>v = {gr.ingate_velocity_m_s:.2f} m/s</td></tr>
        <tr><td>Efektif gate kesiti</td><td>{gr.effective_gate_section}</td><td>-</td></tr>
        <tr><td>Tespit edilen sistem</td><td>{gr.detected_gating_system}</td><td>Cidar: {gr.wall_thickness_category}</td></tr>
        <tr><td>Referans hedef hızları</td><td>{target_text}</td><td>-</td></tr>
        <tr><td>Gate alanı (Ag)</td><td>{gate_flow.area_cm2:.2f} cm²</td><td>{_target_range(gate_flow)}</td></tr>
        <tr><td>Yolluk min kesit alanı (Ar)</td><td>{gr.runner_min_area_cm2:.2f} cm²</td><td>{_target_range(runner_flow)}</td></tr>
        <tr><td>Döküm ağzı boğaz alanı (As)</td><td>{gr.sprue_throat_area_cm2:.2f} cm²</td><td>gerekli {gr.required_sprue_area_cm2:.2f} cm²</td></tr>
        <tr><td>Döküm ağzı taban alanı</td><td>{gr.sprue_base_area_cm2:.2f} cm²</td><td>-</td></tr>
        {_section_flow_rows(gr)}
        <tr><td>Dağıtıcı alanı (Ad)</td><td>{gr.distributor_area_cm2:.2f} cm²</td><td>v = {gr.distributor_velocity_m_s:.2f} m/s</td></tr>
        <tr><td>Curufluk alanı</td><td>{gr.curufluk_area_cm2:.2f} cm²</td><td>v = {gr.curufluk_velocity_m_s:.2f} m/s</td></tr>
        <tr><td>Toplam debi Q</td><td>{gr.ingate_flow_rate_m3_s*1e3:.2f} L/s</td><td>doldurma süresi {gr.ingate_fill_time_s:.2f} s</td></tr>
        <tr><td>Akışkanlık uzunluğu Lf</td><td>{gr.fluidity_length_mm:.1f} mm</td><td>parça boyutu ≤ Lf</td></tr>
        <tr><td>Hız için gerekli seçili kesit alanı</td><td>{gr.required_ingate_area_for_velocity_cm2:.2f} cm²</td><td>mevcut {selected_area:.2f} cm²</td></tr>
        <tr><td>Campbell (Ag/Ar) kontrolü</td><td>{'Geçer' if gr.campbell_ok else 'Geçersiz'}</td><td>hedef oran {target_ratio:.2f}</td></tr>
        <tr><td>Bernoulli döküm ağzı kontrolü</td><td>{'Geçer' if gr.bernoulli_ok else 'Geçersiz'}</td><td>As ≥ gerekli</td></tr>
        <tr><td>Dirsek kaybı (K·v²/2g)</td><td>{gr.elbow_count} dirsek, {gr.head_loss_mm:.1f} mm kayıp</td><td>efektif H={gr.effective_head_mm:.1f} mm</td></tr>
        <tr><td>Meme kalın bölgede</td><td>{'Evet' if gr.ingate_on_thick_region else 'Hayır'} (ortalama M={gr.ingate_avg_m_mm:.2f} mm)</td><td>-</td></tr>
        <tr><td>Meme kalınlığı</td><td>{gr.ingate_thickness_mm:.2f} mm</td><td>-</td></tr>
        <tr><td>Yolluk kalınlığı</td><td>{gr.runner_thickness_mm:.2f} mm</td><td>-</td></tr>
        <tr><td>Dolum süresi</td><td>kullanılan {gr.ingate_fill_time_s:.2f} s</td><td>pratik {auto_fill:.2f} s, Campbell {campbell_fill:.2f} s</td></tr>
        <tr><td>Toplam dökülen kütle / verim</td><td>{gr.total_poured_mass_kg:.3f} kg</td><td>% {gr.pouring_yield*100:.1f}</td></tr>
        <tr><td>Parça / besleyici / yolluk kütle</td><td>parça {gr.part_mass_kg:.3f} kg, besleyici {gr.total_riser_mass_kg:.3f} kg, yolluk {gr.gating_mass_kg:.3f} kg</td><td>oransal {gr.feed_to_part_mass_ratio:.2f}</td></tr>
        <tr><td>Teorik As (sprue taban)</td><td>{gr.design_sprue_base_area_cm2:.2f} cm²</td><td>gerçek {gr.sprue_base_area_cm2:.2f} cm²</td></tr>
        <tr><td>Teorik Ar (yolluk)</td><td>{gr.design_runner_area_cm2:.2f} cm²</td><td>gerçek {gr.runner_min_area_cm2:.2f} cm²</td></tr>
        <tr><td>Teorik Ag (meme toplam)</td><td>{gr.design_gate_total_area_cm2:.2f} cm²</td><td>gerçek {gr.total_ingate_contact_area_cm2:.2f} cm²</td></tr>
        <tr><td>Teorik eşdeğer çaplar</td><td>sprue Ø {gr.design_sprue_diameter_mm:.1f} mm, gate Ø {gr.design_gate_diameter_mm:.1f} mm</td><td>choke v={gr.design_choke_velocity_m_s:.2f} m/s</td></tr>
    </table>
    """


def _format_niyama_table(result: AnalysisResult) -> str:
    if not result.niyama_variants:
        return ""
    rows = []
    part_mask = result.grid == 1  # BodyType.PART == 1
    for k, v in result.niyama_variants.items():
        mask = part_mask if part_mask.size > 0 else np.ones(v.shape, dtype=bool)
        if mask.size == v.size:
            vals = v[mask]
        else:
            vals = v.ravel()
        rows.append(
            f"<tr><td>{k}</td><td>{np.percentile(vals, 5):.3f}</td>"
            f"<td>{np.percentile(vals, 50):.3f}</td><td>{np.percentile(vals, 95):.3f}</td></tr>"
        )
    return "".join(rows)


def _render_html(result: AnalysisResult, screenshot_path: Optional[str] = None) -> str:
    """Render the v8.0 Turkish report as a self-contained HTML string."""
    recs = ""
    if result.recommendations:
        recs = "<ul>" + "".join(f"<li>{_html_escape(r)}</li>" for r in result.recommendations) + "</ul>"
    else:
        recs = "<p>Öneri üretilemedi.</p>"

    screenshot_html = ""
    if screenshot_path and os.path.exists(screenshot_path):
        with open(screenshot_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        screenshot_html = f"""
        <div class="page-break"></div>
        <h2>3D Görünüm</h2>
        <img class="screenshot" src="data:image/png;base64,{img_b64}" />
        """

    params = result.casting_params
    params_html = ""
    if params is not None:
        params_html = f"""
        <table>
            <tr><th>Parametre</th><th>Değer</th></tr>
            <tr><td>Döküm sıcaklığı T_pour</td><td>{params.t_pour_c:.1f} °C</td></tr>
            <tr><td>Liquidus T_liq</td><td>{params.t_liquidus_c:.1f} °C</td></tr>
            <tr><td>Solidus T_sol</td><td>{params.t_solidus_c:.1f} °C</td></tr>
            <tr><td>Kalıp sıcaklığı T_mold</td><td>{params.t_mold_c:.1f} °C</td></tr>
            <tr><td>Döküm süresi t_fill</td><td>{params.t_fill_s:.1f} s</td></tr>
            <tr><td>Sıvı yoğunluk ρ</td><td>{params.rho_liquid_kg_m3:.1f} kg/m³</td></tr>
            <tr><td>Viskozite μ</td><td>{params.viscosity_pa_s:.4f} Pa·s</td></tr>
            <tr><td>Giriş hızı kesiti</td><td>{params.velocity_section_key}</td></tr>
            <tr><td>Giriş hızı v</td><td>{params.ingate_velocity_m_s:.2f} m/s (0 = otomatik)</td></tr>
            <tr><td>Süperheat</td><td>{params.superheat_c:.1f} °C</td></tr>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>JoseCast Analyzer v8.0 Titan - Rapor</title>
    <style>
        @page {{ size: A4; margin: 18mm; }}
        body {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 10pt; color: #222; }}
        h1 {{ font-size: 18pt; text-align: center; color: #00695c; }}
        h2 {{ font-size: 14pt; color: #00695c; border-bottom: 1px solid #00695c; padding-bottom: 2px; }}
        h3 {{ font-size: 12pt; color: #444; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 9pt; }}
        th, td {{ border: 1px solid #aaa; padding: 4px 6px; text-align: left; }}
        th {{ background: #e0f2f1; }}
        ul {{ margin-top: 6px; }}
        li {{ margin-bottom: 4px; }}
        .meta {{ text-align: center; color: #666; margin-bottom: 12px; }}
        .screenshot {{ width: 100%; max-height: 220mm; }}
        .page-break {{ break-before: page; }}
    </style>
</head>
<body>
    <h1>JoseCast Analyzer v8.0 Titan – Geometrik Analiz Raporu</h1>
    <div class="meta">
        Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
        Alaşım: {result.alloy_name} | Kalıp: {result.mold_name} |
        Chvorinov C: {result.chvorinov_c:.4f} s/mm² |
        Voxel: {result.dx_mm:.3f} mm |
        Grid: {result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]} |
        Metal voxel: {int(result.is_metal.sum())} |
        Süre: {result.elapsed_s:.1f} sn
    </div>

    <h2>Döküm Parametreleri</h2>
    {params_html}

    <h2>Voxel Grid ve İstatistik</h2>
    <table>
        <tr><th>Parametre</th><th>Değer</th></tr>
        <tr><td>Voxel boyutu (dx)</td><td>{result.dx_mm:.3f} mm</td></tr>
        <tr><td>Sub-voxel SDF</td><td>Evet</td></tr>
        <tr><td>Grid boyutları</td><td>{result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]}</td></tr>
        <tr><td>Metal voxel sayısı</td><td>{int(result.is_metal.sum())}</td></tr>
        <tr><td>Baskın duvar kalınlığı modülü M (mm)</td><td>{result.dominant_m_mm:.2f}</td></tr>
        <tr><td>Baskın duvar kalınlığı t (mm)</td><td>{result.wall_thickness_mm:.2f}</td></tr>
        <tr><td>SDF ortalama M (mm)</td><td>{result.m_mean_mm:.2f}</td></tr>
        <tr><td>SDF standart sapma (mm)</td><td>{result.m_std_mm:.2f}</td></tr>
        <tr><td>SDF çarpıklık</td><td>{result.m_skewness:.2f}</td></tr>
        <tr><td>Şekil faktörü V²/A³ (global)</td><td>{result.shape_factor_global:.6f}</td></tr>
        <tr><td>Boyut (mm)</td><td>{result.bbox_size_mm[0]:.1f} x {result.bbox_size_mm[1]:.1f} x {result.bbox_size_mm[2]:.1f}</td></tr>
    </table>

    <h2>Gözenek Boyutu Tahmini (µm)</h2>
    {_format_pore_size_summary(result)}

    <h2>Sıcak Noktalar (Hot Spots)</h2>
    <p>Yeterli besleyici/çıkıcı ile çözülen hot spot'lar ekranda gösterilmez; aşağıda sadece çözülmemişler listelenir.</p>
    <table>
        <tr>
            <th>#</th><th>Konum (mm)</th><th>M ± hata (mm)</th><th>t (mm)</th>
            <th>Mesafe (mm)</th><th>Limit (mm)</th><th>Maliyet</th>
            <th>Niyama ens.</th><th>Darcy</th><th>Mean curv</th><th>SF</th><th>Heuver</th><th>Gözenek</th><th>Durum</th>
        </tr>
        {_format_hotspot_table(result)}
    </table>
    <p><em>Genel önleme: kalın kesitlerde yatay yüzey değiştirin, yerel besleyici/çıkıcı ekleyin veya geometriyi inceltin.</em></p>

    <h2>Besleyici (Riser) Değerlendirmesi</h2>
    <table>
        <tr><th>İsim</th><th>Hacim (cm³)</th><th>Yüzey (cm²)</th><th>M (cm)</th><th>Gerekli M (cm)</th><th>Gerekli V (cm³)</th><th>Durum</th></tr>
        {_format_riser_table(result)}
    </table>

    <h2>Otomatik Besleyici / Çıkıcı Önerileri</h2>
    <table>
        <tr><th>Hot Spot</th><th>Şekil</th><th>Konum (cm)</th><th>Boyutlar</th><th>Neden</th></tr>
        {_format_riser_proposal_table(result)}
    </table>

    <h2>Meme / Yolluk / Döküm Ağzı</h2>
    {_format_gate_table(result)}

    <h2>Niyama Varyantları (p5 / p50 / p95)</h2>
    <table>
        <tr><th>Varyant</th><th>p5</th><th>p50</th><th>p95</th></tr>
        {_format_niyama_table(result)}
    </table>

    <div class="page-break"></div>
    <h2>Öneriler</h2>
    {recs}

    {screenshot_html}
</body>
</html>
"""
    return html


def _generate_html(
    result: AnalysisResult,
    output_path: str,
    screenshot_path: Optional[str] = None,
) -> str:
    """Write a self-contained HTML report to disk."""
    html = _render_html(result, screenshot_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_report(
    result: AnalysisResult,
    output_path: str,
    screenshot_path: Optional[str] = None,
) -> str:
    """Render HTML and convert to PDF with WeasyPrint, falling back to fpdf2."""
    html = _render_html(result, screenshot_path)
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(output_path)
    except Exception:
        _generate_report_fpdf2(result, output_path, screenshot_path)
    return output_path


def _generate_report_fpdf2(
    result: AnalysisResult,
    output_path: str,
    screenshot_path: Optional[str] = None,
):
    from fpdf import FPDF

    pdf = FPDF()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", "", 12)
    font = "DejaVu" if os.path.exists(font_path) else "Arial"

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font(font, "", 18)
    pdf.cell(0, 10, "JoseCast Analyzer v8.0 - Geometrik Analiz Raporu", ln=True, align="C")
    pdf.set_font(font, "", 10)
    pdf.cell(0, 6, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.cell(0, 6, f"Alaşım: {result.alloy_name} | Kalıp: {result.mold_name} | C={result.chvorinov_c:.4f}", ln=True, align="C")
    pdf.ln(4)

    if result.casting_params is not None:
        cp = result.casting_params
        pdf.set_font(font, "", 13)
        pdf.cell(0, 8, "Döküm Parametreleri", ln=True)
        pdf.set_font(font, "", 10)
        pdf.cell(0, 6, f"T_pour={cp.t_pour_c:.1f}°C, T_liq={cp.t_liquidus_c:.1f}°C, T_sol={cp.t_solidus_c:.1f}°C", ln=True)
        pdf.cell(0, 6, f"T_mold={cp.t_mold_c:.1f}°C, t_fill={cp.t_fill_s:.1f}s, rho={cp.rho_liquid_kg_m3:.1f} kg/m3, mu={cp.viscosity_pa_s:.4f} Pa.s", ln=True)
        pdf.ln(2)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Voxel Grid ve İstatistik", ln=True)
    pdf.set_font(font, "", 10)
    pdf.cell(0, 6, f"Voxel boyutu (dx): {result.dx_mm:.3f} mm", ln=True)
    pdf.cell(0, 6, f"Grid: {result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]}", ln=True)
    pdf.cell(0, 6, f"Metal voxel: {int(result.is_metal.sum())}", ln=True)
    pdf.cell(0, 6, f"Baskın M: {result.dominant_m_mm:.2f} mm | t: {result.wall_thickness_mm:.2f} mm", ln=True)
    pdf.cell(0, 6, f"SDF ortalama/std/çarpıklık: {result.m_mean_mm:.2f} / {result.m_std_mm:.2f} / {result.m_skewness:.2f}", ln=True)
    pdf.cell(0, 6, f"Şekil faktörü: {result.shape_factor_global:.6f}", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Gözenek Boyutu Tahmini", ln=True)
    pdf.set_font(font, "", 10)
    ps = result.pore_size_um
    if ps is not None and ps.size:
        ps_arr = np.asarray(ps, dtype=np.float64)
        alloy = get_alloy(result.alloy_key)
        macro_limit = float(alloy.macro_pore_limit_um)
        micro_limit = float(alloy.micro_pore_limit_um)
        macro_thr = float(getattr(result, "pore_size_macro_threshold_um", 0.0))
        micro_thr = float(getattr(result, "pore_size_micro_threshold_um", 0.0))
        fine_thr = float(getattr(result, "pore_size_fine_threshold_um", 0.0))
        if macro_thr <= 0.0:
            macro_thr = macro_limit
        if micro_thr <= 0.0:
            micro_thr = micro_limit
        if fine_thr <= 0.0:
            fine_thr = 0.0
        pm = result.pore_size_macro_mask & np.isfinite(ps_arr) & (ps_arr >= macro_thr)
        pmi = result.pore_size_micro_mask & np.isfinite(ps_arr) & (ps_arr >= micro_thr)
        pf = result.pore_size_fine_mask & np.isfinite(ps_arr) & (ps_arr >= fine_thr)
        macro_vox = int(pm.sum())
        micro_vox = int(pmi.sum())
        fine_vox = int(pf.sum())
        macro_max = float(np.max(ps_arr[pm])) if macro_vox else 0.0
        micro_max = float(np.max(ps_arr[pmi])) if micro_vox else 0.0
        fine_max = float(np.max(ps_arr[pf])) if fine_vox else 0.0
        pdf.cell(0, 6, f"Filtre oranlari farkli: Makro ust %60 (esik {macro_thr:.1f} um) | Mikro ust %40 (esik {micro_thr:.1f} um) | Ince ust %20 (esik {fine_thr:.1f} um). Makro (>{macro_limit:.0f} um): {macro_vox} vox (max {macro_max:.1f} um) | Mikro ({micro_limit:.0f}-{macro_limit:.0f} um): {micro_vox} vox (max {micro_max:.1f} um) | Ince (<{micro_limit:.0f} um): {fine_vox} vox (max {fine_max:.1f} um)", ln=True)
    else:
        pdf.cell(0, 6, "Gözenek boyutu hesabi mevcut degil.", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Sıcak Noktalar", ln=True)
    pdf.set_font(font, "", 10)
    pdf.cell(0, 6, "Yeterli besleyici/çıkıcı ile çözülen hot spot'lar gizlendi; sadece çözülmemişler listeleniyor.", ln=True)
    visible_hs = [hs for hs in result.hotspots if not hs.solved]
    if visible_hs:
        for i, hs in enumerate(visible_hs, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            if hs.chill_ok:
                status = "CIKICI (chill) ile cozuldu"
            elif hs.feed_ok:
                status = "OK"
            else:
                status = "UZAK/DARALMA"
            pore = ""
            class_thr = {
                "macro": macro_limit,
                "micro": micro_limit,
                "fine": 0.0,
            }.get(hs.pore_size_class, 0.0)
            if hs.pore_size_class and hs.pore_size_um >= class_thr:
                pore = f" | Gozenek={hs.pore_size_um:.1f} um ({hs.pore_size_class})"
            pdf.cell(0, 6, f"{i}. Konum=({pos}) mm | M={hs.m_value_mm:.2f} ± {hs.m_uncertainty_mm:.2f} mm | "
                           f"t={hs.t_section_mm:.2f} mm | Besleme={hs.dist_to_riser_mm:.1f}/{hs.max_feeding_distance_mm:.1f} mm | "
                           f"Niyama={hs.niyama_ensemble:.2f} | Darcy={hs.darcy_resistance:.2f} | Heuver={'OK' if hs.heuvers_ok else 'FAIL'} | {status}{pore}", ln=True)
    else:
        pdf.cell(0, 6, "Tespit edilmedi.", ln=True)
    pdf.cell(0, 6, "Genel onleme: kalin kesitlerde yatay yuzey degistirin, yerel besleyici/cikici ekleyin veya geometriyi inceltin.", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Besleyici Değerlendirmesi", ln=True)
    pdf.set_font(font, "", 10)
    if result.riser_results:
        for rr in result.riser_results:
            eff_m = max(rr.effective_m_value_mm, rr.m_value_mm)
            type_text = f" [{rr.feeder_type}]" if rr.feeder_type else ""
            pdf.cell(0, 6, f"{rr.name}{type_text}: V={rr.volume_cm3:.2f} cm³, A={rr.surface_area_cm2:.2f} cm², "
                           f"M={rr.m_value_mm / 10.0:.2f} / etkin {eff_m / 10.0:.2f} cm (gerekli {rr.effective_m_required / 10.0:.2f} cm), "
                           f"V gerekli={rr.required_volume_cm3:.2f} cm³", ln=True)
    else:
        pdf.cell(0, 6, "Besleyici body atanmamış.", ln=True)
    pdf.ln(4)

    if result.riser_proposals:
        pdf.set_font(font, "", 13)
        pdf.cell(0, 8, "Otomatik Besleyici / Cikici Onerileri", ln=True)
        pdf.set_font(font, "", 10)
        for i, rp in enumerate(result.riser_proposals, 1):
            pos = ",".join(f"{v / 10.0:.1f}" for v in rp.placement_mm)
            if rp.infeasible:
                pdf.cell(0, 6, f"{i}. UYARI: Hotspot #{rp.target_hotspot_index + 1} icin onerilen besleyici/cikici parcaya sigmiyor.", ln=True)
                pdf.cell(0, 6, f"   Mini exotermik besleyici veya cikici (chill) onerilir; konum=({pos}) cm.", ln=True)
                warning_text = rp.warning if rp.warning else "Cozum kullanici kararidir."
                pdf.cell(0, 6, f"   {warning_text}", ln=True)
            elif rp.shape == "chill":
                pdf.cell(0, 6, f"{i}. cikici (chill): cap={rp.diameter_mm / 10.0:.1f} cm, yukseklik={rp.height_mm / 10.0:.1f} cm, "
                               f"V={rp.volume_cm3:.2f} cm3, konum=({pos}) cm", ln=True)
            elif rp.exothermic:
                pdf.cell(0, 6, f"{i}. ekzotermik mini besleyici: cap={rp.diameter_mm / 10.0:.1f} cm, yukseklik={rp.height_mm / 10.0:.1f} cm, "
                               f"V={rp.volume_cm3:.2f} cm3, konum=({pos}) cm", ln=True)
            else:
                pdf.cell(0, 6, f"{i}. {rp.shape}: cap={rp.diameter_mm / 10.0:.1f} cm, yukseklik={rp.height_mm / 10.0:.1f} cm, "
                               f"V={rp.volume_cm3:.2f} cm3, M={rp.m_required_mm / 10.0:.2f} cm, konum=({pos}) cm", ln=True)
            pdf.cell(0, 6, f"   Neden: {_html_escape(rp.reason)}", ln=True)
        pdf.ln(4)

    if result.gate_result:
        gr = result.gate_result
        section_names = {
            "INGATE": "Meme",
            "RUNNER": "Yolluk",
            "SPRUE_THROAT": "D.AgzI bogazi",
            "SPRUE_BASE": "D.AgzI tabani",
        }
        gate_flow = gr.section_flows.get("INGATE", SectionFlow())
        runner_flow = gr.section_flows.get("RUNNER", SectionFlow())
        pdf.set_font(font, "", 13)
        pdf.cell(0, 8, "Meme / Yolluk / Döküm Ağzı", ln=True)
        pdf.set_font(font, "", 10)
        pdf.cell(0, 6, f"Secili giris kesiti: {getattr(gr, 'selected_section_key', 'INGATE')}, v={gr.ingate_velocity_m_s:.2f} m/s", ln=True)
        pdf.cell(0, 6, f"Efektif gate kesiti: {getattr(gr, 'effective_gate_section', 'INGATE')}", ln=True)
        pdf.cell(0, 6, f"Tespit edilen sistem: {getattr(gr, 'detected_gating_system', '')} (cidar: {getattr(gr, 'wall_thickness_category', '')})", ln=True)
        pdf.cell(0, 6, f"Toplam debi Q: {gr.ingate_flow_rate_m3_s*1e3:.2f} L/s, doldurma suresi: {gr.ingate_fill_time_s:.2f} s", ln=True)
        pdf.cell(0, 6, f"Akiskanlik uzunlugu Lf: {gr.fluidity_length_mm:.1f} mm", ln=True)
        pdf.cell(0, 6, f"Gate alani: {gate_flow.area_cm2:.2f} cm² (hedef {gate_flow.target_area_min_cm2:.2f}-{gate_flow.target_area_max_cm2:.2f})", ln=True)
        pdf.cell(0, 6, f"Yolluk min kesit: {gr.runner_min_area_cm2:.2f} cm² (hedef {runner_flow.target_area_min_cm2:.2f}-{runner_flow.target_area_max_cm2:.2f})", ln=True)
        pdf.cell(0, 6, f"Döküm ağzı boğazı: {gr.sprue_throat_area_cm2:.2f} cm² (gerekli {gr.required_sprue_area_cm2:.2f} cm²)", ln=True)
        pdf.cell(0, 6, f"Döküm ağzı tabanı: {gr.sprue_base_area_cm2:.2f} cm²", ln=True)
        pdf.cell(0, 6, f"Dolum suresi: kullanilan {gr.ingate_fill_time_s:.2f} s, pratik {getattr(gr, 'auto_fill_time_s', gr.recommended_fill_time_s):.2f} s, Campbell {getattr(gr, 'campbell_fill_time_s', 0.0):.2f} s", ln=True)
        pdf.cell(0, 6, f"Toplam dokulen kutle: {gr.total_poured_mass_kg:.3f} kg, verim: {gr.pouring_yield*100:.1f}%", ln=True)
        pdf.cell(0, 6, f"Parca {gr.part_mass_kg:.3f} kg, besleyici {gr.total_riser_mass_kg:.3f} kg, yolluk {gr.gating_mass_kg:.3f} kg; oransal {gr.feed_to_part_mass_ratio:.2f}", ln=True)
        pdf.cell(0, 6, f"Teorik As: {gr.design_sprue_base_area_cm2:.2f} cm², Ar: {gr.design_runner_area_cm2:.2f} cm², Ag: {gr.design_gate_total_area_cm2:.2f} cm²", ln=True)
        pdf.cell(0, 6, f"Teorik caplar: sprue Ø {gr.design_sprue_diameter_mm:.1f} mm, gate Ø {gr.design_gate_diameter_mm:.1f} mm, choke v={gr.design_choke_velocity_m_s:.2f} m/s", ln=True)
        for key, sf in getattr(gr, "section_flows", {}).items():
            if sf.area_cm2 <= 0:
                continue
            name = section_names.get(key, key)
            if key == "INGATE" and getattr(gr, "effective_gate_section", "INGATE").startswith("RUNNER"):
                name = "Yolluk (meme yok)"
            target = ""
            if sf.target_v_min_m_s > 0 and sf.target_v_max_m_s > 0:
                target = (
                    f" hedef v={sf.target_v_min_m_s:.1f}-{sf.target_v_max_m_s:.1f}, "
                    f"A={sf.target_area_min_cm2:.2f}-{sf.target_area_max_cm2:.2f} cm2"
                )
            pdf.cell(0, 6, f"{name}: v={sf.velocity_m_s:.2f}, Re={sf.reynolds:.0f}, Fr={sf.froude:.2f}, A={sf.area_cm2:.2f}{target}, turb={sf.turbulent}", ln=True)
        pdf.cell(0, 6, f"Campbell: {'Geçer' if gr.campbell_ok else 'Geçersiz'}", ln=True)
        pdf.cell(0, 6, f"Bernoulli: {'Geçer' if gr.bernoulli_ok else 'Geçersiz'} (dirsek kaybi: {gr.elbow_count}, {gr.head_loss_mm:.1f} mm)", ln=True)
        pdf.cell(0, 6, f"Meme kalın bölgede: {'Evet' if gr.ingate_on_thick_region else 'Hayır'} "
                       f"(ortalama M={gr.ingate_avg_m_mm:.2f} mm)", ln=True)
        pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Öneriler", ln=True)
    pdf.set_font(font, "", 10)
    if result.recommendations:
        for rec in result.recommendations:
            pdf.multi_cell(190, 6, f"- {rec}", ln=1)
    else:
        pdf.cell(0, 6, "Öneri üretilemedi.", ln=True)

    if screenshot_path and os.path.exists(screenshot_path):
        pdf.add_page()
        pdf.set_font(font, "", 16)
        pdf.cell(0, 10, "3D Görünüm", ln=True, align="C")
        try:
            pdf.image(screenshot_path, x=10, y=25, w=190)
        except Exception:
            pdf.cell(0, 6, "Ekran görüntüsü eklenemedi.", ln=True)

    pdf.output(output_path)
