"""Turkish engineering report generator for JoseCast v8.0."""

import base64
import os
from datetime import datetime
from typing import Optional

import numpy as np

from core.types import AnalysisResult, SectionFlow


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_hotspot_table(result: AnalysisResult) -> str:
    rows = []
    if result.hotspots:
        for i, hs in enumerate(result.hotspots, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            status = "OK" if hs.feed_ok else "UZAK/DARALMA"
            niy = ", ".join(f"{k}={v:.2f}" for k, v in hs.niyama_variants.items())
            rows.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>({pos})</td>"
                f"<td>{hs.m_value_mm:.2f} ± {hs.m_uncertainty_mm:.2f}</td>"
                f"<td>{hs.t_section_mm:.2f}</td>"
                f"<td>{hs.dist_to_riser_mm:.1f}</td>"
                f"<td>{hs.max_feeding_distance_mm:.1f}</td>"
                f"<td>{hs.feeding_cost:.2f}</td>"
                f"<td>{hs.niyama_ensemble:.2f}</td>"
                f"<td>{hs.darcy_resistance:.2f}</td>"
                f"<td>{hs.curvature_mean:.2f}</td>"
                f"<td>{hs.shape_factor:.4f}</td>"
                f"<td>{'Evet' if hs.heuvers_ok else 'Hayır'}</td>"
                f"<td>{status}</td>"
                f"</tr>"
            )
    else:
        rows.append('<tr><td colspan="13">Tespit edilmedi.</td></tr>')
    return "".join(rows)


def _format_riser_table(result: AnalysisResult) -> str:
    rows = []
    if result.riser_results:
        for rr in result.riser_results:
            rows.append(
                f"<tr>"
                f"<td>{_html_escape(rr.name)}</td>"
                f"<td>{rr.volume_cm3:.2f}</td>"
                f"<td>{rr.surface_area_cm2:.2f}</td>"
                f"<td>{rr.m_value_mm:.2f}</td>"
                f"<td>{rr.effective_m_required:.2f}</td>"
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
            pos = ",".join(f"{v:.1f}" for v in rp.placement_mm)
            rows.append(
                f"<tr>"
                f"<td>{rp.target_hotspot_index + 1}</td>"
                f"<td>{_html_escape(rp.shape)}</td>"
                f"<td>({pos})</td>"
                f"<td>{rp.diameter_mm:.1f}</td>"
                f"<td>{rp.height_mm:.1f}</td>"
                f"<td>{rp.volume_cm3:.2f}</td>"
                f"<td>{rp.m_required_mm:.2f}</td>"
                f"<td>{_html_escape(rp.reason)}</td>"
                f"</tr>"
            )
    else:
        rows.append('<tr><td colspan="8">Yeni besleyici önerisi yok.</td></tr>')
    return "".join(rows)


def _section_flow_rows(gr) -> str:
    rows = []
    section_names = {
        "INGATE": "Meme",
        "RUNNER": "Yolluk",
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
    }.get(gr.recommended_gating_system, {})
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

    return f""""
    <table>
        <tr><th>Parametre</th><th>Değer</th><th>Durum / Gerekli</th></tr>
        <tr><td>Seçili giriş kesiti</td><td>{selected_section}</td><td>v = {gr.ingate_velocity_m_s:.2f} m/s</td></tr>
        <tr><td>Efektif gate kesiti</td><td>{gr.effective_gate_section}</td><td>-</td></tr>
        <tr><td>Tespit edilen / Önerilen sistem</td><td>{gr.detected_gating_system} / {gr.recommended_gating_system}</td><td>Cidar: {gr.wall_thickness_category}</td></tr>
        <tr><td>Önerilen sistem hedef hızları</td><td>{target_text}</td><td>-</td></tr>
        <tr><td>Gate alanı (Ag)</td><td>{gate_flow.area_cm2:.2f} cm²</td><td>{_target_range(gate_flow)}</td></tr>
        <tr><td>Yolluk min kesit alanı (Ar)</td><td>{gr.runner_min_area_cm2:.2f} cm²</td><td>{_target_range(runner_flow)}</td></tr>
        <tr><td>Döküm ağzı boğaz alanı (As)</td><td>{gr.sprue_throat_area_cm2:.2f} cm²</td><td>gerekli {gr.required_sprue_area_cm2:.2f} cm²</td></tr>
        <tr><td>Döküm ağzı taban alanı</td><td>{gr.sprue_base_area_cm2:.2f} cm²</td><td>-</td></tr>
        {_section_flow_rows(gr)}
        <tr><td>Toplam debi Q</td><td>{gr.ingate_flow_rate_m3_s*1e3:.2f} L/s</td><td>doldurma süresi {gr.ingate_fill_time_s:.2f} s</td></tr>
        <tr><td>Akışkanlık uzunluğu Lf</td><td>{gr.fluidity_length_mm:.1f} mm</td><td>parça boyutu ≤ Lf</td></tr>
        <tr><td>Hız için gerekli seçili kesit alanı</td><td>{gr.required_ingate_area_for_velocity_cm2:.2f} cm²</td><td>mevcut {selected_area:.2f} cm²</td></tr>
        <tr><td>Campbell (Ag/Ar) kontrolü</td><td>{'Geçer' if gr.campbell_ok else 'Geçersiz'}</td><td>hedef oran {target_ratio:.2f}</td></tr>
        <tr><td>Bernoulli döküm ağzı kontrolü</td><td>{'Geçer' if gr.bernoulli_ok else 'Geçersiz'}</td><td>As ≥ gerekli</td></tr>
        <tr><td>Dirsek kaybı (K·v²/2g)</td><td>{gr.elbow_count} dirsek, {gr.head_loss_mm:.1f} mm kayıp</td><td>efektif H={gr.effective_head_mm:.1f} mm</td></tr>
        <tr><td>Meme kalın bölgede</td><td>{'Evet' if gr.ingate_on_thick_region else 'Hayır'} (ortalama M={gr.ingate_avg_m_mm:.2f} mm)</td><td>Hayır olmalı</td></tr>
        <tr><td>Meme kalınlığı</td><td>{gr.ingate_thickness_mm:.2f} mm</td><td>-</td></tr>
        <tr><td>Yolluk kalınlığı</td><td>{gr.runner_thickness_mm:.2f} mm</td><td>-</td></tr>
    </table>
    """


def _format_niyama_table(result: AnalysisResult) -> str:
    if not result.niyama_variants:
        return ""
    rows = []
    for k, v in result.niyama_variants.items():
        mask = result.is_metal if result.is_metal.size > 0 else np.ones(v.shape, dtype=bool)
        rows.append(
            f"<tr><td>{k}</td><td>{np.percentile(v[mask], 5):.3f}</td>"
            f"<td>{np.percentile(v[mask], 50):.3f}</td><td>{np.percentile(v[mask], 95):.3f}</td></tr>"
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

    <h2>Sıcak Noktalar (Hot Spots)</h2>
    <table>
        <tr>
            <th>#</th><th>Konum (mm)</th><th>M ± hata (mm)</th><th>t (mm)</th>
            <th>Mesafe (mm)</th><th>Limit (mm)</th><th>Maliyet</th>
            <th>Niyama ens.</th><th>Darcy</th><th>Mean curv</th><th>SF</th><th>Heuver</th><th>Durum</th>
        </tr>
        {_format_hotspot_table(result)}
    </table>

    <h2>Besleyici (Riser) Değerlendirmesi</h2>
    <table>
        <tr><th>İsim</th><th>Hacim (cm³)</th><th>Yüzey (cm²)</th><th>M (mm)</th><th>Gerekli M (mm)</th><th>Gerekli V (cm³)</th><th>Durum</th></tr>
        {_format_riser_table(result)}
    </table>

    <h2>Otomatik Besleyici Önerileri</h2>
    <table>
        <tr><th>Hot Spot</th><th>Şekil</th><th>Konum (mm)</th><th>Çap (mm)</th><th>Yükseklik (mm)</th><th>Hacim (cm³)</th><th>M (mm)</th><th>Neden</th></tr>
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
    pdf.cell(0, 8, "Sıcak Noktalar", ln=True)
    pdf.set_font(font, "", 10)
    if result.hotspots:
        for i, hs in enumerate(result.hotspots, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            status = "OK" if hs.feed_ok else "UZAK/DARALMA"
            pdf.cell(0, 6, f"{i}. Konum=({pos}) mm | M={hs.m_value_mm:.2f} ± {hs.m_uncertainty_mm:.2f} mm | "
                           f"t={hs.t_section_mm:.2f} mm | Besleme={hs.dist_to_riser_mm:.1f}/{hs.max_feeding_distance_mm:.1f} mm | "
                           f"Niyama={hs.niyama_ensemble:.2f} | Darcy={hs.darcy_resistance:.2f} | Heuver={'OK' if hs.heuvers_ok else 'FAIL'} | {status}", ln=True)
    else:
        pdf.cell(0, 6, "Tespit edilmedi.", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Besleyici Değerlendirmesi", ln=True)
    pdf.set_font(font, "", 10)
    if result.riser_results:
        for rr in result.riser_results:
            pdf.cell(0, 6, f"{rr.name}: V={rr.volume_cm3:.2f} cm³, A={rr.surface_area_cm2:.2f} cm², "
                           f"M={rr.m_value_mm:.2f} mm (gerekli {rr.effective_m_required:.2f} mm), "
                           f"V gerekli={rr.required_volume_cm3:.2f} cm³", ln=True)
    else:
        pdf.cell(0, 6, "Besleyici body atanmamış.", ln=True)
    pdf.ln(4)

    if result.riser_proposals:
        pdf.set_font(font, "", 13)
        pdf.cell(0, 8, "Otomatik Besleyici Onerileri", ln=True)
        pdf.set_font(font, "", 10)
        for i, rp in enumerate(result.riser_proposals, 1):
            pos = ",".join(f"{v:.1f}" for v in rp.placement_mm)
            pdf.cell(0, 6, f"{i}. {rp.shape}: cap={rp.diameter_mm:.1f} mm, yukseklik={rp.height_mm:.1f} mm, "
                           f"V={rp.volume_cm3:.2f} cm3, M={rp.m_required_mm:.2f} mm, konum=({pos}) mm", ln=True)
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
        pdf.cell(0, 6, f"Tespit edilen / Onerilen sistem: {getattr(gr, 'detected_gating_system', '')} / {getattr(gr, 'recommended_gating_system', '')} (cidar: {getattr(gr, 'wall_thickness_category', '')})", ln=True)
        pdf.cell(0, 6, f"Toplam debi Q: {gr.ingate_flow_rate_m3_s*1e3:.2f} L/s, doldurma suresi: {gr.ingate_fill_time_s:.2f} s", ln=True)
        pdf.cell(0, 6, f"Akiskanlik uzunlugu Lf: {gr.fluidity_length_mm:.1f} mm", ln=True)
        pdf.cell(0, 6, f"Gate alani: {gate_flow.area_cm2:.2f} cm² (hedef {gate_flow.target_area_min_cm2:.2f}-{gate_flow.target_area_max_cm2:.2f})", ln=True)
        pdf.cell(0, 6, f"Yolluk min kesit: {gr.runner_min_area_cm2:.2f} cm² (hedef {runner_flow.target_area_min_cm2:.2f}-{runner_flow.target_area_max_cm2:.2f})", ln=True)
        pdf.cell(0, 6, f"Döküm ağzı boğazı: {gr.sprue_throat_area_cm2:.2f} cm² (gerekli {gr.required_sprue_area_cm2:.2f} cm²)", ln=True)
        pdf.cell(0, 6, f"Döküm ağzı tabanı: {gr.sprue_base_area_cm2:.2f} cm²", ln=True)
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
