"""HTML->PDF Turkish report generator for JoseCast v7 Titan."""

import base64
import os
from datetime import datetime
from typing import Optional

import numpy as np

from core.types import AnalysisResult


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_report(
    result: AnalysisResult,
    output_path: str,
    screenshot_path: Optional[str] = None,
) -> str:
    """Render a Turkish engineering report as HTML and convert to PDF with WeasyPrint."""
    rows_hotspot = []
    if result.hotspots:
        for i, hs in enumerate(result.hotspots, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            status = (
                "KRİTİK: Besleyiciye ulaşmıyor"
                if not np.isfinite(hs.dist_to_riser_mm)
                else ("OK" if hs.feed_ok else "UZAK")
            )
            rows_hotspot.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>({pos}) mm</td>"
                f"<td>{hs.m_value_mm:.2f}</td>"
                f"<td>{hs.dist_to_riser_mm:.1f}</td>"
                f"<td>{hs.max_feeding_distance_mm:.1f}</td>"
                f"<td>{hs.niyama_min:.2f}</td>"
                f"<td>{hs.resistance:.1f}</td>"
                f"<td>{status}</td>"
                f"</tr>"
            )
    else:
        rows_hotspot.append(
            '<tr><td colspan="8">Tespit edilmedi.</td></tr>'
        )

    rows_riser = []
    if result.riser_results:
        for rr in result.riser_results:
            rows_riser.append(
                f"<tr>"
                f"<td>{_html_escape(rr.name)}</td>"
                f"<td>{rr.volume_cm3:.2f}</td>"
                f"<td>{rr.surface_area_cm2:.2f}</td>"
                f"<td>{rr.m_value_mm:.2f}</td>"
                f"<td>{rr.effective_m_required:.2f}</td>"
                f"<td>{'Geçer' if rr.large_enough else 'Geçersiz'}</td>"
                f"</tr>"
            )
    else:
        rows_riser.append('<tr><td colspan="6">Besleyici body atanmamış.</td></tr>')

    gate_rows = ""
    if result.gate_result:
        gr = result.gate_result
        gate_rows = f"""
        <h3>Meme / Yolluk / Döküm Ağzı</h3>
        <table>
            <tr><th>Parametre</th><th>Değer</th><th>Durum</th></tr>
            <tr><td>Toplam meme temas alanı (Ag)</td><td>{gr.total_ingate_contact_area_cm2:.2f} cm²</td><td>-</td></tr>
            <tr><td>Yolluk min kesit alanı (Ar)</td><td>{gr.runner_min_area_cm2:.2f} cm²</td><td>-</td></tr>
            <tr><td>Döküm ağzı taban alanı (As)</td><td>{gr.sprue_base_area_cm2:.2f} cm² (gerekli {gr.required_sprue_area_cm2:.2f})</td><td>{'Geçer' if gr.bernoulli_ok else 'Geçersiz'}</td></tr>
            <tr><td>Campbell (Ag/Ar) kontrolü</td><td>{'Geçer' if gr.campbell_ok else 'Geçersiz'}</td><td>{'Geçer' if gr.campbell_ok else 'Geçersiz'}</td></tr>
            <tr><td>Meme kalın bölgede</td><td>{'Evet' if gr.ingate_on_thick_region else 'Hayır'} (ortalama M={gr.ingate_avg_m_mm:.2f} mm)</td><td>{'Kontrol et' if gr.ingate_on_thick_region else 'OK'}</td></tr>
            <tr><td>Meme kalınlığı</td><td>{gr.ingate_thickness_mm:.2f} mm</td><td>-</td></tr>
            <tr><td>Yolluk kalınlığı</td><td>{gr.runner_thickness_mm:.2f} mm</td><td>-</td></tr>
        </table>
        """

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

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>JoseCast Analyzer v7.0 Titan - Rapor</title>
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
    <h1>JoseCast Analyzer v7.0 Titan – Geometrik Analiz Raporu</h1>
    <div class="meta">Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Malzeme: {result.material_name} | Voxel: {result.dx_mm:.3f} mm | Grid: {result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]} | Metal voxel: {int(result.is_metal.sum())}</div>

    <h2>Voxel Grid Bilgisi</h2>
    <table>
        <tr><th>Parametre</th><th>Değer</th></tr>
        <tr><td>Voxel boyutu (dx)</td><td>{result.dx_mm:.3f} mm</td></tr>
        <tr><td>Grid boyutları</td><td>{result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]}</td></tr>
        <tr><td>Metal voxel sayısı</td><td>{int(result.is_metal.sum())}</td></tr>
        <tr><td>Boyut (mm)</td><td>{result.bbox_size_mm[0]:.1f} x {result.bbox_size_mm[1]:.1f} x {result.bbox_size_mm[2]:.1f}</td></tr>
    </table>

    <h2>Sıcak Noktalar (Hot Spots)</h2>
    <table>
        <tr>
            <th>#</th><th>Konum (mm)</th><th>M (mm)</th>
            <th>Besleme mesafesi (mm)</th><th>Limit (mm)</th>
            <th>Niyama</th><th>Direnç</th><th>Durum</th>
        </tr>
        {''.join(rows_hotspot)}
    </table>

    <h2>Besleyici (Riser) Değerlendirmesi</h2>
    <table>
        <tr><th>İsim</th><th>Hacim (cm³)</th><th>Yüzey (cm²)</th><th>M (mm)</th><th>Gerekli M (mm)</th><th>Durum</th></tr>
        {''.join(rows_riser)}
    </table>

    {gate_rows}

    <div class="page-break"></div>
    <h2>Öneriler</h2>
    {recs}

    {screenshot_html}
</body>
</html>
"""

    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(output_path)
    except Exception:
        # Fallback to fpdf2 if WeasyPrint fails or fonts are missing
        _generate_report_fpdf2(result, output_path, screenshot_path)
    return output_path


def _generate_report_fpdf2(
    result: AnalysisResult,
    output_path: str,
    screenshot_path: Optional[str] = None,
):
    from fpdf import FPDF
    import numpy as np

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
    pdf.cell(0, 10, "JoseCast Analyzer v7.0 - Geometrik Analiz Raporu", ln=True, align="C")
    pdf.set_font(font, "", 10)
    pdf.cell(0, 6, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Voxel Grid Bilgisi", ln=True)
    pdf.set_font(font, "", 10)
    pdf.cell(0, 6, f"Voxel boyutu (dx): {result.dx_mm:.3f} mm", ln=True)
    pdf.cell(0, 6, f"Grid boyutları: {result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]}", ln=True)
    pdf.cell(0, 6, f"Metal voxel sayısı: {int(result.is_metal.sum())}", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Sıcak Noktalar (Hot Spots)", ln=True)
    pdf.set_font(font, "", 10)
    if result.hotspots:
        for i, hs in enumerate(result.hotspots, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            status = "OK" if hs.feed_ok else "UZAK"
            pdf.cell(0, 6, f"{i}. Konum=({pos}) mm | M={hs.m_value_mm:.2f} mm | "
                           f"Besleme={hs.dist_to_riser_mm:.1f} mm | Niyama={hs.niyama_min:.2f} | {status}", ln=True)
    else:
        pdf.cell(0, 6, "Tespit edilmedi.", ln=True)
    pdf.ln(4)

    pdf.set_font(font, "", 13)
    pdf.cell(0, 8, "Besleyici (Riser) Değerlendirmesi", ln=True)
    pdf.set_font(font, "", 10)
    if result.riser_results:
        for rr in result.riser_results:
            pdf.cell(0, 6, f"{rr.name}: V={rr.volume_cm3:.2f} cm3, A={rr.surface_area_cm2:.2f} cm2, "
                           f"M={rr.m_value_mm:.2f} mm (gerekli {rr.effective_m_required:.2f} mm)", ln=True)
    else:
        pdf.cell(0, 6, "Besleyici body atanmamış.", ln=True)
    pdf.ln(4)

    if result.gate_result:
        gr = result.gate_result
        pdf.set_font(font, "", 13)
        pdf.cell(0, 8, "Meme / Yolluk / Döküm Ağzı", ln=True)
        pdf.set_font(font, "", 10)
        pdf.cell(0, 6, f"Toplam meme temas alanı: {gr.total_ingate_contact_area_cm2:.2f} cm2", ln=True)
        pdf.cell(0, 6, f"Yolluk min kesit: {gr.runner_min_area_cm2:.2f} cm2", ln=True)
        pdf.cell(0, 6, f"Döküm ağzı: {gr.sprue_base_area_cm2:.2f} cm2 (gerekli {gr.required_sprue_area_cm2:.2f} cm2)", ln=True)
        pdf.cell(0, 6, f"Campbell: {'Geçer' if gr.campbell_ok else 'Geçersiz'}", ln=True)
        pdf.cell(0, 6, f"Bernoulli: {'Geçer' if gr.bernoulli_ok else 'Geçersiz'}", ln=True)
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
