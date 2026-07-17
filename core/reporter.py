"""Turkish PDF report generator using fpdf2."""

import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from fpdf import FPDF

from core.types import AnalysisResult


def _safe_str(text) -> str:
    if text is None:
        return ""
    return str(text)


def generate_report(
    result: AnalysisResult,
    output_path: str,
    screenshot_path: Optional[str] = None,
) -> str:
    """Write a Turkish engineering report PDF to output_path."""
    pdf = FPDF()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", "", 12)

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 18)
    pdf.cell(0, 10, "JoseCast Analyzer v5.0 - Geometrik Analiz Raporu", ln=True, align="C")
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 10)
    pdf.cell(0, 6, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(4)

    # Grid info
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 13)
    pdf.cell(0, 8, "Voxel Grid Bilgisi", ln=True)
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 10)
    pdf.cell(0, 6, f"Voxel boyutu (dx): {result.dx_mm:.3f} mm", ln=True)
    pdf.cell(0, 6, f"Grid boyutları: {result.grid.shape[0]} x {result.grid.shape[1]} x {result.grid.shape[2]}", ln=True)
    pdf.cell(0, 6, f"Metal voxel sayısı: {int(result.is_metal.sum())}", ln=True)
    pdf.ln(4)

    # Hot spots
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 13)
    pdf.cell(0, 8, "Sıcak Noktalar (Hot Spots)", ln=True)
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 10)
    if result.hotspots:
        for i, hs in enumerate(result.hotspots, 1):
            pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
            if np.isinf(hs.dist_to_riser_mm):
                status = "KRITIK: Besleyiciye ulasmiyor"
            elif hs.feed_ok:
                status = "OK"
            else:
                status = "UZAK"
            pdf.cell(0, 6, f"{i}. Konum=({pos}) mm | M={hs.m_value_mm:.2f} mm | "
                            f"Besleme mesafesi={hs.dist_to_riser_mm:.1f} mm | {status}", ln=True)
    else:
        pdf.cell(0, 6, "Tespit edilmedi.", ln=True)
    pdf.ln(4)

    # Risers
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 13)
    pdf.cell(0, 8, "Besleyici (Riser) Degerlendirmesi", ln=True)
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 10)
    if result.riser_results:
        for rr in result.riser_results:
            pdf.cell(0, 6, f"{rr.name}: V={rr.volume_cm3:.2f} cm3, A={rr.surface_area_cm2:.2f} cm2, "
                            f"M={rr.m_value_mm:.2f} mm (gerekli {rr.target_hotspot_m_mm * 1.2:.2f} mm)", ln=True)
    else:
        pdf.cell(0, 6, "Besleyici body atanmamis.", ln=True)
    pdf.ln(4)

    # Gating
    if result.gate_result:
        gr = result.gate_result
        pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 13)
        pdf.cell(0, 8, "Meme / Yolluk / Dokum Agzi", ln=True)
        pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 10)
        pdf.cell(0, 6, f"Toplam meme temas alani (Ag): {gr.total_ingate_contact_area_cm2:.2f} cm2", ln=True)
        pdf.cell(0, 6, f"Yolluk minimum kesit alani (Ar): {gr.runner_min_area_cm2:.2f} cm2", ln=True)
        pdf.cell(0, 6, f"Dokum agzi taban alani (As): {gr.sprue_base_area_cm2:.2f} cm2 (gerekli {gr.required_sprue_area_cm2:.2f} cm2)", ln=True)
        pdf.cell(0, 6, f"Campbell yolluk kontrolu: {'Gecer' if gr.campbell_ok else 'Gecersiz'}", ln=True)
        pdf.cell(0, 6, f"Bernoulli dokum agzi kontrolu: {'Gecer' if gr.bernoulli_ok else 'Gecersiz'}", ln=True)
        pdf.cell(0, 6, f"Meme kalin bolgede: {'Evet' if gr.ingate_on_thick_region else 'Hayir'} "
                        f"(ortalama M={gr.ingate_avg_m_mm:.2f} mm)", ln=True)
        pdf.ln(4)

    # Recommendations
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 13)
    pdf.cell(0, 8, "Oneriler", ln=True)
    pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 10)
    if result.recommendations:
        for rec in result.recommendations:
            pdf.multi_cell(190, 6, f"- {rec}", ln=1)
    else:
        pdf.cell(0, 6, "Oneri uretilemedi.", ln=True)

    # Screenshot if provided
    if screenshot_path and os.path.exists(screenshot_path):
        pdf.add_page()
        pdf.set_font("DejaVu" if os.path.exists(font_path) else "Arial", "", 16)
        pdf.cell(0, 10, "3D Gorunum", ln=True, align="C")
        try:
            pdf.image(screenshot_path, x=10, y=25, w=190)
        except Exception:
            pdf.cell(0, 6, "Ekran goruntusu eklenemedi.", ln=True)

    pdf.output(output_path)
    return output_path
