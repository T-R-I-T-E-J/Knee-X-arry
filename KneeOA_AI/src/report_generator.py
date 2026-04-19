"""
Report Generator for Knee OA Analysis.
Exports results as JSON, CSV, or formatted PDF medical reports.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates medical reports in multiple formats."""

    SOFTWARE_VERSION = "2.0.0"

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_metadata(self, results: List[Dict]) -> Dict:
        return {
            "analysis_date": datetime.now().isoformat(),
            "software_version": self.SOFTWARE_VERSION,
            "model_version": "EfficientNet-B0 MultiTask v2",
            "total_images": len(results),
        }

    def _build_batch_summary(self, results: List[Dict]) -> Dict:
        """Calculate averaged statistics across all images."""
        import numpy as np

        if not results:
            return {}

        def safe_mean(vals):
            v = [x for x in vals if x is not None]
            return round(float(np.mean(v)), 2) if v else 0.0

        def safe_std(vals):
            v = [x for x in vals if x is not None]
            return round(float(np.std(v)), 2) if v else 0.0

        sharpness_scores = [r.get("sharpness", {}).get("score", 0) for r in results]
        jsw_means = [r.get("jsw_geometric", {}).get("medial", {}).get("mean", 0) for r in results]
        grades = [r.get("kl_grade", 0) for r in results]
        confidences = [r.get("confidence", 0) for r in results]

        grade_dist = {}
        for g in range(5):
            grade_dist[f"grade_{g}"] = grades.count(g)

        return {
            "sharpness_avg": safe_mean(sharpness_scores),
            "sharpness_std": safe_std(sharpness_scores),
            "jsw_avg_mm": safe_mean(jsw_means),
            "jsw_std_mm": safe_std(jsw_means),
            "grade_distribution": grade_dist,
            "most_common_grade": max(set(grades), key=grades.count) if grades else 0,
            "overall_confidence_avg": safe_mean(confidences),
            "overall_confidence_std": safe_std(confidences),
        }

    # ── JSON Export ────────────────────────────────────────────────
    def export_json(self, results: List[Dict], filename: str = None) -> str:
        filename = filename or f"knee_oa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename

        report = {
            "metadata": self._build_metadata(results),
            "images": results,
            "batch_summary": self._build_batch_summary(results),
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report saved: {path}")
        return str(path)

    # ── CSV Export ─────────────────────────────────────────────────
    def export_csv(self, results: List[Dict], filename: str = None) -> str:
        filename = filename or f"knee_oa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = self.output_dir / filename

        headers = [
            "filename", "kl_grade", "confidence",
            "sharpness_score", "sharpness_status",
            "jsw_medial_mean_mm", "jsw_lateral_mean_mm",
            "osteophyte_score", "sclerosis_score", "contour_score",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {
                    "filename": r.get("filename", ""),
                    "kl_grade": r.get("kl_grade", ""),
                    "confidence": r.get("confidence", ""),
                    "sharpness_score": r.get("sharpness", {}).get("score", ""),
                    "sharpness_status": r.get("sharpness", {}).get("status", ""),
                    "jsw_medial_mean_mm": r.get("jsw_geometric", {}).get("medial", {}).get("mean", ""),
                    "jsw_lateral_mean_mm": r.get("jsw_geometric", {}).get("lateral", {}).get("mean", ""),
                    "osteophyte_score": r.get("clinical_params", {}).get("Osteophytes", ""),
                    "sclerosis_score": r.get("clinical_params", {}).get("Sclerosis", ""),
                    "contour_score": r.get("clinical_params", {}).get("Contour", ""),
                }
                writer.writerow(row)

        logger.info(f"CSV report saved: {path}")
        return str(path)

    # ── PDF Export ─────────────────────────────────────────────────
    def export_pdf(self, results: List[Dict], filename: str = None) -> str:
        """Generate a professional medical PDF report using reportlab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            logger.error("reportlab not installed. Run: pip install reportlab")
            return ""

        filename = filename or f"knee_oa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path = self.output_dir / filename

        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                leftMargin=25*mm, rightMargin=25*mm,
                                topMargin=20*mm, bottomMargin=20*mm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                      fontSize=18, spaceAfter=12,
                                      textColor=colors.HexColor("#1e3a5f"))
        heading_style = ParagraphStyle("H2", parent=styles["Heading2"],
                                        fontSize=13, spaceBefore=14, spaceAfter=6,
                                        textColor=colors.HexColor("#2563eb"))
        body_style = ParagraphStyle("Body2", parent=styles["Normal"],
                                     fontSize=10, spaceAfter=4)
        mono_style = ParagraphStyle("Mono", parent=styles["Normal"],
                                     fontName="Courier", fontSize=9)

        elements = []

        # ── Title Page ──
        elements.append(Spacer(1, 30*mm))
        elements.append(Paragraph("Knee Osteoarthritis", title_style))
        elements.append(Paragraph("Parameter Analysis Report", title_style))
        elements.append(Spacer(1, 10*mm))
        elements.append(HRFlowable(width="100%", thickness=2,
                                    color=colors.HexColor("#2563eb")))
        elements.append(Spacer(1, 8*mm))

        meta = self._build_metadata(results)
        info_data = [
            ["Analysis Date:", meta["analysis_date"][:19]],
            ["Software:", f"v{meta['software_version']}"],
            ["Model:", meta["model_version"]],
            ["Images Analyzed:", str(meta["total_images"])],
        ]
        info_table = Table(info_data, colWidths=[45*mm, 100*mm])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(info_table)

        # ── Batch Summary ──
        if len(results) > 1:
            elements.append(Spacer(1, 10*mm))
            elements.append(Paragraph("Batch Summary Statistics", heading_style))

            summary = self._build_batch_summary(results)
            summary_data = [
                ["Parameter", "Mean", "Std Dev"],
                ["Sharpness Score", str(summary["sharpness_avg"]), str(summary["sharpness_std"])],
                ["JSW Medial (mm)", str(summary["jsw_avg_mm"]), str(summary["jsw_std_mm"])],
                ["Overall Confidence (%)", str(round(summary["overall_confidence_avg"]*100, 1)), str(round(summary["overall_confidence_std"]*100, 1))],
            ]

            t = Table(summary_data, colWidths=[60*mm, 35*mm, 35*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(t)

        # ── Individual Results ──
        for i, r in enumerate(results):
            elements.append(PageBreak())
            elements.append(Paragraph(f"Image {i+1}: {r.get('filename', 'Unknown')}", heading_style))
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
            elements.append(Spacer(1, 4*mm))

            # Sharpness
            sh = r.get("sharpness", {})
            elements.append(Paragraph("Image Quality & Sharpness", heading_style))
            sharp_data = [
                ["Sharpness Score:", f"{sh.get('score', 0)} / 100"],
                ["Quality Status:", sh.get("status", "N/A")],
                ["Laplacian Variance:", str(sh.get("laplacian_variance", 0))],
                ["Sobel Score:", str(sh.get("sobel_score", 0))],
                ["Recommendation:", sh.get("recommendation", "")],
            ]
            st = Table(sharp_data, colWidths=[50*mm, 90*mm])
            st.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(st)
            elements.append(Spacer(1, 4*mm))

            # KL Grade
            elements.append(Paragraph("Overall Assessment", heading_style))
            grade_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
            kl = r.get("kl_grade", 0)
            conf = r.get("confidence", 0)
            elements.append(Paragraph(
                f"Estimated KL Grade: <b>Grade {kl} ({grade_names[kl]})</b> — "
                f"Confidence: <b>{conf:.1%}</b>", body_style))
            elements.append(Spacer(1, 4*mm))

            # JSW
            jsw = r.get("jsw_geometric", {})
            elements.append(Paragraph("Femur-Tibia Distance (JSW)", heading_style))
            med = jsw.get("medial", {})
            lat = jsw.get("lateral", {})
            jsw_data = [
                ["Compartment", "Min (mm)", "Max (mm)", "Mean (mm)", "Std"],
                ["Medial", str(med.get("min", 0)), str(med.get("max", 0)),
                 str(med.get("mean", 0)), str(med.get("std", 0))],
                ["Lateral", str(lat.get("min", 0)), str(lat.get("max", 0)),
                 str(lat.get("mean", 0)), str(lat.get("std", 0))],
            ]
            jt = Table(jsw_data, colWidths=[30*mm, 27*mm, 27*mm, 27*mm, 27*mm])
            jt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(jt)
            elements.append(Spacer(1, 4*mm))

            # Clinical params
            cp = r.get("clinical_params", {})
            if cp:
                elements.append(Paragraph("CNN Clinical Parameters", heading_style))
                for name, val in cp.items():
                    elements.append(Paragraph(f"<b>{name}:</b> {val:.3f}", body_style))

        # ── Footer ──
        elements.append(Spacer(1, 15*mm))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d1d5db")))
        elements.append(Paragraph(
            "<i>This report is generated by the Knee OA Parameter Analysis System. "
            "For research and educational purposes only. Not a clinical diagnosis.</i>",
            ParagraphStyle("Footer", parent=body_style, fontSize=8,
                           textColor=colors.grey, alignment=TA_CENTER)))

        doc.build(elements)
        logger.info(f"PDF report saved: {path}")
        return str(path)
