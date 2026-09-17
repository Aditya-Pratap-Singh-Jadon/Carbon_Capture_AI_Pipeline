"""Unified Report Generator producing JSON, CSV, HTML, and PDF reports."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from carbon_capture.reporting.audit_report import AuditReportBuilder
from carbon_capture.reporting.charts import VerificationChartGenerator
from carbon_capture.input.validator import ValidationResult
from carbon_capture.epinn.inference import EPINNInferenceOutput
from carbon_capture.carbon_credit.calculator import CarbonCreditSummary
from carbon_capture.carbon_credit.verification import VerificationDecision
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """Generates all verification deliverables: JSON, CSV, HTML, and PDF."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_gen = VerificationChartGenerator(self.output_dir)
        self.audit_builder = AuditReportBuilder()

    def generate_all(
        self,
        val_result: ValidationResult,
        epinn_output: EPINNInferenceOutput,
        credit_summary: CarbonCreditSummary,
        decision: VerificationDecision,
        input_name: str = "industrial_run",
        pe_aux_mode: str = "verbatim",
    ) -> Dict[str, Path]:
        """Generate complete suite of verification reports."""
        # 1. Build audit record
        audit_record = self.audit_builder.build_audit_record(
            val_result, epinn_output, credit_summary, decision,
            input_file=input_name, pe_aux_mode=pe_aux_mode
        )

        # 2. Generate charts
        chart_residuals = self.chart_gen.plot_residual_distributions(
            epinn_output, f"{input_name}_residuals.png"
        )
        chart_cc = self.chart_gen.plot_carbon_credit_breakdown(
            credit_summary, f"{input_name}_cc_breakdown.png"
        )
        chart_constraints = self.chart_gen.plot_constraint_violations(
            epinn_output, f"{input_name}_constraints.png"
        )

        # 3. Export JSON report
        json_path = self.output_dir / f"{input_name}_verification_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(audit_record, f, indent=2, default=str)

        # 4. Export CSV summary
        csv_path = self.output_dir / f"{input_name}_credit_records.csv"
        records_data = [r.__dict__ for r in credit_summary.records]
        pd.DataFrame(records_data).to_csv(csv_path, index=False)

        # 5. Export HTML report
        html_path = self.output_dir / f"{input_name}_verification_report.html"
        self._generate_html_report(audit_record, html_path, [chart_residuals, chart_cc, chart_constraints])

        # 6. Export PDF report
        pdf_path = self.output_dir / f"{input_name}_verification_report.pdf"
        self._generate_pdf_report(audit_record, pdf_path, [chart_residuals, chart_cc, chart_constraints])

        logger.info(f"Generated all reports for {input_name} in {self.output_dir}")
        return {
            "json": json_path,
            "csv": csv_path,
            "html": html_path,
            "pdf": pdf_path,
        }

    def _generate_html_report(
        self,
        audit: Dict[str, Any],
        out_path: Path,
        chart_paths: list,
    ) -> None:
        dec = audit["verification_decision"]
        hdr = audit["audit_header"]
        cc = audit["deterministic_credit_accounting"]
        phys = audit["epinn_physics_validation"]

        badge_color = "#28a745" if dec["status"] == "VERIFIED" else "#dc3545"

        rejection_html = ""
        if dec["rejection_reasons"]:
            items = "".join(f"<li>{r}</li>" for r in dec["rejection_reasons"])
            rejection_html = f'<div class="alert alert-danger"><strong>Rejection Reasons:</strong><ul>{items}</ul></div>'

        assumptions_items = []
        for a in audit.get("scientific_assumptions_and_ambiguities", []):
            iss = a.get("issue", "")
            note = a.get("note", "")
            assumptions_items.append(f"<li><strong>{iss}:</strong> {note}</li>")
        assumptions_html = "".join(assumptions_items)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Carbon Capture AI Verification Report - {hdr['run_id']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #333; background: #f8f9fa; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        h1, h2, h3 {{ color: #1e293b; }}
        .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: bold; color: #fff; background-color: {badge_color}; font-size: 1.2rem; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 25px 0; }}
        .metric-card {{ background: #f1f5f9; padding: 15px; border-radius: 6px; border-left: 4px solid #0284c7; }}
        .metric-title {{ font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: bold; }}
        .metric-value {{ font-size: 1.5rem; color: #0f172a; font-weight: bold; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; }}
        .chart-container {{ margin: 30px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; }}
        .alert {{ padding: 15px; border-radius: 6px; margin: 15px 0; }}
        .alert-danger {{ background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }}
        .alert-info {{ background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>Industrial Carbon Capture Verification Report</h1>
            <p style="color: #64748b; margin: 0;">Run ID: {hdr['run_id']} | Date: {hdr['timestamp_utc']}</p>
        </div>
        <div class="badge">{dec['status']}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Verified Carbon Credits</div>
            <div class="metric-value">{dec['verified_credits_tco2e']:,.2f} <span style="font-size:0.9rem;">tCO2e</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Provisional Credits</div>
            <div class="metric-value">{dec['provisional_credits_tco2e']:,.2f} <span style="font-size:0.9rem;">tCO2e</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Verification Confidence</div>
            <div class="metric-value">{dec['confidence_score']*100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Mean Physics Residual</div>
            <div class="metric-value">{phys['mean_normalized_residual']:.3f} &sigma;</div>
        </div>
    </div>

    {rejection_html}

    <h2>Deterministic Carbon-Credit Mathematical Accounting</h2>
    <p><em>Authoritative formula: CC_T = [E_reduced + E_removed + E_displaced - PE_aux - PE_lifecycle - L_leakage] &times; F_multiplier</em></p>
    <table>
        <tr><th>Component</th><th>Quantity</th><th>Unit</th><th>Accounting Role</th></tr>
        <tr><td>E_reduced</td><td>{cc['benefit_terms']['E_reduced']:,.2f}</td><td>tCO2e</td><td>Direct Fuel/Combustion Reduction</td></tr>
        <tr><td>E_removed</td><td>{cc['benefit_terms']['E_removed']:,.2f}</td><td>tCO2e</td><td>Direct Capture & Mineralization</td></tr>
        <tr><td>E_displaced</td><td>{cc['benefit_terms']['E_displaced']:,.2f}</td><td>tCO2e</td><td>Circular Byproducts & Heat Recovery</td></tr>
        <tr><td>PE_aux</td><td>{cc['penalty_terms']['PE_aux']:,.2f}</td><td>tCO2e / kWh</td><td>Capture Hardware Energy Penalty</td></tr>
        <tr><td>PE_lifecycle</td><td>{cc['penalty_terms']['PE_lifecycle']:,.2f}</td><td>tCO2e</td><td>Sorbent Lifecycle Penalty</td></tr>
        <tr><td>L_leakage</td><td>{cc['penalty_terms']['L_leakage']:,.2f}</td><td>tCO2e</td><td>Slippage & Transport Leakage</td></tr>
        <tr><td><strong>Net Benefit Bracket</strong></td><td><strong>{(cc['benefit_terms']['E_reduced'] + cc['benefit_terms']['E_removed'] + cc['benefit_terms']['E_displaced'] - cc['penalty_terms']['PE_aux'] - cc['penalty_terms']['PE_lifecycle'] - cc['penalty_terms']['L_leakage']):,.2f}</strong></td><td>tCO2e</td><td>Sum of Benefits minus Penalties</td></tr>
        <tr><td>Mean Multiplier</td><td>{cc['mean_multiplier_product']:.4f}</td><td>ratio</td><td>F_valid &times; F_SME &times; F_perm &times; &alpha;</td></tr>
        <tr style="background:#f1f5f9; font-weight:bold;"><td>Total Authoritative CC_T</td><td>{cc['CC_T_total']:,.2f}</td><td>tCO2e</td><td>Final Calculated Carbon Credits</td></tr>
    </table>

    <div class="chart-container">
        <h3>Carbon-Credit Breakdown</h3>
        <img src="{chart_paths[1].name}" alt="Carbon Credit Breakdown">
    </div>

    <h2>E-PINN Scientific Physics Validation</h2>
    <p>The Extended Physics-Informed Neural Network independently verifies whether process measurements satisfy all 15 governing conservation and thermodynamic equations.</p>
    <div class="chart-container">
        <h3>E-PINN Physics Residual Spectrum</h3>
        <img src="{chart_paths[0].name}" alt="E-PINN Residuals">
    </div>

    <div class="chart-container">
        <h3>Physical & Operational Constraint Checks</h3>
        <img src="{chart_paths[2].name}" alt="Constraint Violations">
    </div>

    <h2>Audit & Reproducibility Metadata</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Pipeline Version</td><td>{hdr['pipeline_version']}</td></tr>
        <tr><td>Model Version</td><td>{hdr['model_version']}</td></tr>
        <tr><td>Total Input Records</td><td>{audit['input_data_integrity']['record_count']}</td></tr>
        <tr><td>Canonical Columns Enforced</td><td>{len(audit['input_data_integrity']['canonical_columns'])} features</td></tr>
        <tr><td>Data Quality Hygiene Passed</td><td>{audit['input_data_integrity']['data_valid']}</td></tr>
    </table>

    <h3>Documented Ambiguities & Engineering Disclosures</h3>
    <ul>
        {assumptions_html}
    </ul>
</div>
</body>
</html>
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_pdf_report(
        self,
        audit: Dict[str, Any],
        out_path: Path,
        chart_paths: list,
    ) -> None:
        """Generate PDF report using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            doc = SimpleDocTemplate(str(out_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            story = []
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1e293b"))
            body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=10, leading=14)
            bold_style = ParagraphStyle("BoldStyle", parent=styles["Normal"], fontSize=10, leading=14, fontName="Helvetica-Bold")

            hdr = audit["audit_header"]
            dec = audit["verification_decision"]
            cc = audit["deterministic_credit_accounting"]

            # Header
            story.append(Paragraph("Industrial Carbon Capture Verification Report", title_style))
            story.append(Paragraph(f"Run ID: {hdr['run_id']} | Date: {hdr['timestamp_utc']}", body_style))
            story.append(Spacer(1, 15))

            # Status Box
            status_color = colors.HexColor("#28a745") if dec["status"] == "VERIFIED" else colors.HexColor("#dc3545")
            status_data = [
                [Paragraph(f"<b>STATUS: {dec['status']}</b>", ParagraphStyle("W", textColor=colors.white, fontName="Helvetica-Bold", fontSize=12)),
                 Paragraph(f"<b>Verified Credits: {dec['verified_credits_tco2e']:,.2f} tCO2e</b>", ParagraphStyle("W2", textColor=colors.white, fontName="Helvetica-Bold", fontSize=12))]
            ]
            t_status = Table(status_data, colWidths=[200, 340])
            t_status.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), status_color),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(t_status)
            story.append(Spacer(1, 15))

            # Credit Accounting Table
            story.append(Paragraph("<b>Deterministic Carbon Credit Accounting Breakdown</b>", styles["Heading2"]))
            table_data = [
                ["Component", "Quantity (tCO2e)", "Role"],
                ["E_reduced", f"{cc['benefit_terms']['E_reduced']:,.2f}", "Fuel / Combustion Reduction"],
                ["E_removed", f"{cc['benefit_terms']['E_removed']:,.2f}", "Direct Capture & Mineralization"],
                ["E_displaced", f"{cc['benefit_terms']['E_displaced']:,.2f}", "Circular Byproduct & Heat"],
                ["PE_aux", f"{cc['penalty_terms']['PE_aux']:,.2f}", "Hardware Energy Penalty"],
                ["PE_lifecycle", f"{cc['penalty_terms']['PE_lifecycle']:,.2f}", "Sorbent Lifecycle Penalty"],
                ["L_leakage", f"{cc['penalty_terms']['L_leakage']:,.2f}", "Slippage & Transport Leakage"],
                ["CC_T Total", f"{cc['CC_T_total']:,.2f}", "Total Net Calculated Credits"],
            ]
            t_acc = Table(table_data, colWidths=[150, 150, 240])
            t_acc.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]))
            story.append(t_acc)
            story.append(Spacer(1, 15))

            # Charts
            for c_path in chart_paths:
                if c_path.exists():
                    story.append(Image(str(c_path), width=480, height=240))
                    story.append(Spacer(1, 10))

            doc.build(story)
            logger.info(f"Generated PDF verification report at {out_path}")
        except Exception as e:
            logger.warning(f"Could not generate PDF report via ReportLab: {e}")
