"""
report_generator.py - PDF, JSON, CSV, Markdown export helpers.
"""

from __future__ import annotations
import csv, io, json
from fpdf import FPDF
from diagnosis_engine import DiagnosisResult
from utils import now_str


class _PDF(FPDF):
    TEAL = (37, 99, 235)
    RED  = (220, 38, 38)
    ORG  = (234, 88, 12)
    GRN  = (22, 163, 74)

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.set_auto_page_break(True, 18)

    def header(self):
        self.set_fill_color(*self.TEAL)
        self.rect(0, 0, 210, 11, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(255, 255, 255)
        self.cell(0, 11, "  Smart Fault Diagnosis System - Industrial AI Report", ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"{self.filename} | Generated {now_str()} | Page {self.page_no()}", align="C")

    def section(self, title: str):
        self.set_x(self.l_margin)
        self.ln(4)
        self.set_fill_color(239, 246, 255)
        self.set_text_color(*self.TEAL)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, f"  {title}", ln=True, fill=True)
        self.set_text_color(30, 30, 30)
        self.ln(1)

    def body(self, text: str, bold: bool = False):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B" if bold else "", 9)
        self.multi_cell(0, 5, _ascii_safe(str(text)))
        self.ln(1)

    def bullet(self, text: str):
        # Always re-anchor to the left margin before indenting, so repeated
        # calls never compound the x-offset (which previously could push
        # the cursor past the right margin and crash multi_cell()).
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, f"-  {_ascii_safe(text)}")
        self.set_x(self.l_margin)


def _ascii_safe(text: str) -> str:
    """FPDF's core fonts only support latin-1; strip anything else safely."""
    if not text:
        return "N/A"
    safe = text.encode("latin-1", errors="replace").decode("latin-1").strip()
    return safe if safe else "N/A"


def generate_pdf_report(diagnosis: DiagnosisResult, filename: str) -> bytes:
    pdf = _PDF(filename)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_PDF.TEAL)
    pdf.ln(4)
    pdf.cell(0, 10, "Fault Diagnosis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, _ascii_safe(f"Equipment: {filename}   |   Date: {now_str()}"), ln=True, align="C")
    pdf.ln(4)

    sc = diagnosis.health_score
    c  = _PDF.GRN if sc >= 60 else (_PDF.ORG if sc >= 40 else _PDF.RED)
    pdf.set_fill_color(*c)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, _ascii_safe(f"Health Score: {sc}/100  |  Severity: {diagnosis.severity}"),
             ln=True, align="C", fill=True)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(4)

    pdf.section("1. Executive Summary")
    pdf.body(diagnosis.summary or "No summary.")

    pdf.section("2. Faults Detected")
    if diagnosis.faults_detected:
        for i, f in enumerate(diagnosis.faults_detected, 1):
            pdf.body(f"{i}. {f.get('fault_name','?')} [{f.get('confidence','?')} confidence]", bold=True)
            pdf.body(f.get("description",""))
            params = f.get("affected_parameters", [])
            if params:
                pdf.body("Affected: " + ", ".join(params))
            pdf.ln(1)
    else:
        pdf.body("No faults detected.")

    pdf.section("3. Root Cause Analysis")
    for rc in diagnosis.root_causes:
        pdf.body(f"Cause: {rc.get('cause','?')}", bold=True)
        pdf.body(rc.get("explanation",""))
        for cf in rc.get("contributing_factors", []):
            pdf.bullet(cf)
        pdf.ln(1)

    pdf.section("4. Corrective Actions")
    pmap = {"Immediate":"URGENT","Within 24h":"HIGH","Within 1 week":"MED","Scheduled":"LOW"}
    for fx in diagnosis.recommended_fixes:
        p = fx.get("priority","?")
        pdf.body(f"[{pmap.get(p,p)}] {fx.get('action','')}", bold=True)
        if fx.get("estimated_downtime"):
            pdf.body(f"Downtime: {fx['estimated_downtime']}  |  Resources: {fx.get('resources_needed','?')}")
        pdf.ln(1)

    pdf.section("5. Risk Assessment")
    ra = diagnosis.risk_assessment
    if ra:
        for k, lbl in [("overall_risk","Overall"),("safety_risk","Safety"),
                        ("production_impact","Production"),("financial_impact","Financial"),
                        ("mtbf_estimate","MTBF")]:
            if ra.get(k):
                pdf.body(f"{lbl}: {ra[k]}")

    pdf.section("6. Failure Prediction")
    fp = diagnosis.failure_prediction
    if fp:
        pdf.body(f"Time to Failure: {fp.get('estimated_time_to_failure','?')}  |  Confidence: {fp.get('confidence','?')}")
        for m in fp.get("failure_modes",[]):
            pdf.bullet(m)

    pdf.section("7. Preventive Actions")
    for pa in diagnosis.preventive_actions:
        pdf.bullet(pa)

    pdf.section("8. Maintenance Schedule")
    for ms in diagnosis.maintenance_schedule:
        pdf.body(f"- {ms.get('task','?')} -- every {ms.get('interval','?')}, next: {ms.get('next_due','?')}")

    return bytes(pdf.output())


def generate_json_report(diagnosis: DiagnosisResult, filename: str) -> bytes:
    payload = {
        "metadata": {"source_file": filename, "generated_at": now_str(), "system": "Smart Fault Diagnosis System (Groq)"},
        "diagnosis": diagnosis.to_dict(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def generate_csv_report(diagnosis: DiagnosisResult) -> bytes:
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Category", "Item", "Value"])
    w.writerow(["Summary", "Health Score", diagnosis.health_score])
    w.writerow(["Summary", "Severity",     diagnosis.severity])
    w.writerow(["Summary", "Summary",      diagnosis.summary])
    for f  in diagnosis.faults_detected:
        w.writerow(["Fault",      f.get("fault_name",""),  f.get("description","")])
    for rc in diagnosis.root_causes:
        w.writerow(["Root Cause", rc.get("cause",""),      rc.get("explanation","")])
    for fx in diagnosis.recommended_fixes:
        w.writerow(["Fix",        fx.get("priority",""),   fx.get("action","")])
    for pa in diagnosis.preventive_actions:
        w.writerow(["Preventive", "",                      pa])
    ra = diagnosis.risk_assessment
    if ra:
        w.writerow(["Risk", "Overall",  ra.get("overall_risk","")])
        w.writerow(["Risk", "Safety",   ra.get("safety_risk","")])
    fp = diagnosis.failure_prediction
    if fp:
        w.writerow(["Prediction", "TTF", fp.get("estimated_time_to_failure","")])
    return buf.getvalue().encode("utf-8")


def generate_markdown_summary(diagnosis: DiagnosisResult, filename: str) -> str:
    lines = [
        "# Fault Diagnosis Report",
        f"**File:** {filename}  |  **Date:** {now_str()}  |  **Health:** {diagnosis.health_score}/100  |  **Severity:** {diagnosis.severity}",
        "", "---", "", "## Summary", diagnosis.summary or "_None._",
        "", "## Faults",
    ]
    for f in diagnosis.faults_detected:
        lines.append(f"- **{f.get('fault_name','?')}** [{f.get('confidence','?')}]: {f.get('description','')}")
    lines += ["", "## Recommended Fixes"]
    for fx in diagnosis.recommended_fixes:
        lines.append(f"- `{fx.get('priority','?')}` - {fx.get('action','')}")
    lines += ["", "## Preventive Actions"]
    for pa in diagnosis.preventive_actions:
        lines.append(f"- {pa}")
    ra = diagnosis.risk_assessment
    if ra:
        lines += ["", "## Risk",
                  f"- Overall: {ra.get('overall_risk','?')}",
                  f"- Safety: {ra.get('safety_risk','?')}"]
    fp = diagnosis.failure_prediction
    if fp:
        lines += ["", "## Failure Prediction",
                  f"- Time to failure: {fp.get('estimated_time_to_failure','?')}",
                  f"- Confidence: {fp.get('confidence','?')}"]
    return "\n".join(lines)
