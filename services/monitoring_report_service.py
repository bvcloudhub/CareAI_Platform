import base64
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO

from PIL import Image as PILImage, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from services.db import query_db


METRIC_META = {
    "spo2": {"label": "SpO2", "unit": "%"},
    "heart_rate": {"label": "Heart rate", "unit": "bpm"},
    "resp_rate": {"label": "Respiratory rate", "unit": "/min"},
    "bp_sys": {"label": "BP systolic", "unit": "mmHg"},
    "bp_dia": {"label": "BP diastolic", "unit": "mmHg"},
    "temperature": {"label": "Temperature", "unit": "C"},
    "glucose": {"label": "Glucose", "unit": "mmol/L"},
    "activity": {"label": "Activity", "unit": "score"},
    "sleep": {"label": "Sleep score", "unit": "score"},
    "weight": {"label": "Weight", "unit": "kg"},
    "body_fat": {"label": "Body fat", "unit": "%"},
    "rr_interval": {"label": "RR interval", "unit": "ms"},
    "pr_interval": {"label": "PR interval", "unit": "ms"},
    "qrs_duration": {"label": "QRS duration", "unit": "ms"},
    "qt_interval": {"label": "QT interval", "unit": "ms"},
    "qtc": {"label": "QTc", "unit": "ms"},
    "rhythm_label": {"label": "Rhythm label", "unit": "", "categorical": True},
}


def _parse_datetime(value):
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _patient_age(birth_date):
    dob = _parse_datetime(birth_date)
    if not dob:
        try:
            dob = datetime.strptime(str(birth_date), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    today = datetime.now(timezone.utc).date()
    born = dob.date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _display_timestamp(value):
    dt = _parse_datetime(value)
    if not dt:
        return str(value or "-")
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def load_monitoring_points(patient_id):
    points = []
    rows = query_db(
        """SELECT kind,value,unit,source,measured_at FROM (
               SELECT id,kind,value,unit,source,measured_at
               FROM vitals WHERE patient_id=?
               ORDER BY measured_at DESC,id DESC LIMIT 5000
           ) recent
           ORDER BY measured_at ASC""",
        (patient_id,),
    )
    for row in rows:
        item = dict(row)
        kind = str(item.get("kind") or "").strip()
        if not kind:
            continue
        meta = METRIC_META.get(kind, {})
        points.append({
            "metric": kind,
            "value": item.get("value"),
            "unit": item.get("unit") or meta.get("unit", ""),
            "source": item.get("source") or "unknown",
            "measured_at": item.get("measured_at"),
            "type": "numeric",
        })

    try:
        ecg_rows = query_db(
            """SELECT measured_at,source,device_identifier,rr_interval_ms,pr_interval_ms,
                      qrs_duration_ms,qt_interval_ms,qtc_ms,rhythm_label
               FROM ecg_recordings WHERE patient_id=?
               ORDER BY measured_at ASC,id ASC LIMIT 1000""",
            (patient_id,),
        )
    except Exception:
        ecg_rows = []

    ecg_map = (
        ("rr_interval", "rr_interval_ms"),
        ("pr_interval", "pr_interval_ms"),
        ("qrs_duration", "qrs_duration_ms"),
        ("qt_interval", "qt_interval_ms"),
        ("qtc", "qtc_ms"),
    )
    for row in ecg_rows:
        item = dict(row)
        source = item.get("source") or item.get("device_identifier") or "ecg"
        for metric, column in ecg_map:
            value = item.get(column)
            if value is None:
                continue
            points.append({
                "metric": metric,
                "value": value,
                "unit": "ms",
                "source": source,
                "measured_at": item.get("measured_at"),
                "type": "numeric",
            })
        rhythm = str(item.get("rhythm_label") or "").strip()
        if rhythm:
            points.append({
                "metric": "rhythm_label",
                "value": rhythm,
                "unit": "",
                "source": source,
                "measured_at": item.get("measured_at"),
                "type": "categorical",
            })
    points.sort(key=lambda x: _parse_datetime(x.get("measured_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return points


def filter_monitoring_points(points, selected_metrics=None, period="all", source="all", from_value=None, to_value=None):
    selected = set(selected_metrics or [])
    now = datetime.now(timezone.utc)
    start = None
    end = None
    if period == "24h":
        start, end = now - timedelta(hours=24), now
    elif period == "7d":
        start, end = now - timedelta(days=7), now
    elif period == "30d":
        start, end = now - timedelta(days=30), now
    elif period == "90d":
        start, end = now - timedelta(days=90), now
    elif period == "custom":
        start = _parse_datetime(from_value)
        end = _parse_datetime(to_value)

    filtered = []
    for point in points:
        if selected and point.get("metric") not in selected:
            continue
        if source and source != "all" and point.get("source") != source:
            continue
        measured = _parse_datetime(point.get("measured_at"))
        if start and measured and measured < start:
            continue
        if end and measured and measured > end:
            continue
        filtered.append(point)
    return filtered


def _metric_label(metric):
    return METRIC_META.get(metric, {}).get("label") or str(metric).replace("_", " ").title()


def _clean_chart_image(data_url):
    if not data_url or not isinstance(data_url, str):
        return None
    match = re.match(r"^data:image/(?:png|jpeg);base64,(.+)$", data_url, flags=re.I | re.S)
    if not match:
        return None
    try:
        raw = base64.b64decode(match.group(1), validate=True)
    except Exception:
        return None
    if len(raw) > 10 * 1024 * 1024:
        return None
    try:
        source = PILImage.open(BytesIO(raw)).convert("RGBA")
        background = PILImage.new("RGBA", source.size, "white")
        background.alpha_composite(source)
        output = BytesIO()
        background.convert("RGB").save(output, format="PNG", optimize=True)
        output.seek(0)
        return output
    except Exception:
        return None


def _summary_rows(points):
    numeric = {}
    rhythms = []
    for point in points:
        metric = point.get("metric")
        if point.get("type") == "categorical" or metric == "rhythm_label":
            rhythms.append(point)
            continue
        try:
            value = float(point.get("value"))
        except (TypeError, ValueError):
            continue
        numeric.setdefault(metric, []).append((value, point))

    rows = []
    for metric, values in numeric.items():
        ordered = sorted(values, key=lambda item: _parse_datetime(item[1].get("measured_at")) or datetime.min.replace(tzinfo=timezone.utc))
        numbers = [item[0] for item in ordered]
        latest_value, latest_point = ordered[-1]
        unit = latest_point.get("unit") or METRIC_META.get(metric, {}).get("unit", "")
        rows.append({
            "metric": metric,
            "label": _metric_label(metric),
            "latest": latest_value,
            "min": min(numbers),
            "max": max(numbers),
            "avg": sum(numbers) / len(numbers),
            "unit": unit,
            "count": len(numbers),
            "latest_at": latest_point.get("measured_at"),
        })
    rows.sort(key=lambda row: row["label"])
    return rows, rhythms


def _fmt_number(value):
    if value is None:
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_monitoring_pdf(
    patient,
    points,
    selected_metrics,
    period,
    source,
    from_value,
    to_value,
    chart_style,
    chart_image_data_url,
    generated_by,
    generated_role,
    conditions=None,
):
    buffer = BytesIO()
    page_size = landscape(A4)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title=f"{patient.get('external_ref', '')} monitoring report",
        author="CareAI",
    )

    stylesheet = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CareAITitle", parent=stylesheet["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#0C5B45"), alignment=TA_LEFT, spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "CareAISubtitle", parent=stylesheet["Normal"], fontSize=8.5, leading=11,
        textColor=colors.HexColor("#47645C"), spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "CareAISection", parent=stylesheet["Heading2"], fontSize=11, leading=14,
        textColor=colors.HexColor("#0C5B45"), spaceBefore=7, spaceAfter=5,
    )
    small = ParagraphStyle("CareAISmall", parent=stylesheet["Normal"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#344A43"))
    small_center = ParagraphStyle("CareAISmallCenter", parent=small, alignment=TA_CENTER)
    body = ParagraphStyle("CareAIBody", parent=stylesheet["Normal"], fontSize=8.5, leading=11)

    patient_name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip()
    age = _patient_age(patient.get("birth_date"))
    generated_at = datetime.now(timezone.utc)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E2DE"))
        canvas.line(12 * mm, 10 * mm, page_size[0] - 12 * mm, 10 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#60756E"))
        canvas.drawString(12 * mm, 6.4 * mm, "CareAI monitoring summary - generated from CareAI records; not a diagnostic interpretation.")
        canvas.drawRightString(page_size[0] - 12 * mm, 6.4 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    story = []
    story.append(Paragraph("CareAI Patient Monitoring Report", title_style))
    story.append(Paragraph(
        f"Patient-linked monitoring summary for review by the patient and authorised care team. Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}.",
        subtitle_style,
    ))

    patient_data = [
        ["Patient", patient_name or "-", "Patient ID", patient.get("external_ref") or "-", "DOB / age", f"{patient.get('birth_date') or '-'}" + (f" / {age}" if age is not None else "")],
        ["Sex", patient.get("sex") or "-", "Location", ", ".join([x for x in [patient.get("city"), patient.get("country")] if x]) or "-", "Living setting", patient.get("living_setting") or "-"],
        ["Assigned nurse", patient.get("assigned_nurse_name") or "Unassigned", "Clinician / GP", patient.get("assigned_clinician_name") or patient.get("gp_name") or "Unassigned", "Generated by", f"{generated_by or '-'} ({generated_role or '-'})"],
    ]
    patient_table = Table(patient_data, colWidths=[25*mm, 48*mm, 27*mm, 45*mm, 28*mm, 60*mm], hAlign="LEFT")
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F6F9F8")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#CBDAD5")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#DCE6E2")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (4,0), (4,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#173F33")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(patient_table)

    active_conditions = [str(c).strip() for c in (conditions or []) if str(c).strip()]
    if active_conditions:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Active conditions: " + ", ".join(active_conditions), small))

    period_labels = {"24h":"Last 24 hours", "7d":"Last 7 days", "30d":"Last 30 days", "90d":"Last 90 days", "all":"All available", "custom":"Custom range"}
    metric_text = ", ".join(_metric_label(key) for key in selected_metrics) or "All available signals"
    filter_data = [
        ["Period", period_labels.get(period, period or "All available"), "Source", source if source and source != "all" else "All sources", "Chart", (chart_style or "line").replace("_", " ").title()],
        ["From", from_value or "-", "To", to_value or "-", "Signals", metric_text],
    ]
    filter_table = Table(filter_data, colWidths=[20*mm, 45*mm, 18*mm, 43*mm, 18*mm, 89*mm], hAlign="LEFT")
    filter_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FFFFFF")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#D7E2DE")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E6EEEB")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (4,0), (4,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.2),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#344A43")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(Spacer(1, 6))
    story.append(filter_table)

    chart_stream = _clean_chart_image(chart_image_data_url)
    if chart_stream:
        story.append(Paragraph("Monitoring chart", section_style))
        image = Image(chart_stream)
        max_w = 260 * mm
        max_h = 88 * mm
        scale = min(max_w / image.imageWidth, max_h / image.imageHeight, 1)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        story.append(image)

    summary, rhythms = _summary_rows(points)
    story.append(Paragraph("Signal summary", section_style))
    if summary:
        table_data = [["Signal", "Latest", "Min", "Max", "Average", "Unit", "Records", "Latest measurement"]]
        for row in summary:
            table_data.append([
                row["label"], _fmt_number(row["latest"]), _fmt_number(row["min"]), _fmt_number(row["max"]),
                _fmt_number(row["avg"]), row["unit"] or "-", str(row["count"]), _display_timestamp(row["latest_at"]),
            ])
        summary_table = Table(table_data, repeatRows=1, colWidths=[37*mm, 20*mm, 18*mm, 18*mm, 20*mm, 18*mm, 18*mm, 49*mm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0C5B45")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7.3),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D7E2DE")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAF9")]),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN", (1,1), (6,-1), "RIGHT"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(summary_table)
    else:
        story.append(Paragraph("No numeric measurements matched the selected report filters.", body))

    if rhythms:
        story.append(Paragraph("ECG rhythm labels", section_style))
        rhythm_data = [["Label", "Measured", "Source"]]
        for point in rhythms[-20:][::-1]:
            rhythm_data.append([str(point.get("value") or "-"), _display_timestamp(point.get("measured_at")), str(point.get("source") or "-")])
        rhythm_table = Table(rhythm_data, repeatRows=1, colWidths=[75*mm, 55*mm, 95*mm])
        rhythm_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF4F0")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#0C5B45")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7.4),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D7E2DE")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(rhythm_table)

    story.append(Spacer(1, 7))
    story.append(KeepTogether([
        Paragraph("Report note", section_style),
        Paragraph(
            "This document summarises measurements stored in CareAI for the selected patient, period, source and signals. "
            "It is intended to support review and communication and does not replace source-device records, clinician assessment or diagnostic interpretation.",
            small,
        ),
    ]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()



def _ecg_waveform_strip_image(ecg, strip_start_seconds, strip_end_seconds):
    """Render one ECG time strip as a high-resolution PNG for the PDF.

    The grid is a visual aid only. It is not presented as a calibrated diagnostic
    ECG paper scale because the source can be simulated or vendor-provided.
    """
    try:
        sampling_rate = max(1, int(ecg.get("sampling_rate_hz") or 250))
    except (TypeError, ValueError):
        sampling_rate = 250
    samples = ecg.get("waveform_samples") or []
    numeric = []
    for value in samples:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            numeric.append(0.0)

    start_index = max(0, int(round(float(strip_start_seconds) * sampling_rate)))
    end_index = min(len(numeric), int(round(float(strip_end_seconds) * sampling_rate)))
    segment = numeric[start_index:end_index]
    if len(segment) < 2:
        return None

    width, height = 1800, 330
    left, right, top, bottom = 72, 28, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # ECG-like grid for readability. It is deliberately labelled as non-calibrated.
    minor = 18
    major = minor * 5
    minor_color = (246, 224, 224)
    major_color = (232, 181, 181)
    axis_color = (101, 118, 112)
    trace_color = (8, 84, 63)
    text_color = (55, 73, 67)

    for x in range(left, left + plot_w + 1, minor):
        draw.line((x, top, x, top + plot_h), fill=major_color if (x-left) % major == 0 else minor_color, width=2 if (x-left) % major == 0 else 1)
    for y in range(top, top + plot_h + 1, minor):
        draw.line((left, y, left + plot_w, y), fill=major_color if (y-top) % major == 0 else minor_color, width=2 if (y-top) % major == 0 else 1)

    max_abs = max(abs(v) for v in segment) if segment else 1.0
    y_limit = max(1.0, max_abs * 1.18)
    center_y = top + plot_h / 2
    x_den = max(1, len(segment) - 1)
    points = []
    for i, value in enumerate(segment):
        x = left + (i / x_den) * plot_w
        y = center_y - (value / y_limit) * (plot_h * 0.46)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=trace_color, width=3, joint="curve")

    # Axis labels / strip timing.
    unit = str(ecg.get("amplitude_unit") or "mV")
    draw.text((8, top + 4), f"+{y_limit:.2g} {unit}", fill=text_color)
    draw.text((8, int(center_y) - 6), f"0 {unit}", fill=text_color)
    draw.text((8, top + plot_h - 13), f"-{y_limit:.2g} {unit}", fill=text_color)
    draw.line((left, center_y, left + plot_w, center_y), fill=axis_color, width=1)

    duration = max(0.001, float(strip_end_seconds) - float(strip_start_seconds))
    tick_count = max(2, min(6, int(round(duration / 2.0)) + 1))
    for index in range(tick_count):
        frac = index / max(1, tick_count - 1)
        x = left + frac * plot_w
        second = float(strip_start_seconds) + frac * duration
        draw.text((int(x) - 12, top + plot_h + 10), f"{second:.1f}s", fill=text_color)

    out = BytesIO()
    image.save(out, format="PNG", optimize=True)
    out.seek(0)
    return out


def build_ecg_pdf(
    patient,
    ecg,
    generated_by,
    generated_role,
    conditions=None,
):
    """Build a separate patient-linked ECG PDF in landscape A4.

    Long recordings are split into readable time strips so 30-second traces are
    not compressed into an unreadable single line. ReportLab naturally carries
    additional strips onto following landscape pages.
    """
    buffer = BytesIO()
    page_size = landscape(A4)
    patient_ref = patient.get("external_ref") or "patient"
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=13 * mm,
        title=f"{patient_ref} ECG report",
        author="CareAI",
    )

    stylesheet = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CareAIEcgTitle", parent=stylesheet["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=22, textColor=colors.HexColor("#0C5B45"), alignment=TA_LEFT, spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "CareAIEcgSubtitle", parent=stylesheet["Normal"], fontSize=8.3, leading=10.5,
        textColor=colors.HexColor("#47645C"), spaceAfter=7,
    )
    section_style = ParagraphStyle(
        "CareAIEcgSection", parent=stylesheet["Heading2"], fontSize=10.5, leading=13,
        textColor=colors.HexColor("#0C5B45"), spaceBefore=6, spaceAfter=4,
    )
    small = ParagraphStyle(
        "CareAIEcgSmall", parent=stylesheet["Normal"], fontSize=7.2, leading=9,
        textColor=colors.HexColor("#344A43"),
    )
    body = ParagraphStyle(
        "CareAIEcgBody", parent=stylesheet["Normal"], fontSize=8.1, leading=10.3,
        textColor=colors.HexColor("#233D35"),
    )

    patient_name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip()
    age = _patient_age(patient.get("birth_date"))
    generated_at = datetime.now(timezone.utc)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E2DE"))
        canvas.line(10 * mm, 9.5 * mm, page_size[0] - 10 * mm, 9.5 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#60756E"))
        canvas.drawString(10 * mm, 6.1 * mm, "CareAI ECG report - patient-linked record; not a diagnostic interpretation.")
        canvas.drawRightString(page_size[0] - 10 * mm, 6.1 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    story = [
        Paragraph("CareAI ECG Report", title_style),
        Paragraph(
            f"Single-lead ECG recording for review by the patient and authorised care team. Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}.",
            subtitle_style,
        ),
    ]

    patient_data = [
        ["Patient", patient_name or "-", "Patient ID", patient_ref, "DOB / age", f"{patient.get('birth_date') or '-'}" + (f" / {age}" if age is not None else "")],
        ["Sex", patient.get("sex") or "-", "Location", ", ".join([x for x in [patient.get("city"), patient.get("country")] if x]) or "-", "Living setting", patient.get("living_setting") or "-"],
        ["Assigned nurse", patient.get("assigned_nurse_name") or "Unassigned", "Clinician / GP", patient.get("assigned_clinician_name") or patient.get("gp_name") or "Unassigned", "Generated by", f"{generated_by or '-'} ({generated_role or '-'})"],
    ]
    patient_table = Table(patient_data, colWidths=[24*mm, 50*mm, 25*mm, 45*mm, 27*mm, 65*mm], hAlign="LEFT")
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F6F9F8")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#CBDAD5")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#DCE6E2")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (4,0), (4,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.3),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#173F33")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(patient_table)

    active_conditions = [str(c).strip() for c in (conditions or []) if str(c).strip()]
    if active_conditions:
        story.append(Spacer(1, 3))
        story.append(Paragraph("Active conditions: " + ", ".join(active_conditions), small))

    story.append(Paragraph("ECG recording details", section_style))
    record_data = [
        ["Measured", _display_timestamp(ecg.get("measured_at")), "Device", ecg.get("device_identifier") or "-", "Source", ecg.get("source") or "-"],
        ["Lead", ecg.get("lead_name") or "Lead I", "Sampling rate", f"{ecg.get('sampling_rate_hz') or '-'} Hz", "Duration", f"{_fmt_number(float(ecg.get('duration_seconds') or 0))} s"],
        ["Amplitude unit", ecg.get("amplitude_unit") or "mV", "ECG record", f"#{ecg.get('id') or '-'}", "Trace points", str(len(ecg.get("waveform_samples") or []))],
    ]
    record_table = Table(record_data, colWidths=[25*mm, 57*mm, 27*mm, 55*mm, 25*mm, 62*mm], hAlign="LEFT")
    record_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#D7E2DE")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E6EEEB")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (4,0), (4,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.2),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#344A43")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(record_table)

    story.append(Paragraph("ECG measurements", section_style))
    metric_pairs = [
        ("Heart rate", ecg.get("heart_rate_bpm"), "bpm"),
        ("RR interval", ecg.get("rr_interval_ms"), "ms"),
        ("PR interval", ecg.get("pr_interval_ms"), "ms"),
        ("QRS duration", ecg.get("qrs_duration_ms"), "ms"),
        ("QT interval", ecg.get("qt_interval_ms"), "ms"),
        ("QTc", ecg.get("qtc_ms"), "ms"),
        ("Rhythm label", ecg.get("rhythm_label"), ""),
    ]
    metric_headers = [label for label, _, _ in metric_pairs]
    metric_values = []
    for _, value, unit in metric_pairs:
        if value in (None, ""):
            metric_values.append("-")
        elif unit:
            try:
                metric_values.append(f"{_fmt_number(float(value))} {unit}")
            except (TypeError, ValueError):
                metric_values.append(f"{value} {unit}")
        else:
            metric_values.append(str(value))
    metric_table = Table([metric_headers, metric_values], colWidths=[36*mm]*7, hAlign="LEFT")
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0C5B45")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#F7FAF9")),
        ("TEXTCOLOR", (0,1), (-1,1), colors.HexColor("#173F33")),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.3),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D7E2DE")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(metric_table)

    samples = ecg.get("waveform_samples") or []
    try:
        rate = max(1, int(ecg.get("sampling_rate_hz") or 250))
    except (TypeError, ValueError):
        rate = 250
    try:
        declared_duration = float(ecg.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        declared_duration = 0
    actual_duration = (len(samples) / rate) if samples and rate else declared_duration
    duration = declared_duration if declared_duration > 0 else actual_duration
    if actual_duration > 0:
        duration = min(duration, actual_duration) if duration > 0 else actual_duration

    story.append(Paragraph("ECG waveform", section_style))
    story.append(Paragraph(
        "Waveform is displayed in readable time strips. The background grid is a visual aid and is not a calibrated diagnostic ECG-paper scale.",
        small,
    ))
    story.append(Spacer(1, 3))

    if samples and duration > 0:
        seconds_per_strip = 10.0
        strip_start = 0.0
        strip_number = 1
        while strip_start < duration - 1e-6:
            strip_end = min(duration, strip_start + seconds_per_strip)
            stream = _ecg_waveform_strip_image(ecg, strip_start, strip_end)
            if stream:
                strip_image = Image(stream)
                max_w = 268 * mm
                max_h = 43 * mm
                scale = min(max_w / strip_image.imageWidth, max_h / strip_image.imageHeight, 1)
                strip_image.drawWidth = strip_image.imageWidth * scale
                strip_image.drawHeight = strip_image.imageHeight * scale
                strip_image.hAlign = "CENTER"
                story.append(KeepTogether([
                    Paragraph(f"Strip {strip_number}: {strip_start:.1f}-{strip_end:.1f} seconds", small),
                    Spacer(1, 2),
                    strip_image,
                    Spacer(1, 4),
                ]))
            strip_number += 1
            strip_start = strip_end
    else:
        story.append(Paragraph("No waveform samples are stored for this ECG record.", body))

    story.append(KeepTogether([
        Paragraph("Report note", section_style),
        Paragraph(
            "This report reproduces the ECG measurements and waveform stored in CareAI for the selected patient record. "
            "The trace may be simulated or vendor-provided. CareAI does not infer a diagnosis from this waveform, and this document does not replace the source-device record, clinician assessment or diagnostic ECG interpretation.",
            small,
        ),
    ]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
