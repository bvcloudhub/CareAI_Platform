"""
PDF report for one AI imaging analysis.

Every analysis the diagnostics workbench shows - lung CT, chest X-ray, Wound AI
(wound type or surgical infection-risk screen) and the rule-based skin demo -
can be downloaded as a PDF, to file with the case or send with a referral.

The result the page rendered is written next to the uploaded image when the
analysis runs (uploads/diagnostics/reports/<run id>.json), so the download
rebuilds exactly the numbers the clinician saw without running the model again.
Nothing is added to the database.

An analysis is not linked to a patient, so the PDF carries no patient identity:
it leaves a line for the case reference and one for the reviewing clinician.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from services.lung_imaging_service import CAM_DIR, UPLOAD_DIR

REPORT_DIR = UPLOAD_DIR / "reports"
_RUN_ID = re.compile(r"^[0-9a-f]{6,32}$")

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

INK = colors.HexColor("#14241d")
MUTED = colors.HexColor("#5d6f66")
LINE = colors.HexColor("#d7e0da")
BRAND = colors.HexColor("#14503c")
SOFT = colors.HexColor("#eef4f0")
WARN_BG = colors.HexColor("#fdf3e7")
WARN_INK = colors.HexColor("#8a5200")
RISK_INK = {"critical": colors.HexColor("#b3261e"), "high": colors.HexColor("#b3261e"),
            "elevated": colors.HexColor("#b3261e"), "moderate": colors.HexColor("#a35b00"),
            "medium": colors.HexColor("#a35b00"), "low": colors.HexColor("#1f6f4f"),
            "normal": colors.HexColor("#1f6f4f")}

FOOTER_NOTE = ("Research prototype - not a medical device. Every output needs qualified clinical review.")
GRADCAM_NOTE = ("Warmer regions influenced the prediction most. These are saliency cues, "
                "not clinical boundaries.")
PIXEL_NOTE = ("Pixels only: the photo has no physical scale, so compare sizes only between photos "
              "taken at the same distance and zoom.")

# Helvetica has no glyphs for the arrows and dashes used in the interface text.
_SUBSTITUTIONS = {"→": "->", "←": "<-", "—": " - ", "–": "-", "≥": ">=",
                  "≤": "<=", "×": "x", "•": "-", "…": "...", "“": '"',
                  "”": '"', "‘": "'", "’": "'", " ": " "}

_STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("careai_body", parent=_STYLES["BodyText"], fontName="Helvetica",
                      fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=0)
SMALL = ParagraphStyle("careai_small", parent=BODY, fontSize=8, leading=11, textColor=MUTED)
SECTION = ParagraphStyle("careai_section", parent=BODY, fontName="Helvetica-Bold", fontSize=8,
                         leading=11, textColor=MUTED, spaceBefore=12, spaceAfter=5)
TITLE = ParagraphStyle("careai_title", parent=BODY, fontName="Helvetica-Bold", fontSize=17, leading=21)
LEAD = ParagraphStyle("careai_lead", parent=BODY, fontSize=11, leading=15.5)
CAPTION = ParagraphStyle("careai_caption", parent=SMALL, alignment=1)
HEADLINE = ParagraphStyle("careai_headline", parent=BODY, fontName="Helvetica-Bold", fontSize=22,
                          leading=25, textColor=BRAND)


# ---------------------------------------------------------------------------
# Storage: the rendered result, kept beside the image it came from
# ---------------------------------------------------------------------------

def save_result(result: Optional[dict]) -> None:
    """Keep a finished analysis so its PDF can be built on request."""
    run_id = str((result or {}).get("run_id") or "")
    if not _RUN_ID.match(run_id):
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")


def load_result(run_id: str) -> Optional[dict]:
    if not _RUN_ID.match(str(run_id or "")):
        return None
    path = REPORT_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def filename_for(result: dict) -> str:
    module = re.sub(r"[^a-z0-9]+", "-", str(result.get("module") or "analysis").lower()).strip("-")
    run_id = re.sub(r"[^0-9a-z]+", "", str(result.get("run_id") or "report").lower())
    return f"careai-{module or 'analysis'}-{run_id or 'report'}.pdf"


# ---------------------------------------------------------------------------
# Flowable helpers
# ---------------------------------------------------------------------------

def _text(value) -> str:
    text = "" if value is None else str(value)
    for bad, good in _SUBSTITUTIONS.items():
        text = text.replace(bad, good)
    # Anything else outside Helvetica's encoding would print as a black box.
    return escape(text.encode("latin-1", "replace").decode("latin-1"))


def _para(value, style=BODY) -> Paragraph:
    return Paragraph(_text(value), style)


def _section(title: str) -> Paragraph:
    return _para(str(title).upper(), SECTION)


def _num(value, spec: str = "{:.2f}", default: str = "-") -> str:
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return default


def _bar(pct, width: float, colour=BRAND) -> Drawing:
    try:
        pct = max(0.0, min(100.0, float(pct)))
    except (TypeError, ValueError):
        pct = 0.0
    drawing = Drawing(width, 5)
    drawing.add(Rect(0, 0, width, 5, fillColor=SOFT, strokeColor=None))
    if pct > 0:
        drawing.add(Rect(0, 0, max(width * pct / 100.0, 1.2), 5, fillColor=colour, strokeColor=None))
    return drawing


def _facts(rows: Sequence[Sequence[str]]) -> Table:
    data = [[_para(label, SMALL), _para(value)] for label, value in rows]
    table = Table(data, colWidths=[38 * mm, CONTENT_WIDTH - 38 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]))
    return table


def _bullets(items: Sequence[str]) -> Optional[ListFlowable]:
    entries = [ListItem(_para(item), leftIndent=10) for item in items if str(item or "").strip()]
    if not entries:
        return None
    return ListFlowable(entries, bulletType="bullet", bulletFontSize=5, bulletOffsetY=2,
                        start="circle", leftIndent=10, spaceBefore=2)


def _score_rows(rows: Sequence[Sequence], colour=BRAND) -> Table:
    """label | bar | value - class probabilities and ensemble components."""
    bar_width = 52 * mm
    data = [[_para(label), _bar(pct, bar_width, colour), _para(value)] for label, pct, value in rows]
    table = Table(data, colWidths=[CONTENT_WIDTH - bar_width - 22 * mm, bar_width, 22 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _headline(value: str, caption: str, badge: Optional[str] = None, badge_colour=BRAND) -> Table:
    badge_style = ParagraphStyle("careai_badge", parent=BODY, fontName="Helvetica-Bold", fontSize=9,
                                 leading=12.5, textColor=badge_colour, alignment=2)
    right = [_para(badge, badge_style)] if badge else [Spacer(1, 1)]
    table = Table([[[_para(value, HEADLINE), _para(caption, SMALL)], right]],
                  colWidths=[CONTENT_WIDTH * 0.6, CONTENT_WIDTH * 0.4], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), SOFT), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def _notice(text: str, background=WARN_BG, ink=WARN_INK) -> Table:
    style = ParagraphStyle("careai_notice", parent=BODY, fontSize=9, leading=12.5, textColor=ink)
    table = Table([[_para(text, style)]], colWidths=[CONTENT_WIDTH], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _image_path(url) -> Optional[Path]:
    """Map a URL used by the page back to a file on disk, by basename only."""
    url = str(url or "")
    name = Path(url).name
    if not name:
        return None
    if url.startswith("/static/cam/"):
        candidate = CAM_DIR / name
    elif url.startswith("/diagnostics-image/"):
        candidate = UPLOAD_DIR / name
    else:
        return None
    return candidate if candidate.exists() else None


def _scaled(path: Path, longest: int = 1300):
    """Embed a print-resolution copy, not the original: a chest X-ray straight
    from the scanner turned a two-page report into 4.7 MB."""
    from PIL import Image as PILImage
    with PILImage.open(path) as source:
        image = source.convert("RGB")
    if max(image.size) > longest:
        image.thumbnail((longest, longest))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=86, optimize=True)
    buffer.seek(0)
    return buffer, image.size


def _figure(path: Path, caption: str, width: float) -> List:
    try:
        stream, (source_w, source_h) = _scaled(path)
    except Exception:
        return []
    if not source_w or not source_h:
        return []
    height = width * source_h / source_w
    if height > 70 * mm:
        height = 70 * mm
        width = height * source_w / source_h
    return [Image(stream, width=width, height=height), Spacer(1, 3), _para(caption, CAPTION)]


def _figures(pairs: Sequence[Sequence]) -> Optional[Table]:
    """The uploaded image beside its overlays."""
    available = [(url, caption) for url, caption in pairs if _image_path(url)]
    if not available:
        return None
    column = CONTENT_WIDTH / len(available)
    cells = []
    for url, caption in available:
        figure = _figure(_image_path(url), caption, column - 6 * mm)
        if figure:
            cells.append(figure)
    if not cells:
        return None
    table = Table([cells], colWidths=[CONTENT_WIDTH / len(cells)] * len(cells), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, -1), 0), ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def _kv_grid(mapping: Dict[str, object]) -> Optional[Table]:
    items = [(key, value) for key, value in (mapping or {}).items() if str(value or "").strip()]
    if not items:
        return None
    rows = []
    for index in range(0, len(items), 2):
        pair = items[index:index + 2]
        cells = [[_para(key, SMALL), _para(value)] for key, value in pair]
        if len(pair) == 1:
            cells.append([Spacer(1, 1)])
        rows.append(cells)
    table = Table(rows, colWidths=[CONTENT_WIDTH / 2] * 2, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (0, -1), 0), ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]))
    return table


def _tiles(values: Sequence[Sequence[str]]) -> Table:
    cells = [[_para(label, SMALL), _para(value, LEAD)] for label, value in values]
    table = Table([cells], colWidths=[CONTENT_WIDTH / len(cells)] * len(cells), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.4, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _signoff() -> Table:
    labels = ["Patient / case reference", "Reviewed by (name, role)", "Date"]
    table = Table([[_para(label, SMALL) for label in labels], [Spacer(1, 12)] * 3],
                  colWidths=[CONTENT_WIDTH / 3] * 3, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LINEBELOW", (0, 1), (-1, 1), 0.6, MUTED),
        ("LEFTPADDING", (0, 0), (0, -1), 0), ("RIGHTPADDING", (-1, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


# ---------------------------------------------------------------------------
# The three kinds of result
# ---------------------------------------------------------------------------

def _probability_rows(probabilities) -> List[Sequence]:
    rows = []
    for item in probabilities or []:
        pct = item.get("pct")
        rows.append((item.get("label"), pct, f"{_num(pct, '{:.1f}')}%"))
    return rows


def _imaging_story(result: dict) -> List:
    risk = str(result.get("risk") or "").lower()
    story = [_headline(f"{_num(result.get('confidence_pct'), '{:.1f}')}%",
                       f"Confidence - {result.get('prediction_display') or result.get('prediction') or ''}",
                       badge=f"RISK {risk.upper()}" if risk else None,
                       badge_colour=RISK_INK.get(risk, BRAND)),
             Spacer(1, 9), _para(result.get("finding"), LEAD)]
    if result.get("low_confidence"):
        story += [Spacer(1, 9), _notice("Low model confidence - treat this output with particular caution.")]
    rows = _probability_rows(result.get("probabilities"))
    if rows:
        story += [_section("Class probabilities"), _score_rows(rows)]
    return story


def _wound_story(result: dict) -> List:
    risk = result.get("surgical_risk") or {}
    review = str(result.get("review_status") or "")
    review_colour = RISK_INK["moderate"] if result.get("uncertain") or risk.get(
        "clinical_review_recommended") else BRAND
    if result.get("mode") == "wound_type":
        story = [_headline(f"{_num(result.get('confidence_pct'), '{:.1f}')}%",
                           f"Confidence - {result.get('prediction_display') or ''}",
                           badge=review, badge_colour=review_colour)]
    else:
        story = [_headline(_num(risk.get("ensemble_score")),
                           f"AI ensemble score - threshold {_num(risk.get('decision_threshold'))}",
                           badge=risk.get("category_display") or review,
                           badge_colour=RISK_INK.get(str(risk.get("category") or "").lower(), BRAND))]
    story += [Spacer(1, 9), _para(result.get("finding"), LEAD)]
    for warning in result.get("warnings") or []:
        story += [Spacer(1, 7), _notice(warning)]

    rows = _probability_rows(result.get("probabilities"))
    if rows:
        story += [_section("Wound type probabilities"), _score_rows(rows)]
        heldout = result.get("heldout") or {}
        if heldout.get("precision") is not None and heldout.get("sensitivity") is not None:
            story += [Spacer(1, 4), _para(
                f"Held-out test for this class: right {_num(heldout['precision'] * 100, '{:.0f}')}% of the times it "
                f"says this, and it finds {_num(heldout['sensitivity'] * 100, '{:.0f}')}% of these wounds.", SMALL)]

    if risk:
        weights = risk.get("weights") or {}
        story += [_section("Surgical infection-risk screen"), _score_rows([
            (f"BiomedCLIP V9 x{_num(weights.get('v9'))}", (risk.get("v9_score") or 0) * 100,
             _num(risk.get("v9_score"))),
            (f"DenseNet121 V8A x{_num(weights.get('v8a'))}", (risk.get("v8a_score") or 0) * 100,
             _num(risk.get("v8a_score")))],
            colour=RISK_INK.get(str(risk.get("category") or "").lower(), BRAND))]
        reasons = _bullets([f"{str(reason)[0].upper()}{str(reason)[1:]}." for reason in risk.get("review_reasons") or []])
        if reasons:
            story += [Spacer(1, 5), reasons]
        story += [Spacer(1, 5), _para(
            f"{risk.get('clinical_note') or ''} This is a model score, not a calibrated probability of infection. "
            "Evaluated on development folds only, never on a held-out test set.", SMALL)]
    return story


def _wound_extras(result: dict) -> List:
    story: List = []
    boundary = result.get("boundary") or {}
    if boundary.get("detected"):
        story += [_section("Wound measurements"), _tiles([
            ("Area", f"{_num(boundary.get('area_px'), '{:,.0f}')} px2"),
            ("Length", f"{_num(boundary.get('length_px'), '{:.0f}')} px"),
            ("Width", f"{_num(boundary.get('width_px'), '{:.0f}')} px"),
            ("Perimeter", f"{_num(boundary.get('perimeter_px'), '{:.0f}')} px"),
            ("Share of frame", f"{_num(boundary.get('coverage_pct'), '{:.1f}')}%")]),
            Spacer(1, 5), _para(PIXEL_NOTE, SMALL)]
    elif result.get("boundary_note"):
        story += [_section("Wound measurements"), _para(result["boundary_note"], SMALL)]

    quality = (result.get("quality") or {}).get("metrics") or {}
    gate = result.get("gate") or {}
    if quality or gate:
        rows = []
        if quality:
            rows.append(("Image quality", f"{str((result.get('quality') or {}).get('status', '')).title()} - "
                                          f"brightness {quality.get('brightness')}, contrast {quality.get('contrast')}, "
                                          f"sharpness {quality.get('blur_score')}, "
                                          f"{quality.get('width')}x{quality.get('height')} px"))
        if gate:
            rows.append(("Wound check", f"{str(gate.get('status', '')).title()} - wound score "
                                        f"{_num((gate.get('wound_probability') or 0) * 100, '{:.0f}')}%, "
                                        f"threshold {_num(gate.get('threshold'))}"))
        story += [_section("Checks before the models ran"), _facts(rows)]

    limitations = _bullets(result.get("limitations") or [])
    if limitations:
        story += [_section("Limitations"), limitations]
    return story


def _demo_story(result: dict) -> List:
    story = [_headline(f"{_num(result.get('confidence'), '{:.0f}')}%", "Confidence - rule-based demo output",
                       badge="SYNTHETIC DEMO", badge_colour=MUTED),
             Spacer(1, 9), _para(result.get("result"), LEAD)]
    return story          # the recommendation prints with every other module's next steps


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def _decorate(canvas, doc, title: str) -> None:
    canvas.saveState()
    canvas.setFillColor(BRAND)
    canvas.rect(0, PAGE_HEIGHT - 17 * mm, PAGE_WIDTH, 17 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 11 * mm, "Care.AI")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 11 * mm, title)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 15 * mm, PAGE_WIDTH - MARGIN, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN, 11 * mm, FOOTER_NOTE)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(result: dict, clinician: Optional[str] = None) -> bytes:
    """Render one analysis as a PDF and return its bytes."""
    if result.get("module") == "wound":
        kind, body = "wound", _wound_story
    elif result.get("kind") == "demo":
        kind, body = "demo", _demo_story
    else:
        kind, body = "imaging", _imaging_story

    header = "AI imaging analysis report"
    story: List = [
        _para(result.get("module_label") or "AI analysis", TITLE),
        _para(f"{result.get('modality') or 'Image analysis'} - Care.AI AI diagnostics", SMALL),
        Spacer(1, 11),
        _facts([("Analysed", result.get("created_at") or "-"),
                ("Report generated", datetime.now().isoformat(sep=" ", timespec="seconds")),
                ("Source image", result.get("source_filename") or "-"),
                ("Run ID", result.get("run_id") or "-"),
                ("Model", result.get("model_version") or result.get("module_label") or "-"),
                ("Downloaded by", clinician or "-")]),
        _section("AI result"),
    ]
    story += body(result)

    recommendations = _bullets(result.get("recommendations") or [])
    if recommendations:
        story += [_section("Recommended next steps"), recommendations]

    figures = _figures([(result.get("upload_url"), "Uploaded image"),
                        (result.get("cam_url"), "Grad-CAM overlay"),
                        ((result.get("boundary") or {}).get("overlay_url"), "Wound boundary")])
    if figures:
        note = GRADCAM_NOTE if result.get("cam_url") else ""
        story += [_section("What the model looked at"), figures]
        if note:
            story += [Spacer(1, 5), _para(note, SMALL)]

    if kind == "wound":
        story += _wound_extras(result)

    technical = dict(result.get("technical") or {})
    if kind == "demo" and result.get("model_version"):
        technical.setdefault("Model version", result["model_version"])
    grid = _kv_grid(technical)
    if grid:
        story += [_section("Technical details"), grid]

    story += [_section("Clinical review"),
              _notice(result.get("disclaimer") or FOOTER_NOTE),
              Spacer(1, 12), _signoff()]

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=25 * mm,
        bottomMargin=20 * mm, title=f"Care.AI {header} {result.get('run_id') or ''}".strip(),
        author="Care.AI", subject=str(result.get("module_label") or ""))
    document.build(story, onFirstPage=lambda c, d: _decorate(c, d, header),
                   onLaterPages=lambda c, d: _decorate(c, d, header))
    return buffer.getvalue()
