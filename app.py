# app.py
# PSA-style Pokémon Card Pre-Grader (FastAPI)
# ------------------------------------------------------------
# Features:
# - /health: simple health check
# - /analyze: fetches front/back images, measures centering,
#             checks corners/edges/surface, runs image QA,
#             aggregates to PSA-style grade with strict JSON.
# - Add ?human=1 to include a human-readable summary.
#
# Notes:
# - Heuristics intentionally conservative for PSA-10 targeting.
# - Computer-vision methods are pragmatic best-effort; they won’t
#   replace PSA but map to the rubric you provided.
# ------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, AnyHttpUrl
from typing import Optional, Dict, Any, List, Tuple
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="PSA-Style Pokémon Card Pre-Grader", version="1.0.0")

# ---------- PSA thresholds (conservative) ----------
FRONT_TOL_STRICT = (55, 45)   # preferred for PSA-10 targeting
FRONT_TOL_LENIENT = (60, 40)  # rare edge case
BACK_TOL = (75, 25)

# ---------- Request model ----------
class AnalyzeBody(BaseModel):
    front_url: AnyHttpUrl
    back_url: Optional[AnyHttpUrl] = None
    set_hint: Optional[str] = None  # e.g., "WOTC holo", "modern FA", "JP back"
    strict_front: Optional[bool] = True  # enforce 55/45 vs allow edge-case 60/40

# ---------- Utilities ----------
def fetch_image(url: str) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        im = Image.open(BytesIO(resp.content)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch or parse image: {e}")

def resize_max(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def detect_card_quad(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return a perspective-rectified top-down view of the card and the skew angle in degrees."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, 0.0
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) != 4:
        # Fallback: minAreaRect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)
    # order points
    pts = approx.reshape(-1,2).astype(np.float32)
    # sort by y then x to approximate tl,tr,br,bl
    srt = pts[np.lexsort((pts[:,0], pts[:,1]))]
    top = srt[:2][np.argsort(srt[:2,0])]
    bottom = srt[2:][np.argsort(srt[2:,0])]
    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]
    # width/height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl,tr,br,bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    # Skew angle estimation
    angle = np.degrees(np.arctan2((tr[1]-tl[1]), (tr[0]-tl[0]) + 1e-6))
    return warped, float(angle)

def find_inner_frame(img: np.ndarray, set_hint: Optional[str]) -> Tuple[int,int,int,int]:
    """
    Estimate the printed inner frame (yellow border box for standard cards).
    Return (left_px, right_px, top_px, bottom_px) margins from outer edge to the inner frame.
    Fallback: use a strong edge map near the perimeter.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Emphasize border changes
    edges = cv2.Canny(gray, 50, 150)
    # For standard yellow-border Pokémon: search for the prominent rectangular ring
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    # scan lines inward to find first strong edge from each side
    def first_edge_from(side: str) -> int:
        if side == "left":
            for x in range(5, w//2):
                col = edges[:,x]
                if col.sum() > (0.02 * 255 * h):  # sufficient edges in column
                    return x
            return w//10
        if side == "right":
            for x in range(w-6, w//2, -1):
                col = edges[:,x]
                if col.sum() > (0.02 * 255 * h):
                    return w-1-x
            return w//10
        if side == "top":
            for y in range(5, h//2):
                row = edges[y,:]
                if row.sum() > (0.02 * 255 * w):
                    return y
            return h//10
        if side == "bottom":
            for y in range(h-6, h//2, -1):
                row = edges[y,:]
                if row.sum() > (0.02 * 255 * w):
                    return h-1-y
            return h//10
        return 0

    left = first_edge_from("left")
    right = first_edge_from("right")
    top = first_edge_from("top")
    bottom = first_edge_from("bottom")
    # Ensure positive
    left = max(1, int(left))
    right = max(1, int(right))
    top = max(1, int(top))
    bottom = max(1, int(bottom))
    return left, right, top, bottom

def ratio_from_borders(a_px: int, b_px: int) -> str:
    if a_px <= 0 or b_px <= 0:
        return "N/A"
    total = a_px + b_px
    a_pct = round((a_px / total) * 100)
    b_pct = 100 - a_pct
    major, minor = max(a_pct, b_pct), min(a_pct, b_pct)
    return f"{major}/{minor}"

def worst_major(r1: str, r2: str) -> int:
    try:
        m1 = int(r1.split("/")[0])
        m2 = int(r2.split("/")[0])
        return max(m1, m2)
    except:
        return 100

def passes_psa10_centering(front_lr: str, front_tb: str, back_lr: str, back_tb: str, strict_front=True) -> Tuple[bool, str]:
    front_limit = 55 if strict_front else 60
    f_worst = worst_major(front_lr, front_tb)
    b_worst = worst_major(back_lr, back_tb)
    notes = []
    if f_worst > front_limit:
        notes.append(f"Front worst axis {f_worst}/{100-f_worst} exceeds {front_limit}/{100-front_limit}.")
    if b_worst > 75:
        notes.append(f"Back worst axis {b_worst}/{100-b_worst} exceeds 75/25.")
    return (len(notes) == 0, "; ".join(notes) if notes else "Within PSA-10 centering thresholds.")

# ---------- Heuristic feature detectors ----------
def detect_glare_and_lowres(img: np.ndarray) -> List[str]:
    flags = []
    h, w = img.shape[:2]
    if min(h, w) < 1200:
        flags.append("low_resolution")
    # glare: proportion of near-white pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    glare_ratio = float((v > 245).sum()) / (h*w)
    if glare_ratio > 0.02:
        flags.append("glare_or_specular_highlights")
    # sleeve/toploader cue: strong parallel reflections near edges
    edges = cv2.Canny(v, 80, 200)
    border_band = np.zeros_like(edges)
    band = 12
    border_band[:band,:] = 1; border_band[-band:,:] = 1; border_band[:,:band] = 1; border_band[:,-band:] = 1
    band_ratio = float((edges * border_band).sum()) / (255 * (2*band*(w+h-2*band)))
    if band_ratio > 0.20:
        flags.append("possible_sleeve_or_reflection_bands")
    return flags

def estimate_corner_whitening(img: np.ndarray) -> Tuple[int, List[str]]:
    """
    Inspect small corner patches for high-L (near white) pixels against colored border.
    Returns severity 0-5 and findings.
    """
    h, w = img.shape[:2]
    pad = int(0.07 * min(h, w))  # small patch
    patches = {
        "tl": img[0:pad, 0:pad],
        "tr": img[0:pad, w-pad:w],
        "bl": img[h-pad:h, 0:pad],
        "br": img[h-pad:h, w-pad:w]
    }
    findings = []
    speck_count = 0
    ding_like = 0
    for name, p in patches.items():
        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        # white speck threshold
        white_mask = (gray > 230).astype(np.uint8)
        count = int(white_mask.sum())
        if count > (0.001 * p.size):  # a few pixels allowed
            speck_count += 1
            findings.append(f"{name} corner: whitening specks detected")
        # ding proxy: contiguous component area
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max([cv2.contourArea(c) for c in cnts], default=0.0)
        if max_area > 12.0:
            ding_like += 1
            findings.append(f"{name} corner: possible ding/rounding (area≈{int(max_area)})")
    # severity mapping
    if speck_count == 0 and ding_like == 0:
        sev = 0
    elif speck_count <= 1 and ding_like == 0:
        sev = 1
    elif speck_count <= 2 and ding_like <= 1:
        sev = 2
    elif speck_count >= 3 or ding_like >= 1:
        sev = 3
    else:
        sev = 2
    return sev, findings

def estimate_edge_whitening(img: np.ndarray) -> Tuple[int, List[str]]:
    """
    Scan narrow edge bands for whitening (near-white runs).
    """
    h, w = img.shape[:2]
    band = max(4, int(0.01 * min(h, w)))
    findings = []

    def band_stats(band_img, label):
        gray = cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY)
        white = (gray > 230).astype(np.uint8)
        # count horizontal/vertical runs via morphology
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1)) if label in ("top","bottom") else cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))
        run = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
        run_len = int(run.sum())
        specks = int(white.sum())
        return specks, run_len

    top = img[:band, :]
    bottom = img[-band:, :]
    left = img[:, :band]
    right = img[:, -band:]

    stats = {
        "top": band_stats(top, "top"),
        "bottom": band_stats(bottom, "bottom"),
        "left": band_stats(left, "left"),
        "right": band_stats(right, "right")
    }

    speck_total = 0
    line_hits = 0
    for side, (specks, run_len) in stats.items():
        speck_total += specks
        if run_len > 300:  # continuous whitening proxy
            line_hits += 1
            findings.append(f"{side} edge: continuous whitening line likely")
        elif specks > 120:
            findings.append(f"{side} edge: multiple whitening flecks")

    # severity mapping
    if speck_total < 80 and line_hits == 0:
        sev = 0 if speck_total < 30 else 1
    elif line_hits == 0 and speck_total < 300:
        sev = 2
    elif line_hits >= 1 or speck_total >= 300:
        sev = 3
    else:
        sev = 2
    return sev, findings

def estimate_surface_defects(img: np.ndarray, hint: Optional[str]) -> Tuple[int, List[str]]:
    """
    Heuristic: detect print/scratch lines and stains/pressure marks.
    """
    findings = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Line-like defects (Holo print lines / scratches)
    edges = cv2.Canny(gray_blur, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=int(0.25*min(img.shape[:2])), maxLineGap=10)
    line_count = 0 if lines is None else len(lines)
    if 1 <= line_count <= 2:
        findings.append("one-to-two faint linear marks (possible holo/roller line)")
        sev_lines = 1
    elif 3 <= line_count <= 6:
        findings.append("multiple linear marks visible")
        sev_lines = 2
    elif line_count > 6:
        findings.append("prominent linear scratches/lines")
        sev_lines = 3
    else:
        sev_lines = 0

    # Stain / blotch proxy: high local variance blobs
    lap = cv2.Laplacian(gray_blur, cv2.CV_64F)
    var_map = cv2.GaussianBlur((lap**2).astype(np.float32), (9,9), 0)
    thresh = np.percentile(var_map, 99.5)
    stain_mask = (var_map > thresh).astype(np.uint8)
    stain_area = int(stain_mask.sum())
    if stain_area > 500:
        findings.append("possible stain/pressure mark region(s)")
        sev_stain = 3
    elif stain_area > 120:
        findings.append("minor localized blemishes")
        sev_stain = 1
    else:
        sev_stain = 0

    sev = max(sev_lines, sev_stain)
    return sev, findings

def human_summary(json_out: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Verdict: {json_out['predicted_grade']} ({json_out['ten_viability']} 10-viability, {json_out['confidence']} confidence)")
    c = json_out["centering"]
    lines.append(f"- Centering — front L/R {c['front_lr']}, T/B {c['front_tb']}; back L/R {c['back_lr']}, T/B {c['back_tb']}. "
                 f"PSA-10 centering pass: {c['passes_psa10_thresholds']}.")
    if c["notes"]:
        lines.append(f"  Notes: {c['notes']}")
    lines.append(f"- Corners (sev {json_out['corners']['severity']}): {', '.join(json_out['corners']['findings']) or 'clean'}")
    lines.append(f"- Edges (sev {json_out['edges']['severity']}): {', '.join(json_out['edges']['findings']) or 'clean'}")
    lines.append(f"- Surface front/back (sev {json_out['surface']['front_severity']}/{json_out['surface']['back_severity']}): "
                 f"{', '.join(json_out['surface']['front_findings']) or 'front clean'}; "
                 f"{', '.join(json_out['surface']['back_findings']) or 'back clean'}")
    if json_out["disqualifiers"]:
        lines.append(f"- Disqualifiers: {', '.join(json_out['disqualifiers'])}")
    if json_out["image_quality_flags"]:
        lines.append(f"- Image quality flags: {', '.join(json_out['image_quality_flags'])}")
    lines.append(f"- Rationale: {json_out['overall_rationale']}")
    return "\n".join(lines)

def aggregate_grade(front_lr, front_tb, back_lr, back_tb,
                    corners_sev, edges_sev, surf_front_sev, surf_back_sev,
                    strict_front: bool) -> Tuple[str, str, List[str]]:
    """
    Returns (predicted_grade, ten_viability, disqualifiers[])
    """
    disq = []
    # Centering pass/fail
    pass_center, center_note = passes_psa10_centering(front_lr, front_tb, back_lr, back_tb, strict_front=strict_front)
    # Determine worst majors for mapping
    f_worst = worst_major(front_lr, front_tb)
    b_worst = worst_major(back_lr, back_tb)

    # Rule-based disqualifiers for 10
    if not pass_center:
        disq.append("centering_exceeds_psa10_threshold")

    # Surface severe implies non-10
    if surf_front_sev >= 3 or surf_back_sev >= 3:
        disq.append("surface_defect_visible")

    if corners_sev >= 3:
        disq.append("corner_damage_visible")

    if edges_sev >= 3:
        disq.append("edge_whitening_continuous_or_heavy")

    # Ten viability
    if disq:
        ten_viability = "none"
    else:
        # all severities 0–1 and centering pass
        core_sev = (0 if f_worst <= (55 if strict_front else 60) else 1) + corners_sev + edges_sev + surf_front_sev
        if (corners_sev <= 1 and edges_sev <= 1 and surf_front_sev <= 1 and surf_back_sev <= 1 and pass_center):
            ten_viability = "high" if core_sev <= 1 else "medium"
        else:
            ten_viability = "low"

    # Predicted grade (conservative mapping)
    max_sev = max(corners_sev, edges_sev, surf_front_sev, surf_back_sev)
    predicted = "PSA 10"
    # centering constraints for 10
    if (strict_front and f_worst > 55) or ((not strict_front) and f_worst > 60) or b_worst > 75:
        predicted = "PSA 9"
    if max_sev >= 1:
        predicted = "PSA 9"
    if max_sev >= 2:
        # if centering also outside 60/40 or back outside 80/20, drop further
        if f_worst > 60 or b_worst > 80:
            predicted = "PSA 8"
        else:
            predicted = "PSA 9"
    if max_sev >= 3:
        predicted = "PSA 8"
    if max_sev >= 4:
        predicted = "≤7"

    # Edge case: front ~60/40 with everything else flawless
    if predicted == "PSA 9" and (f_worst <= 60 and b_worst <= 75) and corners_sev == edges_sev == surf_front_sev == surf_back_sev == 0:
        # borderline 10, keep 9 but note viability (handled above)
        pass

    return predicted, ten_viability, disq

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
def analyze(body: AnalyzeBody, human: int = Query(0, ge=0, le=1)):
    # Load images
    front = fetch_image(str(body.front_url))
    back = fetch_image(str(body.back_url)) if body.back_url else None

    # Resize for compute/time balance
    front = resize_max(front, 2200)
    if back is not None:
        back = resize_max(back, 2200)

    # Rectify & skew
    front_rect, front_angle = detect_card_quad(front)
    front_flags = detect_glare_and_lowres(front_rect)
    if abs(front_angle) > 2.0:
        front_flags.append("skew_over_2deg")

    if back is not None:
        back_rect, back_angle = detect_card_quad(back)
        back_flags = detect_glare_and_lowres(back_rect)
        if abs(back_angle) > 2.0:
            back_flags.append("skew_over_2deg")
    else:
        # If back is missing, we still run with front-only assumptions
        back_rect, back_angle, back_flags = None, 0.0, ["back_image_missing"]

    # Centering measurements
    fl, fr, ft, fb = find_inner_frame(front_rect, body.set_hint)
    front_lr = ratio_from_borders(fl, fr)
    front_tb = ratio_from_borders(ft, fb)

    if back_rect is not None:
        bl, br, bt, bb = find_inner_frame(back_rect, body.set_hint)
        back_lr = ratio_from_borders(bl, br)
        back_tb = ratio_from_borders(bt, bb)
    else:
        # Missing back — pessimistically mark as 75/25 pass, but note uncertainty
        back_lr = "75/25"
        back_tb = "75/25"

    # PSA-10 centering gate
    center_pass, center_notes = passes_psa10_centering(
        front_lr, front_tb, back_lr, back_tb,
        strict_front=(body.strict_front if body.strict_front is not None else True)
    )

    # Corners / edges / surface (front)
    corners_sev_f, corners_find_f = estimate_corner_whitening(front_rect)
    edges_sev_f, edges_find_f = estimate_edge_whitening(front_rect)
    surf_sev_f, surf_find_f = estimate_surface_defects(front_rect, body.set_hint)

    # Back surface & edges/corners are also relevant
    if back_rect is not None:
        corners_sev_b, corners_find_b = estimate_corner_whitening(back_rect)
        edges_sev_b, edges_find_b = estimate_edge_whitening(back_rect)
        surf_sev_b, surf_find_b = estimate_surface_defects(back_rect, body.set_hint)
    else:
        corners_sev_b, corners_find_b = 1, ["back not analyzed"]
        edges_sev_b, edges_find_b = 1, ["back not analyzed"]
        surf_sev_b,  surf_find_b  = 1, ["back not analyzed"]

    # Combine severities (we score corners/edges as the worse of front/back)
    corners_sev = max(corners_sev_f, corners_sev_b)
    edges_sev = max(edges_sev_f, edges_sev_b)
    # Surface: front drives the headline; back matters but often slightly less visually
    surf_front_sev = surf_sev_f
    surf_back_sev = surf_sev_b

    # Aggregate grade
    predicted, ten_viability, disq = aggregate_grade(
        front_lr, front_tb, back_lr, back_tb,
        corners_sev, edges_sev, surf_front_sev, surf_back_sev,
        strict_front=(body.strict_front if body.strict_front is not None else True)
    )

    # Build strict JSON output
    out = {
        "predicted_grade": predicted,
        "ten_viability": ten_viability,
        "centering": {
            "front_lr": front_lr,
            "front_tb": front_tb,
            "back_lr": back_lr,
            "back_tb": back_tb,
            "passes_psa10_thresholds": bool(center_pass),
            "notes": center_notes
        },
        "corners": {
            "severity": int(corners_sev),
            "findings": list(set(corners_find_f + corners_find_b))
        },
        "edges": {
            "severity": int(edges_sev),
            "findings": list(set(edges_find_f + edges_find_b))
        },
        "surface": {
            "front_severity": int(surf_front_sev),
            "front_findings": surf_find_f,
            "back_severity": int(surf_back_sev),
            "back_findings": surf_find_b
        },
        "disqualifiers": disq,
        "overall_rationale": (
            "Conservative PSA-style mapping: centering thresholds prioritize ≤55/45 front and ≤75/25 back; "
            "any stain/indent/continuous whitening/scratch visible at normal view disqualifies a 10; "
            "severity bands aggregate to cap grade when ≥3."
        ),
        "confidence": (
            "high" if all([
                center_pass,
                corners_sev <= 1, edges_sev <= 1, surf_front_sev <= 1, surf_back_sev <= 1,
                len(front_flags) == 0, (back_rect is None or len(back_flags) == 0)
            ]) else ("medium" if max(corners_sev, edges_sev, surf_front_sev, surf_back_sev) <= 2 else "low")
        ),
        "image_quality_flags": list(set(front_flags + (back_flags if back_rect is not None else [])))
    }

    if human == 1:
        return {
            "json": out,
            "summary": human_summary(out)
        }
    return out
