import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os, io, math, datetime, threading

import numpy as np
from PIL import Image, ImageTk
from PIL.ExifTags import TAGS, GPSTAGS
import cv2

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, PageBreak,
    )
    from pypdf import PdfReader, PdfWriter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
APP_TITLE    = "Digital Image Forensics Analyzer"
APP_SUBTITLE = "Image Authenticity Verification Using Digital Image Forensic Techniques"
VERSION      = "v3.0 – ELA + EXIF + Education"
MAX_FILE_MB  = 10
MAX_FILE_SIZE= MAX_FILE_MB * 1024 * 1024
INACTIVITY_TIMEOUT_SECONDS = 30
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
ELA_QUALITY  = 90
ELA_SCALE    = 10
ELA_HIGH     = 60

C_BG      = "#1A1A2E"
C_PANEL   = "#16213E"
C_CARD    = "#0F3460"
C_ACCENT  = "#7C3AED"
C_CYAN    = "#06B6D4"
C_TEXT    = "#E2E8F0"
C_MUTED   = "#94A3B8"
C_SUCCESS = "#22C55E"
C_WARN    = "#F59E0B"
C_DANGER  = "#EF4444"
C_BORDER  = "#1E3A5F"
C_ENTRY   = "#0D2137"

# ════════════════════════════════════════════════════════════════════════════
#  EXIF PARSER  –  Full structured extraction
# ════════════════════════════════════════════════════════════════════════════
class EXIFParser:
    """
    Extracts and structures all EXIF metadata from a JPEG/PNG image.
    Covers: datetime, camera make/model, GPS, camera settings, resolution,
    software, and all other available tags.
    """

    # rational helper 
    @staticmethod
    def _rational_to_float(val) -> float:
        """Convert IFDRational or (num, den) tuple to float."""
        try:
            if hasattr(val, "numerator"):          # IFDRational
                return val.numerator / val.denominator if val.denominator else 0.0
            if isinstance(val, tuple) and len(val) == 2:
                return val[0] / val[1] if val[1] else 0.0
            return float(val)
        except Exception:
            return 0.0

    # GPS decoder 
    @staticmethod
    def _decode_gps(gps_info: dict) -> dict:
        """
        Convert raw GPS IFD dict → human-readable lat/lon/altitude/timestamp.
        Returns dict with keys: latitude, longitude, altitude, gps_timestamp,
                                lat_decimal, lon_decimal, maps_link
        """
        result = {}
        if not gps_info:
            return result

        def dms_to_decimal(dms, ref):
            try:
                d = EXIFParser._rational_to_float(dms[0])
                m = EXIFParser._rational_to_float(dms[1])
                s = EXIFParser._rational_to_float(dms[2])
                dec = d + m / 60.0 + s / 3600.0
                if ref in ("S", "W"):
                    dec = -dec
                return round(dec, 7)
            except Exception:
                return None

        lat_dms = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef", "N")
        lon_dms = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef", "E")

        if lat_dms and lon_dms:
            lat = dms_to_decimal(lat_dms, lat_ref)
            lon = dms_to_decimal(lon_dms, lon_ref)
            if lat is not None and lon is not None:
                result["lat_decimal"] = lat
                result["lon_decimal"] = lon
                result["latitude"]  = (
                    f"{abs(lat):.6f}° {'N' if lat >= 0 else 'S'}")
                result["longitude"] = (
                    f"{abs(lon):.6f}° {'E' if lon >= 0 else 'W'}")
                result["maps_link"] = (
                    f"https://maps.google.com/?q={lat},{lon}")

        alt_raw = gps_info.get("GPSAltitude")
        alt_ref = gps_info.get("GPSAltitudeRef", 0)
        if alt_raw is not None:
            alt = EXIFParser._rational_to_float(alt_raw)
            sign = -1 if alt_ref == 1 else 1
            result["altitude"] = f"{sign * alt:.1f} m"

        ts = gps_info.get("GPSTimeStamp")
        date = gps_info.get("GPSDateStamp", "")
        if ts:
            try:
                h = int(EXIFParser._rational_to_float(ts[0]))
                m = int(EXIFParser._rational_to_float(ts[1]))
                s = int(EXIFParser._rational_to_float(ts[2]))
                result["gps_timestamp"] = f"{date} {h:02d}:{m:02d}:{s:02d} UTC"
            except (TypeError, ValueError, ZeroDivisionError, IndexError):
                result["gps_timestamp"] = "Invalid GPS timestamp format"

        return result

    # ─ aperture / shutter / ISO helpers 
    @staticmethod
    def _format_aperture(val) -> str:
        try:
            f = EXIFParser._rational_to_float(val)
            return f"f/{f:.1f}" if f else "—"
        except Exception:
            return "—"

    @staticmethod
    def _format_shutter(val) -> str:
        try:
            f = EXIFParser._rational_to_float(val)
            if f == 0:
                return "—"
            if f >= 1:
                return f"{f:.1f}s"
            return f"1/{round(1/f)}s"
        except Exception:
            return "—"

    @staticmethod
    def _format_focal(val) -> str:
        try:
            f = EXIFParser._rational_to_float(val)
            return f"{f:.1f} mm" if f else "—"
        except Exception:
            return "—"

    #  MAIN EXTRACTION 
    @classmethod
    def extract(cls, image_path: str) -> dict:
        """
        Returns a structured dict:
        {
          "basic"    : { format, mode, width, height, filesize },
          "datetime" : { date_taken, date_modified, date_digitized },
          "camera"   : { make, model, lens, flash, focal_length },
          "settings" : { iso, aperture, shutter_speed, exposure_bias,
                         metering_mode, white_balance, exposure_program },
          "resolution": { x_dpi, y_dpi, unit, pixel_width, pixel_height },
          "software" : { software, artist, copyright, host_computer },
          "gps"      : { latitude, longitude, altitude, gps_timestamp,
                         lat_decimal, lon_decimal, maps_link },
          "raw"      : { all_tag: value, ... },
          "warnings" : [ list of forensic flag strings ],
        }
        """
        result = {
            "basic":      {},
            "datetime":   {},
            "camera":     {},
            "settings":   {},
            "resolution": {},
            "software":   {},
            "gps":        {},
            "raw":        {},
            "warnings":   [],
        }

        try:
            img  = Image.open(image_path)
            size = os.path.getsize(image_path)

            #  Basic 
            result["basic"] = {
                "Format"      : img.format or "Unknown",
                "Color Mode"  : img.mode,
                "Width (px)"  : img.size[0],
                "Height (px)" : img.size[1],
                "File Size"   : f"{size/1024:.1f} KB  ({size:,} bytes)",
                "Megapixels"  : f"{img.size[0]*img.size[1]/1_000_000:.2f} MP",
            }

            #  Raw EXIF via PIL 
            raw_exif = {}
            gps_raw  = {}

            exif_data = img._getexif() if hasattr(img, "_getexif") else None
            if exif_data:
                for tag_id, val in exif_data.items():
                    tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
                    if tag_name == "GPSInfo" and isinstance(val, dict):
                        for gk, gv in val.items():
                            gps_raw[GPSTAGS.get(gk, f"GPS_{gk}")] = gv
                    else:
                        if isinstance(val, bytes):
                            try:
                                val = val.decode("utf-8",
                                                 errors="replace").strip("\x00")
                            except Exception:
                                val = val.hex()
                        raw_exif[tag_name] = val

            result["raw"] = raw_exif

            #  DateTime 
            dt = {}
            for key, label in [("DateTimeOriginal",  "Date Taken"),
                                ("DateTime",          "Date Modified"),
                                ("DateTimeDigitized", "Date Digitized")]:
                v = raw_exif.get(key, "")
                if v:
                    dt[label] = str(v)
                else:
                    dt[label] = "Not available"
            result["datetime"] = dt

            #  Camera 
            result["camera"] = {
                "Make"         : str(raw_exif.get("Make",        "Not available")),
                "Model"        : str(raw_exif.get("Model",       "Not available")),
                "Lens Model"   : str(raw_exif.get("LensModel",   "Not available")),
                "Flash"        : str(raw_exif.get("Flash",       "Not available")),
                "Focal Length" : cls._format_focal(raw_exif["FocalLength"])
                                 if "FocalLength" in raw_exif else "Not available",
            }

            #  Camera Settings 
            result["settings"] = {
                "ISO Speed"        : str(raw_exif.get("ISOSpeedRatings",
                                         raw_exif.get("ISO", "Not available"))),
                "Aperture"         : cls._format_aperture(raw_exif["FNumber"])
                                     if "FNumber" in raw_exif else "Not available",
                "Shutter Speed"    : cls._format_shutter(raw_exif["ExposureTime"])
                                     if "ExposureTime" in raw_exif
                                     else "Not available",
                "Exposure Bias"    : str(raw_exif.get("ExposureBiasValue",
                                         "Not available")),
                "Metering Mode"    : str(raw_exif.get("MeteringMode",
                                         "Not available")),
                "White Balance"    : ("Auto" if raw_exif.get("WhiteBalance") == 0
                                      else "Manual"
                                      if raw_exif.get("WhiteBalance") == 1
                                      else str(raw_exif.get("WhiteBalance",
                                               "Not available"))),
                "Exposure Program" : str(raw_exif.get("ExposureProgram",
                                         "Not available")),
            }

            #  Resolution 
            xres = cls._rational_to_float(raw_exif["XResolution"]) \
                   if "XResolution" in raw_exif else None
            yres = cls._rational_to_float(raw_exif["YResolution"]) \
                   if "YResolution" in raw_exif else None
            unit_map = {1: "No unit", 2: "DPI", 3: "DPC"}
            unit = unit_map.get(raw_exif.get("ResolutionUnit", 2), "DPI")
            result["resolution"] = {
                "X Resolution"  : f"{xres:.0f} {unit}" if xres else "Not available",
                "Y Resolution"  : f"{yres:.0f} {unit}" if yres else "Not available",
                "Pixel Width"   : str(img.size[0]),
                "Pixel Height"  : str(img.size[1]),
                "Orientation"   : str(raw_exif.get("Orientation", "Not available")),
            }

            #  Software 
            result["software"] = {
                "Software"      : str(raw_exif.get("Software",     "Not available")),
                "Artist"        : str(raw_exif.get("Artist",       "Not available")),
                "Copyright"     : str(raw_exif.get("Copyright",    "Not available")),
                "Host Computer" : str(raw_exif.get("HostComputer", "Not available")),
                "Processing"    : str(raw_exif.get("ProcessingSoftware",
                                       "Not available")),
            }

            #  GPS 
            result["gps"] = cls._decode_gps(gps_raw)
            if not result["gps"]:
                result["gps"]["status"] = "No GPS data embedded in this image."

            #  Forensic Warnings 
            warnings = []

            if not raw_exif:
                warnings.append("⚠  No EXIF metadata found — "
                                 "metadata may have been stripped after editing.")

            sw = raw_exif.get("Software", "")
            edit_keywords = ["photoshop", "gimp", "lightroom", "affinity",
                             "picsart", "canva", "pixlr", "snapseed",
                             "illustrator", "paint"]
            if sw and any(kw in sw.lower() for kw in edit_keywords):
                warnings.append(f"⚠  Editing software detected: '{sw}' — "
                                 "image may have been post-processed.")
            elif sw and sw.strip().lower() not in ("", "not available"):
                warnings.append(f"ℹ  Software tag present: '{sw}'.")

            dt_orig   = raw_exif.get("DateTimeOriginal", "")
            dt_modify = raw_exif.get("DateTime", "")
            if dt_orig and dt_modify and dt_orig != dt_modify:
                warnings.append(
                    f"⚠  Date inconsistency detected:\n"
                    f"    Original  : {dt_orig}\n"
                    f"    Modified  : {dt_modify}\n"
                    f"    Image was modified after capture.")

            if not result["gps"] or "status" in result["gps"]:
                warnings.append("ℹ  No GPS data — "
                                 "location cannot be verified from metadata.")
            else:
                warnings.append(
                    f"✔  GPS location present: "
                    f"{result['gps'].get('latitude','?')}, "
                    f"{result['gps'].get('longitude','?')}")

            cam_make  = raw_exif.get("Make",  "")
            cam_model = raw_exif.get("Model", "")
            if not cam_make and not cam_model:
                warnings.append("⚠  No camera make/model — "
                                 "device origin cannot be confirmed.")
            else:
                warnings.append(f"✔  Camera identified: {cam_make} {cam_model}".strip())

            result["warnings"] = warnings

        except Exception as e:
            result["warnings"].append(f"❌  EXIF extraction error: {e}")

        return result


# ════════════════════════════════════════════════════════════════════════════
#  ELA ENGINE
# ════════════════════════════════════════════════════════════════════════════
class ELAEngine:
    @staticmethod
    def analyse(image_path: str) -> tuple:
        """
        Perform Error Level Analysis.
        Returns (ela_numpy_array, stats_dict)
        """
        original    = Image.open(image_path).convert("RGB")
        buf         = io.BytesIO()
        original.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr  = np.array(original,      dtype=np.float32)
        recomp_arr= np.array(recompressed,  dtype=np.float32)
        diff      = np.abs(orig_arr - recomp_arr) * ELA_SCALE
        diff      = np.clip(diff, 0, 255).astype(np.uint8)

        flat      = diff.flatten().astype(np.float32)
        mean_e    = float(np.mean(flat))
        max_e     = float(np.max(flat))
        std_e     = float(np.std(flat))
        hi_px     = int(np.sum(diff > ELA_HIGH))
        total_px  = diff.size // 3
        hi_ratio  = (hi_px / total_px * 100) if total_px else 0

        stats = {
            "Mean Error Level"        : round(mean_e,   2),
            "Max Error Level"         : round(max_e,    2),
            "Std Deviation"           : round(std_e,    2),
            "High Error Pixels"       : hi_px,
            "High Error Ratio (%)"    : round(hi_ratio, 2),
            "ELA Quality Setting"     : ELA_QUALITY,
            "Amplification Factor"    : ELA_SCALE,
        }
        return diff, stats


# ════════════════════════════════════════════════════════════════════════════
#  VERDICT ENGINE
# ════════════════════════════════════════════════════════════════════════════
class VerdictEngine:

    EDIT_KEYWORDS = ["photoshop", "gimp", "lightroom", "affinity",
                     "picsart", "canva", "pixlr", "snapseed",
                     "illustrator", "paint.net"]

    @classmethod
    def compute(cls, ela_stats: dict, exif: dict) -> dict:
        """
        Combined ELA + EXIF scoring.

        This system uses a project-based weighted scoring model with a
        maximum score of 100. The score starts at 0 and points are added only
        when a suspicious condition is triggered. Some conditions are
        alternative ranges, not cumulative values.

        Score breakdown (maximum 100):
          ELA mean error > 40         → +25
          ELA mean error 20–40        → +10
          ELA high-error ratio > 5 %  → +20
          ELA high-error ratio 1–5 %  → +10
          ELA std deviation > 20      → +15
          No EXIF metadata            → +15
          Known editing software      → +10
          General software tag        → +5
          Date inconsistency          → +10
          No camera make/model        → +5
        """
        score  = 0
        flags  = []
        raw    = exif.get("raw", {})
        sw_sec = exif.get("software", {})

        #  ELA 
        me = ela_stats["Mean Error Level"]
        hr = ela_stats["High Error Ratio (%)"]
        sd = ela_stats["Std Deviation"]

        if me > 40:
            score += 25
            flags.append(
                f"🔴 ELA: High mean error ({me}) — strong compression "
                "inconsistency. Likely manipulation present.")
        elif me > 20:
            score += 10
            flags.append(
                f"🟡 ELA: Moderate mean error ({me}) — possible "
                "compression irregularity detected.")
        else:
            flags.append(
                f"🟢 ELA: Low mean error ({me}) — "
                "compression appears uniform and consistent.")

        if hr > 5:
            score += 20
            flags.append(
                f"🔴 ELA: {hr}% of pixels show elevated error — "
                "suspicious regions detected in the image.")
        elif hr > 1:
            score += 10
            flags.append(
                f"🟡 ELA: {hr}% of pixels show slightly elevated error — "
                "minor irregularities present.")
        else:
            flags.append(
                f"🟢 ELA: Only {hr}% high-error pixels — "
                "image appears compression-consistent.")

        if sd > 20:
            score += 15
            flags.append(
                f"🔴 ELA: High standard deviation ({sd}) — "
                "uneven error distribution suggests localised editing.")
        else:
            flags.append(
                f"🟢 ELA: Standard deviation ({sd}) is within the normal range — "
                "error distribution is relatively even.")

        #  EXIF 
        if not raw:
            score += 15
            flags.append(
                "🔴 EXIF: No metadata found — metadata may have been "
                "stripped, which is common after image editing or online sharing.")
        else:
            flags.append(f"🟢 EXIF: {len(raw)} metadata fields found.")

        sw = sw_sec.get("Software", "Not available")
        if sw and sw != "Not available":
            if any(kw in sw.lower() for kw in cls.EDIT_KEYWORDS):
                score += 10
                flags.append(
                    f"🔴 EXIF: Known editing software detected — '{sw}'. "
                    "Image was likely processed in post-production.")
            else:
                score += 5
                flags.append(
                    f"🟡 EXIF: Software tag present — '{sw}'. "
                    "May indicate image processing.")
        else:
            flags.append("🟢 EXIF: No software tag detected.")

        dt_orig   = raw.get("DateTimeOriginal", "")
        dt_modify = raw.get("DateTime", "")
        if dt_orig and dt_modify and dt_orig != dt_modify:
            score += 10
            flags.append(
                f"🔴 EXIF: Date inconsistency — "
                f"Original: {dt_orig} | Modified: {dt_modify}. "
                "Image was saved or altered after capture.")
        else:
            flags.append("🟢 EXIF: Date fields are consistent or unavailable.")

        cam_make  = raw.get("Make",  "")
        cam_model = raw.get("Model", "")
        if not cam_make and not cam_model:
            score += 5
            flags.append(
                "🟡 EXIF: No camera make/model found — "
                "device origin cannot be confirmed.")
        else:
            flags.append(
                f"🟢 EXIF: Camera identified as {cam_make} {cam_model}.".strip())

        gps = exif.get("gps", {})
        if "lat_decimal" in gps:
            flags.append(
                f"🟢 EXIF: GPS coordinates embedded — "
                f"{gps.get('latitude','?')}, {gps.get('longitude','?')}.")
        else:
            flags.append(
                "ℹ  EXIF: No GPS data embedded in this image.")

      
        score = min(score, 100)
        if score >= 60:
            verdict, colour = "LIKELY MANIPULATED",   C_DANGER
        elif score >= 30:
            verdict, colour = "POSSIBLY MANIPULATED", C_WARN
        else:
            verdict, colour = "LIKELY AUTHENTIC",     C_SUCCESS

        return {"score": score, "verdict": verdict,
                "colour": colour, "flags": flags}


# ════════════════════════════════════════════════════════════════════════════
#  REPORT BUILDER  –  Produces a read-only, locked PDF forensic report
# ════════════════════════════════════════════════════════════════════════════
class ReportBuilder:
    """
    Generates a professional, read-only PDF forensic report.

    Security:
      • Printing    : allowed (so the report can be printed for submission)
      • Copying     : disabled
      • Editing     : disabled
      • Annotations : disabled
    The PDF is encrypted with a random owner password so no one can unlock
    editing permissions, while remaining freely openable (no user password).
    """

    #  Colour palette (hex → reportlab Color) 
    _COL_HEADER   = rl_colors.HexColor("#7C3AED")   # C_ACCENT purple
    _COL_SUBHDR   = rl_colors.HexColor("#0F3460")   # C_CARD dark blue
    _COL_CYAN     = rl_colors.HexColor("#06B6D4")
    _COL_DANGER   = rl_colors.HexColor("#EF4444")
    _COL_WARN     = rl_colors.HexColor("#F59E0B")
    _COL_SUCCESS  = rl_colors.HexColor("#22C55E")
    _COL_BG_ALT   = rl_colors.HexColor("#F1F5F9")   # light row alt
    _COL_BG_HDR   = rl_colors.HexColor("#E0E7FF")   # table header row
    _COL_TEXT     = rl_colors.HexColor("#1E293B")
    _COL_MUTED    = rl_colors.HexColor("#64748B")
    _COL_WHITE    = rl_colors.white
    _COL_BLACK    = rl_colors.black
    _COL_LINE     = rl_colors.HexColor("#CBD5E1")

    #  Verdict colour map 
    _VERDICT_COLOUR = {
        "LIKELY AUTHENTIC":     rl_colors.HexColor("#22C55E"),
        "POSSIBLY MANIPULATED": rl_colors.HexColor("#F59E0B"),
        "LIKELY MANIPULATED":   rl_colors.HexColor("#EF4444"),
    }

    # ════════════════════════════════════════════════════════════════════════
    #  PUBLIC ENTRY POINT
    # ════════════════════════════════════════════════════════════════════════
    @classmethod
    def build_pdf(cls,
                  image_path: str,
                  ela_image:  object,          # PIL Image or None
                  ela_stats:  dict,
                  exif:       dict,
                  verdict:    dict,
                  out_path:   str) -> None:
        """
        Build the full PDF report at *out_path* then lock it.

        Parameters
        ----------
        image_path  : str   – path of the analysed image
        ela_image   : PIL.Image or None – ELA result array as PIL image
        ela_stats   : dict  – from ELAEngine.analyse()
        exif        : dict  – from EXIFParser.extract()
        verdict     : dict  – from VerdictEngine.compute()
        out_path    : str   – destination path (must end in .pdf)
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError(
                "reportlab and pypdf are required for PDF export.\n"
                "Run:  pip install reportlab pypdf")

        tmp_path = out_path + "_tmp_unlocked.pdf"
        try:
            cls._write_pdf(image_path, ela_image, ela_stats,
                           exif, verdict, tmp_path)
            cls._lock_pdf(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ════════════════════════════════════════════════════════════════════════
    #  PDF CONTENT BUILDER
    # ════════════════════════════════════════════════════════════════════════
    @classmethod
    def _write_pdf(cls, image_path, ela_image, ela_stats,
                   exif, verdict, path):
        """Compose the full story and write to *path* (unlocked)."""
        import tempfile, io as _io

        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=22*mm,  bottomMargin=22*mm,
            title="Digital Image Forensic Analysis Report",
            author="Image Forensics Analyzer v3.0",
            subject="Forensic Report — " + os.path.basename(image_path),
            creator="UniKL MIIT FYP — Aida Yusreena Binti Abd Halim",
        )

        W = A4[0] - 36*mm   # usable width
        styles = cls._make_styles()
        story  = []

        #  Cover / header band 
        story += cls._cover_section(styles, image_path, verdict, W)

        #  ELA images side-by-side 
        story += cls._ela_images_section(styles, image_path, ela_image, W)

        #  Score breakdown table 
        story += cls._score_section(styles, ela_stats, exif, verdict, W)

        #  ELA statistics 
        story += cls._kv_section(styles, "📊  Error Level Analysis (ELA)",
                                  ela_stats, W,
                                  note=(
                                      "ELA re-compresses the image at quality=90 and measures "
                                      "pixel-level differences (amplified ×10). "
                                      "Higher values indicate compression inconsistency "
                                      "which may signal manipulation."
                                  ))

        #  EXIF sections 
        story += cls._kv_section(styles, "📅  Date & Time",
                                  exif.get("datetime", {}), W,
                                  note="DateTimeOriginal = shutter press. "
                                       "DateTime = last file save. "
                                       "A mismatch suggests re-saving after capture.")

        story += cls._kv_section(styles, "📷  Camera Information",
                                  exif.get("camera", {}), W,
                                  note="Make/Model confirms the capture device. "
                                       "Missing camera info may indicate a non-native capture.")

        story += cls._kv_section(styles, "⚙  Camera Settings",
                                  exif.get("settings", {}), W,
                                  note="ISO, aperture, and shutter speed are set by the "
                                       "camera hardware at capture time. Missing values "
                                       "suggest the image was not shot by a real camera.")

        story += cls._kv_section(styles, "🖼  Image Resolution",
                                  exif.get("resolution", {}), W)

        story += cls._kv_section(styles, "🧾  Software & Origin",
                                  exif.get("software", {}), W,
                                  note="The Software field records which program last "
                                       "saved this file. Editing software names are "
                                       "a strong forensic indicator.")

        story += cls._gps_section(styles, exif.get("gps", {}), W)

        #  Forensic flags 
        story += cls._flags_section(styles, verdict, exif, W)

        #  Raw EXIF (truncated) 
        story += cls._raw_section(styles, exif.get("raw", {}), W)

        #  Disclaimer / footer 
        story += cls._disclaimer_section(styles, W)

        doc.build(story,
                  onFirstPage=cls._page_frame,
                  onLaterPages=cls._page_frame)

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE FRAME  (header + footer on every page)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _page_frame(canvas_obj, doc):
        W, H = A4
        canvas_obj.saveState()

        # Top rule
        canvas_obj.setStrokeColor(rl_colors.HexColor("#7C3AED"))
        canvas_obj.setLineWidth(2)
        canvas_obj.line(18*mm, H - 14*mm, W - 18*mm, H - 14*mm)

        # Footer rule
        canvas_obj.setLineWidth(0.5)
        canvas_obj.setStrokeColor(rl_colors.HexColor("#CBD5E1"))
        canvas_obj.line(18*mm, 14*mm, W - 18*mm, 14*mm)

        # Footer text
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.setFillColor(rl_colors.HexColor("#64748B"))
        canvas_obj.drawString(18*mm, 9*mm,
            "Digital Image Forensics Analyzer v3.0  ·  "
            "FYP – Universiti Kuala Lumpur (MIIT)  ·  "
            "Aida Yusreena Binti Abd Halim (52215123089)")
        canvas_obj.drawRightString(W - 18*mm, 9*mm,
            f"Page {doc.page}  ·  READ-ONLY — DO NOT ALTER")

        canvas_obj.restoreState()

    # ════════════════════════════════════════════════════════════════════════
    #  STYLES
    # ════════════════════════════════════════════════════════════════════════
    @classmethod
    def _make_styles(cls):
        base = getSampleStyleSheet()

        def ps(name, parent="Normal", **kw):
            kw.setdefault("fontName",  "Helvetica")
            kw.setdefault("textColor", cls._COL_TEXT)
            return ParagraphStyle(name, parent=base[parent], **kw)

        return {
            "title":    ps("RTitle",  "Title",
                           fontSize=20, textColor=cls._COL_WHITE,
                           fontName="Helvetica-Bold", alignment=TA_CENTER,
                           spaceAfter=4),
            "subtitle": ps("RSub",    fontSize=9,  textColor=rl_colors.HexColor("#DDD6FE"),
                           alignment=TA_CENTER, spaceAfter=2),
            "verdict":  ps("RVerdict",fontSize=22, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, spaceAfter=2),
            "score":    ps("RScore",  fontSize=12, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, spaceAfter=6),
            "sechead":  ps("RSecHead",fontSize=11, fontName="Helvetica-Bold",
                           textColor=cls._COL_WHITE, alignment=TA_LEFT,
                           spaceAfter=0, spaceBefore=6),
            "note":     ps("RNote",   fontSize=8,  textColor=cls._COL_MUTED,
                           fontName="Helvetica-Oblique",
                           spaceBefore=2, spaceAfter=4),
            "kv_key":   ps("RKey",    fontSize=9,  fontName="Helvetica-Bold"),
            "kv_val":   ps("RVal",    fontSize=9),
            "flag":     ps("RFlag",   fontSize=9,  spaceBefore=2, spaceAfter=1),
            "small":    ps("RSmall",  fontSize=7.5,textColor=cls._COL_MUTED),
            "normal":   ps("RNormal", fontSize=9,  spaceAfter=3),
            "bold":     ps("RBold",   fontSize=9,  fontName="Helvetica-Bold"),
        }

    # ════════════════════════════════════════════════════════════════════════
    #  SECTION BUILDERS
    # ════════════════════════════════════════════════════════════════════════

    #  Helpers 
    @classmethod
    def _section_header(cls, styles, title, W):
        tbl = Table([[Paragraph(title, styles["sechead"])]],
                    colWidths=[W])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), cls._COL_SUBHDR),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[cls._COL_SUBHDR]),
        ]))
        return [Spacer(1, 6*mm), tbl]

    @classmethod
    def _kv_table(cls, styles, data: dict, W):
        if not data:
            return [Paragraph("  No data available.", styles["small"])]
        rows = []
        for i, (k, v) in enumerate(data.items()):
            bg = cls._COL_BG_ALT if i % 2 == 0 else cls._COL_WHITE
            rows.append([
                Paragraph(str(k), styles["kv_key"]),
                Paragraph(str(v)[:120], styles["kv_val"]),
            ])
        tbl = Table(rows, colWidths=[W*0.38, W*0.62])
        style_cmds = [
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("GRID",         (0,0), (-1,-1), 0.3, cls._COL_LINE),
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ]
        for i in range(len(rows)):
            bg = cls._COL_BG_ALT if i % 2 == 0 else cls._COL_WHITE
            style_cmds.append(("BACKGROUND", (0,i), (-1,i), bg))
        tbl.setStyle(TableStyle(style_cmds))
        return [tbl]

    @classmethod
    def _kv_section(cls, styles, title, data, W, note=""):
        elems = cls._section_header(styles, title, W)
        if note:
            elems.append(Paragraph(note, styles["note"]))
        elems += cls._kv_table(styles, data, W)
        return elems

    #  Cover 
    @classmethod
    def _cover_section(cls, styles, image_path, verdict, W):
        ts  = datetime.datetime.now().strftime("%d %B %Y  ·  %H:%M:%S")
        fname = os.path.basename(image_path)
        v_text  = verdict.get("verdict", "UNKNOWN")
        v_score = verdict.get("score",   0)
        v_col   = cls._VERDICT_COLOUR.get(v_text, cls._COL_MUTED)

        # Banner
        banner = Table(
            [[Paragraph("DIGITAL IMAGE FORENSIC ANALYSIS REPORT", styles["title"])],
             [Paragraph("Image Authenticity Verification Using Digital Image Forensic Techniques",
                        styles["subtitle"])],
             [Paragraph(f"Generated: {ts}", styles["subtitle"])],
             [Paragraph(f"Image: {fname}", styles["subtitle"])],
            ],
            colWidths=[W])
        banner.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), cls._COL_HEADER),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ]))

        # Verdict badge
        badge_data = [
            [Paragraph("FORENSIC VERDICT", ParagraphStyle(
                "vl", fontName="Helvetica", fontSize=8,
                textColor=cls._COL_MUTED, alignment=TA_CENTER))],
            [Paragraph(v_text, ParagraphStyle(
                "vv", fontName="Helvetica-Bold", fontSize=20,
                textColor=v_col, alignment=TA_CENTER))],
            [Paragraph(f"Manipulation Score:  {v_score} / 100",
                       ParagraphStyle("vs", fontName="Helvetica-Bold",
                                      fontSize=12, textColor=v_col,
                                      alignment=TA_CENTER))],
            [Paragraph(
                "Score  0–29 = Likely Authentic  |  "
                "30–59 = Possibly Manipulated  |  "
                "60–100 = Likely Manipulated",
                ParagraphStyle("vn", fontName="Helvetica", fontSize=7.5,
                               textColor=cls._COL_MUTED, alignment=TA_CENTER))],
        ]
        badge = Table(badge_data, colWidths=[W])
        badge.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), rl_colors.HexColor("#F8FAFC")),
            ("BOX",           (0,0), (-1,-1), 1.5, v_col),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))

        return [banner, Spacer(1, 5*mm), badge, Spacer(1, 4*mm)]

    #  ELA side-by-side images 
    @classmethod
    def _ela_images_section(cls, styles, image_path, ela_pil, W):
        import tempfile, io as _io
        from reportlab.platypus import Image as RLImage

        elems = cls._section_header(styles, "🔬  Visual Analysis — Original vs ELA Result", W)
        elems.append(Paragraph(
            "LEFT: Original image.   "
            "RIGHT: ELA result — bright areas indicate higher compression error (suspicious).",
            styles["note"]))

        img_w = (W - 6*mm) / 2
        img_h = img_w * 0.65

        cells = []
        # Original
        try:
            orig_rl = RLImage(image_path, width=img_w, height=img_h,
                              kind="proportional")
            cells.append(orig_rl)
        except Exception:
            cells.append(Paragraph("Original image unavailable.", styles["small"]))

        # ELA
        if ela_pil is not None:
            try:
                buf = _io.BytesIO()
                ela_pil.save(buf, format="PNG")
                buf.seek(0)
                ela_rl = RLImage(buf, width=img_w, height=img_h,
                                 kind="proportional")
                cells.append(ela_rl)
            except Exception:
                cells.append(Paragraph("ELA image unavailable.", styles["small"]))
        else:
            cells.append(Paragraph("ELA image unavailable.", styles["small"]))

        img_tbl = Table([cells], colWidths=[img_w, img_w],
                        rowHeights=[img_h + 4*mm])
        img_tbl.setStyle(TableStyle([
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("BOX",          (0,0), (0,0),   0.5, cls._COL_LINE),
            ("BOX",          (1,0), (1,0),   0.5, cls._COL_LINE),
            ("LEFTPADDING",  (0,0), (-1,-1), 2),
            ("RIGHTPADDING", (0,0), (-1,-1), 2),
            ("COLPADDING",   (0,0), (-1,-1), 3),
        ]))
        elems.append(img_tbl)

        # Labels row
        label_tbl = Table(
            [[Paragraph("Original Image", ParagraphStyle(
                  "lbl", fontSize=8, fontName="Helvetica-Bold",
                  alignment=TA_CENTER, textColor=cls._COL_MUTED)),
              Paragraph("ELA Result  (brighter = more suspicious)", ParagraphStyle(
                  "lbl2", fontSize=8, fontName="Helvetica-Bold",
                  alignment=TA_CENTER, textColor=cls._COL_DANGER))]],
            colWidths=[img_w, img_w])
        label_tbl.setStyle(TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        elems.append(label_tbl)
        return elems

    #  Score breakdown 
    @classmethod
    def _score_section(cls, styles, ela_stats, exif, verdict, W):
        """Build the PDF score breakdown table using the final 100-point model."""
        elems = cls._section_header(styles, "🧮  Score Breakdown — How the Verdict Was Calculated", W)
        elems.append(Paragraph(
            "The score starts at 0. Risk points are added only when a suspicious "
            "trigger condition is met. Rules under the same category are alternative "
            "ranges, so only one rule from that category can apply. The overall "
            "rule-based weighted model totals 100 points.",
            styles["note"]))

        raw  = exif.get("raw", {})
        sw   = exif.get("software", {}).get("Software", "Not available")
        me   = ela_stats.get("Mean Error Level", 0)
        hr   = ela_stats.get("High Error Ratio (%)", 0)
        sd   = ela_stats.get("Std Deviation", 0)

        def indicator_row(indicator, value, trigger, pts, max_pts, note, col):
            pts_str = f"+{pts}" if pts > 0 else "0"
            return [
                Paragraph(indicator, styles["kv_key"]),
                Paragraph(str(value), styles["kv_val"]),
                Paragraph(trigger, styles["small"]),
                Paragraph(pts_str, ParagraphStyle(
                    "pts", fontName="Helvetica-Bold", fontSize=8.5,
                    textColor=col, alignment=TA_CENTER)),
                Paragraph(note, styles["small"]),
            ]

        hdr_style = ParagraphStyle("sh", fontName="Helvetica-Bold",
                                   fontSize=7.0, textColor=cls._COL_WHITE)
        rows = [[
            Paragraph("Indicator", hdr_style),
            Paragraph("Current Value", hdr_style),
            Paragraph("Trigger Rule", hdr_style),
            Paragraph("Risk Pts", hdr_style),
            Paragraph("Why This Point Was Added", hdr_style),
        ]]

        # ELA Mean Error Level (maximum 25)
        if me > 40:
            rows.append(indicator_row("ELA Mean Error", me, "Mean > 40", 25, 25,
                "Strong compression inconsistency", cls._COL_DANGER))
        elif me > 20:
            rows.append(indicator_row("ELA Mean Error", me, "20 < Mean ≤ 40", 10, 25,
                "Moderate compression irregularity", cls._COL_WARN))
        else:
            rows.append(indicator_row("ELA Mean Error", me, "Mean ≤ 20", 0, 25,
                "Compression looks uniform", cls._COL_SUCCESS))

        # ELA High Error Ratio (maximum 20)
        if hr > 5:
            rows.append(indicator_row("High Error Ratio", f"{hr}%", "Ratio > 5%", 20, 20,
                "Many suspicious regions", cls._COL_DANGER))
        elif hr > 1:
            rows.append(indicator_row("High Error Ratio", f"{hr}%", "1% < Ratio ≤ 5%", 10, 20,
                "Some irregularities detected", cls._COL_WARN))
        else:
            rows.append(indicator_row("High Error Ratio", f"{hr}%", "Ratio ≤ 1%", 0, 20,
                "Very few anomalies", cls._COL_SUCCESS))

        # ELA Standard Deviation (maximum 15)
        if sd > 20:
            rows.append(indicator_row("ELA Std Deviation", sd, "Std Dev > 20", 15, 15,
                "Uneven error distribution", cls._COL_DANGER))
        else:
            rows.append(indicator_row("ELA Std Deviation", sd, "Std Dev ≤ 20", 0, 15,
                "Error distribution is even", cls._COL_SUCCESS))

        # EXIF Metadata Availability (maximum 15)
        if not raw:
            rows.append(indicator_row("EXIF Metadata", "MISSING", "No EXIF metadata", 15, 15,
                "Metadata may have been stripped", cls._COL_DANGER))
        else:
            rows.append(indicator_row("EXIF Metadata", f"{len(raw)} fields", "EXIF metadata present", 0, 15,
                "Metadata is available", cls._COL_SUCCESS))

        # Software Tag (maximum 10)
        edit_kws = VerdictEngine.EDIT_KEYWORDS
        if sw and sw != "Not available":
            if any(k in sw.lower() for k in edit_kws):
                rows.append(indicator_row("Software Tag", sw[:40], "Known editing software", 10, 10,
                    "Editing software detected", cls._COL_DANGER))
            else:
                rows.append(indicator_row("Software Tag", sw[:40], "General software tag", 5, 10,
                    "Some processing may have occurred", cls._COL_WARN))
        else:
            rows.append(indicator_row("Software Tag", "None", "No software tag", 0, 10,
                "No software tag detected", cls._COL_SUCCESS))

        # Date Consistency (maximum 10)
        dt_orig   = raw.get("DateTimeOriginal", "")
        dt_modify = raw.get("DateTime", "")
        if dt_orig and dt_modify and dt_orig != dt_modify:
            rows.append(indicator_row("Date Consistency", "INCONSISTENT", "Taken date ≠ modified date", 10, 10,
                "Image was saved after capture", cls._COL_DANGER))
        else:
            rows.append(indicator_row("Date Consistency", "OK", "Consistent or unavailable", 0, 10,
                "No date mismatch triggered", cls._COL_SUCCESS))

        # Camera Make/Model (maximum 5)
        make  = raw.get("Make",  "")
        model = raw.get("Model", "")
        if not make and not model:
            rows.append(indicator_row("Camera Make/Model", "MISSING", "Camera info missing", 5, 5,
                "Device origin cannot be confirmed", cls._COL_WARN))
        else:
            rows.append(indicator_row("Camera Make/Model",
                f"{make} {model}".strip(), "Camera info present", 0, 5,
                "Device identified", cls._COL_SUCCESS))

        # Total row
        score = verdict.get("score", 0)
        v_col = cls._VERDICT_COLOUR.get(verdict.get("verdict",""), cls._COL_MUTED)
        rows.append([
            Paragraph("TOTAL SCORE", ParagraphStyle(
                "tot", fontName="Helvetica-Bold", fontSize=8.5,
                textColor=cls._COL_WHITE)),
            Paragraph("", styles["kv_val"]),
            Paragraph("Overall model total = 100", ParagraphStyle(
                "totn", fontName="Helvetica-Bold", fontSize=7.0,
                textColor=cls._COL_WHITE)),
            Paragraph(str(score), ParagraphStyle(
                "totp", fontName="Helvetica-Bold", fontSize=11,
                textColor=cls._COL_WHITE, alignment=TA_CENTER)),
            Paragraph(verdict.get("verdict",""), ParagraphStyle(
                "totv", fontName="Helvetica-Bold", fontSize=8.0,
                textColor=cls._COL_WHITE)),
        ])

        cw = [W*0.19, W*0.16, W*0.27, W*0.10, W*0.28]
        tbl = Table(rows, colWidths=cw)

        style_cmds = [
            ("BACKGROUND",    (0,0), (-1,0),  cls._COL_SUBHDR),
            ("BACKGROUND",    (0,-1),(-1,-1), v_col),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 3),
            ("RIGHTPADDING",  (0,0), (-1,-1), 3),
            ("GRID",          (0,0), (-1,-2), 0.3, cls._COL_LINE),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]
        for i in range(1, len(rows) - 1):
            bg = cls._COL_BG_ALT if i % 2 == 1 else cls._COL_WHITE
            style_cmds.append(("BACKGROUND", (0,i), (-1,i), bg))
        tbl.setStyle(TableStyle(style_cmds))
        elems.append(tbl)
        return elems

    #  GPS 
    @classmethod
    def _gps_section(cls, styles, gps, W):
        elems = cls._section_header(styles, "🌍  GPS Location Data", W)
        elems.append(Paragraph(
            "GPS coordinates are embedded by smartphones and GPS-enabled cameras. "
            "Absence may indicate editing or GPS was disabled.",
            styles["note"]))
        if "lat_decimal" in gps:
            data = {
                "Latitude":      gps.get("latitude",      "—"),
                "Longitude":     gps.get("longitude",     "—"),
                "Altitude":      gps.get("altitude",      "Not available"),
                "GPS Timestamp": gps.get("gps_timestamp", "Not available"),
                "Google Maps":   gps.get("maps_link",     "—"),
            }
            elems += cls._kv_table(styles, data, W)
        else:
            elems.append(Paragraph(
                "⚠  " + gps.get("status", "No GPS data embedded in this image."),
                styles["flag"]))
        return elems

    #  Forensic flags 
    @classmethod
    def _flags_section(cls, styles, verdict, exif, W):
        elems = cls._section_header(styles, "🚩  Forensic Indicators", W)
        elems.append(Paragraph(
            "🔴 = suspicious  |  🟡 = possible concern  |  🟢 = likely authentic  |  ℹ = informational",
            styles["note"]))
        for i, flag in enumerate(verdict.get("flags", []), 1):
            elems.append(Paragraph(f"{i}.  {flag}", styles["flag"]))
        elems.append(Spacer(1, 3*mm))
        elems += cls._section_header(styles, "⚠  EXIF Forensic Warnings", W)
        for w in exif.get("warnings", []):
            elems.append(Paragraph(w, styles["flag"]))
        return elems

    #  Raw EXIF 
    @classmethod
    def _raw_section(cls, styles, raw, W):
        elems = cls._section_header(styles, "🗂  Complete Raw EXIF Data", W)
        elems.append(Paragraph(
            f"All {len(raw)} metadata fields extracted from the image file.",
            styles["note"]))
        if raw:
            # Limit to 80 fields to keep PDF manageable
            items = list(raw.items())[:80]
            data = {k: str(v)[:100] for k, v in items}
            elems += cls._kv_table(styles, data, W)
            if len(raw) > 80:
                elems.append(Paragraph(
                    f"… {len(raw)-80} additional fields omitted for brevity.",
                    styles["small"]))
        else:
            elems.append(Paragraph("No raw EXIF data found.", styles["small"]))
        return elems

    #  Disclaimer 
    @classmethod
    def _disclaimer_section(cls, styles, W):
        elems = [Spacer(1, 6*mm)]
        tbl = Table([[Paragraph(
            "DISCLAIMER  —  This report is generated for educational and research purposes only. "
            "Results should be interpreted alongside other forensic evidence. "
            "This tool is NOT intended for professional or legal investigations.  "
            "FYP – Universiti Kuala Lumpur (MIIT)  ·  "
            "Aida Yusreena Binti Abd Halim  52215123089",
            ParagraphStyle("disc", fontSize=7.5, fontName="Helvetica-Oblique",
                           textColor=cls._COL_MUTED, alignment=TA_CENTER))
        ]], colWidths=[W])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cls._COL_BG_ALT),
            ("BOX",           (0,0), (-1,-1), 0.5, cls._COL_LINE),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        elems.append(tbl)
        return elems

    # ════════════════════════════════════════════════════════════════════════
    #  PDF LOCKING  (disable editing & copying; allow printing)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _lock_pdf(src_path: str, dst_path: str) -> None:
        """
        Read the unlocked PDF, apply encryption that:
          • Sets NO user password  (anyone can open it)
          • Sets a random owner password  (prevents permission override)
          • Grants: print (full quality)
          • Denies: copy, edit, annotations, form filling
        """
        import secrets
        owner_pw = secrets.token_hex(32)   # 64-char random 

        reader = PdfReader(src_path)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        # Copy metadata
        if reader.metadata:
            writer.add_metadata(reader.metadata)

        writer.encrypt(
            user_password="",          # nosec B106 - intentionally openable without user password
            owner_password=owner_pw,   # random — locks permissions
            use_128bit=True,
            permissions_flag=(
                # Allow print (bit 3 = 4) + print high quality (bit 12 = 2048)
                4 | 2048
                # Everything else (copy=16, edit=8, annotations=32,
                # form fill=256, etc.) is NOT set → denied
            ),
        )

        with open(dst_path, "wb") as f:
            writer.write(f)

    # ════════════════════════════════════════════════════════════════════════
    #  LEGACY TEXT BUILDER  (kept for reference / fallback)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def build(image_path, ela_stats, exif, verdict):
        """Plain-text fallback (used only if reportlab is unavailable)."""
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 66
        lines = [sep,
                 "  DIGITAL IMAGE FORENSIC ANALYSIS REPORT",
                 f"  Generated: {ts}",
                 f"  Image: {image_path}",
                 sep]
        lines += [f"  VERDICT: {verdict.get('verdict','—')}",
                  f"  Score:   {verdict.get('score',0)} / 100", ""]
        for section, data in [
            ("ELA STATISTICS",   ela_stats),
            ("DATE & TIME",      exif.get("datetime",   {})),
            ("CAMERA",           exif.get("camera",     {})),
            ("SETTINGS",         exif.get("settings",   {})),
            ("RESOLUTION",       exif.get("resolution", {})),
            ("SOFTWARE",         exif.get("software",   {})),
        ]:
            lines += [sep, f"  {section}", sep]
            for k, v in data.items():
                lines.append(f"  {k:<28}: {v}")
        lines += [sep, "  FORENSIC INDICATORS", sep]
        for i, f in enumerate(verdict.get("flags", []), 1):
            lines.append(f"  {i}. {f}")
        lines += [sep,
                  "  This report is for educational purposes only.",
                  sep]
        return "\n".join(lines)





# ════════════════════════════════════════════════════════════════════════════
#  EDUCATIONAL CONTENT  –  Info texts, glossary, tutorial steps
# ════════════════════════════════════════════════════════════════════════════

INFO_TEXTS = {
    "ela": {
        "title": "📊 What is Error Level Analysis (ELA)?",
        "body": (
            "ELA is a technique that detects whether parts of a JPEG image have\n"
            "been tampered with by checking inconsistencies in compression.\n\n"
            "HOW IT WORKS (step-by-step):\n"
            "  1. The original image is re-saved at a known JPEG quality (90%).\n"
            "  2. The difference between the original and re-saved version is\n"
            "     calculated pixel-by-pixel.\n"
            "  3. The difference is amplified (×10) so it becomes visible.\n"
            "  4. Bright areas in the ELA image = high error = suspicious.\n\n"
            "FORMULA:\n"
            "  ELA pixel = |Original pixel − Re-compressed pixel| × 10\n\n"
            "WHY IT WORKS:\n"
            "  Authentic images have uniform compression error throughout.\n"
            "  Edited/pasted regions were compressed separately, so they\n"
            "  produce a different (usually higher) error level.\n\n"
            "WHAT TO LOOK FOR:\n"
            "  🔴 Bright white/yellow patches → likely edited region\n"
            "  🟢 Uniform dark image          → likely authentic"
        ),
    },
    "ela_mean": {
        "title": "📐 Mean Error Level — What does it mean?",
        "body": (
            "The Mean Error Level is the AVERAGE brightness of the ELA image\n"
            "across all pixels.\n\n"
            "FORMULA:\n"
            "  Mean Error = sum of all pixel differences ÷ total pixel count\n\n"
            "THRESHOLDS USED IN THIS SYSTEM:\n"
            "  Mean Error > 40   →  +25 points  (Strong manipulation signal)\n"
            "  Mean Error 20–40  →  +10 points  (Possible irregularity)\n"
            "  Mean Error < 20   →  +0  points  (Compression looks uniform)\n\n"
            "PLAIN ENGLISH:\n"
            "  Think of it like the average 'noisiness' of the image after\n"
            "  re-compression. A clean, unedited photo will have low noise.\n"
            "  A photo with pasted-in objects will have high noise."
        ),
    },
    "ela_high_ratio": {
        "title": "📈 High Error Ratio — What does it mean?",
        "body": (
            "This is the PERCENTAGE of pixels that have an unusually high\n"
            "error level (above the threshold of 60).\n\n"
            "FORMULA:\n"
            "  High-Error Pixels = pixels where ELA value > 60\n"
            "  High Error Ratio  = (High-Error Pixels ÷ Total Pixels) × 100\n\n"
            "THRESHOLDS USED IN THIS SYSTEM:\n"
            "  Ratio > 5%   →  +20 points  (Many suspicious regions)\n"
            "  Ratio 1–5%   →  +10 points  (Some irregularities)\n"
            "  Ratio < 1%   →  +0  points  (Very few anomalies)\n\n"
            "PLAIN ENGLISH:\n"
            "  If 10% of your image has high error, that means 1 in every\n"
            "  10 pixels looks suspicious — very likely manipulation."
        ),
    },
    "ela_std": {
        "title": "📉 Standard Deviation — What does it mean?",
        "body": (
            "Standard Deviation (Std Dev) measures how SPREAD OUT the error\n"
            "levels are across the image.\n\n"
            "FORMULA:\n"
            "  Std Dev = sqrt( average of (each pixel − mean)² )\n\n"
            "THRESHOLD USED IN THIS SYSTEM:\n"
            "  Std Dev > 20  →  +15 points  (Uneven error distribution)\n\n"
            "PLAIN ENGLISH:\n"
            "  A low Std Dev means the whole image has similar error levels\n"
            "  (consistent, likely authentic).\n"
            "  A HIGH Std Dev means some parts have very different error\n"
            "  levels from others — suggesting localised editing (e.g. a\n"
            "  person was pasted into the background)."
        ),
    },
    "exif": {
        "title": "🗂 What is EXIF Metadata?",
        "body": (
            "EXIF (Exchangeable Image File Format) is hidden data embedded\n"
            "inside every photo taken with a camera or smartphone.\n\n"
            "IT RECORDS THINGS LIKE:\n"
            "  📅 Date & Time the photo was taken\n"
            "  📷 Camera make, model, and lens\n"
            "  ⚙  Aperture, shutter speed, ISO settings\n"
            "  🌍 GPS location coordinates\n"
            "  🧾 Software used to save/edit the file\n\n"
            "WHY IT MATTERS FOR FORENSICS:\n"
            "  • Missing EXIF → metadata was stripped (common after editing)\n"
            "  • Photoshop/GIMP in Software field → image was post-processed\n"
            "  • Date Original ≠ Date Modified → image was changed after capture\n"
            "  • No camera info → photo may not be from a real camera\n\n"
            "IMPORTANT: EXIF alone cannot prove authenticity — it can be\n"
            "manually edited. It's one piece of the forensic puzzle."
        ),
    },
    "datetime": {
        "title": "📅 Date & Time Fields — What do they mean?",
        "body": (
            "Three timestamp fields are extracted from the image:\n\n"
            "  DATE TAKEN (DateTimeOriginal):\n"
            "  → When the camera shutter was pressed.\n"
            "  → Set automatically by the camera — hard to fake.\n\n"
            "  DATE MODIFIED (DateTime):\n"
            "  → When the file was last saved.\n"
            "  → If this differs from Date Taken, the image was re-saved\n"
            "    after capture — a potential sign of editing.\n\n"
            "  DATE DIGITIZED (DateTimeDigitized):\n"
            "  → When an analogue photo was digitised (scanned).\n"
            "  → Usually the same as Date Taken for digital photos.\n\n"
            "RED FLAG:\n"
            "  If Date Taken ≠ Date Modified, the system adds +10 to the\n"
            "  manipulation score.\n\n"
            "EXAMPLE:\n"
            "  Date Taken:    2024:03:10 08:30:00\n"
            "  Date Modified: 2024:06:15 14:22:00  ← SUSPICIOUS GAP"
        ),
    },
    "camera": {
        "title": "📷 Camera Information — What does it tell us?",
        "body": (
            "This section shows which device captured the image.\n\n"
            "FIELDS EXPLAINED:\n"
            "  Make    → Manufacturer (e.g. Apple, Samsung, Canon)\n"
            "  Model   → Device name  (e.g. iPhone 14, Galaxy S23)\n"
            "  Lens    → Lens model if recorded\n"
            "  Flash   → Whether flash fired (0 = no, 1 = yes)\n"
            "  Focal Length → Zoom level in millimetres\n\n"
            "FORENSIC MEANING:\n"
            "  ✅ Camera info present → image was captured by a real device\n"
            "  ⚠  Missing make/model → +5 to manipulation score\n"
            "     (Could mean image was downloaded, screenshotted,\n"
            "      or generated by AI)\n\n"
            "NOTE: Camera info can be manually added or removed using\n"
            "metadata editors, so treat it as supporting evidence."
        ),
    },
    "settings": {
        "title": "⚙ Camera Settings — What do they mean?",
        "body": (
            "These are the exposure settings recorded at the moment the\n"
            "photo was taken.\n\n"
            "ISO SPEED:\n"
            "  Light sensitivity of the sensor. Low (100) = bright scene.\n"
            "  High (3200+) = dark scene, more grain/noise.\n\n"
            "APERTURE (f-number):\n"
            "  Size of the lens opening. f/1.8 = wide (blurry background).\n"
            "  f/16 = narrow (everything in focus).\n\n"
            "SHUTTER SPEED:\n"
            "  How long the sensor was exposed. 1/1000s = freezes motion.\n"
            "  1s = motion blur on moving subjects.\n\n"
            "FORENSIC MEANING:\n"
            "  These settings are set by the camera hardware at capture time.\n"
            "  Missing values suggest the image was not captured by a camera\n"
            "  (e.g. generated, screenshotted, or exported from software)."
        ),
    },
    "resolution": {
        "title": "🖼 Resolution — What does it mean?",
        "body": (
            "Resolution describes the image's size and pixel density.\n\n"
            "FIELDS EXPLAINED:\n"
            "  Pixel Width / Height → Dimensions in pixels\n"
            "  X / Y Resolution     → Dots Per Inch (DPI) for printing\n"
            "  Orientation          → Rotation metadata (1 = normal)\n\n"
            "FORENSIC MEANING:\n"
            "  • Very low DPI (e.g. 72 DPI) often means a web/screenshot.\n"
            "  • Inconsistent X vs Y DPI may indicate resampling.\n"
            "  • An image resized after export may show generic 72 DPI\n"
            "    even if originally taken at a higher resolution.\n\n"
            "NOTE: Resolution alone is weak evidence — focus on ELA and\n"
            "EXIF combined for a stronger forensic conclusion."
        ),
    },
    "software": {
        "title": "🧾 Software Field — Why does it matter?",
        "body": (
            "The Software field in EXIF records the last program that\n"
            "saved or processed the image file.\n\n"
            "EXAMPLES:\n"
            "  'Apple iPhone'   → saved natively from iPhone camera\n"
            "  'Adobe Photoshop CS6' → opened and re-saved in Photoshop\n"
            "  'GIMP 2.10'      → edited in GIMP (free image editor)\n"
            "  'Snapseed'       → edited on a smartphone app\n\n"
            "WHAT THIS SYSTEM FLAGS:\n"
            "  Known editing software detected  →  +10 to score\n"
            "  Any software tag present         →  +5 to score\n\n"
            "EDITING KEYWORDS CHECKED:\n"
            "  photoshop, gimp, lightroom, affinity, picsart,\n"
            "  canva, pixlr, snapseed, illustrator, paint.net\n\n"
            "IMPORTANT: Software editing does not always mean the content\n"
            "was manipulated — photos may be edited for colour correction\n"
            "without any content being faked."
        ),
    },
    "gps": {
        "title": "🌍 GPS Data — What does it tell us?",
        "body": (
            "GPS data records where the photo was taken using coordinates\n"
            "embedded by a smartphone or GPS-enabled camera.\n\n"
            "FIELDS EXPLAINED:\n"
            "  Latitude  → North/South position (e.g. 3.1390° N)\n"
            "  Longitude → East/West position   (e.g. 101.6869° E)\n"
            "  Altitude  → Height above sea level in metres\n"
            "  GPS Timestamp → Time recorded by GPS (usually UTC)\n\n"
            "FORENSIC MEANING:\n"
            "  ✅ GPS present → location is verifiable via Google Maps link\n"
            "  ⚠  GPS absent → location cannot be confirmed from metadata\n"
            "     (May indicate privacy stripping, editing, or AI generation)\n\n"
            "HOW TO USE:\n"
            "  Click the Google Maps link (if present) to verify whether\n"
            "  the claimed location matches the image content.\n\n"
            "CAUTION: GPS can be spoofed using metadata editors."
        ),
    },
    "verdict": {
        "title": "🏆 How is the Verdict Score Calculated?",
        "body": (
            "The manipulation score (0–100) starts at 0 and adds points\n"
            "only when a suspicious trigger rule is met. Conditions inside\n"
            "the same category are alternative rules, not cumulative rules.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  CATEGORY / TRIGGER RULE                 POINTS   MAX\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ELA Mean Error > 40                       +25     25\n"
            "  ELA Mean Error 20–40                      +10     25\n"
            "  ELA Mean Error ≤ 20                        +0     25\n"
            "  ELA High Error Ratio > 5%                 +20     20\n"
            "  ELA High Error Ratio 1–5%                 +10     20\n"
            "  ELA High Error Ratio ≤ 1%                  +0     20\n"
            "  ELA Standard Deviation > 20               +15     15\n"
            "  EXIF Metadata missing                     +15     15\n"
            "  Known editing software detected           +10     10\n"
            "  General software tag detected              +5     10\n"
            "  Date taken and modified date different    +10     10\n"
            "  Camera make/model missing                  +5      5\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  TOTAL MAXIMUM SCORE                               100\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "EXAMPLE:\n"
            "  ELA Mean Error can only be +0, +10, or +25, not all\n"
            "  three. This prevents duplicate scoring in the same category.\n\n"
            "VERDICT THRESHOLDS:\n"
            "  Score  0–29  →  🟢 LIKELY AUTHENTIC\n"
            "  Score 30–59  →  🟡 POSSIBLY MANIPULATED\n"
            "  Score 60–100 →  🔴 LIKELY MANIPULATED\n\n"
            "IMPORTANT: This score is a preliminary risk indicator, not\n"
            "definitive proof. Always interpret alongside context."
        ),
    },
    "raw": {
        "title": "🗂 Raw EXIF Data — What is this?",
        "body": (
            "This tab shows ALL metadata fields extracted from the image,\n"
            "exactly as stored in the file — unfiltered and unformatted.\n\n"
            "WHAT YOU MIGHT SEE:\n"
            "  • Standard EXIF tags (Make, Model, DateTime, etc.)\n"
            "  • Manufacturer-specific private tags (MakerNote)\n"
            "  • Colour profile data, thumbnail info, compression type\n"
            "  • Any custom or non-standard tags the camera recorded\n\n"
            "WHO IS THIS FOR?\n"
            "  This section is mainly for advanced users and forensic\n"
            "  investigators who want to inspect all available metadata.\n\n"
            "BEGINNER TIP:\n"
            "  You don't need to understand every field here. Focus on\n"
            "  the other tabs and the Verdict card for the main findings."
        ),
    },
}

TUTORIAL_STEPS = [
    {
        "step": 1,
        "title": "👋 Welcome to the Digital Image Forensics Analyzer!",
        "body": (
            "This tutorial will guide you through using the application\n"
            "step-by-step. It is designed to be simple for beginners.\n\n"
            "WHAT THIS APP DOES:\n"
            "  It helps you detect whether an image has been digitally\n"
            "  manipulated (edited, faked, or tampered with).\n\n"
            "IT USES TWO MAIN METHODS:\n"
            "  1. ELA (Error Level Analysis) — analyses pixel compression\n"
            "  2. EXIF Metadata — reads hidden data inside the image file\n\n"
            "SUPPORTED FILE TYPES:\n"
            "  .jpg  .jpeg  .png   (up to 10 MB)\n\n"
            "Click  ▶ Next Step  to continue."
        ),
    },
    {
        "step": 2,
        "title": "📂 Step 1 — Select an Image",
        "body": (
            "To begin, you need to load an image into the application.\n\n"
            "HOW TO DO IT:\n"
            "  1. Click the  📂 Select Image  button in the top toolbar.\n"
            "  2. A file browser window will open.\n"
            "  3. Navigate to your image file and click  Open.\n\n"
            "WHAT HAPPENS NEXT:\n"
            "  • Your image will appear in the left panel (Original Image).\n"
            "  • The filename and size will appear in the toolbar.\n"
            "  • The  🔬 Analyze  button will become clickable.\n\n"
            "TIPS FOR BEST RESULTS:\n"
            "  • Use the original file (not a screenshot of the image).\n"
            "  • JPG/JPEG files contain the most EXIF metadata.\n"
            "  • PNG files may have less EXIF data.\n\n"
            "Click  ▶ Next Step  when ready."
        ),
    },
    {
        "step": 3,
        "title": "🔬 Step 2 — Run the Analysis",
        "body": (
            "Once your image is loaded, you can run the forensic analysis.\n\n"
            "HOW TO DO IT:\n"
            "  1. Click the  🔬 Analyze  button in the toolbar.\n"
            "  2. A progress bar will appear at the bottom of the screen.\n"
            "  3. Wait a few seconds — the analysis runs in the background.\n\n"
            "WHAT THE APP IS DOING:\n"
            "  ① Performing ELA — re-compressing image and measuring\n"
            "    pixel differences.\n"
            "  ② Extracting EXIF metadata — reading all hidden data\n"
            "    embedded in the file.\n"
            "  ③ Computing the Verdict Score — adding up all suspicious\n"
            "    indicators found.\n\n"
            "WHEN IT FINISHES:\n"
            "  • The ELA image appears on the left (brighter = suspicious).\n"
            "  • The Verdict and Score appear in the right panel.\n"
            "  • All tabs fill with detailed results.\n\n"
            "Click  ▶ Next Step  to learn how to read the results."
        ),
    },
    {
        "step": 4,
        "title": "🏆 Step 3 — Read the Verdict",
        "body": (
            "After analysis, the main result is shown in the FORENSIC\n"
            "VERDICT card (top-right panel).\n\n"
            "WHAT YOU WILL SEE:\n"
            "  Verdict label — one of three possible results:\n"
            "    🟢 LIKELY AUTHENTIC      (Score 0–29)\n"
            "    🟡 POSSIBLY MANIPULATED  (Score 30–59)\n"
            "    🔴 LIKELY MANIPULATED    (Score 60–100)\n\n"
            "  Manipulation Score — a number from 0 to 100:\n"
            "    Low score  = fewer suspicious indicators found\n"
            "    High score = many suspicious indicators found\n\n"
            "  Progress bar — visual representation of the score.\n\n"
            "IMPORTANT:\n"
            "  The verdict is a PROBABILITY, not a fact. A high score\n"
            "  means manipulation is likely — not guaranteed. Use it as\n"
            "  one piece of evidence alongside other information.\n\n"
            "  Click the  ℹ Score  button next to the verdict to see\n"
            "  the exact score breakdown formula.\n\n"
            "Click  ▶ Next Step  to explore the detail tabs."
        ),
    },
    {
        "step": 5,
        "title": "📑 Step 4 — Explore the Detail Tabs",
        "body": (
            "The right panel contains 8 tabs with detailed results.\n"
            "Each tab has an  ℹ  button — click it to learn what the\n"
            "data means.\n\n"
            "TABS OVERVIEW:\n"
            "  📊 ELA & Flags  → Raw ELA statistics + all forensic flags\n"
            "  📅 Date & Time  → Timestamps: taken, modified, digitized\n"
            "  📷 Camera       → Device make, model, lens, flash\n"
            "  ⚙  Settings     → ISO, aperture, shutter speed\n"
            "  🖼  Resolution   → Image dimensions and DPI\n"
            "  🧾  Software     → Software that last saved this file\n"
            "  🌍  GPS          → Location coordinates (if available)\n"
            "  🗂  Raw EXIF     → All metadata fields, unfiltered\n\n"
            "TIP:\n"
            "  Start with  📊 ELA & Flags  tab — it shows you the most\n"
            "  important forensic indicators all in one place.\n\n"
            "Click  ▶ Next Step  to learn about the ELA image."
        ),
    },
    {
        "step": 6,
        "title": "🔬 Step 5 — Read the ELA Image",
        "body": (
            "The ELA image (bottom-left panel) is a visual map of where\n"
            "the compression error is highest in the image.\n\n"
            "HOW TO READ IT:\n"
            "  ⬜ Bright white / light areas:\n"
            "     → High error level → SUSPICIOUS region\n"
            "     → May indicate pasted, drawn, or altered content\n\n"
            "  ⬛ Dark / black areas:\n"
            "     → Low error level → likely original content\n"
            "     → Compression is consistent with a real photo\n\n"
            "COMMON PATTERNS:\n"
            "  • A face that is much brighter than the background\n"
            "    → face may have been pasted in\n"
            "  • Text or watermarks appearing bright\n"
            "    → text was added after the original was taken\n"
            "  • Uniform dark image with a bright patch\n"
            "    → that bright patch is the suspicious region\n\n"
            "NOTE: Edges of objects and highly textured areas naturally\n"
            "have higher ELA — not every bright area is manipulation."
        ),
    },
    {
        "step": 7,
        "title": "📄 Step 6 — Export the Report",
        "body": (
            "Once analysis is complete, you can save a full forensic\n"
            "report as a protected read-only PDF file.\n\n"
            "HOW TO DO IT:\n"
            "  1. Click  📄 Export Report  in the toolbar.\n"
            "  2. Choose a location to save the file.\n"
            "  3. The report is saved as a protected .pdf file.\n\n"
            "WHAT IS IN THE REPORT:\n"
            "  • Verdict and score with full legend\n"
            "  • All ELA statistics\n"
            "  • All EXIF metadata sections\n"
            "  • All forensic flags and warnings\n"
            "  • Complete raw EXIF data\n"
            "  • Disclaimer and academic attribution\n\n"
            "USE CASES:\n"
            "  → Submit as part of an academic report\n"
            "  → Archive analysis results for a collection of images\n"
            "  → Share findings with others for review\n\n"
            "🎉 TUTORIAL COMPLETE!\n"
            "You now know how to use the full application.\n"
            "Click  ℹ  buttons anywhere in the app to learn more."
        ),
    },
]

GLOSSARY_TERMS = [
    ("ELA", "Error Level Analysis — a forensic technique that detects image manipulation by measuring JPEG compression inconsistencies across different regions of an image."),
    ("EXIF", "Exchangeable Image File Format — hidden metadata embedded in image files that records camera settings, timestamps, GPS, and software information."),
    ("JPEG Compression", "A method of reducing image file size by discarding some visual data. Every time a JPEG is re-saved, quality is slightly lost. ELA exploits this."),
    ("Mean Error Level", "The average pixel difference between the original image and its re-compressed version. Higher values suggest more inconsistency in the image."),
    ("Standard Deviation", "A measure of how spread out the ELA error values are. High std dev means some regions have very different error from others — a sign of localised editing."),
    ("High Error Ratio", "The percentage of pixels with an error level above 60. A high ratio means many pixels look suspicious after re-compression."),
    ("Manipulation Score", "A number from 0–100 that summarises how many suspicious indicators were found. 0 = very clean, 100 = highly suspicious."),
    ("Metadata Stripping", "The removal of EXIF data from an image, often done by editing software or social media platforms. Missing metadata raises suspicion."),
    ("DateTimeOriginal", "The EXIF field recording when the camera shutter was pressed. This is set by the camera itself."),
    ("DateTime", "The EXIF field recording when the file was last modified. If different from DateTimeOriginal, the image was re-saved after capture."),
    ("GPS Coordinates", "Location data (latitude/longitude) embedded by a smartphone or GPS camera. Allows you to verify where a photo was supposedly taken."),
    ("Software Field", "An EXIF tag recording which program last saved the image. Editing software names (Photoshop, GIMP, etc.) are a forensic red flag."),
    ("Amplification Factor", "In ELA, the pixel differences are multiplied (×10 in this system) to make them visible to the human eye."),
    ("Authentic Image", "An image that has not been digitally manipulated — it shows what the camera originally captured, without alteration."),
    ("Digital Forensics", "The science of investigating digital data to detect fraud, tampering, or illegal activity. Image forensics is one specialised branch."),
]


# ════════════════════════════════════════════════════════════════════════════
#  REUSABLE POPUP HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _make_popup(parent, title: str, width: int = 640, height: int = 500) -> tk.Toplevel:
    """Create a themed modal-style Toplevel window."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.configure(bg=C_BG)
    win.resizable(True, True)
    # Centre relative to parent
    parent.update_idletasks()
    px = parent.winfo_rootx() + (parent.winfo_width()  - width)  // 2
    py = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    win.geometry(f"{width}x{height}+{px}+{py}")
    win.grab_set()
    return win


def show_info_popup(parent, key: str):
    """Show a 'Tap for Info' popup for the given key from INFO_TEXTS."""
    entry = INFO_TEXTS.get(key, {
        "title": "ℹ Information",
        "body":  "No information available for this section."
    })
    win = _make_popup(parent, entry["title"], 620, 480)

    # Title bar
    hdr = tk.Frame(win, bg=C_ACCENT, height=44)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text=f"  {entry['title']}",
             font=("Segoe UI", 11, "bold"),
             bg=C_ACCENT, fg="white").pack(side="left", padx=12, pady=8)

    # Body
    body_frame = tk.Frame(win, bg=C_BG)
    body_frame.pack(fill="both", expand=True, padx=14, pady=10)

    txt = scrolledtext.ScrolledText(
        body_frame, font=("Segoe UI", 10), bg=C_ENTRY, fg=C_TEXT,
        relief="flat", wrap="word", state="normal", padx=12, pady=10)
    txt.pack(fill="both", expand=True)
    txt.insert("end", entry["body"])
    txt.config(state="disabled")

    # Close button
    tk.Button(win, text="✔  Got it, close",
              font=("Segoe UI", 10, "bold"),
              bg=C_ACCENT, fg="white", relief="flat",
              cursor="hand2", padx=16, pady=6,
              command=win.destroy).pack(pady=(0, 14))


# ════════════════════════════════════════════════════════════════════════════
#  TUTORIAL WINDOW
# ════════════════════════════════════════════════════════════════════════════

class TutorialWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("📖 Tutorial — Step-by-Step Guide")
        self.configure(bg=C_BG)
        self.resizable(True, True)
        self._step_index = 0
        self._steps = TUTORIAL_STEPS

        parent.update_idletasks()
        w, h = 660, 520
        px = parent.winfo_rootx() + (parent.winfo_width()  - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{px}+{py}")
        self.grab_set()

        self._build()
        self._render()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C_ACCENT, height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  📖  Step-by-Step Tutorial",
                 font=("Segoe UI", 10, "bold"),
                 bg=C_ACCENT, fg="white").pack(side="left", padx=14)
        self.lbl_step_counter = tk.Label(
            hdr, text="", font=("Segoe UI", 9),
            bg=C_ACCENT, fg="#DDD6FE")
        self.lbl_step_counter.pack(side="right", padx=14)

        # Progress bar
        prog_frame = tk.Frame(self, bg=C_PANEL, height=6)
        prog_frame.pack(fill="x")
        self.prog = ttk.Progressbar(
            prog_frame, maximum=len(self._steps), length=660)
        style = ttk.Style()
        style.configure("T.Horizontal.TProgressbar",
                        troughcolor=C_PANEL, background=C_CYAN, thickness=6)
        self.prog.configure(style="T.Horizontal.TProgressbar")
        self.prog.pack(fill="x")

        # Step title
        self.lbl_title = tk.Label(
            self, text="", font=("Segoe UI", 13, "bold"),
            bg=C_BG, fg=C_TEXT, wraplength=600, justify="left")
        self.lbl_title.pack(anchor="w", padx=20, pady=(14, 4))

        # Body text
        body_frame = tk.Frame(self, bg=C_BG)
        body_frame.pack(fill="both", expand=True, padx=14)
        self.txt = scrolledtext.ScrolledText(
            body_frame, font=("Segoe UI", 10), bg=C_ENTRY, fg=C_TEXT,
            relief="flat", wrap="word", state="normal", padx=12, pady=10)
        self.txt.pack(fill="both", expand=True)

        # Navigation buttons
        nav = tk.Frame(self, bg=C_BG)
        nav.pack(fill="x", padx=14, pady=12)

        self.btn_prev = tk.Button(
            nav, text="◀ Previous",
            font=("Segoe UI", 10, "bold"),
            bg=C_CARD, fg=C_TEXT, relief="flat",
            cursor="hand2", padx=14, pady=6,
            command=self._prev)
        self.btn_prev.pack(side="left")

        self.btn_next = tk.Button(
            nav, text="Next Step ▶",
            font=("Segoe UI", 10, "bold"),
            bg=C_ACCENT, fg="white", relief="flat",
            cursor="hand2", padx=14, pady=6,
            command=self._next)
        self.btn_next.pack(side="right")

        tk.Button(nav, text="✖  Close Tutorial",
                  font=("Segoe UI", 9),
                  bg=C_BG, fg=C_MUTED, relief="flat",
                  cursor="hand2", padx=10,
                  command=self.destroy).pack(side="right", padx=10)

    def _render(self):
        step = self._steps[self._step_index]
        total = len(self._steps)

        self.lbl_step_counter.config(
            text=f"Step {step['step']} of {total}")
        self.lbl_title.config(text=step["title"])

        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("end", step["body"])
        self.txt.config(state="disabled")

        self.prog["value"] = self._step_index + 1

        self.btn_prev.config(state="normal" if self._step_index > 0 else "disabled")
        last = self._step_index == total - 1
        self.btn_next.config(
            text="✔  Finish Tutorial" if last else "Next Step ▶",
            command=self.destroy if last else self._next)

    def _next(self):
        if self._step_index < len(self._steps) - 1:
            self._step_index += 1
            self._render()

    def _prev(self):
        if self._step_index > 0:
            self._step_index -= 1
            self._render()


# ════════════════════════════════════════════════════════════════════════════
#  GLOSSARY WINDOW
# ════════════════════════════════════════════════════════════════════════════

class GlossaryWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("📚 Forensics Glossary")
        self.configure(bg=C_BG)
        self.resizable(True, True)

        parent.update_idletasks()
        w, h = 680, 560
        px = parent.winfo_rootx() + (parent.winfo_width()  - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{px}+{py}")
        self.grab_set()
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=C_CARD, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  📚  Digital Forensics Glossary",
                 font=("Segoe UI", 11, "bold"),
                 bg=C_CARD, fg=C_TEXT).pack(side="left", padx=14)
        tk.Label(hdr, text=f"  {len(GLOSSARY_TERMS)} terms",
                 font=("Segoe UI", 9),
                 bg=C_CARD, fg=C_MUTED).pack(side="right", padx=14)

        tk.Label(self,
                 text="  Key terms used in this application, explained simply.",
                 font=("Segoe UI", 9, "italic"),
                 bg=C_BG, fg=C_MUTED).pack(anchor="w", padx=14, pady=(8, 2))

        frame = tk.Frame(self, bg=C_BG)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        canvas = tk.Canvas(frame, bg=C_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical",
                                  command=canvas.yview)
        inner = tk.Frame(canvas, bg=C_BG)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, (term, definition) in enumerate(GLOSSARY_TERMS):
            row_bg = C_CARD if i % 2 == 0 else C_PANEL
            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill="x", padx=4, pady=2)
            tk.Label(row, text=f"  {term}",
                     font=("Segoe UI", 10, "bold"),
                     bg=row_bg, fg=C_CYAN,
                     width=22, anchor="w").pack(side="left", pady=6)
            tk.Label(row, text=definition,
                     font=("Segoe UI", 9),
                     bg=row_bg, fg=C_TEXT,
                     wraplength=440, justify="left", anchor="w").pack(
                         side="left", padx=6, pady=6)

        tk.Button(self, text="✖  Close",
                  font=("Segoe UI", 10, "bold"),
                  bg=C_CARD, fg=C_TEXT, relief="flat",
                  cursor="hand2", padx=16, pady=6,
                  command=self.destroy).pack(pady=(0, 10))


# ════════════════════════════════════════════════════════════════════════════
#  SCORE EXPLAINER WINDOW  –  Real-time formula breakdown
# ════════════════════════════════════════════════════════════════════════════

class ScoreExplainerWindow(tk.Toplevel):
    """Shows how the rule-based weighted scoring model produced the verdict."""

    def __init__(self, parent, ela_stats: dict, exif: dict, verdict: dict):
        super().__init__(parent)
        self.title("🧮 Score Breakdown — Rule-Based Weighted Scoring Model")
        self.configure(bg=C_BG)
        self.resizable(True, True)

        parent.update_idletasks()
        w, h = 1060, 700
        px = parent.winfo_rootx() + (parent.winfo_width()  - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{px}+{py}")
        self.grab_set()

        self._ela   = ela_stats
        self._exif  = exif
        self._verd  = verdict
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C_ACCENT, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text="  🧮  Rule-Based Weighted Score Breakdown",
            font=("Segoe UI", 11, "bold"),
            bg=C_ACCENT,
            fg="white"
        ).pack(side="left", padx=14)
        tk.Label(
            hdr,
            text="Score starts at 0. Points are added only when a trigger condition is met.",
            font=("Segoe UI", 9),
            bg=C_ACCENT,
            fg="#DDD6FE"
        ).pack(side="left", padx=6)

        # Summary card
        verdict_colour = self._verd.get("colour", C_MUTED)
        summary = tk.Frame(self, bg=C_CARD)
        summary.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(
            summary,
            text=f"  FINAL VERDICT:  {self._verd.get('verdict','—')}",
            font=("Segoe UI", 13, "bold"),
            bg=C_CARD,
            fg=verdict_colour
        ).pack(side="left", pady=10, padx=8)
        tk.Label(
            summary,
            text=f"Score: {self._verd.get('score', 0)} / 100",
            font=("Segoe UI", 11, "bold"),
            bg=C_CARD,
            fg=verdict_colour
        ).pack(side="right", pady=10, padx=16)

        tk.Label(
            self,
            text=(
                "  This system uses a rule-based weighted scoring model. "
                "Rules under the same category are alternative, not cumulative. "
                "For example, ELA Mean Error can only trigger one value: +0, +10, or +25."
            ),
            font=("Segoe UI", 9, "italic"),
            bg=C_BG,
            fg=C_MUTED,
            wraplength=1020,
            justify="left"
        ).pack(anchor="w", padx=12, pady=(4, 6))

        style = ttk.Style()
        style.configure("Score.TNotebook", background=C_BG, borderwidth=0)
        style.configure(
            "Score.TNotebook.Tab",
            background=C_PANEL,
            foreground=C_MUTED,
            font=("Segoe UI", 9, "bold"),
            padding=[10, 5]
        )
        style.map(
            "Score.TNotebook.Tab",
            background=[("selected", C_ACCENT)],
            foreground=[("selected", "white")]
        )

        nb = ttk.Notebook(self, style="Score.TNotebook")
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        tab_applied = tk.Frame(nb, bg=C_BG)
        tab_rules = tk.Frame(nb, bg=C_BG)
        nb.add(tab_applied, text="A. Applied Score for This Image")
        nb.add(tab_rules, text="B. Complete Score Rules")

        self._build_applied_tab(tab_applied)
        self._build_rules_tab(tab_rules)

        sep = tk.Frame(self, bg=C_BORDER, height=2)
        sep.pack(fill="x", padx=10, pady=4)

        tk.Label(
            self,
            text=(
                f"  TOTAL SCORE = {self._verd.get('score',0)} / 100\n"
                f"  Verdict thresholds: 0–29 = Likely Authentic  |  "
                f"30–59 = Possibly Manipulated  |  60–100 = Likely Manipulated"
            ),
            font=("Segoe UI", 9),
            bg=C_BG,
            fg=C_TEXT,
            justify="left"
        ).pack(anchor="w", padx=14, pady=4)

        tk.Button(
            self,
            text="✔  Close",
            font=("Segoe UI", 10, "bold"),
            bg=C_ACCENT,
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
            command=self.destroy
        ).pack(pady=(2, 12))

    def _build_applied_tab(self, parent):
        tk.Label(
            parent,
            text=(
                "  This tab shows only the trigger condition applied to the current image in each category. "
                "The total score is calculated from these applied rows only."
            ),
            font=("Segoe UI", 9, "italic"),
            bg=C_BG,
            fg=C_MUTED,
            wraplength=1000,
            justify="left"
        ).pack(anchor="w", padx=8, pady=(8, 4))

        outer, inner = self._make_scroll_area(parent)

        self._add_header(inner, [
            ("Indicator Category", 26),
            ("Current Value", 18),
            ("Triggered Condition", 36),
            ("Points Added", 15),
            ("Why This Point Was Added", 54),
        ])

        for i, (indicator, value, trigger, pts, _weight, note, colour) in enumerate(self._build_rows()):
            row_bg = C_CARD if i % 2 == 0 else C_PANEL
            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill="x", padx=2, pady=1)
            pts_text = f"+{pts}" if pts > 0 else "+0"
            pts_colour = C_DANGER if pts >= 15 else (C_WARN if pts > 0 else C_SUCCESS)

            self._cell(row, f"  {indicator}", 26, colour, row_bg, bold=True)
            self._cell(row, value, 18, C_TEXT, row_bg, font=("Consolas", 9))
            self._cell(row, trigger, 36, C_MUTED, row_bg, wrap=300)
            self._cell(row, pts_text, 15, pts_colour, row_bg, center=True, bold=True)
            self._cell(row, note, 54, C_MUTED, row_bg, wrap=420)

    def _build_rules_tab(self, parent):
        tk.Label(
            parent,
            text=(
                "  This tab shows all possible trigger rules used by the system. "
                "Only one rule from each indicator category can be applied during analysis."
            ),
            font=("Segoe UI", 9, "italic"),
            bg=C_BG,
            fg=C_MUTED,
            wraplength=1000,
            justify="left"
        ).pack(anchor="w", padx=8, pady=(8, 4))

        outer, inner = self._make_scroll_area(parent)

        self._add_header(inner, [
            ("Applied?", 10),
            ("Indicator Category", 28),
            ("Trigger Condition", 42),
            ("Points Added", 15),
            ("Why This Point Is Added", 54),
        ])

        applied_lookup = {(row[0], row[2], row[3]) for row in self._build_rows()}
        for i, (indicator, trigger, pts, note, colour) in enumerate(self._all_rule_rows()):
            row_bg = C_CARD if i % 2 == 0 else C_PANEL
            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill="x", padx=2, pady=1)
            is_applied = (indicator, trigger, pts) in applied_lookup
            applied_text = "✓" if is_applied else ""
            pts_text = f"+{pts}" if pts > 0 else "+0"
            pts_colour = C_DANGER if pts >= 15 else (C_WARN if pts > 0 else C_SUCCESS)

            self._cell(row, applied_text, 10, C_SUCCESS if is_applied else C_MUTED, row_bg, center=True, bold=True)
            self._cell(row, f"  {indicator}", 28, colour, row_bg, bold=True)
            self._cell(row, trigger, 42, C_MUTED, row_bg, wrap=360)
            self._cell(row, pts_text, 15, pts_colour, row_bg, center=True, bold=True)
            self._cell(row, note, 54, C_MUTED, row_bg, wrap=420)

    def _make_scroll_area(self, parent):
        outer = tk.Frame(parent, bg=C_BG)
        outer.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        canvas = tk.Canvas(outer, bg=C_BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C_BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return outer, inner

    def _add_header(self, parent, columns):
        header = tk.Frame(parent, bg=C_BORDER)
        header.pack(fill="x", padx=2, pady=(0, 2))
        for text, width in columns:
            tk.Label(
                header,
                text=text,
                font=("Segoe UI", 9, "bold"),
                bg=C_BORDER,
                fg=C_TEXT,
                width=width,
                anchor="w"
            ).pack(side="left", padx=3, pady=5)

    def _cell(self, parent, text, width, fg, bg, font=None, bold=False,
              center=False, wrap=None):
        if font is None:
            font = ("Segoe UI", 9, "bold") if bold else ("Segoe UI", 8)
        tk.Label(
            parent,
            text=text,
            font=font,
            bg=bg,
            fg=fg,
            width=width,
            anchor="center" if center else "w",
            wraplength=wrap,
            justify="left"
        ).pack(side="left", pady=4, padx=3)

    def _all_rule_rows(self):
        """Return every possible trigger rule in the 100-point weighted model."""
        return [
            ("ELA Mean Error Level", "Mean error below or equal to 20", 0,
             "Compression looks uniform", C_SUCCESS),
            ("ELA Mean Error Level", "Mean error between 20 and 40", 10,
             "Moderate compression irregularity", C_WARN),
            ("ELA Mean Error Level", "Mean error above 40", 25,
             "Strong compression inconsistency", C_DANGER),

            ("ELA High Error Ratio", "High error ratio below or equal to 1%", 0,
             "Very few suspicious pixel regions", C_SUCCESS),
            ("ELA High Error Ratio", "High error ratio between 1% and 5%", 10,
             "Some irregularities are detected", C_WARN),
            ("ELA High Error Ratio", "High error ratio above 5%", 20,
             "Many suspicious pixel regions are detected", C_DANGER),

            ("ELA Standard Deviation", "Standard deviation below or equal to 20", 0,
             "Error distribution is even", C_SUCCESS),
            ("ELA Standard Deviation", "Standard deviation above 20", 15,
             "Uneven error distribution may suggest localised editing", C_DANGER),

            ("EXIF Metadata Availability", "EXIF metadata is available", 0,
             "Metadata is available", C_SUCCESS),
            ("EXIF Metadata Availability", "No EXIF metadata found", 15,
             "Metadata may have been stripped", C_DANGER),

            ("Software Tag", "No software tag found", 0,
             "No software tag detected", C_SUCCESS),
            ("Software Tag", "General software tag detected", 5,
             "Some processing may have occurred", C_WARN),
            ("Software Tag", "Known editing software detected", 10,
             "Editing software detected", C_DANGER),

            ("Date Consistency", "Dates are consistent or unavailable", 0,
             "No date mismatch triggered", C_SUCCESS),
            ("Date Consistency", "Date taken and modified date are different", 10,
             "Image may have been saved after capture", C_DANGER),

            ("Camera Make/Model", "Camera make/model is available", 0,
             "Device origin is available", C_SUCCESS),
            ("Camera Make/Model", "Camera make/model is missing", 5,
             "Device origin cannot be confirmed", C_WARN),
        ]

    def _build_rows(self):
        """Return list of (indicator, value, trigger_rule, points, category_weight, note, colour)."""
        rows = []
        ela  = self._ela
        exif = self._exif
        raw  = exif.get("raw", {})
        sw   = exif.get("software", {}).get("Software", "Not available")

        me = ela.get("Mean Error Level", 0)
        hr = ela.get("High Error Ratio (%)", 0)
        sd = ela.get("Std Deviation", 0)

        if me > 40:
            pts, trigger, note, col = 25, "Mean error above 40", "Strong compression inconsistency", C_DANGER
        elif me > 20:
            pts, trigger, note, col = 10, "Mean error between 20 and 40", "Moderate compression irregularity", C_WARN
        else:
            pts, trigger, note, col = 0, "Mean error below or equal to 20", "Compression looks uniform", C_SUCCESS
        rows.append(("ELA Mean Error Level", str(me), trigger, pts, 25, note, col))

        if hr > 5:
            pts, trigger, note, col = 20, "High error ratio above 5%", "Many suspicious pixel regions", C_DANGER
        elif hr > 1:
            pts, trigger, note, col = 10, "High error ratio between 1% and 5%", "Some irregularities detected", C_WARN
        else:
            pts, trigger, note, col = 0, "High error ratio below or equal to 1%", "Very few anomalies", C_SUCCESS
        rows.append(("ELA High Error Ratio", f"{hr}%", trigger, pts, 20, note, col))

        if sd > 20:
            pts, trigger, note, col = 15, "Standard deviation above 20", "Uneven error distribution", C_DANGER
        else:
            pts, trigger, note, col = 0, "Standard deviation below or equal to 20", "Error distribution is even", C_SUCCESS
        rows.append(("ELA Standard Deviation", str(sd), trigger, pts, 15, note, col))

        if not raw:
            pts, trigger, note, col = 15, "No EXIF metadata found", "Metadata may have been stripped", C_DANGER
        else:
            pts, trigger, note, col = 0, "EXIF metadata is available", f"{len(raw)} fields found", C_SUCCESS
        rows.append(("EXIF Metadata Availability", "Available" if raw else "Missing", trigger, pts, 15, note, col))

        edit_kws = VerdictEngine.EDIT_KEYWORDS
        if sw and sw != "Not available":
            if any(k in sw.lower() for k in edit_kws):
                pts, trigger, note, col = 10, "Known editing software detected", "Editing software detected", C_DANGER
            else:
                pts, trigger, note, col = 5, "General software tag detected", "Some processing may have occurred", C_WARN
        else:
            pts, trigger, note, col = 0, "No software tag found", "No software tag detected", C_SUCCESS
        rows.append(("Software Tag", sw[:30] if sw else "—", trigger, pts, 10, note, col))

        dt_orig   = raw.get("DateTimeOriginal", "")
        dt_modify = raw.get("DateTime", "")
        if dt_orig and dt_modify and dt_orig != dt_modify:
            pts, trigger, note, col = 10, "Date taken and modified date are different", "Image was saved after capture", C_DANGER
        else:
            pts, trigger, note, col = 0, "Dates are consistent or unavailable", "No date mismatch triggered", C_SUCCESS
        rows.append(("Date Consistency", "Different" if pts else "OK", trigger, pts, 10, note, col))

        make  = raw.get("Make",  "")
        model = raw.get("Model", "")
        if not make and not model:
            pts, trigger, note, col = 5, "Camera make/model is missing", "Device origin cannot be confirmed", C_WARN
        else:
            pts, trigger, note, col = 0, "Camera make/model is available", f"{make} {model}".strip(), C_SUCCESS
        rows.append(("Camera Make/Model", "Missing" if pts else "Available", trigger, pts, 5, note, col))

        return rows


# ════════════════════════════════════════════════════════════════════════════
#  FILE SIGNATURE VALIDATION – Magic Number Check
# ════════════════════════════════════════════════════════════════════════════
def is_valid_magic_number(path: str, ext: str) -> bool:
    """
    Validate the real file signature/magic number before image processing.
    This prevents spoofed files such as fake.jpg that are not real images.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(8)

        # JPEG/JPG magic number: FF D8 FF
        if ext in {".jpg", ".jpeg"}:
            return header.startswith(b"\xFF\xD8\xFF")

        # PNG magic number: 89 50 4E 47 0D 0A 1A 0A
        if ext == ".png":
            return header.startswith(b"\x89PNG\r\n\x1a\n")

        return False

    except Exception:
        return False


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE}  ·  {VERSION}")
        # Fit the GUI inside the laptop screen but use almost full width for clear demo view
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w = min(1360, max(1180, sw - 20))
        win_h = min(630,  max(560,  sh - 170))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(1040, 540)
        self.configure(bg=C_BG)

        # State
        self.image_path     = None
        self.orig_tk        = None
        self.ela_tk         = None
        self.ela_stats      = {}
        self.exif_data      = {}
        self.verdict        = {}
        self._done          = False
        self._last_ela_arr  = None

        # Inactivity security timer state
        self._inactivity_seconds_left = INACTIVITY_TIMEOUT_SECONDS
        self._inactivity_job = None
        self._security_results_visible = False

        self._build_ui()
        self._center()
        self._bind_activity_events()
        self._update_inactivity_countdown_label()
        # Show welcome tutorial offer to first-time users
        self.after(600, self._welcome_prompt)

    def _welcome_prompt(self):
        """Offer tutorial on first launch."""
        ans = messagebox.askyesno(
            "👋 Welcome to Image Forensics Analyzer v3.0!",
            "This app checks whether an image has been digitally manipulated.\n\n"
            "It looks like your first time here.\n"
            "Would you like to start the Step-by-Step Tutorial?\n\n"
            "(You can also access it anytime via the  📖 Tutorial  button.)",
            icon="question")
        if ans:
            self.open_tutorial()

    #  window center 
    def _center(self):
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w,  h  = self.winfo_width(),       self.winfo_height()
        self.geometry(f"+{max(0, (sw-w)//2)}+{max(0, (sh-h)//2 - 35)}")

    # ════════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self._build_header()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()

    #  Header 
    def _build_header(self):
        hdr = tk.Frame(self, bg=C_ACCENT, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  🔍  {APP_TITLE}",
                 font=("Segoe UI", 13, "bold"),
                 bg=C_ACCENT, fg="white").pack(side="left", padx=16)
        tk.Label(hdr, text=APP_SUBTITLE,
                 font=("Segoe UI", 9), bg=C_ACCENT, fg="#DDD6FE").pack(side="left")

        #  Shortcut buttons in header
        def hbtn(text, cmd):
            b = tk.Button(hdr, text=text, command=cmd,
                          font=("Segoe UI", 10, "bold"),
                          bg="#5B21B6", fg="white", relief="flat",
                          cursor="hand2", padx=10, pady=3,
                          activebackground="#6D28D9", activeforeground="white")
            b.pack(side="right", padx=(0, 6), pady=12)
            return b

        hbtn("📚 Glossary",   self.open_glossary)
        hbtn("📖 Tutorial",   self.open_tutorial)
        tk.Label(hdr, text=f"  {VERSION}  ",
                 font=("Segoe UI", 9), bg=C_ACCENT, fg="#DDD6FE").pack(side="right")

    #  Toolbar 
    def _build_toolbar(self):
        bar = tk.Frame(self, bg=C_PANEL, height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        def btn(text, cmd, bg, state="normal"):
            b = tk.Button(bar, text=text, command=cmd,
                          font=("Segoe UI", 9, "bold"),
                          bg=bg, fg="white", relief="flat",
                          cursor="hand2", padx=10, pady=4,
                          activebackground=bg, activeforeground="white",
                          state=state)
            b.pack(side="left", padx=(8, 2), pady=7)
            return b

        self.btn_open    = btn("📂  Select Image",  self.open_image,   C_ACCENT)
        self.btn_analyze = btn("🔬  Analyze",       self.run_analysis, C_CYAN,    "disabled")
        self.btn_export  = btn("📄  Export Report", self.export_report,"#334155", "disabled")
        self.btn_clear   = btn("🗑  Clear",         self.clear_all,    "#374151")

        # Visible inactivity timer
        self.lbl_inactivity = tk.Label(
            bar,
            text="  Timer: inactive / no results  ",
            font=("Segoe UI", 10, "bold"),
            bg="#7F1D1D", fg="white",
            padx=8, pady=3)
        self.lbl_inactivity.pack(side="right", padx=(6, 10), pady=7)

        # Beginner quick-help hint
        tk.Label(bar, text="❓ New here?",
                 font=("Segoe UI", 8), bg=C_PANEL, fg=C_MUTED).pack(side="left", padx=(14, 2))
        tk.Button(bar, text="Start Tutorial ▶",
                  font=("Segoe UI", 8, "bold"),
                  bg=C_CARD, fg=C_CYAN, relief="flat",
                  cursor="hand2", padx=8, pady=2,
                  command=self.open_tutorial).pack(side="left")

        self.lbl_file = tk.Label(bar, text="No image loaded.",
                                 font=("Segoe UI", 9),
                                 bg=C_PANEL, fg=C_MUTED)
        self.lbl_file.pack(side="left", padx=20)

    #  Body 
    def _build_body(self):
        body = tk.Frame(self, bg=C_BG)
        body.pack(fill="both", expand=True, padx=6, pady=2)


        body.grid_columnconfigure(0, weight=1, uniform="main")
        body.grid_columnconfigure(1, weight=5, uniform="main")
        body.grid_rowconfigure(0, weight=1)

        #  Left: image panels 
        left = tk.Frame(body, bg=C_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self.lbl_orig = self._img_frame(left, "📷  Original Image")
        self.lbl_ela  = self._img_frame(left, "🔬  ELA Result  (bright = suspicious)")

        #  Right: enlarged forensic verdict + results notebook 
        right = tk.Frame(body, bg=C_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self._build_verdict_card(right)
        self._build_notebook(right)

    def _img_frame(self, parent, title):
        frm = tk.LabelFrame(parent, text=f"  {title}  ",
                            font=("Segoe UI", 11, "bold"),
                            bg=C_PANEL, fg=C_TEXT,
                            bd=1, relief="solid", labelanchor="n")
        frm.pack(fill="both", expand=True, pady=(0, 6))

        #  Full-screen shortcut
        if "Original" in title:
            orig_bar = tk.Frame(frm, bg=C_PANEL)
            orig_bar.pack(fill="x")
            tk.Button(orig_bar, text="⛶ Full Screen",
                      font=("Segoe UI", 9, "bold"),
                      bg=C_CARD, fg=C_CYAN, relief="flat",
                      cursor="hand2", padx=10, pady=2,
                      command=lambda: self._open_fullscreen_image("orig")
                      ).pack(side="right", padx=6, pady=2)

        # ELA image gets a real-time feedback hint bar
        if "ELA" in title:
            hint_bar = tk.Frame(frm, bg=C_PANEL)
            hint_bar.pack(fill="x")
            self.lbl_ela_hint = tk.Label(
                hint_bar,
                text="  Run analysis to see ELA result.",
                font=("Segoe UI", 10, "italic"),
                bg=C_PANEL, fg=C_MUTED, anchor="w")
            self.lbl_ela_hint.pack(fill="x", padx=6, pady=2)

            # Full screen button for ELA result 
            tk.Button(hint_bar, text="⛶ Full Screen",
                      font=("Segoe UI", 9, "bold"),
                      bg=C_CARD, fg=C_CYAN, relief="flat",
                      cursor="hand2", padx=10, pady=2,
                      command=lambda: self._open_fullscreen_image("ela")
                      ).pack(side="right", padx=6, pady=2)

            # ℹ info button for ELA
            tk.Button(hint_bar, text="ℹ What is ELA?",
                      font=("Segoe UI", 9, "bold"),
                      bg=C_CARD, fg=C_CYAN, relief="flat",
                      cursor="hand2", padx=10, pady=2,
                      command=lambda: show_info_popup(self, "ela")
                      ).pack(side="right", padx=6, pady=2)

        lbl = tk.Label(frm, text="Upload an image to begin.",
                       font=("Segoe UI", 10), bg=C_PANEL, fg=C_MUTED)
        lbl.pack(fill="both", expand=True, padx=4, pady=4)
        return lbl

    #  Verdict card 
    def _build_verdict_card(self, parent):
        # Clear verdict card, kept compact so the analysis explanation below has more room.
        card = tk.Frame(parent, bg=C_CARD, bd=1, relief="solid", height=122)
        card.pack(fill="x", pady=(0, 0))
        card.pack_propagate(False)

        top = tk.Frame(card, bg=C_CARD)
        top.pack(fill="x", padx=18, pady=(6, 2))

        tk.Label(top, text="FORENSIC VERDICT",
                 font=("Segoe UI", 11, "bold"),
                 bg=C_CARD, fg=C_TEXT).pack(side="left")

        self.btn_score_info = tk.Button(
            top, text="ℹ Score",
            font=("Segoe UI", 9, "bold"),
            bg=C_PANEL, fg=C_CYAN, relief="flat",
            cursor="hand2", padx=8, pady=2,
            command=self._open_score_explainer,
            state="disabled")
        self.btn_score_info.pack(side="right")

        tk.Label(top, text="How was this calculated? →",
                 font=("Segoe UI", 9), bg=C_CARD, fg=C_MUTED).pack(side="right", padx=6)

        self.lbl_verdict = tk.Label(card, text="—  RUN ANALYSIS TO SEE RESULT",
                                    font=("Segoe UI", 17, "bold"),
                                    bg=C_CARD, fg=C_MUTED)
        self.lbl_verdict.pack(pady=(2, 0))

        self.lbl_score = tk.Label(card, text="MANIPULATION SCORE: —",
                                  font=("Segoe UI", 11, "bold"),
                                  bg=C_CARD, fg=C_MUTED)
        self.lbl_score.pack(pady=(0, 2))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("V.Horizontal.TProgressbar",
                        troughcolor=C_BG, background=C_ACCENT,
                        thickness=10)

        self.prog = ttk.Progressbar(card, style="V.Horizontal.TProgressbar",
                                    length=700, maximum=100)
        self.prog.pack(fill="x", padx=22, pady=(1, 4))

    #  Notebook (tabs) 
    def _build_notebook(self, parent):
        style = ttk.Style()
        style.configure("Dark.TNotebook",
                        background=C_BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=C_PANEL, foreground=C_MUTED,
                        font=("Segoe UI", 9, "bold"),
                        padding=[4, 3])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", C_ACCENT)],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(parent, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, pady=(0, 0))

        def make_tab(label, info_key):
            """Create a tab frame with a scrolled text area and an ℹ info button."""
            frame = tk.Frame(nb, bg=C_BG)
            nb.add(frame, text=label)

            # Top bar with ℹ button
            bar = tk.Frame(frame, bg=C_PANEL, height=28)
            bar.pack(fill="x")
            bar.pack_propagate(False)
            tk.Label(bar, text=f"  {label}",
                     font=("Segoe UI", 10, "bold"), bg=C_PANEL, fg=C_MUTED).pack(side="left", padx=10)
            tk.Button(bar, text="ℹ  Tap for Info",
                      font=("Segoe UI", 9, "bold"),
                      bg=C_CARD, fg=C_CYAN, relief="flat",
                      cursor="hand2", padx=10, pady=2,
                      command=lambda k=info_key: show_info_popup(self, k)
                      ).pack(side="right", padx=6, pady=3)

            txt = scrolledtext.ScrolledText(
                frame, font=("Consolas", 13),
                bg=C_ENTRY, fg=C_TEXT,
                insertbackground=C_TEXT,
                relief="flat", wrap="word",
                state="disabled")
            txt.pack(fill="both", expand=True, padx=4, pady=4)
            return txt

        self.txt_flags      = make_tab("ELA & Flags",  "ela")
        self.txt_datetime   = make_tab("Date & Time",   "datetime")
        self.txt_camera     = make_tab("Camera",        "camera")
        self.txt_settings   = make_tab("Settings",      "settings")
        self.txt_resolution = make_tab("Resolution",    "resolution")
        self.txt_software   = make_tab("Software",      "software")
        self.txt_gps        = make_tab("GPS",           "gps")
        self.txt_raw        = make_tab("Raw EXIF",      "raw")

    def _scrolled(self, parent) -> scrolledtext.ScrolledText:
        txt = scrolledtext.ScrolledText(parent,
                                        font=("Consolas", 13),
                                        bg=C_ENTRY, fg=C_TEXT,
                                        insertbackground=C_TEXT,
                                        relief="flat", wrap="word",
                                        state="disabled")
        txt.pack(fill="both", expand=True, padx=4, pady=4)
        return txt

    #  Status bar 
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C_BORDER, height=18)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.lbl_status = tk.Label(bar, text="  Ready — select an image to begin.",
                                   font=("Segoe UI", 8),
                                   bg=C_BORDER, fg=C_TEXT)
        self.lbl_status.pack(side="left")

        tk.Label(bar, text="FYP · UniKL MIIT · Aida Yusreena  |  v3.0  ",
                 font=("Segoe UI", 7),
                 bg=C_BORDER, fg=C_MUTED).pack(side="right", padx=(0, 12))

    # ════════════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ════════════════════════════════════════════════════════════════════════
    def _bind_activity_events(self):
        """Reset inactivity timer whenever the user is actively using the app."""
        activity_events = (
            "<Motion>", "<Button>", "<ButtonPress>", "<ButtonRelease>",
            "<MouseWheel>", "<KeyPress>", "<KeyRelease>",
            "<FocusIn>", "<Configure>",
        )
        for event_name in activity_events:
            self.bind_all(event_name, self._on_user_activity, add="+")

    def _on_user_activity(self, event=None):
        """Reset timer only when analysed results are visible."""
        if not self._security_results_visible:
            return

        widget = getattr(event, "widget", None)

        # Reset only on real user input
        if event is not None and getattr(event, "type", None) == tk.EventType.Configure:
            if widget is self:
                return

        self._reset_inactivity_timer()

    def _cancel_inactivity_timer(self):
        if self._inactivity_job is not None:
            try:
                self.after_cancel(self._inactivity_job)
            except tk.TclError as error:
                print(f"Timer cancellation skipped: {error}")
            self._inactivity_job = None

    def _start_inactivity_timer(self):
        self._security_results_visible = True
        self._reset_inactivity_timer()

    def _reset_inactivity_timer(self):
        self._cancel_inactivity_timer()
        self._inactivity_seconds_left = INACTIVITY_TIMEOUT_SECONDS
        self._update_inactivity_countdown_label()
        self._inactivity_job = self.after(1000, self._tick_inactivity_timer)

    def _tick_inactivity_timer(self):
        if not self._security_results_visible:
            self._inactivity_job = None
            self._update_inactivity_countdown_label()
            return

        self._inactivity_seconds_left -= 1

        if self._inactivity_seconds_left <= 0:
            self._inactivity_job = None
            self._auto_clear_results_due_to_inactivity()
            return

        self._update_inactivity_countdown_label()
        self._inactivity_job = self.after(1000, self._tick_inactivity_timer)

    def _stop_inactivity_timer(self):
        self._security_results_visible = False
        self._cancel_inactivity_timer()
        self._inactivity_seconds_left = INACTIVITY_TIMEOUT_SECONDS
        self._update_inactivity_countdown_label()

    def _format_seconds(self, total_seconds: int) -> str:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _update_inactivity_countdown_label(self):
        if self._security_results_visible:
            remaining = self._format_seconds(max(self._inactivity_seconds_left, 0))
            self.lbl_inactivity.config(
                text=f"  Timer: auto-clear in {remaining}",
                fg=C_DANGER if self._inactivity_seconds_left <= 10 else C_WARN
            )
        else:
            self.lbl_inactivity.config(
                text="  Timer: inactive / no results",
                fg=C_WARN
            )

    def _clear_results_only(self):
        """Clear displayed forensic results while keeping the selected image loaded."""
        self._done         = False
        self.ela_stats     = {}
        self.exif_data     = {}
        self.verdict       = {}
        self.ela_tk        = None
        self._last_ela_arr = None

        self.lbl_ela.config(image="", text="ELA result was removed for security. Run analysis again.")
        self._reset_panels()
        self.btn_export.config(state="disabled")
        self.btn_score_info.config(state="disabled")
        self.lbl_verdict.config(text="—  Results cleared after inactivity", fg=C_MUTED)
        self.lbl_score.config(text="MANIPULATION SCORE: —", fg=C_MUTED)
        self.lbl_ela_hint.config(text="  Results were auto-cleared. Analyze again to continue.")
        self.prog["value"] = 0

        if self.image_path:
            self.btn_analyze.config(state="normal")

    def _auto_clear_results_due_to_inactivity(self):
        self._stop_inactivity_timer()
        self.clear_all()
        self.set_status("🔒  Image and results auto-cleared after 30 seconds of inactivity for data security.")
        messagebox.showinfo(
            "Security Auto-Clear",
            "The loaded image and displayed forensic results were automatically removed after 30 seconds of inactivity.\n\n"
            "Please re-upload the image to continue."
        )

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"),
                       ("All Files", "*.*")])
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTS:
            messagebox.showerror("Invalid File",
                f"Unsupported type: '{ext}'\nAllowed: JPG, JPEG, PNG")
            return

        #  validate real file signature / magic number
        if not is_valid_magic_number(path, ext):
            messagebox.showerror(
                "Invalid File Signature",
                "The file extension is supported, but the actual file content "
                "does not match a valid JPG, JPEG, or PNG image.\n\n"
                "Possible spoofed or invalid image file rejected."
            )
            return

        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            messagebox.showerror("File Too Large",
                f"Size {size/1024/1024:.1f} MB exceeds {MAX_FILE_MB} MB limit.")
            return

        self._stop_inactivity_timer()
        self.image_path = path
        self._done      = False
        fname = os.path.basename(path)
        self.lbl_file.config(text=f"📁  {fname}  ({size/1024:.1f} KB)")
        self.set_status(f"Loaded: {fname}")
        self._display_image(path, self.lbl_orig, "orig")
        self.lbl_ela.config(image="",
                            text="ELA result will appear here after analysis.")
        self._reset_panels()
        self.btn_analyze.config(state="normal")
        self.btn_export.config(state="disabled")

    def run_analysis(self):
        if not self.image_path:
            return
        self.btn_analyze.config(state="disabled")
        self.set_status("⏳  Running forensic analysis…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            ela_arr, ela_stats = ELAEngine.analyse(self.image_path)
            exif_data          = EXIFParser.extract(self.image_path)
            verdict            = VerdictEngine.compute(ela_stats, exif_data)

            self.ela_stats       = ela_stats
            self.exif_data       = exif_data
            self.verdict         = verdict
            self._last_ela_arr   = ela_arr   # stored for PDF export

            self.after(0, self._update_ui, ela_arr)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.btn_analyze.config(state="normal"))

    def _update_ui(self, ela_arr: np.ndarray):
        # ELA image
        self._display_image_pil(Image.fromarray(ela_arr), self.lbl_ela, "ela")

        # Verdict
        v = self.verdict
        self.lbl_verdict.config(text=v["verdict"], fg=v["colour"])
        self.lbl_score.config(
            text=f"MANIPULATION SCORE: {v['score']} / 100", fg=v["colour"])
        self.prog["value"] = v["score"]
        ttk.Style().configure("V.Horizontal.TProgressbar",
                              background=v["colour"])

        # Enable score info button
        self.btn_score_info.config(state="normal")

        # Real-time ELA feedback caption
        me  = self.ela_stats.get("Mean Error Level", 0)
        hr  = self.ela_stats.get("High Error Ratio (%)", 0)
        if me > 40 or hr > 5:
            ela_hint = "⚠  High error detected — bright areas above may indicate manipulation."
        elif me > 20 or hr > 1:
            ela_hint = "🟡  Moderate error — some irregularities. Review bright regions carefully."
        else:
            ela_hint = "🟢  Low error — ELA image looks uniform. Compression is consistent."
        self.lbl_ela_hint.config(text=ela_hint)

        # Tab 1 – ELA Stats + Flags
        self._fill(self.txt_flags, self._ela_flags_text())

        # Tab 2 – Date & Time
        self._fill(self.txt_datetime, self._section_text(
            "📅  DATE & TIME",
            self.exif_data.get("datetime", {}),
            "Date & time indicate when the photo was taken, modified, or digitized.\n"
            "Inconsistency between these values may suggest post-capture editing."))

        # Tab 3 – Camera
        self._fill(self.txt_camera, self._section_text(
            "📷  CAMERA INFORMATION",
            self.exif_data.get("camera", {}),
            "Camera make and model confirm the device that captured the image.\n"
            "Missing camera info may indicate the image was not captured natively."))

        # Tab 4 – Settings
        self._fill(self.txt_settings, self._section_text(
            "⚙  CAMERA SETTINGS",
            self.exif_data.get("settings", {}),
            "ISO, aperture, and shutter speed are set at capture time by the camera.\n"
            "Missing values may indicate the image was generated or heavily edited."))

        # Tab 5 – Resolution
        self._fill(self.txt_resolution, self._section_text(
            "🖼  IMAGE RESOLUTION",
            self.exif_data.get("resolution", {}),
            "Resolution and DPI values describe the image's physical dimensions.\n"
            "Inconsistent values may indicate the image was resized or resampled."))

        # Tab 6 – Software
        self._fill(self.txt_software, self._software_text())

        # Tab 7 – GPS
        self._fill(self.txt_gps, self._gps_text())

        # Tab 8 – Raw EXIF
        self._fill(self.txt_raw, self._raw_text())

        self._done = True
        self.btn_analyze.config(state="normal")
        self.btn_export.config(state="normal")
        self._start_inactivity_timer()
        self.set_status(
            f"✅  Analysis complete  |  Verdict: {v['verdict']}  |  "
            f"Score: {v['score']}/100")

    #  Text builders 
    def _ela_flags_text(self) -> str:
        sep  = "─" * 52
        ela  = self.ela_stats

        def stat_hint(key):
            """Return a plain-English one-liner for each ELA stat."""
            hints = {
                "Mean Error Level": (
                    "Average pixel difference after re-compression. "
                    "< 20 = good  |  20–40 = moderate  |  > 40 = suspicious"
                ),
                "Max Error Level": (
                    "Highest single-pixel error found. Very high values "
                    "pinpoint the most suspicious region."
                ),
                "Std Deviation": (
                    "How unevenly spread the error is. "
                    "> 20 = uneven (possible pasting)  |  Low = consistent."
                ),
                "High Error Pixels": (
                    "Number of pixels with error level > 60 (threshold)."
                ),
                "High Error Ratio (%)": (
                    "% of pixels that look suspicious. "
                    "< 1% = clean  |  1–5% = minor  |  > 5% = suspicious"
                ),
                "ELA Quality Setting": (
                    "Re-compression quality used (90%). Fixed for consistency."
                ),
                "Amplification Factor": (
                    "Differences are multiplied by this (×10) to make them visible."
                ),
            }
            return hints.get(key, "")

        lines = [
            "ELA STATISTICS",
            "  (Click  ℹ Tap for Info  above to learn what each value means)",
            sep,
        ]
        for k, v in ela.items():
            lines.append(f"  {k:<34}: {v}")
            hint = stat_hint(k)
            if hint:
                lines.append(f"    → {hint}")

        lines += ["", sep,
                  "FORENSIC INDICATORS",
                  "  (🔴 = suspicious  |  🟡 = possible  |  🟢 = likely authentic)",
                  sep]
        for i, flag in enumerate(self.verdict.get("flags", []), 1):
            lines.append(f"\n  {i:>2}. {flag}")

        lines += ["", sep,
                  "EXIF FORENSIC WARNINGS",
                  sep]
        for w in self.exif_data.get("warnings", []):
            lines.append(f"\n  {w}")

        return "\n".join(lines)

    def _section_text(self, title: str, data: dict, note: str = "") -> str:
        sep   = "─" * 52
        lines = [title, sep]
        if note:
            lines += [f"  ℹ  {note}", sep]
        if data:
            for k, v in data.items():
                lines.append(f"  {k:<24}: {v}")
        else:
            lines.append("  No data available.")
        return "\n".join(lines)

    def _software_text(self) -> str:
        sep  = "─" * 52
        sw   = self.exif_data.get("software", {})
        lines = [
            "🧾  SOFTWARE & ORIGIN",
            sep,
            "  The Software field reveals which program last saved this image.",
            "  Editing software tags (Photoshop, GIMP, etc.) are strong",
            "  indicators that the image has been post-processed.",
            sep,
        ]
        for k, v in sw.items():
            marker = "  ⚠  " if (k == "Software" and v != "Not available"
                                  and any(kw in v.lower()
                                  for kw in VerdictEngine.EDIT_KEYWORDS)) else "  "
            lines.append(f"{marker}{k:<22}: {v}")
        return "\n".join(lines)

    def _gps_text(self) -> str:
        sep  = "─" * 52
        gps  = self.exif_data.get("gps", {})
        lines = [
            "🌍  GPS LOCATION DATA",
            sep,
            "  GPS coordinates are embedded by smartphones and GPS-enabled cameras.",
            "  Presence of GPS data helps verify where the image was taken.",
            "  Absence may indicate the image was edited or GPS was disabled.",
            sep,
        ]
        if "lat_decimal" in gps:
            lines += [
                f"  {'Latitude':<22}: {gps.get('latitude', '—')}",
                f"  {'Longitude':<22}: {gps.get('longitude', '—')}",
                f"  {'Altitude':<22}: {gps.get('altitude', 'Not available')}",
                f"  {'GPS Timestamp':<22}: {gps.get('gps_timestamp', 'Not available')}",
                "",
                sep,
                "  🗺  Google Maps Link:",
                f"  {gps.get('maps_link', '—')}",
            ]
        else:
            lines.append(
                f"\n  ⚠  {gps.get('status', 'No GPS data found.')}")
        return "\n".join(lines)

    def _raw_text(self) -> str:
        sep   = "─" * 52
        raw   = self.exif_data.get("raw", {})
        lines = [
            "🗂  COMPLETE RAW EXIF DATA",
            sep,
            f"  Total fields extracted: {len(raw)}",
            sep,
        ]
        if raw:
            for k, v in raw.items():
                lines.append(f"  {k:<28}: {str(v)[:70]}")
        else:
            lines.append("  No raw EXIF metadata found.")
        return "\n".join(lines)

    #  Educational feature methods 
    def open_tutorial(self):
        TutorialWindow(self)

    def open_glossary(self):
        GlossaryWindow(self)

    def _open_score_explainer(self):
        if self._done:
            ScoreExplainerWindow(self, self.ela_stats,
                                 self.exif_data, self.verdict)

    # Full-screen image preview 
    def _open_fullscreen_image(self, slot: str):
        """Open the selected image or ELA result in full screen for clearer demo view."""
        if slot == "ela":
            if self._last_ela_arr is None:
                messagebox.showinfo("No ELA Result", "Run analysis first to view the ELA result in full screen.")
                return
            img = Image.fromarray(self._last_ela_arr)
            title = "Full Screen ELA Result"
        else:
            if not self.image_path:
                messagebox.showinfo("No Image", "Select an image first.")
                return
            img = Image.open(self.image_path)
            title = "Full Screen Original Image"

        win = tk.Toplevel(self)
        win.title(title)
        win.configure(bg="#1E1E1E")
        win.attributes("-fullscreen", True)
        win.focus_force()
        win.grab_set()

        def close_preview(event=None):
            try:
                win.grab_release()
            except tk.TclError as error:
                print(f"Full-screen grab release skipped: {error}")
            win.destroy()
            return "break"

        # Bind both normal and global escape 
        win.bind("<Escape>", close_preview)
        win.bind_all("<Escape>", close_preview)
        win.bind("<Button-1>", close_preview)

        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        preview = img.copy()
        preview.thumbnail((sw - 50, sh - 110), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(preview)

        lbl = tk.Label(win, image=tk_img, bg="#2A2A2A", bd=2, relief="solid")
        lbl.image = tk_img
        lbl.pack(expand=True)

        close_btn = tk.Button(
            win,
            text="✖ Close Full Screen (ESC)",
            font=("Segoe UI", 14, "bold"),
            bg="#7C3AED", fg="white", relief="flat",
            cursor="hand2", padx=18, pady=6,
            command=close_preview)
        close_btn.pack(side="bottom", pady=14)

    #  Export 
    def export_report(self):
        if not self._done:
            messagebox.showinfo("No Results", "Run an analysis first.")
            return

        if REPORTLAB_AVAILABLE:
            self._export_pdf()
        else:

            messagebox.showwarning(
                "PDF Library Missing",
                "reportlab and pypdf are not installed.\n\n"
                "Install them with:\n"
                "  pip install reportlab pypdf\n\n"
                "Falling back to plain-text export (editable).")
            self._export_txt_fallback()

    def _export_pdf(self):
        """Export a locked, read-only PDF forensic report."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="Save Forensic Report (PDF)",
            defaultextension=".pdf",
            filetypes=[("PDF Report", "*.pdf"), ("All Files", "*.*")],
            initialfile=f"forensic_report_{ts}.pdf")
        if not path:
            return

        self.set_status("⏳  Generating PDF report…")
        self.btn_export.config(state="disabled")

        def _build():
            try:
                # Pass ELA image as PIL if available
                ela_pil = None
                if hasattr(self, "_last_ela_arr") and self._last_ela_arr is not None:
                    ela_pil = Image.fromarray(self._last_ela_arr)

                ReportBuilder.build_pdf(
                    image_path=self.image_path,
                    ela_image=ela_pil,
                    ela_stats=self.ela_stats,
                    exif=self.exif_data,
                    verdict=self.verdict,
                    out_path=path,
                )
                self.after(0, lambda: self._export_done(path))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Export Error", f"Could not generate PDF:\n{e}"))
            finally:
                self.after(0, lambda: self.btn_export.config(state="normal"))

        threading.Thread(target=_build, daemon=True).start()

    def _export_done(self, path):
        messagebox.showinfo(
            "PDF Report Saved",
            f"Read-only PDF report saved to:\n{path}\n\n"
            "🔒  The file is locked — editing and copying are disabled.\n"
            "    Printing is still allowed.")
        self.set_status(f"📄  PDF exported (read-only): {os.path.basename(path)}")

    def _export_txt_fallback(self):
        """Plain-text fallback when reportlab is not installed."""
        path = filedialog.asksaveasfilename(
            title="Save Forensic Report (Text)",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("All Files", "*.*")],
            initialfile=f"forensic_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
        if not path:
            return
        report = ReportBuilder.build(self.image_path,
                                     self.ela_stats,
                                     self.exif_data,
                                     self.verdict)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        messagebox.showinfo("Report Saved", f"Saved to:\n{path}")
        self.set_status(f"📄  Report exported: {os.path.basename(path)}")

    #  Clear 
    def clear_all(self):
        self._stop_inactivity_timer()
        self.image_path    = None
        self._done         = False
        self.ela_stats     = {}
        self.exif_data     = {}
        self.verdict       = {}
        self.orig_tk       = None
        self.ela_tk        = None
        self._last_ela_arr = None

        for lbl, msg in [(self.lbl_orig, "Upload an image to begin."),
                         (self.lbl_ela,  "ELA result will appear here.")]:
            lbl.config(image="", text=msg)

        self._reset_panels()
        self.btn_analyze.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.btn_score_info.config(state="disabled")
        self.lbl_file.config(text="No image loaded.")
        self.lbl_verdict.config(text="—  Run analysis to see result", fg=C_MUTED)
        self.lbl_score.config(text="MANIPULATION SCORE: —", fg=C_MUTED)
        self.lbl_ela_hint.config(text="  Run analysis to see ELA result.")
        self.prog["value"] = 0
        self.set_status("Cleared. Ready for new analysis.")

    #  Helpers 
    def set_status(self, msg: str):
        self.lbl_status.config(text=f"  {msg}")

    def _reset_panels(self):
        for txt in (self.txt_flags, self.txt_datetime, self.txt_camera,
                    self.txt_settings, self.txt_resolution,
                    self.txt_software, self.txt_gps, self.txt_raw):
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.config(state="disabled")

    def _fill(self, widget: scrolledtext.ScrolledText, content: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content)
        widget.config(state="disabled")

    def _display_image(self, path: str, label: tk.Label, slot: str):
        img = Image.open(path)
        self._display_image_pil(img, label, slot)

    def _display_image_pil(self, img: Image.Image,
                            label: tk.Label, slot: str):
        label.update_idletasks()
        w = max(label.winfo_width(),  380)
        h = max(label.winfo_height(), 200)
        img.thumbnail((w - 6, h - 6), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img, text="")
        if slot == "orig":
            self.orig_tk = tk_img
        else:
            self.ela_tk  = tk_img


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()