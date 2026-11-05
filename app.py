import os
import io
import zipfile
import tempfile
import subprocess
from datetime import datetime

import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import imageio_ffmpeg
import json
import math
import folium
import osmnx as ox
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    KeepTogether,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aerial Flood Water Mapping",
    page_icon="🌊",
    layout="wide",
)

st.title("Aerial Flood Water Mapping")
st.write(
    "Detect visible standing water in aerial images and drone videos "
    "using U-Net ResNet34 pixel-level segmentation."
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "best_unet.pth")
HISTORY_DIR = os.path.join(PROJECT_DIR, "prediction_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
IMG_SIZE = 256
MAX_SIDE = 1600
FRAME_INTERVAL_SECONDS = 1


# ============================================================
# MODEL AND IMAGE TRANSFORMS
# ============================================================

VAL_TFMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(state)
    model.eval()
    return model


try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


def resize_image(img):
    height, width = img.shape[:2]
    longest_side = max(height, width)

    if longest_side > MAX_SIDE:
        scale = MAX_SIDE / longest_side
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        img = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    return img


def decode_image(image_bytes):
    array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return resize_image(img)


def predict_rgb(img):
    img = resize_image(img)
    height, width = img.shape[:2]

    tensor = VAL_TFMS(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0][0].cpu().numpy()

    probability = cv2.resize(
        probability,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return img, probability


@st.cache_data(show_spinner=False)
def predict(image_bytes):
    img = decode_image(image_bytes)
    if img is None:
        return None, None
    return predict_rgb(img)


def build_views(img, prob, threshold, alpha):
    mask = prob >= threshold

    shown = np.where(mask, prob, 0.0)
    heatmap_bgr = cv2.applyColorMap(
        (shown * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap[~mask] = 0

    blended = cv2.addWeighted(img, 1.0 - alpha, heatmap, alpha, 0)
    overlay = img.copy()
    overlay[mask] = blended[mask]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    return mask, heatmap, overlay


def get_region_count(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return len(contours)


def calculate_metrics(mask, prob):
    water_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    coverage = (water_pixels / total_pixels * 100.0) if total_pixels else 0.0
    mean_confidence = float(prob[mask].mean()) if mask.any() else 0.0

    return {
        "coverage_percent": coverage,
        "water_pixels": water_pixels,
        "total_pixels": total_pixels,
        "mean_confidence": mean_confidence,
        "regions": get_region_count(mask),
    }


# ============================================================
# PDF REPORT HELPERS
# ============================================================

def image_to_png_bytes(image):
    if image.ndim == 2:
        ok, encoded = cv2.imencode(".png", image)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)

    if not ok:
        raise RuntimeError("Could not encode image for PDF report.")

    return io.BytesIO(encoded.tobytes())


def add_pdf_image(story, title, image_buffer, styles, width=480):
    story.append(Paragraph(title, styles["Heading2"]))
    image_buffer.seek(0)
    story.append(PDFImage(image_buffer, width=width, height=270, kind="proportional"))
    story.append(Spacer(1, 12))


def build_pdf_table(rows):
    table = Table(rows, colWidths=[285, 185], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EEF3F8")]),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def generate_image_pdf(img, mask, heatmap, overlay, prob, threshold, image_name):
    metrics = calculate_metrics(mask, prob)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "Aerial Flood Water Mapping - Image Analysis Report",
        styles["Title"],
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Image: {image_name}", styles["Normal"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"],
    ))
    story.append(Paragraph(f"Detection threshold: {threshold:.2f}", styles["Normal"]))
    story.append(Spacer(1, 14))

    rows = [
        ["Metric", "Result"],
        ["Water coverage (image pixels)", f"{metrics['coverage_percent']:.2f}%"],
        ["Detected water pixels", f"{metrics['water_pixels']:,}"],
        ["Total image pixels", f"{metrics['total_pixels']:,}"],
        ["Mean confidence in detected pixels", f"{metrics['mean_confidence']:.3f}"],
        ["Detected connected regions", str(metrics["regions"])],
        ["Image dimensions", f"{img.shape[1]} × {img.shape[0]} pixels"],
    ]
    story.append(build_pdf_table(rows))
    story.append(Spacer(1, 18))

    original_buffer = image_to_png_bytes(img)
    overlay_buffer = image_to_png_bytes(overlay)
    mask_buffer = image_to_png_bytes(mask.astype(np.uint8) * 255)
    heatmap_buffer = image_to_png_bytes(heatmap)

    add_pdf_image(story, "Original Image", original_buffer, styles)
    add_pdf_image(story, "Water Detection Overlay", overlay_buffer, styles)
    add_pdf_image(story, "Binary Water Mask", mask_buffer, styles)
    add_pdf_image(story, "Confidence Heatmap", heatmap_buffer, styles)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(prob.ravel(), bins=30)
    ax.set_title("Model Confidence Distribution")
    ax.set_xlabel("Predicted water probability")
    ax.set_ylabel("Pixel count")
    ax.grid(True, alpha=0.25)

    graph_buffer = io.BytesIO()
    fig.savefig(graph_buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    graph_buffer.seek(0)
    add_pdf_image(story, "Confidence Distribution", graph_buffer, styles)

    story.append(Paragraph(
        "Limitations: Coverage is a percentage of image pixels, not a "
        "real-world geographical area measurement. The model detects visible "
        "standing water and may also detect permanent rivers, lakes, ponds, "
        "reflections, or visually similar surfaces. It cannot independently "
        "distinguish floodwater from permanent water.",
        styles["Italic"],
    ))

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    doc.build(story)
    output.seek(0)
    return output.getvalue()


def generate_video_pdf(df, video_name, threshold):
    styles = getSampleStyleSheet()
    story = []

    average_coverage = float(df["water_coverage_percent"].mean())
    maximum_row = df.loc[df["water_coverage_percent"].idxmax()]
    minimum_row = df.loc[df["water_coverage_percent"].idxmin()]

    story.append(Paragraph(
        "Aerial Flood Water Mapping - Video Analysis Report",
        styles["Title"],
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Video: {video_name}", styles["Normal"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"],
    ))
    story.append(Paragraph(f"Detection threshold: {threshold:.2f}", styles["Normal"]))
    story.append(Spacer(1, 14))

    rows = [
        ["Metric", "Result"],
        ["Processed sampled frames", str(len(df))],
        ["Average sampled-frame coverage", f"{average_coverage:.2f}%"],
        ["Maximum sampled-frame coverage", f"{maximum_row['water_coverage_percent']:.2f}%"],
        ["Timestamp of maximum coverage", f"{maximum_row['timestamp_seconds']:.2f} sec"],
        ["Minimum sampled-frame coverage", f"{minimum_row['water_coverage_percent']:.2f}%"],
        ["Timestamp of minimum coverage", f"{minimum_row['timestamp_seconds']:.2f} sec"],
    ]
    story.append(build_pdf_table(rows))
    story.append(Spacer(1, 18))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(
        df["timestamp_seconds"],
        df["water_coverage_percent"],
        marker="o",
        markersize=3,
    )
    ax.set_title("Water Coverage Across Sampled Video Frames")
    ax.set_xlabel("Video time (seconds)")
    ax.set_ylabel("Water coverage (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    graph_buffer = io.BytesIO()
    fig.savefig(graph_buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    graph_buffer.seek(0)
    add_pdf_image(story, "Frame-wise Water Coverage", graph_buffer, styles)

    story.append(Paragraph(
        "Limitations: Coverage is calculated from image pixels in each sampled "
        "frame. Camera movement, viewing angle, altitude, scene changes, "
        "reflections, and permanent water bodies can affect the values. "
        "These percentages are not a georeferenced flood area measurement "
        "and do not by themselves prove flood expansion.",
        styles["Italic"],
    ))

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    doc.build(story)
    output.seek(0)
    return output.getvalue()


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(video_path, output_dir, threshold, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video could not be opened. Try another video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        cap.release()
        raise ValueError("Could not detect video FPS.")

    duration = total_frames / fps if total_frames > 0 else 0
    frame_step = max(1, int(round(fps * FRAME_INTERVAL_SECONDS)))

    original_dir = os.path.join(output_dir, "original_frames")
    mask_dir = os.path.join(output_dir, "water_masks")
    overlay_dir = os.path.join(output_dir, "overlay_frames")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    raw_video_path = os.path.join(output_dir, "processed_flood_video_raw.mp4")
    video_output_path = os.path.join(output_dir, "processed_flood_video.mp4")

    writer = None
    records = []
    frame_index = 0
    processed_count = 0

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            if frame_index % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img, prob = predict_rgb(frame_rgb)
                mask, _, overlay = build_views(img, prob, threshold, 0.55)

                height, width = img.shape[:2]

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        raw_video_path,
                        fourcc,
                        1.0 / FRAME_INTERVAL_SECONDS,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not create temporary output video.")

                frame_number = processed_count + 1
                frame_name = f"frame_{frame_number:05d}.jpg"
                mask_name = f"mask_{frame_number:05d}.png"
                overlay_name = f"overlay_{frame_number:05d}.jpg"

                cv2.imwrite(
                    os.path.join(original_dir, frame_name),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(mask_dir, mask_name),
                    mask.astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    os.path.join(overlay_dir, overlay_name),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                metrics = calculate_metrics(mask, prob)
                records.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": round(frame_index / fps, 2),
                    "water_coverage_percent": round(metrics["coverage_percent"], 2),
                    "mean_confidence": round(metrics["mean_confidence"], 4),
                    "water_pixels": metrics["water_pixels"],
                    "total_pixels": metrics["total_pixels"],
                    "detected_regions": metrics["regions"],
                })
                processed_count += 1

            frame_index += 1

            # Update progress every approximately one second of source video.
            if total_frames > 0 and frame_index % max(1, int(round(fps))) == 0:
                progress_bar.progress(min(frame_index / total_frames, 1.0))
                status_text.text(
                    f"Processed {processed_count} sampled frames | "
                    f"Video time: {frame_index / fps:.1f} sec"
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if processed_count == 0 or not os.path.exists(raw_video_path):
        raise ValueError("No frames were extracted from this video.")

    # Convert to browser-friendly H.264 MP4.
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    conversion = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i", raw_video_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            video_output_path,
        ],
        capture_output=True,
        text=True,
    )

    if conversion.returncode != 0 or not os.path.exists(video_output_path):
        error_text = conversion.stderr[-2500:] if conversion.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"H.264 conversion failed:\n{error_text}")

    if os.path.exists(raw_video_path):
        os.remove(raw_video_path)

    df = pd.DataFrame(records)
    df.attrs["threshold"] = threshold
    df.attrs["duration_seconds"] = duration

    csv_path = os.path.join(output_dir, "water_coverage_report.csv")
    df.to_csv(csv_path, index=False)

    graph_path = os.path.join(output_dir, "water_coverage_graph.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        df["timestamp_seconds"],
        df["water_coverage_percent"],
        marker="o",
        markersize=3,
    )
    ax.set_title("Water Coverage Across Sampled Video Frames")
    ax.set_xlabel("Video time (seconds)")
    ax.set_ylabel("Water coverage (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.savefig(graph_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    pdf_bytes = generate_video_pdf(df, os.path.basename(video_path), threshold)
    pdf_path = os.path.join(output_dir, "video_analysis_report.pdf")
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)

    maximum_row = df.loc[df["water_coverage_percent"].idxmax()]
    summary_path = os.path.join(output_dir, "video_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as report:
        report.write("AERIAL FLOOD WATER MAPPING - VIDEO REPORT\n")
        report.write("-----------------------------------------\n")
        report.write(f"Original video duration: {duration:.2f} seconds\n")
        report.write(f"Original FPS: {fps:.2f}\n")
        report.write(f"Processed sampled frames: {processed_count}\n")
        report.write(f"Sampling interval: {FRAME_INTERVAL_SECONDS} second(s)\n")
        report.write(f"Average water coverage: {df['water_coverage_percent'].mean():.2f}%\n")
        report.write(f"Maximum water coverage: {maximum_row['water_coverage_percent']:.2f}%\n")
        report.write(f"Maximum coverage timestamp: {maximum_row['timestamp_seconds']:.2f} sec\n")
        report.write("\nCoverage is a percentage of frame pixels, not geographic area.\n")

    progress_bar.progress(1.0)
    status_text.text(
        f"Completed: {processed_count} sampled frames from {duration:.1f} seconds."
    )

    return df, video_output_path, pdf_path




# ============================================================
# PRELIMINARY POND-LIKE / IRREGULAR WATER ANALYSIS
# ============================================================

def classify_water_components(mask, min_component_pixels=80):
    """
    Describe connected water-mask components using image morphology only.
    These labels are visual screening cues, NOT verified flood labels.
    """
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    pond_like_pixels = 0
    irregular_pixels = 0
    uncertain_pixels = 0
    records = []
    annotated = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB)

    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < min_component_pixels:
            uncertain_pixels += area
            continue

        component = (labels == component_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            uncertain_pixels += area
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        circularity = (
            4.0 * math.pi * contour_area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
        aspect_ratio = max(width, height) / max(1, min(width, height))

        # Conservative morphology-only groups. Irregular does not mean flood.
        if circularity >= 0.45 and solidity >= 0.75 and aspect_ratio <= 2.5:
            label = "Pond-like shape (not confirmed)"
            pond_like_pixels += area
            color = (40, 190, 90)
        elif area >= 250 and (circularity < 0.35 or aspect_ratio > 3.0 or solidity < 0.60):
            label = "Irregular/spread water (review; not confirmed flood)"
            irregular_pixels += area
            color = (255, 165, 0)
        else:
            label = "Uncertain water"
            uncertain_pixels += area
            color = (255, 230, 0)

        cv2.drawContours(annotated, [contour], -1, color, 2)
        cv2.putText(
            annotated, f"C{component_id}",
            (x, max(15, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
        records.append({
            "component_id": component_id,
            "area_pixels": area,
            "bbox_width": width,
            "bbox_height": height,
            "circularity": round(circularity, 3),
            "solidity": round(solidity, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "morphology_label": label,
        })

    return {
        "components": records,
        "pond_like_pixels": pond_like_pixels,
        "irregular_pixels": irregular_pixels,
        "uncertain_pixels": uncertain_pixels,
        "annotated_mask": annotated,
    }


def analyze_uploaded_video_morphology(video_bytes, filename, threshold,
                                      sample_interval_seconds=1,
                                      max_frames=300):
    """Run U-Net on sampled frames and compute preliminary shape descriptors."""
    suffix = os.path.splitext(filename)[1] or ".mp4"
    temp_path = None
    rows = []
    previews = []
    total_frames = 0

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Video could not be opened. Try an MP4 file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fps if fps and fps > 0 else 25.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        frame_step = max(1, int(round(fps * sample_interval_seconds)))

        frame_index = 0
        sampled = 0
        while sampled < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = resize_image(rgb)
            image, probability = predict_rgb(rgb)
            mask = probability >= threshold
            metrics = calculate_metrics(mask, probability)
            morphology = classify_water_components(mask)

            total_pixels = int(mask.size)
            pond_pct = morphology["pond_like_pixels"] / max(1, total_pixels) * 100
            irregular_pct = morphology["irregular_pixels"] / max(1, total_pixels) * 100
            uncertain_pct = morphology["uncertain_pixels"] / max(1, total_pixels) * 100

            timestamp = frame_index / fps
            rows.append({
                "timestamp_seconds": round(timestamp, 2),
                "water_coverage_percent": round(metrics["coverage_percent"], 3),
                "pond_like_shape_pixels_percent": round(pond_pct, 3),
                "irregular_spread_water_pixels_percent": round(irregular_pct, 3),
                "uncertain_water_pixels_percent": round(uncertain_pct, 3),
                "detected_components": len(morphology["components"]),
            })

            if len(previews) < 6:
                overlay = image.copy()
                colored = morphology["annotated_mask"]
                # Draw outlines from the morphology result on the original frame.
                outline_mask = np.any(colored != np.repeat(
                    (mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2
                ), axis=2)
                # Use contour colors generated above as a simple preview layer.
                overlay[outline_mask] = (
                    0.55 * overlay[outline_mask] + 0.45 * colored[outline_mask]
                ).astype(np.uint8)
                previews.append((timestamp, overlay, morphology["annotated_mask"]))

            sampled += 1
            frame_index += frame_step

        cap.release()
        if not rows:
            raise ValueError("No frames could be sampled from this video.")

        return pd.DataFrame(rows), previews, duration, frame_count, fps

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# ZIP EXPORT
# ============================================================

def create_zip(output_dir):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(output_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, output_dir)
                zip_file.write(full_path, relative_path)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Detection Controls")
threshold = st.sidebar.slider(
    "Detection threshold",
    0.05,
    0.95,
    0.50,
    0.05,
    help="Lower values detect more water but may increase false detections.",
)
overlay_alpha = st.sidebar.slider(
    "Image overlay opacity",
    0.0,
    1.0,
    0.55,
    0.05,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Model scope:** The model segments visible standing water. Permanent "
    "rivers, lakes and ponds may also be detected. It cannot distinguish "
    "normal water from floodwater. Without georeferencing, coverage is a "
    "share of image pixels, not real-world ground area."
)
st.sidebar.caption("Inference device: CPU")




# ============================================================
# OPENSTREETMAP GIS HELPERS
# ============================================================

OSM_APP_USER_AGENT = "flood_extent_mapping_group16_iitp_demo"


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_osm_place(place_name):
    """Geocode one place entered by the user; cache the result."""
    geolocator = Nominatim(
        user_agent=OSM_APP_USER_AGENT,
        timeout=20
    )
    result = geolocator.geocode(
        place_name,
        exactly_one=True,
        timeout=20
    )
    if result is None:
        return None
    return (
        float(result.latitude),
        float(result.longitude),
        result.address
    )


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_osm_water_features(latitude, longitude, radius_m):
    """
    Fetch community-mapped OSM water features around a selected place.
    This is a reference layer, not a georeferenced flood prediction.
    """
    ox.settings.use_cache = True
    ox.settings.log_console = False

    lat_delta = radius_m / 111320.0
    cos_lat = max(math.cos(math.radians(latitude)), 0.15)
    lon_delta = radius_m / (111320.0 * cos_lat)

    west = longitude - lon_delta
    south = latitude - lat_delta
    east = longitude + lon_delta
    north = latitude + lat_delta

    bbox = (west, south, east, north)
    tags = {
        "natural": "water",
        "waterway": "riverbank",
        "landuse": "reservoir",
    }

    gdf = ox.features_from_bbox(
        bbox=bbox,
        tags=tags
    )

    if gdf is None or gdf.empty:
        return None

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        return None

    useful_columns = [
        column for column in
        ["name", "natural", "water", "waterway", "landuse", "geometry"]
        if column in gdf.columns
    ]
    gdf = gdf[useful_columns].copy()

    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def render_osm_gis_tab():
    st.header("OpenStreetMap GIS – Reference Water Bodies")
    st.write(
        "Search a place to display mapped ponds, lakes, reservoirs and "
        "riverbank features from OpenStreetMap."
    )

    place = st.text_input(
        "Location to inspect",
        value="Patna, Bihar, India",
        help=(
            "Your drone-video location is unknown. Patna is only a sample "
            "GIS demonstration area. Replace it if you later identify the "
            "actual video location."
        ),
        key="osm_gis_place"
    )

    radius_km = st.slider(
        "Search radius (km)",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        key="osm_gis_radius"
    )

    st.info(
        "The selected OSM location is a reference-map demo only. "
        "It is not assumed to be the location where your drone video was recorded."
    )

    if st.button(
        "Fetch OSM Water Bodies",
        type="primary",
        key="fetch_osm_water"
    ):
        if not place.strip():
            st.error("Enter a city, village or place name.")
        else:
            try:
                with st.spinner("Finding the selected location..."):
                    location = geocode_osm_place(place.strip())

                if location is None:
                    st.error(
                        "Location not found. Try a more specific place name."
                    )
                else:
                    latitude, longitude, address = location
                    st.caption(f"Matched location: {address}")

                    with st.spinner(
                        "Downloading mapped water features from OpenStreetMap..."
                    ):
                        water_gdf = fetch_osm_water_features(
                            latitude,
                            longitude,
                            radius_km * 1000
                        )

                    if water_gdf is None or water_gdf.empty:
                        st.warning(
                            "No mapped water features were returned. "
                            "OSM mapping may be incomplete in this area; "
                            "this does not prove that no water bodies exist."
                        )
                    else:
                        st.session_state["osm_gis_geojson"] = water_gdf.to_json()
                        st.session_state["osm_gis_center"] = (
                            latitude,
                            longitude
                        )
                        st.session_state["osm_gis_address"] = address
                        st.session_state["osm_gis_count"] = len(water_gdf)

            except Exception as exc:
                st.error(
                    "OSM request failed. Public geocoding/Overpass services "
                    "may be temporarily busy. Wait briefly and try once more."
                )
                st.exception(exc)

    geojson_text = st.session_state.get("osm_gis_geojson")
    center = st.session_state.get("osm_gis_center")

    if geojson_text and center:
        geojson_data = json.loads(geojson_text)
        latitude, longitude = center

        st.success(
            f"Loaded {st.session_state.get('osm_gis_count', 0)} "
            "mapped water feature(s)."
        )

        map_obj = folium.Map(
            location=[latitude, longitude],
            zoom_start=14,
            control_scale=True,
            tiles="OpenStreetMap"
        )

        folium.Marker(
            [latitude, longitude],
            tooltip="Selected search location",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(map_obj)

        tooltip_fields = [
            field for field in
            ["name", "natural", "water", "waterway", "landuse"]
            if any(
                field in feature.get("properties", {})
                for feature in geojson_data.get("features", [])
            )
        ]
        tooltip_labels = {
            "name": "Name",
            "natural": "Natural",
            "water": "Water type",
            "waterway": "Waterway",
            "landuse": "Land use",
        }

        folium.GeoJson(
            data=geojson_data,
            name="OSM mapped water features",
            style_function=lambda feature: {
                "color": "#1769aa",
                "weight": 2,
                "fillColor": "#35a7d5",
                "fillOpacity": 0.35,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[
                    tooltip_labels[field]
                    for field in tooltip_fields
                ],
                localize=True,
                sticky=False
            ) if tooltip_fields else None,
        ).add_to(map_obj)

        folium.LayerControl().add_to(map_obj)

        st_folium(
            map_obj,
            use_container_width=True,
            height=520
        )

        st.download_button(
            "Download OSM water features (GeoJSON)",
            data=geojson_text.encode("utf-8"),
            file_name="osm_reference_water_bodies.geojson",
            mime="application/geo+json",
            key="download_osm_geojson"
        )

        st.caption(
            "Map data © OpenStreetMap contributors · ODbL. "
            "OSM features are community-mapped and may be incomplete or outdated."
        )

        with st.expander("Important interpretation"):
            st.write(
                "Blue features are water objects mapped in OSM. They are not "
                "guaranteed to be permanent water. Areas absent from OSM are "
                "not automatically floodwater. Accurate comparison with the "
                "U-Net mask requires the drone image/frame to be georeferenced "
                "to the same coordinate reference system."
            )


# ============================================================
# MAIN TABS
# ============================================================

image_tab, video_tab, history_tab, gis_tab, pond_flood_tab = st.tabs([
    "Image Detection",
    "Video Detection",
    "Prediction History",
    "GIS Mapping",
    "Pond vs Floodwater Analysis",
])


# ============================================================
# IMAGE DETECTION
# ============================================================

with image_tab:
    st.header("Flood Detection from Aerial Image")
    st.write("Upload an aerial or drone image to identify visible water.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_file is None:
        st.info("Upload an aerial image to begin.")
    else:
        with st.spinner("Running image prediction..."):
            img, prob = predict(uploaded_file.getvalue())

        if img is None:
            st.error("Image could not be read. Please upload a JPG or PNG.")
        else:
            mask, heatmap, overlay = build_views(
                img,
                prob,
                threshold,
                overlay_alpha,
            )
            metrics = calculate_metrics(mask, prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Water coverage", f"{metrics['coverage_percent']:.1f}%")
            c2.metric(
                "Mean confidence",
                f"{metrics['mean_confidence']:.2f}" if mask.any() else "—",
            )
            c3.metric("Distinct regions", str(metrics["regions"]))

            st.caption(
                f"Detected water pixels: {metrics['water_pixels']:,} / "
                f"{metrics['total_pixels']:,} | Image size: "
                f"{img.shape[1]} × {img.shape[0]} px"
            )

            if metrics["coverage_percent"] < 0.5:
                st.warning(
                    "No or very little water detected at this threshold. "
                    "Try lowering the threshold or inspect the input image."
                )
            elif metrics["coverage_percent"] > 85:
                st.warning(
                    "Most of the frame is flagged as water. Inspect the overlay "
                    "carefully before interpreting this result."
                )

            st.subheader("Original vs Detected Result")
            comparison_col1, comparison_col2 = st.columns(2)
            with comparison_col1:
                st.image(img, caption="Original image", use_container_width=True)
            with comparison_col2:
                st.image(overlay, caption="Water detection overlay", use_container_width=True)

            t1, t2, t3 = st.tabs([
                "Confidence Heatmap",
                "Binary Mask",
                "Original Image",
            ])
            with t1:
                st.image(
                    heatmap,
                    caption="Model confidence inside detected regions",
                    use_container_width=True,
                )
            with t2:
                st.image(
                    mask.astype(np.uint8) * 255,
                    caption="Binary water mask",
                    use_container_width=True,
                )
            with t3:
                st.image(img, use_container_width=True)

            st.subheader("Advanced Image Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric("Water pixels", f"{metrics['water_pixels']:,}")
            stat_col2.metric("Total pixels", f"{metrics['total_pixels']:,}")
            stat_col3.metric("Detected regions", str(metrics["regions"]))

            st.write("**Pixel confidence distribution**")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(prob.ravel(), bins=30)
            ax.set_xlabel("Predicted water probability")
            ax.set_ylabel("Pixel count")
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

            png_ok, png_buffer = cv2.imencode(
                ".png",
                mask.astype(np.uint8) * 255,
            )
            if png_ok:
                st.download_button(
                    "Download Binary Mask (PNG)",
                    data=png_buffer.tobytes(),
                    file_name="flood_water_mask.png",
                    mime="image/png",
                )

            if st.button("Generate Image PDF Report", key="generate_image_pdf"):
                try:
                    with st.spinner("Generating image PDF report..."):
                        pdf_bytes = generate_image_pdf(
                            img,
                            mask,
                            heatmap,
                            overlay,
                            prob,
                            threshold,
                            uploaded_file.name,
                        )
                    st.session_state["image_pdf_bytes"] = pdf_bytes
                    st.success("Image PDF report generated successfully.")
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")

            if st.session_state.get("image_pdf_bytes"):
                st.download_button(
                    "Download Image Analysis PDF",
                    data=st.session_state["image_pdf_bytes"],
                    file_name="flood_image_analysis_report.pdf",
                    mime="application/pdf",
                    key="download_image_pdf",
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

with video_tab:
    st.header("Flood Detection from Drone Video")
    st.write(
        "Upload a drone video. The application samples approximately one "
        "frame per second and performs water segmentation on each sampled frame."
    )
    st.info(
        "Video processing uses CPU. Long or high-resolution videos may take "
        "several minutes. The output video contains sampled frames at about "
        "one frame per second; it is not a full-frame-rate copy of the input."
    )

    uploaded_video = st.file_uploader(
        "Upload drone video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video is not None:
        st.video(uploaded_video.getvalue())

        if st.button("Start Video Flood Detection", type="primary"):
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_dir = os.path.join(HISTORY_DIR, f"video_{run_id}")
            os.makedirs(output_dir, exist_ok=True)

            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            temp_video_path = None
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_video:
                    temp_video.write(uploaded_video.getvalue())
                    temp_video_path = temp_video.name

                with st.spinner("Processing video frames..."):
                    df, processed_video_path, report_path = process_video(
                        temp_video_path,
                        output_dir,
                        threshold,
                        progress_bar,
                        status_text,
                    )

                st.success("Video processing completed successfully.")

                maximum_row = df.loc[df["water_coverage_percent"].idxmax()]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Processed Frames", len(df))
                c2.metric(
                    "Average Water Coverage",
                    f"{df['water_coverage_percent'].mean():.2f}%",
                )
                c3.metric(
                    "Maximum Water Coverage",
                    f"{maximum_row['water_coverage_percent']:.2f}%",
                )
                c4.metric(
                    "Maximum Coverage Time",
                    f"{maximum_row['timestamp_seconds']:.1f} sec",
                )

                st.subheader("Processed H.264 Video")
                with open(processed_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                st.download_button(
                    "Download Processed Video (MP4)",
                    data=video_bytes,
                    file_name="processed_flood_video.mp4",
                    mime="video/mp4",
                )

                st.subheader("Water Coverage Over Video Time")
                chart_df = df.set_index("timestamp_seconds")[
                    ["water_coverage_percent"]
                ]
                st.line_chart(chart_df)

                st.subheader("Frame-wise Analysis Report")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download Coverage Report (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="water_coverage_report.csv",
                    mime="text/csv",
                )

                with open(report_path, "rb") as pdf_file:
                    video_pdf_bytes = pdf_file.read()
                st.download_button(
                    "Download Video Analysis PDF",
                    data=video_pdf_bytes,
                    file_name="video_analysis_report.pdf",
                    mime="application/pdf",
                )

                st.subheader("Extracted Frame Results")
                overlay_dir = os.path.join(output_dir, "overlay_frames")
                overlay_files = sorted(os.listdir(overlay_dir))

                preview_count = min(6, len(overlay_files))
                if preview_count:
                    preview_cols = st.columns(min(3, preview_count))
                    for i, filename in enumerate(overlay_files[:preview_count]):
                        frame_path = os.path.join(overlay_dir, filename)
                        frame_img = cv2.imread(frame_path)
                        if frame_img is not None:
                            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                            with preview_cols[i % len(preview_cols)]:
                                st.image(
                                    frame_img,
                                    caption=filename,
                                    use_container_width=True,
                                )

                st.subheader("Complete Results")
                zip_bytes = create_zip(output_dir)
                st.download_button(
                    "Download Complete Results (ZIP)",
                    data=zip_bytes,
                    file_name="flood_video_analysis.zip",
                    mime="application/zip",
                )
                st.caption(f"Results saved locally in: {output_dir}")

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
                st.exception(exc)

            finally:
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)


# ============================================================
# PREDICTION HISTORY
# ============================================================

with history_tab:
    st.header("Prediction History")
    st.write("Browse video-analysis folders saved on this computer.")

    history_items = []
    if os.path.isdir(HISTORY_DIR):
        for folder_name in sorted(os.listdir(HISTORY_DIR), reverse=True):
            folder_path = os.path.join(HISTORY_DIR, folder_name)
            if os.path.isdir(folder_path):
                history_items.append((folder_name, folder_path))

    if not history_items:
        st.info("No saved video analysis results found yet.")
    else:
        for folder_name, folder_path in history_items:
            with st.expander(folder_name):
                csv_path = os.path.join(folder_path, "water_coverage_report.csv")
                pdf_path = os.path.join(folder_path, "video_analysis_report.pdf")
                video_path = os.path.join(folder_path, "processed_flood_video.mp4")

                if os.path.exists(csv_path):
                    try:
                        history_df = pd.read_csv(csv_path)
                        if not history_df.empty:
                            st.metric(
                                "Average water coverage",
                                f"{history_df['water_coverage_percent'].mean():.2f}%",
                            )
                            st.dataframe(
                                history_df.head(10),
                                use_container_width=True,
                            )
                    except Exception as exc:
                        st.warning(f"Could not read CSV: {exc}")

                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            "Download PDF report",
                            data=pdf_file.read(),
                            file_name=f"{folder_name}_report.pdf",
                            mime="application/pdf",
                            key=f"history_pdf_{folder_name}",
                        )

                if os.path.exists(video_path):
                    with open(video_path, "rb") as video_file:
                        st.download_button(
                            "Download processed video",
                            data=video_file.read(),
                            file_name=f"{folder_name}_processed.mp4",
                            mime="video/mp4",
                            key=f"history_video_{folder_name}",
                        )


# ============================================================
# GIS MAPPING TAB
# ============================================================

with gis_tab:
    render_osm_gis_tab()




# ============================================================
# POND VS FLOODWATER PRELIMINARY ANALYSIS TAB
# ============================================================

with pond_flood_tab:
    st.header("Pond-like vs Irregular/Spread Water Analysis")
    st.warning(
        "This is a preliminary morphology-based screening tool, not a trained "
        "pond-versus-flood classifier. Shape alone cannot prove flooding. "
        "Without GPS/georeferencing or before-and-after imagery, results must "
        "be interpreted as visual cues for manual review."
    )
    st.write(
        "Upload the ORIGINAL drone video (not a previously processed/annotated "
        "MP4). The app samples frames, runs the existing U-Net, and describes "
        "connected water regions by shape."
    )

    st.info(
        "Labels mean: green = pond-like shape; orange = irregular/spread water "
        "that may deserve inundation review; yellow = uncertain. Orange does "
        "not mean confirmed floodwater. Patna's OSM layer is not compared with "
        "this video because the video location is unknown."
    )

    analysis_video = st.file_uploader(
        "Upload original drone video",
        type=["mp4", "avi", "mov", "mkv"],
        key="pond_flood_video_upload",
    )
    sample_interval = st.selectbox(
        "Frame sampling interval",
        options=[1, 2, 5],
        index=0,
        format_func=lambda value: f"Every {value} second(s)",
        key="pond_flood_sample_interval",
    )
    max_sample_frames = st.slider(
        "Maximum sampled frames",
        min_value=30,
        max_value=300,
        value=120,
        step=30,
        key="pond_flood_max_frames",
    )

    if analysis_video is not None:
        st.caption(f"Selected file: {analysis_video.name}")
        if st.button(
            "Run Pond vs Floodwater Preliminary Analysis",
            type="primary",
            key="run_pond_flood_analysis",
        ):
            try:
                with st.spinner(
                    "Sampling video and running U-Net + shape analysis. "
                    "CPU processing may take a few minutes..."
                ):
                    analysis_df, preview_frames, duration, frame_count, fps = (
                        analyze_uploaded_video_morphology(
                            analysis_video.getvalue(),
                            analysis_video.name,
                            threshold,
                            sample_interval_seconds=sample_interval,
                            max_frames=max_sample_frames,
                        )
                    )

                st.session_state["pond_flood_analysis_df"] = analysis_df
                st.session_state["pond_flood_preview_frames"] = preview_frames
                st.session_state["pond_flood_video_name"] = analysis_video.name
                st.session_state["pond_flood_video_meta"] = {
                    "duration": duration,
                    "frame_count": frame_count,
                    "fps": fps,
                }
                st.success("Preliminary analysis completed.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

    analysis_df = st.session_state.get("pond_flood_analysis_df")
    if analysis_df is not None and not analysis_df.empty:
        meta = st.session_state.get("pond_flood_video_meta", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Sampled frames", len(analysis_df))
        col2.metric("Average water coverage",
                    f"{analysis_df['water_coverage_percent'].mean():.2f}%")
        col3.metric("Video duration", f"{meta.get('duration', 0):.1f} sec")

        st.subheader("Frame-wise water morphology indicators")
        chart_cols = [
            "pond_like_shape_pixels_percent",
            "irregular_spread_water_pixels_percent",
            "uncertain_water_pixels_percent",
        ]
        chart_df = analysis_df.set_index("timestamp_seconds")[chart_cols]
        st.line_chart(chart_df)

        st.subheader("Analysis table")
        st.dataframe(analysis_df, use_container_width=True)
        st.download_button(
            "Download morphology analysis CSV",
            data=analysis_df.to_csv(index=False).encode("utf-8"),
            file_name="pond_vs_floodwater_preliminary_analysis.csv",
            mime="text/csv",
            key="download_pond_flood_csv",
        )

        st.subheader("Sampled frame previews")
        previews = st.session_state.get("pond_flood_preview_frames", [])
        for start in range(0, len(previews), 2):
            cols = st.columns(2)
            for offset, (timestamp, frame, mask_preview) in enumerate(
                previews[start:start + 2]
            ):
                with cols[offset]:
                    st.image(
                        frame,
                        caption=f"Original frame with detected component outlines — {timestamp:.1f}s",
                        use_container_width=True,
                    )
                    st.image(
                        mask_preview,
                        caption="Component morphology preview (green/orange/yellow outlines)",
                        use_container_width=True,
                    )

        st.caption(
            "Interpretation: pond-like geometry is not proof of a pond; irregular "
            "or spread geometry is not proof of flooding. Camera movement, "
            "perspective, vegetation, shadows, reflections, threshold choice, "
            "and segmentation errors can change these indicators. The video "
            "does not provide geographic area without georeferencing."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Aerial Flood Water Mapping | U-Net ResNet34 | "
    "Pixel-level standing-water segmentation | CPU inference"
)
