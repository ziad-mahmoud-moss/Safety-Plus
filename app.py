from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import threading
import time
import uuid
import tempfile
import atexit
import webbrowser
import zipfile
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent
OUTPUT = Path(tempfile.mkdtemp(prefix="safety-plus-"))
atexit.register(shutil.rmtree, OUTPUT, ignore_errors=True)
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
MODEL_PATH = BASE / "model.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
lock = threading.Lock()
detectors = None
CLASS_NAMES = ("person", "helmet", "vest", "shoes", "cigarette", "phone", "fire_extinguisher")
NMS_IOU = 0.50


def load_detectors():
    """Unpack the two original checkpoints into temporary storage once."""
    from ultralytics import YOLO

    extracted = OUTPUT / "models"
    extracted.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(MODEL_PATH) as bundle:
            metadata = json.loads(bundle.read("manifest.json"))
            if metadata.get("format") != "safety-plus-ensemble-v1":
                raise ValueError("صيغة حزمة model.pt غير متوافقة مع التطبيق.")
            members = metadata["models"]
            if [member["file"] for member in members] != ["best.pt", "best-gazar.pt"]:
                raise ValueError("حزمة model.pt لا تحتوي على الموديلين المطلوبين.")
            loaded = []
            for member in members:
                destination = extracted / member["file"]
                digest = hashlib.sha256()
                with bundle.open(member["file"]) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != member["sha256"]:
                    raise ValueError("بيانات model.pt غير مكتملة. أعد نسخ الحزمة الأصلية.")
                loaded.append((member["file"], YOLO(str(destination)), int(member["imgsz"])))
            return loaded
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise ValueError("تعذّر قراءة حزمة model.pt. تأكد من تنزيل الملف الكامل، بما فيه ملفات Git LFS.") from error


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/style.css")
def stylesheet():
    return send_from_directory(BASE, "style.css")


@app.get("/api/status")
def engine_status():
    return jsonify(ready=MODEL_PATH.is_file())


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="حجم الملف أكبر من 100 ميجابايت."), 413


def update(job_id, **values):
    with lock:
        jobs[job_id].update(values)


def detect(job_id, source, confidence, kind):
    global detectors
    started = time.perf_counter()
    capture = writer = None
    try:
        import cv2
        import numpy as np
        update(job_id, status="processing", message="جاري تجهيز محرك الكشف…", progress=2)
        if detectors is None:
            detectors = load_detectors()
        folder = source.parent
        counts = defaultdict(lambda: {"count": 0, "sum": 0.0, "max": 0.0})
        records = []

        def predict(frame, frame_index=0, timestamp=0.0):
            candidates = defaultdict(list)
            for model_name, model, image_size in detectors:
                result = model.predict(frame, conf=confidence, imgsz=image_size, verbose=False)[0]
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    label = str(result.names[int(box.cls.item())]).strip().lower().replace(" ", "_").replace("-", "_")
                    candidates[label].append({"label": label, "confidence": float(box.conf.item()),
                                              "box": box.xyxy[0].tolist(), "source_model": model_name})
            # Suppress duplicate boxes within each normalized class only.
            # Different classes (for example person and vest) stay independent.
            detections = []
            for label, items in candidates.items():
                boxes = [[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                         for x1, y1, x2, y2 in (item["box"] for item in items)]
                scores = [item["confidence"] for item in items]
                indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, NMS_IOU)
                for index in np.asarray(indices, dtype=int).reshape(-1):
                    item = items[int(index)]
                    score = item["confidence"]
                    counts[label]["count"] += 1
                    counts[label]["sum"] += score
                    counts[label]["max"] = max(counts[label]["max"], score)
                    detections.append({"label": label, "confidence": round(score, 4),
                                       "box": [round(v, 1) for v in item["box"]],
                                       "source_model": item["source_model"]})
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            records.append({"frame": frame_index, "time": round(timestamp, 3), "detections": detections})
            # Use clear percentage labels instead of tiny decimal confidence labels.
            annotated = frame.copy()
            colors = [(217, 178, 80), (111, 205, 71), (86, 208, 237), (204, 158, 106),
                      (102, 119, 242), (199, 134, 226), (162, 211, 69)]
            line_width = max(2, round(min(frame.shape[:2]) / 350))
            font_scale = max(0.5, min(1.2, min(frame.shape[:2]) / 1000))
            for item in detections:
                label = item["label"]
                class_id = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
                color = colors[int(class_id) % len(colors)]
                x1, y1, x2, y2 = map(int, item["box"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
                caption = f"{label.replace('_', ' ')}  {item['confidence'] * 100:.1f}%"
                (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                label_x = max(0, min(x1, frame.shape[1] - tw - 12))
                label_y = max(th + 12, y1)
                cv2.rectangle(annotated, (label_x, label_y - th - 12), (label_x + tw + 12, label_y + baseline), color, -1)
                cv2.putText(annotated, caption, (label_x + 6, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (20, 35, 30), 1, cv2.LINE_AA)
            return annotated

        if kind == "image":
            frame = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("تعذّر قراءة الصورة. جرّب صورة بصيغة JPG أو PNG.")
            if frame.shape[0] * frame.shape[1] > 40_000_000:
                raise ValueError("أبعاد الصورة كبيرة جدًا. قلّل حجمها إلى أقل من 40 ميجابكسل.")
            update(job_id, message="جاري تحليل الصورة…", progress=25)
            annotated = predict(frame)
            success, encoded = cv2.imencode(".jpg", annotated)
            if not success:
                raise ValueError("تعذّر حفظ الصورة الناتجة.")
            encoded.tofile(str(folder / "result.jpg"))
            result_name = "result.jpg"
            frame_count = 1
        else:
            capture = cv2.VideoCapture(str(source))
            fps = capture.get(cv2.CAP_PROP_FPS)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or total <= 0:
                raise ValueError("تعذّر قراءة الفيديو. جرّب ملف MP4 آخر.")
            if total / fps > 120:
                raise ValueError("ارفع فيديو مدته دقيقتان أو أقل لتجربة الكشف.")
            frame_count = 0
            raw_output = folder / "annotated.mp4"
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_count / fps > 120:
                    raise ValueError("مدة الفيديو تتجاوز دقيقتين.")
                # Keep processing practical on CPU while preserving the aspect ratio.
                scale = min(1.0, 1280 / max(frame.shape[:2]))
                width = max(2, int(frame.shape[1] * scale) // 2 * 2)
                height = max(2, int(frame.shape[0] * scale) // 2 * 2)
                frame = cv2.resize(frame, (width, height))
                if writer is None:
                    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise ValueError("تعذّر إنشاء ملف الفيديو الناتج.")
                annotated = predict(frame, frame_count, frame_count / fps)
                writer.write(annotated)
                frame_count += 1
                update(job_id, progress=min(92, 5 + round(frame_count / total * 87)),
                       message=f"جاري تحليل الفيديو: {frame_count} / {total} إطار")
            capture.release()
            writer.release() if writer is not None else None
            if frame_count == 0:
                raise ValueError("الفيديو لا يحتوي على إطارات قابلة للقراءة.")
            update(job_id, progress=95, message="جاري تجهيز الفيديو للعرض…")
            import imageio_ffmpeg

            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_output),
                "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                "-shortest", str(folder / "result.mp4"),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            raw_output.unlink(missing_ok=True)
            result_name = "result.mp4"

        summary = [{"label": label, "count": values["count"],
                    "confidence": round(values["sum"] / values["count"], 4),
                    "max_confidence": round(values["max"], 4)} for label, values in counts.items()]
        summary.sort(key=lambda row: row["count"], reverse=True)
        detection_count = sum(row["count"] for row in summary)
        report = {"model": "Safety Plus Ensemble", "source_models": [item[0] for item in detectors],
                  "fusion": {"method": "class-wise NMS", "iou": NMS_IOU, "confidence": "retained source score"},
                  "threshold": confidence, "kind": kind,
                  "frames": frame_count, "total_detections": detection_count,
                  "average_confidence": round(sum(v["sum"] for v in counts.values()) / detection_count, 4) if detection_count else 0,
                  "seconds": round(time.perf_counter() - started, 1), "summary": summary,
                  "records": records}
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        public_report = {key: value for key, value in report.items() if key != "records"}
        update(job_id, status="done", progress=100, message="اكتمل التحليل",
               result={**public_report, "media_url": f"/results/{job_id}/{result_name}",
                       "report_url": f"/results/{job_id}/report.json"})
    except ValueError as error:
        update(job_id, status="error", message=str(error))
    except Exception:
        logging.exception("Detection failed for %s", job_id)
        update(job_id, status="error", message="تعذّر إكمال التحليل. تأكد من تثبيت المتطلبات وتحديث ultralytics إلى 8.4.149 أو أحدث؛ التفاصيل في نافذة التشغيل.")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        source.unlink(missing_ok=True)


@app.post("/api/detect")
def start_detection():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify(error="اختَر صورة أو فيديو أولًا."), 400
    extension = Path(upload.filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return jsonify(error="صيغة الملف غير مدعومة."), 400
    if not MODEL_PATH.is_file():
        return jsonify(error="ضع ملف model.pt بجوار app.py ثم حاول مجددًا."), 400
    try:
        confidence = float(request.form.get("confidence", "0.25"))
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
            raise ValueError()
    except ValueError:
        return jsonify(error="حدّ الثقة يجب أن يكون بين 5% و95%."), 400
    with lock:
        if sum(job["status"] in {"queued", "processing"} for job in jobs.values()) >= 3:
            return jsonify(error="هناك ملفات قيد التحليل. حاول مجددًا بعد قليل."), 429
        # Expire completed session files after 24 hours when a new upload arrives.
        expired = [key for key, job in jobs.items() if job["status"] in {"done", "error"} and time.time() - job["created"] > 86400]
        for key in expired:
            shutil.rmtree(OUTPUT / key, ignore_errors=True)
            del jobs[key]
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "queued", "progress": 0, "message": "في انتظار بدء التحليل…", "created": time.time()}
    folder = OUTPUT / job_id
    source = folder / f"input{extension}"
    try:
        folder.mkdir()
        upload.save(source)
        executor.submit(detect, job_id, source, confidence,
                        "image" if extension in IMAGE_EXTENSIONS else "video")
    except Exception:
        with lock:
            jobs.pop(job_id, None)
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify(error="تعذّر حفظ الملف. تأكد من توفر مساحة كافية."), 500
    return jsonify(job_id=job_id), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with lock:
        job = jobs.get(job_id)
        return jsonify(job) if job else (jsonify(error="جلسة التحليل غير موجودة. ارفع الملف مجددًا."), 404)


@app.get("/results/<job_id>/<filename>")
def result_file(job_id, filename):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        return jsonify(error="الملف غير موجود."), 404
    if filename not in {"result.jpg", "result.mp4", "report.json"}:
        return jsonify(error="الملف غير موجود."), 404
    return send_from_directory(OUTPUT / job_id, filename, as_attachment=request.args.get("download") == "1")


if __name__ == "__main__":
    browser_timer = threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000"))
    browser_timer.daemon = True
    browser_timer.start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
