import json
import time
import subprocess
import platform
from pathlib import Path
import cv2
import numpy as np
from .events import DEFAULT_ZONE, Incidents, polygon, inside

def process(source, output, tracker='bytetrack', confidence=0.35, zone=None,
            model_path='yolo11n.pt', device='cpu', max_frames=0, callback=None, cooldown=5.0, track_experiment=False, engine='ultralytics', imgsz=320, threads=2):
    """One model and tracker per run. Timestamps use source frame index / FPS."""
    from ultralytics import YOLO
    if engine not in ('ultralytics','pytorch','onnx'):
        raise ValueError('Unsupported inference engine')
    if device!='cpu':
        raise ValueError('This project version runs on CPU')
    if tracker not in ('bytetrack', 'botsort'):
        raise ValueError('Unsupported tracker')
    if not 0.1 <= confidence <= 0.95:
        raise ValueError('Confidence must be between 0.1 and 0.95')
    if cooldown < 0:
        raise ValueError('Cooldown must be nonnegative')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not retain an earlier run's success marker.
    (out/'summary.json').unlink(missing_ok=True)
    (out/'mlflow.json').unlink(missing_ok=True)
    cap = cv2.VideoCapture(source)
    writer = None
    log = Incidents(cooldown=cooldown)
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video/camera. Try a local MP4 file.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            if isinstance(source, int):
                fps = 25.0
            else:
                raise ValueError('Video has no usable FPS metadata; transcode it first.')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if engine=='ultralytics':
            model = YOLO(model_path)
            model.predict(np.zeros((384, 640, 3), np.uint8), imgsz=640, device='cpu', verbose=False)
        else:
            from .engines import TorchEngine, OnnxEngine
            from .tracker_adapter import create_tracker,NumpyDetections
            detector=TorchEngine(model_path,threads,imgsz) if engine=='pytorch' else OnnxEngine(model_path,threads)
            if detector.size!=imgsz: raise ValueError('Input size must match fixed ONNX export')
            mot=create_tracker(tracker,fps)
            detector.detect(np.zeros((imgsz,imgsz,3),np.uint8))
        unique, latencies = set(), []
        frames = 0
        started = time.perf_counter()
        contour = None
        while not max_frames or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if contour is None:
                contour = polygon(DEFAULT_ZONE if zone is None else zone, w, h)
                writer = cv2.VideoWriter(str(out/'annotated_raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                if not writer.isOpened():
                    raise RuntimeError('MP4 video encoder is unavailable.')
            seconds = frames/fps
            # Keep low-confidence candidates for ByteTrack's second association pass.
            if engine=='ultralytics':
                r = model.track(frame, persist=True, tracker=tracker+'.yaml', classes=[0],
                                conf=0.1, imgsz=640, device='cpu', verbose=False)[0]
                latencies.append(float(r.speed['inference']))
                tracked_rows=[] if r.boxes is None or r.boxes.id is None else list(zip(
                    r.boxes.xyxy.cpu().numpy(),r.boxes.id.int().cpu().tolist(),r.boxes.conf.cpu().tolist()))
            else:
                detections,latency=detector.detect(frame,confidence=.1)
                latencies.append(latency)
                tracked=mot.update(NumpyDetections(detections),frame)
                tracked_rows=[(row[:4],int(row[4]),float(row[5])) for row in tracked]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [contour], (35, 70, 180))
            frame = cv2.addWeighted(frame, .82, overlay, .18, 0)
            cv2.polylines(frame, [contour], True, (50,170,255), 2)
            active = 0
            if tracked_rows:
                for b, tid, score in tracked_rows:
                    if score < confidence:
                        continue
                    unique.add(tid)
                    violation = inside(b, contour)
                    log.update(tid, violation, seconds)
                    active += int(violation)
                    color = (65,65,245) if violation else (100,220,70)
                    x1,y1,x2,y2 = map(int,b)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    label = f'ID {tid} | {score:.2f}' + (' | ZONE ALERT' if violation else '')
                    cv2.putText(frame,label,(max(0,x1),max(45,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
                    cv2.circle(frame, (int((x1+x2)/2),y2),4,color,-1)
            # Missing detections are unknown, not evidence of exiting the zone.
            cv2.rectangle(frame,(0,0),(w,34),(23,29,39),-1)
            cv2.putText(frame,f'SAFEVISION | {seconds:.1f}s | IDs {len(unique)} | Incidents {len(log.rows)} | In zone {active}',
                        (10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
            writer.write(frame)
            if frames == 0 or (active and frames % 10 == 0):
                cv2.imwrite(str(out/'preview.jpg'),frame)
            frames += 1
            if callback and frames % 10 == 0:
                callback(frames, total, frame)
        elapsed = time.perf_counter()-started
        if not frames:
            raise ValueError('No decodable frames found in the source.')
    finally:
        cap.release()
        if writer:
            writer.release()
        log.save(out/'incidents.csv')
    # Browser-compatible H.264. Fail clearly instead of presenting a broken player.
    import imageio_ffmpeg
    encoding = time.perf_counter()
    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel','error',
        '-i',str(out/'annotated_raw.mp4'),'-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v','libx264','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',
        str(out/'annotated.mp4')],capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError('Video encoding failed: '+proc.stderr[-1000:])
    (out/'annotated_raw.mp4').unlink()
    summary = dict(frames=frames, source_fps=fps, unique_track_ids=len(unique), incidents=len(log.rows),
                   processing_fps=frames/elapsed, inference_ms=float(np.mean(latencies)),
                   processing_seconds=elapsed, encoding_seconds=time.perf_counter()-encoding,
                   tracker=tracker, confidence=confidence, model=str(model_path), device=device,
                   zone=DEFAULT_ZONE if zone is None else zone, cooldown_seconds=cooldown)
    summary.update(inference_engine=engine,inference_imgsz=640 if engine=='ultralytics' else imgsz,
                   cpu_threads=None if engine=='ultralytics' else threads)
    import torch
    import ultralytics
    summary['environment'] = dict(python=platform.python_version(),platform=platform.platform(),
                                 processor=platform.processor(),torch=torch.__version__,
                                 ultralytics=ultralytics.__version__,opencv=cv2.__version__)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if track_experiment:
        from .experiment import log_result
        summary["mlflow"] = log_result(out, source=source)
    return summary
