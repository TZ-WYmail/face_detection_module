"""face_dedup_pipeline.py

命令行入口：从视频/目录提取高质量人脸并进行去重（track 聚合 + embedding dedup）。
使用 `face_dedup_utils` 中的组件，优先使用 insightface，如果缺失则回退。
"""
from __future__ import annotations
import os
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from face_dedup_utils import (
    Detector, SimpleEmbedder, Deduper, align_face, is_high_quality_face
)


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)


def process_video(video_path: str, output_dir: str, args):
    os.makedirs(output_dir, exist_ok=True)
    det = Detector(backend=args.detector, device='cuda' if args.cuda else 'cpu')
    model_track = None
    device_str = 'cuda' if args.cuda else 'cpu'
    use_half = bool(args.cuda)
    if args.use_tracks:
        try:
            # 优先复用 Detector 内的 YOLO 模型（若存在），否则新建 ultralytics YOLO 实例
            if getattr(det, 'model', None) is not None:
                model_track = det.model
            else:
                from ultralytics import YOLO
                model_track = YOLO('yolov11n-face.pt')
            print("使用 ultralytics YOLO + ByteTrack 进行跟踪")
        except Exception as e:
            print("错误: 未能初始化 ultralytics YOLO 或 ByteTrack 跟踪器。请安装 ultralytics + YOLOX/ByteTrack。参考 scripts/face_detection/README.md")
            print("导入/初始化错误详情:", e)
            return 0
    deduper = Deduper(metric=args.metric, threshold=args.threshold)
    embedder = SimpleEmbedder()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_interval = args.sample_interval if args.sample_interval and args.sample_interval > 0 else max(int(round(fps)), 1)

    preview_out = os.path.join(output_dir, 'preview.mp4')
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return 0
    H, W = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vw = cv2.VideoWriter(preview_out, cv2.VideoWriter_fourcc(*'mp4v'), max(fps / sample_interval, 1), (W, H))

    # record paths
    rec_path = os.path.join(output_dir, 'face_records_dedup.txt')
    if not os.path.exists(rec_path) or os.path.getsize(rec_path) == 0:
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write('persona_id,track_id,time,timestamp,confidence,image_path,frame_count\n')

    frame_count = 0
    face_id = 0
    saved_count = 0

    # track aggregator state
    track_db = {}  # tid -> {'embs':[], 'best_conf':float, 'best_img':np.ndarray, 'first_saved':bool}
    previous_active = set()

    pbar = tqdm(total=total_frames if total_frames>0 else None, desc=f"Dedup: {Path(video_path).name}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if total_frames > 0:
                pbar.update(1)

            if frame_count % sample_interval != 0:
                frame_count += 1
                continue

            timestamp = frame_count / fps
            time_str = time.strftime('%H:%M:%S', time.gmtime(int(timestamp)))

            current_active = set()

            if args.use_tracks and model_track is not None:
                # 使用 ultralytics 的 model.track 接口（内部可配置 ByteTrack）进行检测+关联
                try:
                    results = model_track.track(frame, persist=True, conf=args.conf, tracker="bytetrack.yaml", verbose=False, half=use_half, device=device_str)
                except Exception as e:
                    print("错误: 调用 model.track 失败：", e)
                    results = None

                xyxy = []
                ids = []
                confs = []
                saved_flags = []

                if results and len(results) > 0 and getattr(results[0], 'boxes', None) is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    if getattr(boxes, 'id', None) is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        ids = np.array([-1] * len(xyxy))

                    for j in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[j])
                        tid = int(ids[j])
                        conf = float(confs[j])
                        if tid == -1:
                            continue
                        current_active.add(tid)
                        clipped = (max(0, min(x1, W-1)), max(0, min(y1, H-1)), max(0, min(x2, W-1)), max(0, min(y2, H-1)))
                        cx1, cy1, cx2, cy2 = clipped
                        if cx2 <= cx1 or cy2 <= cy1:
                            continue
                        face_crop = frame[cy1:cy2, cx1:cx2]

                        # 尝试用 Detector 在 face_crop 上获得关键点与 embedding（始终尝试，避免质量筛选误杀）
                        det_info = None
                        try:
                            found = det.detect(face_crop, conf_threshold=args.conf)
                            if found:
                                # 取第一个检测结果作为该 crop 的关键点/embedding 信息
                                det_info = found[0]
                        except Exception:
                            det_info = None

                        # 质量过滤（若启用）
                        if args.require_quality:
                            if det_info is None or not is_high_quality_face(det_info):
                                continue

                        aligned = align_face(face_crop, det_info.get('kps') if det_info is not None else None)
                        emb = None
                        if det_info is not None and det_info.get('embedding') is not None:
                            emb = np.asarray(det_info.get('embedding'), dtype=np.float32)
                        else:
                            emb = SimpleEmbedder.get_embedding_from_detection({}, aligned)

                        rec = track_db.setdefault(tid, {'embs': [], 'best_conf': 0.0, 'best_img': None, 'first_saved': False})
                        if emb is not None:
                            rec['embs'].append(emb)
                        if conf > rec['best_conf']:
                            rec['best_conf'] = conf
                            rec['best_img'] = aligned.copy() if aligned is not None else face_crop.copy()

                        # 策略：first
                        if args.save_strategy == 'first' and not rec['first_saved']:
                            pid = deduper.find_match(emb) if emb is not None else None
                            if pid is None and emb is not None:
                                pid = deduper.add(emb)
                                img_path = os.path.join(output_dir, f"face_{pid:05d}_track{tid}_f{frame_count}.jpg")
                                if save_image(aligned, img_path):
                                    saved_count += 1
                                    rec['first_saved'] = True
                                    with open(rec_path, 'a', encoding='utf-8') as f:
                                        f.write(f"{pid},{tid},{time_str},{timestamp:.2f},{conf:.4f},{img_path},{frame_count}\n")

                # 将当前帧的 tracking 输出用于可视化（写在循环尾部）


            else:
                # 不使用 tracker：逐检测去重
                dets = det.detect(frame, conf_threshold=args.conf)
                for i, d in enumerate(dets):
                    if args.require_quality and not is_high_quality_face(d):
                        continue
                    x1,y1,x2,y2 = d['bbox']
                    face_crop = frame[y1:y2, x1:x2]
                    aligned = align_face(face_crop, d.get('kps', None))
                    emb = SimpleEmbedder.get_embedding_from_detection(d, aligned)
                    pid = deduper.find_match(emb) if emb is not None else None
                    if pid is None and emb is not None:
                        pid = deduper.add(emb)
                        img_path = os.path.join(output_dir, f"face_{pid:05d}_f{frame_count}.jpg")
                        if save_image(aligned, img_path):
                            saved_count += 1
                            with open(rec_path, 'a', encoding='utf-8') as f:
                                f.write(f"{pid},-1,{time_str},{timestamp:.2f},{d.get('confidence',0):.4f},{img_path},{frame_count}\n")

            # 处理 track 结束（best_end）
            if args.use_tracks and model_track is not None and args.save_strategy == 'best_end':
                prev = previous_active
                ended = prev - current_active
                for tid in list(ended):
                    rec = track_db.get(tid)
                    if not rec:
                        continue
                    if len(rec.get('embs', [])) == 0:
                        continue
                    agg = np.mean(np.stack(rec['embs'], axis=0), axis=0)
                    pid = deduper.find_match(agg)
                    if pid is None:
                        pid = deduper.add(agg)
                    img_path = os.path.join(output_dir, f"face_{pid:05d}_track{tid}_BEST_END.jpg")
                    if rec.get('best_img') is not None and save_image(rec['best_img'], img_path):
                        saved_count += 1
                        with open(rec_path, 'a', encoding='utf-8') as f:
                            f.write(f"{pid},{tid},{time_str},{timestamp:.2f},{rec.get('best_conf',0):.4f},{img_path},{frame_count}\n")
                    track_db.pop(tid, None)

            previous_active = current_active

            # 预览写帧（简单版本）
            vw.write(frame)

            frame_count += 1

    finally:
        pbar.close()
        cap.release()
        vw.release()

    print(f"完成: {saved_count} 张人脸已保存 (输出: {output_dir})")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='高质量人脸去重流水线（track + embedding）')
    parser.add_argument('input', type=str, help='视频文件或包含视频的目录')
    parser.add_argument('--output-dir', '-o', type=str, default='detected_faces_dedup', help='输出顶层目录')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--sample-interval', type=int, default=1)
    parser.add_argument('--use-tracks', dest='use_tracks', action='store_true')
    parser.add_argument('--no-tracks', dest='use_tracks', action='store_false')
    parser.set_defaults(use_tracks=True)
    parser.add_argument('--save-strategy', type=str, choices=['first', 'best', 'best_end'], default='best_end')
    parser.add_argument('--require-quality', dest='require_quality', action='store_true')
    parser.add_argument('--detector', type=str, default='auto', help='auto/insightface/yolo/haar')
    parser.add_argument('--threshold', type=float, default=0.6, help='Dedup 相似阈值（cosine）')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--iou', type=float, default=0.3)
    parser.add_argument('--max-age', type=int, default=30)
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()

    inp = args.input
    to_process = []
    exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    if os.path.isfile(inp):
        to_process = [inp]
    elif os.path.isdir(inp):
        for root, _, files in os.walk(inp):
            for fn in files:
                if fn.lower().endswith(exts):
                    to_process.append(os.path.join(root, fn))
    else:
        print(f"输入路径不存在: {inp}")
        return

    total = 0
    for vid in to_process:
        video_name = Path(vid).stem
        per_out = os.path.join(args.output_dir, video_name)
        os.makedirs(per_out, exist_ok=True)
        process_video(vid, per_out, args)


if __name__ == '__main__':
    main()
