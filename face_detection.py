import argparse
import cv2
import glob
import os
import time
import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

# 默认输出目录
OUTPUT_DIR = 'detected_faces'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载YOLOv11模型
def load_model():
    # 使用预训练的YOLOv11模型，专注于人脸检测
    model = YOLO('yolov11n-face.pt')
    # 确保使用GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model.to(device)
    return model

# 批量处理帧并检测人脸
def process_batch(frames_data, model, confidence_threshold=0.5):
    batch_frames = [data['frame'] for data in frames_data]
    
    # 批量推理
    results = model(batch_frames, verbose=False)
    
    face_data = []
    for i, result in enumerate(results):
        frame_info = frames_data[i]
        frame = frame_info['frame']
        timestamp = frame_info['timestamp']
        frame_count = frame_info['frame_count']
        
        # 计算时间字符串
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # 处理检测结果
        boxes = result.boxes
        for box in boxes:
            # 获取置信度
            confidence = float(box.conf[0])
            
            # 如果置信度足够高
            if confidence > confidence_threshold:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 裁剪人脸区域
                face_img = frame[y1:y2, x1:x2]
                
                face_data.append({
                    'face_img': face_img,
                    'time_str': time_str,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'frame_count': frame_count
                })
    
    return face_data

# 保存检测到的人脸图片
def save_face_images(face_data, start_face_count=0, output_dir=OUTPUT_DIR):
    """保存检测到的人脸图片到指定目录并返回记录列表与新的计数."""
    os.makedirs(output_dir, exist_ok=True)
    face_records = []
    face_count = start_face_count

    for data in face_data:
        face_img = data['face_img']
        time_str = data['time_str']
        timestamp = data['timestamp']
        confidence = data['confidence']

        # 保存人脸图片
        face_filename = os.path.join(output_dir, f"face_{face_count:04d}_time_{time_str.replace(':', '_')}.jpg")
        cv2.imwrite(face_filename, face_img)

        # 记录人脸出现的时间
        face_records.append({
            'face_id': face_count,
            'time': time_str,
            'timestamp': timestamp,
            'confidence': confidence,
            'image_path': face_filename
        })

        face_count += 1

    return face_records, face_count

# 处理视频并检测人脸
def process_video(video_path, model, output_dir=OUTPUT_DIR, batch_size=8, sample_interval=8, confidence_threshold=0.5):
    # 打开视频文件
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频: {video_path}")
    except Exception as e:
        print(f"错误: {str(e)}")
        return None
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频FPS: {fps}")
    print(f"总帧数: {total_frames}")
    
    # 使用传入的批处理大小与采样间隔
    
    # 用于记录人脸出现的时间
    all_face_records = []
    face_count = 0

    # 为该视频创建独立输出目录
    video_stem = Path(video_path).stem
    per_video_out = os.path.join(output_dir, video_stem)
    os.makedirs(per_video_out, exist_ok=True)
    
    # 创建进度条
    pbar = tqdm(total=total_frames, desc="处理视频")
    
    # 处理视频帧
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        frames_batch = []
        batch_count = 0
        
        # 收集一批帧
        while batch_count < batch_size and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 更新进度条
            pbar.update(1)
            
            # 每隔几帧处理一次，提高效率
            if frame_count % sample_interval == 0:
                # 计算当前帧的时间戳
                timestamp = frame_count / fps
                
                frames_batch.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_count': frame_count
                })
                batch_count += 1
            
            frame_count += 1
        
        # 如果没有帧可处理，退出循环
        if not frames_batch:
            break
        
        # 批量处理帧
        face_data = process_batch(frames_batch, model, confidence_threshold=confidence_threshold)

        # 保存检测到的人脸到本视频的输出目录
        if face_data:
            face_records, face_count = save_face_images(face_data, face_count, output_dir=per_video_out)
            all_face_records.extend(face_records)
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    cap.release()
    
    # 保存人脸记录到文件（视频级）
    save_records(all_face_records, output_dir=per_video_out, video_basename=video_stem)
    
    print(f"处理完成! 共检测到 {face_count} 个人脸")
    return all_face_records

# 保存人脸记录到文件
def save_records(face_records, output_dir=OUTPUT_DIR, video_basename: str = "records"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{video_basename}_face_records.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("人脸ID,时间(MM:SS),时间戳(秒),置信度,图片路径\n")
            for record in face_records:
                f.write(f"{record['face_id']},{record['time']},{record['timestamp']:.2f},{record['confidence']:.4f},{record['image_path']}\n")
    except Exception as e:
        print(f"保存记录时出错: {str(e)}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description='批量人脸检测（支持单个视频或目录）')
    parser.add_argument('input', help='视频文件路径或包含视频的目录')
    parser.add_argument('--output-dir', '-o', default=OUTPUT_DIR, help='输出目录（默认: detected_faces）')
    parser.add_argument('--batch-size', type=int, default=8, help='批处理大小（默认: 8）')
    parser.add_argument('--sample-interval', type=int, default=8, help='采样间隔（帧）（默认: 8）')
    parser.add_argument('--confidence', type=float, default=0.5, help='检测置信度阈值（默认: 0.5）')
    parser.add_argument('--recursive', action='store_true', help='递归查找目录中的视频文件')
    args = parser.parse_args()

    print("开始加载YOLOv11模型...")
    model = load_model()
    print("模型加载完成!")

    # 设置PyTorch以使用更多线程
    if torch.cuda.is_available():
        torch.set_num_threads(4)

    input_path = args.input
    to_process = []
    if os.path.isfile(input_path):
        to_process = [input_path]
    elif os.path.isdir(input_path):
        exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        if args.recursive:
            for root, _, files in os.walk(input_path):
                for fn in files:
                    if fn.lower().endswith(exts):
                        to_process.append(os.path.join(root, fn))
        else:
            for ext in exts:
                to_process.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
    else:
        print(f"输入路径不存在: {input_path}")
        return

    total_start = time.time()
    for vid in to_process:
        print(f"开始处理视频: {vid}")
        start_time = time.time()
        process_video(vid, model, output_dir=args.output_dir, batch_size=args.batch_size, sample_interval=args.sample_interval, confidence_threshold=args.confidence)
        end_time = time.time()
        print(f"处理完成: {vid} 用时 {end_time - start_time:.2f} 秒")
    total_end = time.time()
    print(f"全部完成，总用时 {total_end - total_start:.2f} 秒，视频数量: {len(to_process)}")

if __name__ == "__main__":
    main()