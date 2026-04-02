#!/bin/bash

# 项目使用示例脚本

echo "=========================================="
echo "人脸检测与去重流水线 - 示例运行"
echo "=========================================="
echo ""

# 设置项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# 示例 1: 处理单个视频（GPU）
echo "示例 1: 处理单个视频（GPU 加速）"
echo "命令: python face_dedup_pipeline.py videos/your_video.mp4 --cuda"
echo ""
# python face_dedup_pipeline.py videos/your_video.mp4 --cuda

# 示例 2: 处理整个文件夹（CPU）
echo "示例 2: 处理整个文件夹（CPU）"
echo "命令: python face_dedup_pipeline.py videos/ --output custom_output"
echo ""
# python face_dedup_pipeline.py videos/ --output custom_output

# 示例 3: 高质量模式（身份证照采集）
echo "示例 3: 高质量模式（身份证照采集）"
echo "命令: python face_dedup_pipeline.py videos/id_photo.mp4 --cuda \\"
echo "        --yaw-threshold 10 --pitch-threshold 10 --roll-threshold 5"
echo ""
# python face_dedup_pipeline.py videos/id_photo.mp4 --cuda \
#     --yaw-threshold 10 --pitch-threshold 10 --roll-threshold 5

# 示例 4: 快速处理模式
echo "示例 4: 快速处理模式（速度优先）"
echo "命令: python face_dedup_pipeline.py videos/ --cuda \\"
echo "        --detector yolo --sample-interval 5 --confidence 0.3"
echo ""
# python face_dedup_pipeline.py videos/ --cuda \
#     --detector yolo --sample-interval 5 --confidence 0.3

echo ""
echo "=========================================="
echo "提示："
echo "  • 将你的视频放到 videos/ 文件夹"
echo "  • 取消注释（删除 #）以运行相应示例"
echo "  • 结果保存到 detected_faces_frontal/ 文件夹"
echo "  • 查看所有参数：python face_dedup_pipeline.py --help"
echo "=========================================="
