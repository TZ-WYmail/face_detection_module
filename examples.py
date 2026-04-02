#!/usr/bin/env python3
"""
使用示例脚本

展示如何以编程方式使用人脸检测和去重流水线
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from face_dedup_pipeline import FrontalFaceExtractor, process_video
import argparse


def example_1_basic_usage():
    """示例1：基本使用 - 处理单个视频"""
    print("\n" + "="*60)
    print("示例1：基本使用")
    print("="*60)
    
    video_path = "sample_video.mp4"
    output_dir = "output/example1"
    
    if not Path(video_path).exists():
        print(f"⚠️  示例视频不存在: {video_path}")
        print("请提供你的视频文件路径")
        return
    
    print(f"输入视频: {video_path}")
    print(f"输出目录: {output_dir}")
    
    # 使用argparse创建args对象（模拟命令行参数）
    args = argparse.Namespace(
        input=video_path,
        output_dir=output_dir,
        conf=0.5,
        sample_interval=1,
        use_tracks=True,
        save_strategy='best_end',
        yaw_threshold=25.0,
        pitch_threshold=25.0,
        roll_threshold=15.0,
        threshold=0.8,  # 标准模式：0.8（阈值越高，匹配标准越严格，漏检越多）
        metric='cosine',
        detector='auto',
        cuda=False
    )
    
    # 处理视频
    process_video(video_path, output_dir, args)
    
    print("✅ 处理完成！")
    print(f"结果已保存到: {output_dir}")


def example_2_high_quality():
    """示例2：高质量提取 - 严格的正脸过滤"""
    print("\n" + "="*60)
    print("示例2：高质量正脸提取")
    print("="*60)
    
    video_path = "sample_video.mp4"
    output_dir = "output/example2_high_quality"
    
    if not Path(video_path).exists():
        print(f"⚠️  示例视频不存在: {video_path}")
        return
    
    print("使用严格的正脸过滤参数")
    print("- yaw_threshold: 10.0°")
    print("- pitch_threshold: 10.0°")
    print("- roll_threshold: 5.0°")
    
    args = argparse.Namespace(
        input=video_path,
        output_dir=output_dir,
        conf=0.6,
        sample_interval=1,
        use_tracks=True,
        save_strategy='best_end',
        yaw_threshold=10.0,      # 严格
        pitch_threshold=10.0,    # 严格
        roll_threshold=5.0,      # 严格
        threshold=0.85,          # 严格模式：0.85（非常严格，只保留高置信度匹配）
        metric='cosine',
        detector='insightface',
        cuda=True
    )
    
    process_video(video_path, output_dir, args)
    print("✅ 处理完成！")


def example_3_fast_processing():
    """示例3：快速处理 - 优先处理速度"""
    print("\n" + "="*60)
    print("示例3：快速处理模式")
    print("="*60)
    
    video_path = "sample_video.mp4"
    output_dir = "output/example3_fast"
    
    if not Path(video_path).exists():
        print(f"⚠️  示例视频不存在: {video_path}")
        return
    
    print("使用快速处理参数")
    print("- sample_interval: 5 (每5帧处理一帧)")
    print("- detector: yolo")
    print("- confidence: 0.3")
    
    args = argparse.Namespace(
        input=video_path,
        output_dir=output_dir,
        conf=0.3,              # 较低的检测阈值
        sample_interval=5,     # 较大的采样间隔
        use_tracks=True,
        save_strategy='first',  # 首个正脸即保存
        yaw_threshold=40.0,    # 宽松阈值
        pitch_threshold=40.0,
        roll_threshold=30.0,
        threshold=0.8,         # 即使宽松模式也使用0.8（保持严格的相似度匹配）
        metric='cosine',
        detector='yolo',       # 使用快速检测器
        cuda=True
    )
    
    process_video(video_path, output_dir, args)
    print("✅ 处理完成！")


def example_4_batch_processing():
    """示例4：批处理 - 处理多个视频"""
    print("\n" + "="*60)
    print("示例4：批处理模式")
    print("="*60)
    
    video_dir = "videos"
    output_dir = "output/batch_results"
    
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        print(f"⚠️  视频目录不存在: {video_dir}")
        print("请创建 'videos' 目录并放入视频文件")
        return
    
    # 查找所有视频文件
    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    video_files = []
    for ext in video_exts:
        video_files.extend(video_dir_path.glob(f'*{ext}'))
        video_files.extend(video_dir_path.glob(f'**/*{ext}'))
    
    if not video_files:
        print(f"⚠️  未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    args = argparse.Namespace(
        output_dir=output_dir,
        conf=0.5,
        sample_interval=1,
        use_tracks=True,
        save_strategy='best_end',
        yaw_threshold=25.0,
        pitch_threshold=25.0,
        roll_threshold=15.0,
        threshold=0.8,
        metric='cosine',
        detector='auto',
        cuda=True
    )
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n处理第 {i}/{len(video_files)} 个视频: {video_file.name}")
        video_name = video_file.stem
        per_out = os.path.join(output_dir, video_name)
        
        try:
            process_video(str(video_file), per_out, args)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            continue
    
    print("\n✅ 所有视频处理完成！")


def example_5_using_extractor_class():
    """示例5：直接使用FrontalFaceExtractor类"""
    print("\n" + "="*60)
    print("示例5：使用FrontalFaceExtractor类")
    print("="*60)
    
    import argparse
    
    # 创建配置对象
    args = argparse.Namespace(
        detector='insightface',
        cuda=True,
        metric='cosine',
        threshold=0.8,
        yaw_threshold=25.0,
        pitch_threshold=25.0,
        roll_threshold=15.0,
        conf=0.5,
        save_strategy='best_end',
        use_tracks=True
    )
    
    # 创建提取器对象
    extractor = FrontalFaceExtractor(args)
    
    print("✅ FrontalFaceExtractor 初始化成功")
    print(f"  检测器: {extractor.det.backend}")
    print(f"  跟踪: {'启用' if extractor.model_track else '禁用'}")
    print(f"  去重阈值: {extractor.deduper.threshold}")
    
    # 这里你可以使用 extractor 进行自定义处理
    # 例如：
    # result = extractor.process_frame_with_tracking(frame, ...)


def main():
    """主函数 - 列出所有示例"""
    print("\n" + "="*60)
    print("人脸检测与去重流水线 - 使用示例")
    print("="*60)
    print("\n可用的示例：")
    print("  1. 基本使用 - 处理单个视频")
    print("  2. 高质量提取 - 严格的正脸过滤")
    print("  3. 快速处理 - 优先处理速度")
    print("  4. 批处理 - 处理多个视频")
    print("  5. 使用提取器类 - 自定义处理")
    
    # 如果有命令行参数，运行对应示例
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_1_basic_usage,
            '2': example_2_high_quality,
            '3': example_3_fast_processing,
            '4': example_4_batch_processing,
            '5': example_5_using_extractor_class,
        }
        
        if example_num in examples:
            try:
                examples[example_num]()
            except KeyboardInterrupt:
                print("\n\n⚠️  用户中断")
            except Exception as e:
                print(f"\n❌ 错误发生: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ 未知的示例: {example_num}")
            sys.exit(1)
    else:
        print("\n使用方法:")
        print("  python examples.py <示例号>")
        print("\n示例：")
        print("  python examples.py 1")
        print("  python examples.py 2")


if __name__ == '__main__':
    main()
