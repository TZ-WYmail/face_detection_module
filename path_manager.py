#!/usr/bin/env python3
"""
统一的路径管理系统

功能:
  ✓ 集中定义所有输出路径
  ✓ 自动创建必要的目录
  ✓ 路径验证和规范化
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PathManager:
    """统一的路径管理器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化路径管理器
        
        Args:
            output_dir: 输出根目录（如果为None，使用默认的detected_faces_frontal）
        """
        if output_dir is None:
            output_dir = 'detected_faces_frontal'
        
        self.output_dir = Path(output_dir).resolve()
        self.embeddings_dir = self.output_dir / 'embeddings'
        self.debug_frames_dir = self.output_dir / 'debug_frames'
        self.records_file = self.output_dir / 'face_records_frontal.txt'
        self.preview_video_file = self.output_dir / 'preview.mp4'
        
        # 初始化目录结构
        self._init_directories()
    
    def _init_directories(self) -> None:
        """创建所有必需的目录"""
        directories = [
            self.output_dir,
            self.embeddings_dir,
            self.debug_frames_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"✓ 路径管理器初始化完成:")
        logger.debug(f"  - 输出根目录: {self.output_dir}")
        logger.debug(f"  - Embedding目录: {self.embeddings_dir}")
        logger.debug(f"  - 调试帧目录: {self.debug_frames_dir}")
    
    def get_face_image_path(self, pid: int, tid: int, frame_count: int, 
                           suffix: str = '') -> Path:
        """
        获取人脸图像保存路径
        
        Args:
            pid: 人脸ID
            tid: 轨迹ID
            frame_count: 帧号
            suffix: 后缀（如'FRONTAL', 'BEST'）
            
        Returns:
            人脸图像路径
        """
        if suffix:
            filename = f"face_{pid:05d}_track{tid}_{suffix}.jpg"
        else:
            filename = f"face_{pid:05d}_track{tid}_f{frame_count}.jpg"
        
        return self.output_dir / filename
    
    def get_embedding_path(self, pid: int, tid: int, frame_count: int) -> Path:
        """
        获取Embedding保存路径
        
        Args:
            pid: 人脸ID
            tid: 轨迹ID
            frame_count: 帧号
            
        Returns:
            Embedding文件路径
        """
        filename = f"face_{pid:05d}_track{tid}_f{frame_count}.npy"
        return self.embeddings_dir / filename
    
    def get_debug_frame_path(self, pid: int, tid: int, frame_count: int) -> Path:
        """
        获取调试帧保存路径
        
        Args:
            pid: 人脸ID
            tid: 轨迹ID
            frame_count: 帧号
            
        Returns:
            调试帧文件路径
        """
        filename = f"frame_{pid:05d}_track{tid}_f{frame_count}.jpg"
        return self.debug_frames_dir / filename
    
    def get_debug_info_path(self, pid: int, tid: int, frame_count: int) -> Path:
        """
        获取调试信息保存路径
        
        Args:
            pid: 人脸ID
            tid: 轨迹ID
            frame_count: 帧号
            
        Returns:
            调试信息文件路径
        """
        filename = f"frame_{pid:05d}_track{tid}_f{frame_count}_info.txt"
        return self.debug_frames_dir / filename
    
    def verify_structure(self) -> bool:
        """
        验证路径结构完整性
        
        Returns:
            是否完整
        """
        required_dirs = [
            self.output_dir,
            self.embeddings_dir,
            self.debug_frames_dir,
        ]
        
        all_exist = all(d.exists() for d in required_dirs)
        
        if not all_exist:
            logger.warning("✗ 路径结构不完整，尝试重新初始化...")
            self._init_directories()
            all_exist = True
        
        return all_exist
    
    def get_status_summary(self) -> str:
        """获取路径状态摘要"""
        summary = [
            "\n" + "="*60,
            "路径结构状态",
            "="*60,
            f"根目录: {self.output_dir}",
            f"Embedding: {self.embeddings_dir}",
            f"调试帧: {self.debug_frames_dir}",
            f"记录文件: {self.records_file}",
            f"预览视频: {self.preview_video_file}",
            "="*60,
        ]
        return "\n".join(summary)
    
    def get_file_count(self) -> dict:
        """获取各目录文件统计"""
        return {
            'face_images': len(list(self.output_dir.glob('face_*.jpg'))),
            'embeddings': len(list(self.embeddings_dir.glob('*.npy'))),
            'debug_frames': len(list(self.debug_frames_dir.glob('*.jpg'))),
        }


def main():
    """测试入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    # 测试路径管理器
    pm = PathManager('test_output')
    print(pm.get_status_summary())
    
    # 示例路径生成
    print("\n示例路径生成:")
    print(f"人脸图像: {pm.get_face_image_path(1, 100, 0, 'FRONTAL')}")
    print(f"Embedding: {pm.get_embedding_path(1, 100, 0)}")
    print(f"调试帧: {pm.get_debug_frame_path(1, 100, 0)}")
    print(f"文件统计: {pm.get_file_count()}")


if __name__ == '__main__':
    main()
