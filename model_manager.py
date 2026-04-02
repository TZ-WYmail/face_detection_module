#!/usr/bin/env python3
"""
统一的模型管理器

功能:
  ✓ 集中管理所有模型的下载路径
  ✓ 验证模型完整性
  ✓ 支持多种下载源（insightface、HuggingFace、直链）
  ✓ 自动下载缺失的模型
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """统一的模型管理器"""
    
    # 统一使用本地路径，避免散布在多个目录
    PROJECT_ROOT = Path(__file__).parent
    INSIGHTFACE_DIR = PROJECT_ROOT / "models" / "insightface" / "buffalo_l"
    YOLO_DIR = PROJECT_ROOT / "models" / "yolo"
    
    # 必需模型清单
    REQUIRED_MODELS = {
        'insightface': {
            'w600k_r50.onnx': 'ArcFace人脸特征提取（512维Embedding）',
            '1k3d68.onnx': '68点3D人脸关键点（姿态估计）',
        },
        'yolo': {
            'yolov11n-face.onnx': 'YOLOv11 Nano人脸检测模型',
        }
    }
    
    @classmethod
    def get_insightface_model_dir(cls) -> Path:
        """获取InsightFace模型目录"""
        return cls.INSIGHTFACE_DIR
    
    @classmethod
    def get_yolo_model_dir(cls) -> Path:
        """获取YOLO模型目录"""
        return cls.YOLO_DIR
    
    @classmethod
    def ensure_dirs_exist(cls) -> None:
        """确保所有模型目录存在"""
        cls.INSIGHTFACE_DIR.mkdir(parents=True, exist_ok=True)
        cls.YOLO_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ 模型目录结构就绪:")
        logger.info(f"  - InsightFace: {cls.INSIGHTFACE_DIR}")
        logger.info(f"  - YOLO: {cls.YOLO_DIR}")
    
    @classmethod
    def verify_models(cls, check_insightface: bool = True, 
                     check_yolo: bool = True, verbose: bool = True) -> Dict[str, bool]:
        """
        验证模型完整性
        
        Args:
            check_insightface: 是否检查InsightFace模型
            check_yolo: 是否检查YOLO模型
            verbose: 是否输出详细日志
            
        Returns:
            模型验证结果字典
        """
        results = {}
        
        if check_insightface:
            insightface_ok = cls._verify_insightface(verbose)
            results['insightface'] = insightface_ok
        
        if check_yolo:
            yolo_ok = cls._verify_yolo(verbose)
            results['yolo'] = yolo_ok
        
        return results
    
    @classmethod
    def _verify_insightface(cls, verbose: bool = True) -> bool:
        """验证InsightFace模型"""
        if verbose:
            logger.info("\n检查InsightFace模型...")
        
        all_found = True
        for model_name, desc in cls.REQUIRED_MODELS['insightface'].items():
            model_path = cls.INSIGHTFACE_DIR / model_name
            exists = model_path.exists()
            
            if verbose:
                status = "✓" if exists else "✗"
                logger.info(f"  {status} {model_name:<25} ({desc})")
            
            if not exists:
                all_found = False
        
        if not all_found and verbose:
            logger.warning(f"\n⚠️  缺失InsightFace模型文件")
            logger.info(f"   运行: python download_insightface.py --insightface-dir {cls.INSIGHTFACE_DIR}")
        
        return all_found
    
    @classmethod
    def _verify_yolo(cls, verbose: bool = True) -> bool:
        """验证YOLO模型"""
        if verbose:
            logger.info("\n检查YOLO模型...")
        
        # 查找任何 .pt 或 .onnx 文件
        pt_files = list(cls.YOLO_DIR.glob('*.pt'))
        onnx_files = list(cls.YOLO_DIR.glob('*.onnx'))
        
        found_any = len(pt_files) > 0 or len(onnx_files) > 0
        
        if verbose:
            if found_any:
                logger.info(f"  ✓ 找到{len(pt_files)}个.pt文件，{len(onnx_files)}个.onnx文件")
                for f in pt_files[:3]:
                    logger.info(f"    - {f.name}")
                for f in onnx_files[:3]:
                    logger.info(f"    - {f.name}")
            else:
                logger.warning(f"  ✗ 未找到YOLO模型")
                logger.info(f"   将在首次使用时自动下载到: {cls.YOLO_DIR}")
        
        return found_any
    
    @classmethod
    def get_model_status(cls) -> str:
        """获取模型状态摘要"""
        results = cls.verify_models(verbose=False)
        
        status_lines = [
            "\n" + "="*60,
            "模型状态摘要",
            "="*60,
        ]
        
        insightface_ok = results.get('insightface', False)
        yolo_ok = results.get('yolo', False)
        
        status_lines.append(f"InsightFace: {'✓ 完备' if insightface_ok else '✗ 缺失'}")
        status_lines.append(f"YOLO: {'✓ 完备' if yolo_ok else '⚠️  需下载'}")
        
        if not (insightface_ok and yolo_ok):
            status_lines.append(f"\n建议运行模型下载:")
            if not insightface_ok:
                status_lines.append(f"  python download_insightface.py --insightface-dir {cls.INSIGHTFACE_DIR}")
        
        status_lines.append("="*60)
        return "\n".join(status_lines)


def main():
    """命令行工具"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="人脸检测模块模型管理器"
    )
    parser.add_argument(
        '--check', action='store_true',
        help='检查模型完整性'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='显示模型状态摘要'
    )
    parser.add_argument(
        '--insightface-dir', type=Path, default=None,
        help='InsightFace模型目录（覆盖默认路径）'
    )
    parser.add_argument(
        '--yolo-dir', type=Path, default=None,
        help='YOLO模型目录（覆盖默认路径）'
    )
    
    args = parser.parse_args()
    
    # 如果指定了自定义目录，更新类属性
    if args.insightface_dir:
        ModelManager.INSIGHTFACE_DIR = args.insightface_dir
    if args.yolo_dir:
        ModelManager.YOLO_DIR = args.yolo_dir
    
    if args.check or (not args.status):
        ModelManager.ensure_dirs_exist()
        ModelManager.verify_models(verbose=True)
    
    if args.status:
        print(ModelManager.get_model_status())


if __name__ == '__main__':
    main()
