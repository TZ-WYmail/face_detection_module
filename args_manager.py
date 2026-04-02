#!/usr/bin/env python3
"""
统一的参数管理系统

功能:
  ✓ 集中管理所有命令行参数
  ✓ 参数验证和范围检查
  ✓ 严格模式支持（--strict-mode 启用所有严格阈值）
  ✓ 配置文件支持（YAML）
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import json
import sys

# 尝试导入YAML支持
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

import config as cfg

logger = logging.getLogger(__name__)


class ArgumentValidator:
    """参数验证器"""
    
    # 参数范围定义
    RANGES = {
        'conf': (0.0, 1.0, '检测置信度应在 0.0-1.0 之间'),
        'threshold': (0.0, 1.0, '去重相似度阈值应在 0.0-1.0 之间'),
        'yaw_threshold': (0.0, 180.0, '偏航角应在 0.0-180.0° 之间'),
        'pitch_threshold': (0.0, 180.0, '俯仰角应在 0.0-180.0° 之间'),
        'roll_threshold': (0.0, 180.0, '翻滚角应在 0.0-180.0° 之间'),
        'sample_interval': (1, 1000, '采样间隔应在 1-1000 之间'),
    }
    
    @staticmethod
    def validate_argument(name: str, value: Any) -> Tuple[bool, str]:
        """
        验证单个参数
        
        Args:
            name: 参数名
            value: 参数值
            
        Returns:
            (是否有效, 错误信息)
        """
        if name not in ArgumentValidator.RANGES:
            return True, ""
        
        min_val, max_val, msg = ArgumentValidator.RANGES[name]
        
        if not (min_val <= value <= max_val):
            return False, f"{msg}，当前值: {value}"
        
        return True, ""
    
    @staticmethod
    def validate_all(args) -> bool:
        """验证所有参数"""
        is_valid = True
        
        for param_name, (min_val, max_val, msg) in ArgumentValidator.RANGES.items():
            if not hasattr(args, param_name):
                continue
            
            value = getattr(args, param_name)
            valid, err_msg = ArgumentValidator.validate_argument(param_name, value)
            
            if not valid:
                logger.error(f"✗ 参数错误: {param_name} = {value}")
                logger.error(f"  {err_msg}")
                is_valid = False
            else:
                logger.debug(f"✓ 参数有效: {param_name} = {value}")
        
        return is_valid


class ArgumentManager:
    """统一的参数管理器"""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        
        parser = argparse.ArgumentParser(
            description='人脸检测和去重系统',
            formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="""
示例:
    # 基础使用
    python face_dedup_pipeline.py videos/video.mp4
  
    # 严格模式（更严格的正脸筛选）
    python face_dedup_pipeline.py videos/video.mp4 --strict-mode
  
    # 自定义参数
    python face_dedup_pipeline.py videos/video.mp4 --threshold 0.6 --conf 0.6
  
    # 从配置文件加载
    python face_dedup_pipeline.py videos/video.mp4 --config config.yaml
                        """
        )
        
        # ============================================================
        # 输入/输出
        # ============================================================
        parser.add_argument(
            'input',
            help='输入视频文件或目录'
        )
        parser.add_argument(
            '-o', '--output-dir',
            default=cfg.DEFAULT_OUTPUT_DIR,
            help=f'输出目录（默认: {cfg.DEFAULT_OUTPUT_DIR}）'
        )
        parser.add_argument(
            '--config',
            type=Path,
            help='从YAML配置文件加载参数（会覆盖其他命令行参数）'
        )
        
        # ============================================================
        # 检测参数
        # ============================================================
        parser.add_argument(
            '--detector',
            choices=['auto', 'insightface', 'yolo', 'haar'],
            default=cfg.DEFAULT_DETECTOR,
            help=f'检测器选择（默认: {cfg.DEFAULT_DETECTOR}）'
        )
        parser.add_argument(
            '--conf',
            type=float,
            default=cfg.DEFAULT_CONFIDENCE_THRESHOLD,
            help=f'检测置信度阈值（默认: {cfg.DEFAULT_CONFIDENCE_THRESHOLD}），范围: 0.0-1.0'
        )
        parser.add_argument(
            '--metric',
            choices=['cosine', 'euclidean'],
            default=cfg.SIMILARITY_METRIC,
            help=f'相似度计算方法（默认: {cfg.SIMILARITY_METRIC}）'
        )
        
        # ============================================================
        # 去重参数
        # ============================================================
        parser.add_argument(
            '--threshold',
            type=float,
            default=cfg.DEDUP_THRESHOLD,
            help=f'去重相似度阈值（默认: {cfg.DEDUP_THRESHOLD}），范围: 0.0-1.0'
        )
        parser.add_argument(
            '--strict-mode',
            action='store_true',
            help='启用严格模式：使用所有严格的阈值（STRICT_* 参数）'
        )
        
        # ============================================================
        # 正脸筛选参数
        # ============================================================
        parser.add_argument(
            '--yaw-threshold',
            type=float,
            default=cfg.YAW_THRESHOLD,
            help=f'左右转动角度阈值（默认: {cfg.YAW_THRESHOLD}°）'
        )
        parser.add_argument(
            '--pitch-threshold',
            type=float,
            default=cfg.PITCH_THRESHOLD,
            help=f'上下转动角度阈值（默认: {cfg.PITCH_THRESHOLD}°）'
        )
        parser.add_argument(
            '--roll-threshold',
            type=float,
            default=cfg.ROLL_THRESHOLD,
            help=f'头部倾斜角度阈值（默认: {cfg.ROLL_THRESHOLD}°）'
        )
        
        # ============================================================
        # 视频处理参数
        # ============================================================
        parser.add_argument(
            '--sample-interval',
            type=int,
            default=cfg.DEFAULT_SAMPLE_INTERVAL,
            help=f'采样间隔（默认: {cfg.DEFAULT_SAMPLE_INTERVAL}），每N帧处理一帧'
        )
        parser.add_argument(
            '--save-strategy',
            choices=['best_end', 'first', 'all'],
            default='best_end',
            help='保存策略：best_end(最佳质量), first(首个正脸), all(全部保存)'
        )
        parser.add_argument(
            '--use-tracks',
            action='store_true',
            default=True,
            help='是否使用轨迹跟踪（默认: 开启）'
        )
        parser.add_argument(
            '--no-tracks',
            dest='use_tracks',
            action='store_false',
            help='禁用轨迹跟踪'
        )
        
        # ============================================================
        # 其他参数
        # ============================================================
        parser.add_argument(
            '--cuda',
            action='store_true',
            help='使用CUDA加速'
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='启用调试模式（输出详细日志）'
        )
        parser.add_argument(
            '--verify-models',
            action='store_true',
            help='启动前验证所需模型'
        )
        
        return parser
    
    @staticmethod
    def load_config_file(config_path: Path) -> Dict[str, Any]:
        """
        从配置文件加载参数
        
        Args:
            config_path: YAML配置文件路径
            
        Returns:
            参数字典
        """
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return {}
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    logger.warning("PyYAML未安装，无法读取 .yaml 文件")
                    logger.info("请运行: pip install pyyaml")
                    return {}
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"✓ 从配置文件加载参数: {config_path}")
                    return config or {}
            
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"✓ 从配置文件加载参数: {config_path}")
                    return config or {}
            
            else:
                logger.warning(f"不支持的配置文件格式: {config_path.suffix}")
                return {}
        
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            return {}
    
    @staticmethod
    def apply_strict_mode(args) -> None:
        """
        应用严格模式（使用所有 STRICT_* 阈值）
        
        Args:
            args: 参数对象
        """
        if args.strict_mode:
            logger.info("⚙️  启用严格模式，应用严格阈值:")
            args.yaw_threshold = cfg.STRICT_YAW_THRESHOLD
            args.pitch_threshold = cfg.STRICT_PITCH_THRESHOLD
            args.roll_threshold = cfg.STRICT_ROLL_THRESHOLD
            args.threshold = cfg.STRICT_DEDUP_THRESHOLD
            
            logger.info(f"  - 偏航角: {cfg.STRICT_YAW_THRESHOLD}°")
            logger.info(f"  - 俯仰角: {cfg.STRICT_PITCH_THRESHOLD}°")
            logger.info(f"  - 翻滚角: {cfg.STRICT_ROLL_THRESHOLD}°")
            logger.info(f"  - 去重阈值: {cfg.STRICT_DEDUP_THRESHOLD}")
    
    @staticmethod
    def parse_args(argv=None) -> argparse.Namespace:
        """
        解析命令行参数，应用配置文件，返回最终的参数对象
        
        Args:
            argv: 命令行参数列表（默认: sys.argv[1:]）
            
        Returns:
            参数对象
        """
        parser = ArgumentManager.create_parser()
        args = parser.parse_args(argv)
        
        # 如果指定了配置文件，加载其参数
        if args.config:
            config = ArgumentManager.load_config_file(args.config)
            
            # 将配置文件参数应用到args（命令行参数优先级更高）
            for key, value in config.items():
                if key == 'input':  # input不能从配置文件覆盖
                    continue
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    logger.debug(f"配置文件中的未知参数: {key}")
        
        # 应用严格模式
        ArgumentManager.apply_strict_mode(args)
        
        # 验证所有参数
        if not ArgumentValidator.validate_all(args):
            logger.error("参数验证失败，请检查参数值范围")
            sys.exit(1)
        
        return args


def main():
    """测试入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    # 测试参数解析
    parser = ArgumentManager.create_parser()
    print(parser.format_help())


if __name__ == '__main__':
    main()
