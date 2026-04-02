#!/usr/bin/env python3
"""
检查保存的人脸图片质量检测是否真实有效
"""
import cv2
import numpy as np
import os
from pathlib import Path


def check_image_quality(img_path: str, verbose: bool = True) -> dict:
    """
    检查图像的实际质量指标
    
    Returns:
        dict: 包含各项质量指标
    """
    if not os.path.exists(img_path):
        return {'error': f'文件不存在: {img_path}'}
    
    img = cv2.imread(img_path)
    if img is None:
        return {'error': f'无法读取图像: {img_path}'}
    
    h, w = img.shape[:2]
    
    # 1. 亮度检查
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    brightness_status = "✅ 正常" if 50 <= brightness <= 210 else "❌ 过暗/过亮"
    
    # 2. 对比度检查
    contrast = np.std(gray)
    contrast_status = "✅ 足够" if contrast >= 20 else "❌ 对比度低"
    
    # 3. 清晰度检查（Laplacian方差）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    sharpness_status = "✅ 清晰" if sharpness >= 100 else "❌ 模糊"
    
    # 4. 直方图均衡度检查
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_entropy = -np.sum(hist / hist.sum() * np.log2(hist / hist.sum() + 1e-10))
    entropy_status = "✅ 信息量足" if hist_entropy >= 4.0 else "❌ 信息量不足"
    
    # 5. 边缘检测（Canny）
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w) if h * w > 0 else 0
    edge_status = "✅ 边缘足够" if edge_density >= 0.01 else "❌ 边缘太少"
    
    # 6. 肤色检查
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 肤色范围：H(0-20或160-180), S(10-40), V(60-255)
    lower_skin1 = np.array([0, 10, 60])
    upper_skin1 = np.array([20, 40, 255])
    lower_skin2 = np.array([160, 10, 60])
    upper_skin2 = np.array([180, 40, 255])
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_ratio = np.count_nonzero(skin_mask) / (h * w) if h * w > 0 else 0
    skin_status = "✅ 肤色比例足" if skin_ratio >= 0.3 else "❌ 肤色比例低"
    
    # 综合评分
    quality_scores = {
        '亮度(50-210)': brightness,
        '对比度(>=20)': contrast,
        '清晰度方差(>=100)': sharpness,
        '直方图熵(>=4.0)': hist_entropy,
        '边缘密度(>=0.01)': edge_density,
        '肤色比例(>=0.3)': skin_ratio,
    }
    
    result = {
        'file': img_path,
        'size': f'{w}x{h}',
        '亮度': {'值': f'{brightness:.1f}', '状态': brightness_status},
        '对比度': {'值': f'{contrast:.1f}', '状态': contrast_status},
        '清晰度': {'值': f'{sharpness:.1f}', '状态': sharpness_status},
        '直方图熵': {'值': f'{hist_entropy:.2f}', '状态': entropy_status},
        '边缘密度': {'值': f'{edge_density:.4f}', '状态': edge_status},
        '肤色比例': {'值': f'{skin_ratio:.2f}', '状态': skin_status},
        'quality_scores': quality_scores,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📸 图片质量检查: {Path(img_path).name}")
        print(f"{'='*70}")
        print(f"尺寸: {result['size']}")
        print(f"\n亮度 (推荐范围 50-210): {result['亮度']['值']} {result['亮度']['状态']}")
        print(f"对比度 (推荐 >= 20): {result['对比度']['值']} {result['对比度']['状态']}")
        print(f"清晰度/方差 (推荐 >= 100): {result['清晰度']['值']} {result['清晰度']['状态']}")
        print(f"直方图熵 (推荐 >= 4.0): {result['直方图熵']['值']} {result['直方图熵']['状态']}")
        print(f"边缘密度 (推荐 >= 0.01): {result['边缘密度']['值']} {result['边缘密度']['状态']}")
        print(f"肤色比例 (推荐 >= 0.3): {result['肤色比例']['值']} {result['肤色比例']['状态']}")
        
        # 综合评价
        failed_checks = [k for k, v in result.items() if isinstance(v, dict) and '❌' in v.get('状态', '')]
        if failed_checks:
            print(f"\n⚠️  质量风险 ({len(failed_checks)}项):")
            for check in failed_checks:
                print(f"   - {check}")
        else:
            print(f"\n✅ 所有检查都通过")
        print(f"{'='*70}\n")
    
    return result


def check_output_directory(output_dir: str):
    """检查输出目录中所有保存的人脸"""
    if not os.path.isdir(output_dir):
        print(f"❌ 目录不存在: {output_dir}")
        return
    
    jpg_files = sorted(Path(output_dir).glob("*.jpg"))
    
    if not jpg_files:
        print(f"❌ 没有找到 JPG 文件")
        return
    
    print(f"\n📁 检查目录: {output_dir}")
    print(f"   找到 {len(jpg_files)} 张图片\n")
    
    results = []
    failed_count = 0
    
    for img_path in jpg_files:
        result = check_image_quality(str(img_path), verbose=False)
        results.append(result)
        
        # 统计失败检查项
        if 'quality_scores' in result:
            failed = 0
            if result['quality_scores']['亮度(50-210)'] < 50 or result['quality_scores']['亮度(50-210)'] > 210:
                failed += 1
            if result['quality_scores']['对比度(>=20)'] < 20:
                failed += 1
            if result['quality_scores']['清晰度方差(>=100)'] < 100:
                failed += 1
            if result['quality_scores']['直方图熵(>=4.0)'] < 4.0:
                failed += 1
            if result['quality_scores']['边缘密度(>=0.01)'] < 0.01:
                failed += 1
            if result['quality_scores']['肤色比例(>=0.3)'] < 0.3:
                failed += 1
            
            if failed > 0:
                failed_count += 1
            
            status = "⚠️" if failed > 0 else "✅"
            print(f"{status} {Path(img_path).name:40s} | 亮度:{result['quality_scores']['亮度(50-210)']:6.1f} | 对比:{result['quality_scores']['对比度(>=20)']:5.1f} | 清晰:{result['quality_scores']['清晰度方差(>=100)']:7.1f} | 边缘:{result['quality_scores']['边缘密度(>=0.01)']:7.4f}")
    
    print(f"\n{'='*70}")
    print(f"📊 统计汇总:")
    print(f"   总数: {len(jpg_files)}")
    print(f"   ✅ 质量合格: {len(jpg_files) - failed_count}")
    print(f"   ⚠️  质量风险: {failed_count}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    
    # 检查特定图片或整个目录
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            check_image_quality(path, verbose=True)
        elif os.path.isdir(path):
            check_output_directory(path)
    else:
        # 默认检查输出目录
        output_dir = './output/video-2'
        if os.path.isdir(output_dir):
            check_output_directory(output_dir)
            # 也检查最后一张图片
            jpg_files = sorted(Path(output_dir).glob("*.jpg"))
            if jpg_files:
                print(f"\n详细检查最后一张图片:\n")
                check_image_quality(str(jpg_files[-1]), verbose=True)
        else:
            print(f"❌ 默认输出目录不存在: {output_dir}")
            print(f"   用法: python check_face_quality.py <图片路径或目录>")
