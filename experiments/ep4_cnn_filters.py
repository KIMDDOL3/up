#!/usr/bin/env python3
"""딥러닝 기초 4편 CNN 실측: ①합성곱 필터(엣지 검출) ②파라미터 수 비교 ③풀링 축소.
데이터: sklearn digits(8×8 손글씨 숫자 실데이터). 근거: LeCun et al. 1998(LeNet), Goodfellow ch.9."""
import numpy as np
np.set_printoptions(linewidth=200)

from sklearn.datasets import load_digits
digits = load_digits()
img = digits.images[0]  # 8×8, 0~16 밝기
print("=== 실험 재료: 손글씨 '0' (8×8, 실데이터) ===")
print(img.astype(int))

def conv2d(x, k):
    """valid 합성곱 (실은 cross-correlation, 딥러닝 관례)."""
    kh, kw = k.shape
    h, w = x.shape
    out = np.zeros((h-kh+1, w-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i+kh, j:j+kw] * k)
    return out

kernels = {
    "수직 엣지": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),
    "수평 엣지": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),
}
print("\n=== 실험 1: 3×3 필터가 특징을 찾는다 (합성곱 출력, |값| 상위 강조) ===")
for name, k in kernels.items():
    out = conv2d(img, k)
    strong = int((np.abs(out) > np.abs(out).max() * 0.5).sum())
    print(f"\n[{name}] 출력 {out.shape}, 강한 반응 셀 {strong}개")
    print(np.round(out).astype(int))

print("\n=== 실험 2: 파라미터 수 — 펼치기(FC) vs 합성곱(CNN) ===")
# 28×28 이미지 기준 표준 비교
fc = 28*28*100 + 100          # 784→100 완전연결
conv = 3*3*1*16 + 16          # 3×3 필터 16개
print(f"완전연결(784→100): {fc:,}개 파라미터")
print(f"합성곱(3×3 필터 16개): {conv:,}개 파라미터")
print(f"절감: {fc/conv:.0f}배")

print("\n=== 실험 3: 맥스풀링 — 크기 1/4, 최댓값만 남긴다 ===")
out = conv2d(img, kernels["수직 엣지"])
h2, w2 = out.shape[0]//2*2, out.shape[1]//2*2
pooled = out[:h2, :w2].reshape(h2//2, 2, w2//2, 2).max(axis=(1, 3))
print(f"합성곱 출력 {out.shape} → 풀링 후 {pooled.shape} (셀 수 {out.size}→{pooled.size})")
print(np.round(pooled).astype(int))
