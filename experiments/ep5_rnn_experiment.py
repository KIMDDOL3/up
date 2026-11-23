#!/usr/bin/env python3
"""딥러닝 기초 5편 RNN 실측: ①은닉 상태가 앞 역사의 요약임을 관측(같은 입력, 다른 문맥)
②기억 감쇠 곡선(초기 입력 영향력이 시간에 따라 사라진다) ③BPTT 기울기 소실/폭발(W_h 크기별).
근거: Elman 1990, Rumelhart et al. 1986(BPTT), Pascanu et al. 2013, Goodfellow ch.10."""
import numpy as np

rng = np.random.default_rng(42)
np.set_printoptions(linewidth=200, suppress=True)

# ── 실험 공통: 단일층 RNN 순전파 ──
# h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
def rnn_forward(seq, Wx, Wh, b, h0=None):
    h = np.zeros(Wx.shape[0]) if h0 is None else h0.copy()
    hs = []
    for x in seq:
        h = np.tanh(Wx @ x + Wh @ h + b)
        hs.append(h.copy())
    return np.array(hs)

D, H = 4, 8
Wx = rng.normal(0, 0.4, (H, D))
b = np.zeros(H)

print("=== 실험 1: 같은 입력 x_t, 다른 과거 → 다른 은닉 상태 ===")
# 공통 접미사 [A, B] 앞에 서로 다른 접두사 문맥 P1, P2를 붙인 두 시퀀스
A = rng.normal(0, 1, D); B = rng.normal(0, 1, D)
P1 = rng.normal(0, 1, (6, D)); P2 = rng.normal(0, 1, (6, D))
Wh_m = rng.normal(0, 0.5, (H, H))  # 중간 크기 W_h (반복 감쇠 완만)
h_P1 = rnn_forward(np.vstack([P1, A, B]), Wx, Wh_m, b)
h_P2 = rnn_forward(np.vstack([P2, A, B]), Wx, Wh_m, b)
for label, idx in [("A 입력 직후(h_6)", 6), ("B 입력 직후(h_7)", 7)]:
    v1, v2 = h_P1[idx], h_P2[idx]
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"  {label}: 코사인 유사도 = {cos:.3f} (1.0=문맥을 잊음, 0=문맥을 구별)")

print("\n=== 실험 2: 기억 감쇠 — 초기 입력이 t스텝 뒤 상태에 남기는 감도 ===")
# 감도 S(t) = ||∂h_t/∂h_0|| 를 200회 난수 평균 — 1.0에서 시작해 (ρ·tanh')^t 비율로 감쇠
for radius, tag in [(0.5, "작은 W_h(ρ≈0.5)"), (0.9, "큰 W_h(ρ≈0.9)")]:
    Wh = rng.normal(0, radius / np.sqrt(H), (H, H))
    trials = {5: [], 10: [], 20: [], 40: []}
    for _ in range(200):
        h = np.tanh(Wx @ rng.normal(0, 1, D) + b)
        J = np.eye(H)
        for t in range(1, 41):
            J = np.diag(1 - h ** 2) @ Wh @ J   # ∂h_t/∂h_{t-1} 곱
            h = np.tanh(Wx @ rng.normal(0, 1, D) + Wh @ h + b)
            if t in trials:
                trials[t].append(np.linalg.norm(J))
    print(f"  {tag}: 감도 5스텝={np.mean(trials[5]):.3f} 10스텝={np.mean(trials[10]):.3f} "
          f"20스텝={np.mean(trials[20]):.4f} 40스텝={np.mean(trials[40]):.5f}")

print("\n=== 실험 3: BPTT 기울기 소실/폭발 — W_h 크기별 100스텝 역전파 ===")
T = 100
for radius in [0.5, 1.0, 1.5]:
    Wh = rng.normal(0, radius / np.sqrt(H), (H, H))
    dh = rng.normal(0, 1, H)   # 마지막 시점에서 흘러들어오는 그래디언트
    g = dh.copy()
    norms = []
    for t in range(T):
        h_prev = np.tanh(Wx @ rng.normal(0, 1, D) + Wh @ g + b)  # 대략적 중간 상태 크기용
        g = Wh.T @ (g * (1 - np.tanh(rng.normal(0, 1, H)) ** 2))  # dL/dh_{t-1}
        norms.append(np.linalg.norm(g))
    print(f"  ρ={radius}: 스텝 10={norms[9]:.2e} 30={norms[29]:.2e} 60={norms[59]:.2e} 100={norms[99]:.2e}")

print("\n=== 실험 4: 한계의 숫자 — RNN vs 펼치기 파라미터 비교 ===")
for Tlen, D_, H_ in [(50, 8, 16), (200, 16, 32)]:
    rnn_p = D_ * H_ + H_ * H_ + H_          # Wx + Wh + b
    fc_p = Tlen * D_ * H_ + H_              # 시퀀스를 펼쳐 완전연결(길이 고정)
    print(f"  길이 {Tlen}: RNN = {rnn_p}개 (길이와 무관) / 펼치기 = {fc_p}개 ({fc_p/rnn_p:.0f}배)")
