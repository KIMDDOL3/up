"""DL 기초 3편(역전파) 재현 코드 — tistory 글의 모든 수치는 이 스크립트에서 나왔다.
실행: python3 ep3_backprop_xor.py (numpy만 필요)
"""
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
Y = np.array([[0],[1],[1],[0]], float)
def sigmoid(z): return 1/(1+np.exp(-z))

def run(seed, lr, epochs, hidden, act):
    """성공(정확도 100% + 여유 0.45) 도달 에포크 반환. 실패 시 None."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2,hidden))*0.8; b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden,1))*0.8; b2 = np.zeros(1)
    def fwd(X):
        h = np.maximum(0, X@W1+b1) if act=="relu" else sigmoid(X@W1+b1)
        return h, sigmoid(h@W2+b2)
    for ep in range(1, epochs+1):
        h, out = fwd(X)
        d_out = (out - Y)/4
        dW2 = h.T@d_out; db2 = d_out.sum(0)
        d_h = d_out@W2.T * ((h>0) if act=="relu" else h*(1-h))
        dW1 = X.T@d_h; db1 = d_h.sum(0)
        W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1; b1 -= lr*db1
        _, out = fwd(X)
        if np.mean((out>0.5).astype(int)==Y)==1.0 and np.abs(out-0.5).min()>0.45:
            return ep
    return None

if __name__ == "__main__":
    for label, kw in [("sigmoid 은닉2 lr=3", dict(lr=3.0, epochs=5000, hidden=2, act="sigmoid")),
                      ("ReLU 은닉4 lr=0.5", dict(lr=0.5, epochs=3000, hidden=4, act="relu"))]:
        results = [run(s, **kw) for s in range(10)]
        succ = [r for r in results if r]
        print(f"{label}: 성공 {len(succ)}/10", (f", 평균 {np.mean(succ):.0f}에포크" if succ else ""))
    # 기울기 소실
    for depth in (3, 6, 10):
        g = 0.25 ** depth
        print(f"sigmoid {depth}층 기울기 신호 = 초기의 {g:.2e}")
