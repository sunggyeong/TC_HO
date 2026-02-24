from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


class DecisionPolicy:
    def decide_sequence(self, tensors: Dict[str, Any]) -> np.ndarray:
        """
        Return planned serving node index for each time slot.
        -1 means no selection.
        """
        raise NotImplementedError


@dataclass
class StickyReactiveGreedyPolicy(DecisionPolicy):
    """
    Sustainable reactive proxy:
    - current slot available node among A_final only
    - greedy by Q_proxy with hysteresis (switch margin)
    """
    score_key: str = "Q_proxy"
    switch_margin: float = 3.0

    def decide_sequence(self, tensors: Dict[str, Any]) -> np.ndarray:
        A = np.asarray(tensors["A_final"], dtype=int)
        S = np.asarray(tensors[self.score_key], dtype=float)
        T, N = A.shape

        decisions = np.full(T, -1, dtype=int)
        current = -1

        for t in range(T):
            valid = np.where(A[t] == 1)[0]
            if len(valid) == 0:
                decisions[t] = -1
                current = -1
                continue

            # best candidate by current score
            scores = S[t, valid].copy()
            scores[~np.isfinite(scores)] = -1e9
            best_idx = int(valid[int(np.argmax(scores))])

            if 0 <= current < N and A[t, current] == 1 and np.isfinite(S[t, current]):
                if S[t, current] + self.switch_margin >= S[t, best_idx]:
                    decisions[t] = current
                else:
                    decisions[t] = best_idx
            else:
                decisions[t] = best_idx

            current = decisions[t]

        return decisions


@dataclass
class PredictedPolicyConfig:
    horizon_steps: int = 20
    discount: float = 0.93
    switch_margin: float = 1.5
    switch_cost: float = 2.0
    prediction_noise_std: float = 1.0
    availability_false_negative: float = 0.02
    availability_false_positive: float = 0.01
    seed: int = 0


class PredictedLookaheadPolicy(DecisionPolicy):
    """
    Predicted+Diffusion / Predicted+Consistency shared decision proxy.
    실제 차이는 runtime_eval의 inference latency trace에서 반영.
    """
    def __init__(self, cfg: PredictedPolicyConfig, score_key: str = "Q_proxy"):
        self.cfg = cfg
        self.score_key = score_key

    def _build_predicted(self, A_final: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.cfg.seed)

        A_hat = A_final.copy().astype(int)
        score_hat = score.copy().astype(float)

        finite = np.isfinite(score_hat)
        score_hat[finite] += rng.normal(0.0, self.cfg.prediction_noise_std, size=score_hat.shape)[finite]

        fn = (A_hat == 1) & (rng.random(A_hat.shape) < self.cfg.availability_false_negative)
        fp = (A_hat == 0) & (rng.random(A_hat.shape) < self.cfg.availability_false_positive)
        A_hat[fn] = 0
        A_hat[fp] = 1

        return A_hat, score_hat

    def decide_sequence(self, tensors: Dict[str, Any]) -> np.ndarray:
        A = np.asarray(tensors["A_final"], dtype=int)
        S = np.asarray(tensors[self.score_key], dtype=float)
        T, N = A.shape

        H = max(1, int(self.cfg.horizon_steps))
        gamma = float(self.cfg.discount)

        A_hat, S_hat = self._build_predicted(A, S)

        decisions = np.full(T, -1, dtype=int)
        current = -1

        for t in range(T):
            valid_now = np.where(A[t] == 1)[0]
            if len(valid_now) == 0:
                decisions[t] = -1
                current = -1
                continue

            current_valid = (0 <= current < N and A[t, current] == 1)

            best_node = -1
            best_value = -1e18

            for n in valid_now:
                total = 0.0
                for h in range(H):
                    tau = t + h
                    if tau >= T:
                        break
                    if A_hat[tau, n] == 1 and np.isfinite(S_hat[tau, n]):
                        total += (gamma ** h) * float(S_hat[tau, n])
                    else:
                        total -= (gamma ** h) * 5.0

                if current_valid and n != current:
                    total -= self.cfg.switch_cost
                if current_valid and n == current:
                    total += self.cfg.switch_margin

                if total > best_value:
                    best_value = total
                    best_node = int(n)

            decisions[t] = best_node
            current = best_node

        return decisions