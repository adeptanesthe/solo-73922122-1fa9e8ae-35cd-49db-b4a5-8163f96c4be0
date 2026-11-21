"""巡检复核修正：冻结 layer1，仅微调 layer2 的权重与偏置。

建模
----
网络结构固定（见 app.conv）。冻结 layer1 后，对任一输出网格点 p，
layer1 激活 a1 不再随可调参数变化，layer2 输出分数对 layer2 的 36 个
权重 + 1 个偏置是**线性**的：

    s_p(d) = s0_p + F_p · d

其中 d 为参数改变量（与 w2[0].ravel() C 序对齐，末位为偏置），F_p 为
该点 layer2 各 tap 的 layer1 激活（图外零填充按 0 处理），s0_p 为原分数。

修正要求化为 d 上的线性约束：
  * 保留点（keep）    : F_p·d = 0
  * 裂缝点（crack）   : s0_p + F_p·d ≥ 0.1
  * 非裂缝点（noncrack）: s0_p + F_p·d ≤ -0.1

目标为最小化参数改变量平方和 ‖d‖²，配合线性（不）等式约束，是凸二次规划，
Hessian 为单位阵，解唯一。先用 HiGHS 线性规划做严格可行性判定，再用
SLSQP 求最小范数解；解出后经 JSON 落盘→重新载入→整图基准推理复查。
全部计算 float64。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize

from .conv import (
    L2_IN_CH,
    KERNEL,
    L2_DILATION,
    L2_PAD,
    layer1_out_size,
    run_full_baseline,
    validate_weights,
)

LABEL_CRACK = "crack"
LABEL_NONCRACK = "noncrack"
LABEL_KEEP = "keep"
LABELS = (LABEL_CRACK, LABEL_NONCRACK, LABEL_KEEP)
LABEL_CN = {LABEL_CRACK: "裂缝", LABEL_NONCRACK: "非裂缝", LABEL_KEEP: "保留原分数"}

# layer2 可调参数数：36 个权重 + 1 个偏置
N_PARAMS = L2_IN_CH * KERNEL * KERNEL + 1
CRACK_THRESHOLD = 0.1
NONCRACK_THRESHOLD = -0.1
# 求解器内部要求的约束余量（比复查容差更严，吸收 JSON 往返的舍入）
SOLVER_TOL = 1e-10
# 题目规定的重新载入后整图推理复查容差
RECHECK_TOL = 1e-8
MAX_POINTS = 32


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class MarkedPoint:
    image_index: int
    row: int
    col: int
    label: str


@dataclass
class PointResult:
    image_index: int
    row: int
    col: int
    label: str
    score_before: float
    score_after: float
    ok: bool
    violation: float  # 复查约束的违例量（<= RECHECK_TOL 即通过）


@dataclass
class RefineResult:
    success: bool
    reason: str | None
    points: list[PointResult] = field(default_factory=list)
    objective: float | None = None          # 参数改变量平方和 ‖d‖²
    max_violation: float | None = None
    new_weights: dict | None = None
    new_weights_json: str | None = None
    n_crack: int = 0
    n_noncrack: int = 0
    n_keep: int = 0


# --------------------------------------------------------------------------
# 冻结 layer1 后的线性化
# --------------------------------------------------------------------------
def layer1_activations(image: np.ndarray, w1, b1) -> np.ndarray:
    """重算 layer1 输出（含 ReLU），返回 (4, ho, wo) float64。

    与 run_full_baseline 同一条第一层计算路径，保证特征与整图基准一致。
    """
    x = np.asarray(image, dtype=np.float64)[None, :, :]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    wins = np.lib.stride_tricks.sliding_window_view(xp, (KERNEL, KERNEL),
                                                    axis=(1, 2))
    wins = wins[:, ::2, ::2]                            # (1,ho,wo,3,3)
    a1 = np.einsum("ihwxy,oixy->ohw", wins, w1)
    return np.maximum(a1 + b1[:, None, None], 0.0)


def build_feature_matrix(a1: np.ndarray, points: list[tuple[int, int]]
                         ) -> np.ndarray:
    """对同一图上的输出网格点 (r,c) 构造 layer2 线性特征矩阵 F。

    行序与 w2[0].ravel()（C 序：ic→kh→kw）对齐，末列为偏置项 1.0；
    图外 tap 按零填充处理，与 run_full_baseline 完全一致。
    """
    a1p = np.pad(a1, ((0, 0), (L2_PAD, L2_PAD), (L2_PAD, L2_PAD)),
                 mode="constant")
    m = len(points)
    feat = np.zeros((m, N_PARAMS), dtype=np.float64)
    for i, (r, c) in enumerate(points):
        j = 0
        for ic in range(L2_IN_CH):
            for kh in range(KERNEL):
                for kw in range(KERNEL):
                    rr = r + kh * L2_DILATION
                    cc = c + kw * L2_DILATION
                    feat[i, j] = a1p[ic, rr, cc]
                    j += 1
        feat[i, N_PARAMS - 1] = 1.0
    return feat


def _param_vector(w2, b2) -> np.ndarray:
    v = np.empty(N_PARAMS, dtype=np.float64)
    v[:N_PARAMS - 1] = w2[0].ravel()
    v[N_PARAMS - 1] = b2[0]
    return v


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------
def validate_points(points: list[MarkedPoint],
                    grid_sizes: list[tuple[int, int]]) -> None:
    if not points:
        raise ValueError("至少需要 1 个标点")
    if len(points) > MAX_POINTS:
        raise ValueError(f"标点总数至多 {MAX_POINTS} 个，收到 {len(points)} 个")
    n_crack = sum(p.label == LABEL_CRACK for p in points)
    n_noncrack = sum(p.label == LABEL_NONCRACK for p in points)
    n_keep = sum(p.label == LABEL_KEEP for p in points)
    if n_crack + n_noncrack == 0:
        raise ValueError("至少需要 1 个修正点（标为裂缝或非裂缝）")
    if n_keep == 0:
        raise ValueError("至少需要 1 个保留点（保留原分数）")
    seen: set[tuple[int, int, int]] = set()
    for p in points:
        if p.label not in LABELS:
            raise ValueError(f"未知标注类型：{p.label}")
        if not (0 <= p.image_index < len(grid_sizes)):
            raise ValueError(f"标点引用了不存在的图序号 {p.image_index}")
        ho, wo = grid_sizes[p.image_index]
        if not (0 <= p.row < ho and 0 <= p.col < wo):
            raise ValueError(
                f"图 {p.image_index} 的标点 ({p.row},{p.col}) 超出输出网格 "
                f"{ho}x{wo}")
        key = (p.image_index, p.row, p.col)
        if key in seen:
            raise ValueError(f"重复标点：图 {p.image_index} ({p.row},{p.col})")
        seen.add(key)


# --------------------------------------------------------------------------
# 二次规划
# --------------------------------------------------------------------------
def _build_constraints(s0: np.ndarray, F: np.ndarray, labels: list[str]
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造 G d <= h（不等式）与 E d = e（等式，保留点）。"""
    g_rows: list[np.ndarray] = []
    h_vals: list[float] = []
    e_rows: list[np.ndarray] = []
    e_vals: list[float] = []
    for i, label in enumerate(labels):
        if label == LABEL_KEEP:
            e_rows.append(F[i])
            e_vals.append(0.0)
        elif label == LABEL_CRACK:
            # s0 + F d >= 0.1  =>  -F d <= s0 - 0.1
            g_rows.append(-F[i])
            h_vals.append(float(s0[i] - CRACK_THRESHOLD))
        else:  # noncrack: s0 + F d <= -0.1
            g_rows.append(F[i])
            h_vals.append(float(NONCRACK_THRESHOLD - s0[i]))
    G = np.asarray(g_rows, dtype=np.float64).reshape(-1, N_PARAMS)
    h = np.asarray(h_vals, dtype=np.float64)
    E = (np.asarray(e_rows, dtype=np.float64).reshape(-1, N_PARAMS)
         if e_rows else None)
    e = np.asarray(e_vals, dtype=np.float64) if e_vals else None
    return G, h, E, e


def check_feasible(G, h, E, e) -> tuple[bool, str, np.ndarray | None]:
    """用 HiGHS 线性规划严格判定约束是否可行（纯可行性问题）。

    可行时一并返回一个可行点（作为后续 SLSQP 的兜底初始点）。
    """
    try:
        lp = linprog(
            c=np.zeros(N_PARAMS),
            A_ub=G if G.shape[0] else None,
            b_ub=h if G.shape[0] else None,
            A_eq=E, b_eq=e,
            bounds=[(None, None)] * N_PARAMS,
            method="highs",
        )
    except Exception as exc:  # 数值问题等，交给调用方按失败处理
        return False, f"可行性线性规划异常：{exc}", None
    if lp.status == 0:
        return True, "", np.asarray(lp.x, dtype=np.float64)
    message = getattr(lp, "message", "未知状态") or "未知状态"
    if lp.status == 2:
        return False, "线性规划判定约束不可行（infeasible）", None
    return (False,
            f"可行性线性规划未成功（status={lp.status}：{message}）", None)


def solve_min_change(s0: np.ndarray, F: np.ndarray, labels: list[str]
                     ) -> tuple[np.ndarray | None, str | None]:
    """求满足全部标点约束且 ‖d‖² 最小的参数改变量 d。"""
    G, h, E, e = _build_constraints(s0, F, labels)
    feasible, why, lp_point = check_feasible(G, h, E, e)
    if not feasible:
        return None, why

    cons = []
    for k in range(G.shape[0]):
        cons.append({
            "type": "ineq",
            "fun": (lambda d, g=G[k], hh=h[k]: hh - g @ d),
            "jac": (lambda d, g=G[k]: -g),
        })
    if E is not None:
        for k in range(E.shape[0]):
            cons.append({
                "type": "eq",
                "fun": (lambda d, q=E[k], bb=e[k]: bb - q @ d),
                "jac": (lambda d, q=E[k]: -q),
            })

    def obj(d):
        return 0.5 * float(d @ d)

    # 原点优先（可行时它就是最优解）；HiGHS 可行点作为兜底初始点
    starts = [np.zeros(N_PARAMS)]
    if lp_point is not None and np.linalg.norm(lp_point) > 1e-12:
        starts.append(lp_point)
    best: np.ndarray | None = None
    last_msg = ""
    for d0 in starts:
        res = minimize(obj, d0, jac=lambda d: d, method="SLSQP",
                       constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 1000})
        d = res.x
        viol = _constraint_violation(d, G, h, E, e)
        last_msg = getattr(res, "message", "")
        if viol <= SOLVER_TOL:
            if best is None or d @ d < best @ best:
                best = d
    if best is None:
        return None, f"约束可行但二次规划求解未达到精度要求（{last_msg}）"
    return best, None


def _constraint_violation(d: np.ndarray, G, h, E, e) -> float:
    v = 0.0
    if G.shape[0]:
        v = max(v, float(np.max(G @ d - h)))
    if E is not None:
        v = max(v, float(np.max(np.abs(E @ d - e))))
    return max(v, 0.0)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run_refine(images: list[np.ndarray], weights: dict,
               points: list[MarkedPoint]) -> RefineResult:
    """完整修正流程：线性化 -> QP -> 写 JSON -> 重新载入 -> 整图复查。"""
    if not 1 <= len(images) <= 2:
        raise ValueError("需要 1～2 张图像")
    w1, b1, w2, b2 = validate_weights(weights)

    # 每张图：layer1 激活、整图基准输出、输出网格尺寸
    per_image: list[dict] = []
    grid_sizes: list[tuple[int, int]] = []
    for idx, img in enumerate(images):
        ho, wo = layer1_out_size(*img.shape)
        grid_sizes.append((ho, wo))
        a1 = layer1_activations(img, w1, b1)
        out = run_full_baseline(img, w1, b1, w2, b2)[0]
        per_image.append({"a1": a1, "out": out})
    validate_points(points, grid_sizes)

    labels = [p.label for p in points]
    # 按图分组构造特征，再按标点顺序拼回
    F = np.zeros((len(points), N_PARAMS), dtype=np.float64)
    s0 = np.zeros(len(points), dtype=np.float64)
    for idx in range(len(images)):
        idxs = [i for i, p in enumerate(points) if p.image_index == idx]
        if not idxs:
            continue
        rc = [(points[i].row, points[i].col) for i in idxs]
        sub = build_feature_matrix(per_image[idx]["a1"], rc)
        F[idxs] = sub
        for k, i in enumerate(idxs):
            s0[i] = per_image[idx]["out"][points[i].row, points[i].col]

    # 线性化自检：F·原参数（含偏置列）必须复现整图基准分数
    lin_check = np.abs(F @ _param_vector(w2, b2) - s0)
    if float(lin_check.max()) > 1e-12:
        raise RuntimeError(
            f"layer2 线性化与整图基准不一致，最大偏差 {lin_check.max():.3e}")

    d, err = solve_min_change(s0, F, labels)
    if d is None:
        return RefineResult(
            success=False,
            reason=_infeasible_reason(err, s0, labels),
            n_crack=labels.count(LABEL_CRACK),
            n_noncrack=labels.count(LABEL_NONCRACK),
            n_keep=labels.count(LABEL_KEEP),
        )

    # 组装新权重：layer1 原样保留，layer2 写入新参数
    new_params = _param_vector(w2, b2) + d
    new_w2 = new_params[:N_PARAMS - 1].reshape(1, L2_IN_CH, KERNEL, KERNEL)
    new_b2 = new_params[N_PARAMS - 1:N_PARAMS]
    new_weights = {
        "layer1": weights["layer1"] if isinstance(weights.get("layer1"), dict)
        else {"weight": np.asarray(weights["layer1_weight"]).tolist(),
              "bias": np.asarray(weights["layer1_bias"]).tolist()},
        "layer2": {"weight": new_w2.tolist(), "bias": new_b2.tolist()},
    }
    payload = json.dumps(
        new_weights, indent=2, ensure_ascii=False).encode("utf-8")

    # ---- 落盘前复查：模拟“写 JSON -> 重新载入 -> 整图推理” ----
    reloaded = json.loads(payload.decode("utf-8"))
    rw1, rb1, rw2, rb2 = validate_weights(reloaded)
    if not (np.array_equal(rw1, w1) and np.array_equal(rb1, b1)):
        return RefineResult(success=False,
                            reason="复查失败：第一层参数在 JSON 往返后发生变化")

    results: list[PointResult] = []
    max_violation = 0.0
    for idx, img in enumerate(images):
        out_new = run_full_baseline(img, rw1, rb1, rw2, rb2)[0]
        for p in (p for p in points if p.image_index == idx):
            before = float(per_image[idx]["out"][p.row, p.col])
            after = float(out_new[p.row, p.col])
            if p.label == LABEL_CRACK:
                viol = max(0.0, CRACK_THRESHOLD - after)
                ok = after >= CRACK_THRESHOLD - RECHECK_TOL
            elif p.label == LABEL_NONCRACK:
                viol = max(0.0, after - NONCRACK_THRESHOLD)
                ok = after <= NONCRACK_THRESHOLD + RECHECK_TOL
            else:
                viol = abs(after - before)
                ok = viol <= RECHECK_TOL
            max_violation = max(max_violation, viol)
            results.append(PointResult(
                image_index=idx, row=p.row, col=p.col, label=p.label,
                score_before=before, score_after=after, ok=ok,
                violation=viol))

    if max_violation > RECHECK_TOL or any(not r.ok for r in results):
        return RefineResult(
            success=False,
            reason=(f"重新载入后整图推理复查未通过：最大约束违例 "
                    f"{max_violation:.3e} 超过容差 {RECHECK_TOL:.0e}"),
            points=results, objective=float(d @ d),
            max_violation=max_violation,
            n_crack=labels.count(LABEL_CRACK),
            n_noncrack=labels.count(LABEL_NONCRACK),
            n_keep=labels.count(LABEL_KEEP),
        )

    return RefineResult(
        success=True, reason=None, points=results,
        objective=float(d @ d), max_violation=max_violation,
        new_weights=reloaded, new_weights_json=payload.decode("utf-8"),
        n_crack=labels.count(LABEL_CRACK),
        n_noncrack=labels.count(LABEL_NONCRACK),
        n_keep=labels.count(LABEL_KEEP),
    )


def _infeasible_reason(solver_reason: str | None, s0: np.ndarray,
                       labels: list[str]) -> str:
    """给出不可行/失败原因，并列出距目标差距最大的约束供巡检员定位。"""
    parts = [solver_reason or "求解失败"]
    gaps = []
    for i, label in enumerate(labels):
        if label == LABEL_CRACK:
            gap = CRACK_THRESHOLD - float(s0[i])
            gaps.append((gap, f"图点#{i} 裂缝要求≥0.1，原分数 {s0[i]:.4f}，"
                              f"还差 {gap:.4f}"))
        elif label == LABEL_NONCRACK:
            gap = float(s0[i]) - NONCRACK_THRESHOLD
            gaps.append((gap, f"图点#{i} 非裂缝要求≤-0.1，原分数 {s0[i]:.4f}，"
                              f"还差 {gap:.4f}"))
    gaps.sort(reverse=True)
    if gaps:
        parts.append("最吃力的修正要求：" + "；".join(t for _, t in gaps[:3]))
    parts.append("可调整标点标注、放宽彼此冲突的要求后重试；"
                 "保留点要求分数严格不变，可能与修正要求冲突。")
    return "。".join(parts)
