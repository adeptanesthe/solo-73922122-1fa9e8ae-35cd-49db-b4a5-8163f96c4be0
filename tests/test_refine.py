"""复核修正功能自测：线性化、QP 最小性/可行性、复查、产物管理、HTTP 接口。"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app.artifact import ArtifactStore
from app.conv import (
    L2_IN_CH,
    KERNEL,
    run_full_baseline,
    validate_weights,
)
from app.io_utils import REFINE_MAX_SIDE, load_grayscale_png
from app.refine import (
    LABEL_CRACK,
    LABEL_KEEP,
    LABEL_NONCRACK,
    MAX_POINTS,
    N_PARAMS,
    RECHECK_TOL,
    MarkedPoint,
    _build_constraints,
    build_feature_matrix,
    layer1_activations,
    run_refine,
    solve_min_change,
)


# --------------------------------------------------------------------- helpers
def make_weights(rng):
    w1 = rng.normal(0, 0.2, (4, 1, 3, 3))
    b1 = rng.normal(0, 0.1, 4)
    w2 = rng.normal(0, 0.2, (1, 4, 3, 3))
    b2 = rng.normal(0, 0.1, 1)
    return {"layer1": {"weight": w1.tolist(), "bias": b1.tolist()},
            "layer2": {"weight": w2.tolist(), "bias": b2.tolist()}}


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(42)
    img = rng.random((48, 40))
    wts = make_weights(rng)
    w1, b1, w2, b2 = validate_weights(wts)
    out = run_full_baseline(img, w1, b1, w2, b2)[0]
    return rng, img, wts, (w1, b1, w2, b2), out


def extreme_points(out, n):
    flat = out.ravel()
    low = np.argsort(flat)[:n]
    high = np.argsort(flat)[-n:]
    wo = out.shape[1]
    return ([divmod(int(i), wo) for i in low],
            [divmod(int(i), wo) for i in high])


# ----------------------------------------------------------- 1. 线性化正确性
class TestLinearization:
    def test_feature_reproduces_baseline_score(self, case):
        _, img, _, (w1, b1, w2, b2), out = case
        a1 = layer1_activations(img, w1, b1)
        rc = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1])]
        F = build_feature_matrix(a1, rc)
        assert F.shape == (out.size, N_PARAMS)
        assert N_PARAMS == L2_IN_CH * KERNEL * KERNEL + 1 == 37
        params = np.concatenate([w2[0].ravel(), b2])
        assert np.allclose(F @ params, out.ravel(), atol=1e-12, rtol=0)

    def test_feature_matches_tiled_path(self, case):
        # 与分块实现交叉核对若干点的分数
        from app.conv import TiledRunner
        _, img, _, (w1, b1, w2, b2), out = case
        tiled = TiledRunner(img, w1, b1, w2, b2, 7).run()[0]
        assert np.abs(tiled - out).max() <= 1e-8
        a1 = layer1_activations(img, w1, b1)
        for r, c in [(0, 0), (3, 5), (out.shape[0] - 1, out.shape[1] - 1)]:
            f = build_feature_matrix(a1, [(r, c)])[0]
            params = np.concatenate([w2[0].ravel(), b2])
            assert abs(f @ params - out[r, c]) <= 1e-12

    def test_bias_column_is_one(self, case):
        _, img, _, (w1, b1, *_), _ = case
        F = build_feature_matrix(layer1_activations(img, w1, b1), [(0, 0)])
        assert F[0, -1] == 1.0


# ----------------------------------------------------------- 2. 二次规划
class TestQP:
    def _points(self, out, extra_keep=()):
        lows, highs = extreme_points(out, 2)
        pts = ([MarkedPoint(0, *lows[0], LABEL_CRACK),
                MarkedPoint(0, *highs[0], LABEL_NONCRACK)]
               + [MarkedPoint(0, r, c, LABEL_KEEP) for r, c in extra_keep])
        return pts

    def test_min_norm_solution_and_thresholds(self, case):
        _, img, wts, (w1, b1, w2, b2), out = case
        pts = self._points(out, extra_keep=[(5, 5), (6, 7)])
        res = run_refine([img], wts, pts)
        assert res.success, res.reason
        # 阈值条件（重新载入 + 整图推理后的复查结果）
        for p in res.points:
            if p.label == LABEL_CRACK:
                assert p.score_after >= 0.1 - RECHECK_TOL
            elif p.label == LABEL_NONCRACK:
                assert p.score_after <= -0.1 + RECHECK_TOL
            else:
                assert abs(p.score_after - p.score_before) <= RECHECK_TOL
            assert p.ok and p.violation <= RECHECK_TOL
        assert res.max_violation <= RECHECK_TOL

    def test_objective_is_minimum_via_kkt(self, case):
        # 独立 KKT 校验：2d = A'λ，不等式乘子非负
        _, img, wts, (w1, b1, w2, b2), out = case
        pts = self._points(out, extra_keep=[(5, 5), (6, 7)])
        a1 = layer1_activations(img, w1, b1)
        rc = [(p.row, p.col) for p in pts]
        F = build_feature_matrix(a1, rc)
        s0 = np.array([out[r, c] for r, c in rc])
        d, err = solve_min_change(s0, F, [p.label for p in pts])
        assert err is None
        G, h, E, e = _build_constraints(s0, F, [p.label for p in pts])
        active = np.where(h - G @ d <= 1e-8)[0]
        rows = [-G[k] for k in active]
        if E is not None:
            rows += [E[k] for k in range(E.shape[0])]
        A = np.asarray(rows)
        lam, *_ = np.linalg.lstsq(A.T, 2 * d, rcond=None)
        assert np.linalg.norm(A.T @ lam - 2 * d) <= 1e-9
        assert np.all(lam[:len(active)] >= -1e-7)

    def test_no_change_when_constraints_already_met(self, case):
        # 原分数本身已满足阈值时，最优 d 应为 0（平方和最小）。
        # 用混合正负号的 layer2 权重，保证整图分数同时跨越 ±0.1。
        rng = np.random.default_rng(99)
        img = rng.random((32, 32))
        wts = make_weights(rng)
        w2 = np.asarray(wts["layer2"]["weight"])
        w2[0, 0] = -np.abs(w2[0, 0]) - 0.5   # 通道 0 负贡献
        w2[0, 1] = np.abs(w2[0, 1]) + 0.5    # 通道 1 正贡献
        wts["layer2"]["weight"] = w2.tolist()
        wts["layer2"]["bias"] = [0.0]
        w1, b1, w2, b2 = validate_weights(wts)
        out = run_full_baseline(img, w1, b1, w2, b2)[0]
        assert out.max() >= 0.1 and out.min() <= -0.1
        flat = out.ravel()
        crack_rc = divmod(int(np.argmax(flat)), out.shape[1])
        non_rc = divmod(int(np.argmin(flat)), out.shape[1])
        pts = [MarkedPoint(0, *crack_rc, LABEL_CRACK),
               MarkedPoint(0, *non_rc, LABEL_NONCRACK),
               MarkedPoint(0, 2, 2, LABEL_KEEP)]
        res = run_refine([img], wts, pts)
        assert res.success
        assert res.objective <= 1e-20
        # 分数应与原分数逐位一致
        by_key = {(p.row, p.col): p for p in res.points}
        assert by_key[crack_rc].score_after == float(out[crack_rc])
        assert by_key[non_rc].score_after == float(out[non_rc])

    def test_infeasible_conflicting_constraints(self, case):
        # 全零图内部点特征完全相同：同特征既要求 ≥0.1 又要求 ≤-0.1 => 不可行
        rng = np.random.default_rng(7)
        wts = make_weights(rng)
        img = np.zeros((32, 32))
        pts = [MarkedPoint(0, 8, 8, LABEL_CRACK),
               MarkedPoint(0, 9, 9, LABEL_NONCRACK),
               MarkedPoint(0, 5, 5, LABEL_KEEP)]
        res = run_refine([img], wts, pts)
        assert not res.success
        assert res.new_weights is None and res.new_weights_json is None
        assert "不可行" in res.reason

    def test_keep_equality_conflict_detectable(self, case):
        # 保留点与修正点共享同一特征，且原分数不满足修正阈值 => 不可行
        rng = np.random.default_rng(8)
        wts = make_weights(rng)
        img = np.zeros((32, 32))
        w1, b1, w2, b2 = validate_weights(wts)
        out = run_full_baseline(img, w1, b1, w2, b2)[0]
        s = float(out[8, 8])
        if s >= 0.1:
            # 原分数已达标则改要求一个达不到的方向（非裂缝阈值 -0.1）
            if s > -0.1:
                pts = [MarkedPoint(0, 8, 8, LABEL_NONCRACK),
                       MarkedPoint(0, 9, 9, LABEL_KEEP)]
            else:
                pts = [MarkedPoint(0, 8, 8, LABEL_CRACK),
                       MarkedPoint(0, 9, 9, LABEL_KEEP)]
        else:
            pts = [MarkedPoint(0, 8, 8, LABEL_CRACK),
                   MarkedPoint(0, 9, 9, LABEL_KEEP)]
        res = run_refine([img], wts, pts)
        assert not res.success and "不可行" in res.reason


# ----------------------------------------------------------- 3. 输入校验
class TestValidation:
    def test_need_correction_and_keep(self, case):
        _, img, wts, _, out = case
        only_keep = [MarkedPoint(0, 1, 1, LABEL_KEEP)]
        with pytest.raises(ValueError, match="修正点"):
            run_refine([img], wts, only_keep)
        only_fix = [MarkedPoint(0, 1, 1, LABEL_CRACK),
                    MarkedPoint(0, 1, 2, LABEL_NONCRACK)]
        with pytest.raises(ValueError, match="保留点"):
            run_refine([img], wts, only_fix)

    def test_max_32_points(self, case):
        _, img, wts, _, _ = case
        pts = [MarkedPoint(0, i, 0, LABEL_KEEP) for i in range(MAX_POINTS + 1)]
        with pytest.raises(ValueError, match="32"):
            run_refine([img], wts, pts)

    def test_duplicate_and_out_of_grid(self, case):
        _, img, wts, _, out = case
        ho, wo = out.shape
        dup = [MarkedPoint(0, 1, 1, LABEL_CRACK),
               MarkedPoint(0, 1, 1, LABEL_KEEP),
               MarkedPoint(0, 2, 2, LABEL_KEEP)]
        with pytest.raises(ValueError, match="重复"):
            run_refine([img], wts, dup)
        bad = [MarkedPoint(0, ho, 0, LABEL_CRACK),
               MarkedPoint(0, 0, 0, LABEL_KEEP)]
        with pytest.raises(ValueError, match="输出网格"):
            run_refine([img], wts, bad)

    def test_image_side_limit_128(self):
        big = (np.random.default_rng(0).random((129, 64)) * 255).astype(np.uint8)
        with pytest.raises(ValueError, match="128"):
            load_grayscale_png(png_bytes(big), max_side=REFINE_MAX_SIDE)
        ok = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
        assert load_grayscale_png(png_bytes(ok),
                                  max_side=REFINE_MAX_SIDE).shape == (128, 128)

    def test_two_images_and_image_index(self, case):
        rng, img, wts, _, out = case
        img2 = rng.random((24, 24))
        pts = [MarkedPoint(0, 1, 1, LABEL_CRACK),
               MarkedPoint(1, 2, 2, LABEL_KEEP)]
        res = run_refine([img, img2], wts, pts)
        assert res.success
        idxs = {p.image_index for p in res.points}
        assert idxs == {0, 1}
        bad = [MarkedPoint(0, 1, 1, LABEL_CRACK),
               MarkedPoint(2, 0, 0, LABEL_KEEP)]
        with pytest.raises(ValueError, match="图序号"):
            run_refine([img, img2], wts, bad)


# ----------------------------------------------------------- 4. 复查与权重写回
class TestRecheckAndWeights:
    def test_layer1_frozen_and_format_preserved(self, case, tmp_path):
        _, img, wts, (w1, b1, w2, b2), out = case
        lows, highs = extreme_points(out, 1)
        pts = [MarkedPoint(0, *lows[0], LABEL_CRACK),
               MarkedPoint(0, *highs[0], LABEL_NONCRACK),
               MarkedPoint(0, 4, 4, LABEL_KEEP)]
        res = run_refine([img], wts, pts)
        assert res.success
        rw1, rb1, rw2, rb2 = validate_weights(json.loads(res.new_weights_json))
        assert np.array_equal(rw1, w1) and np.array_equal(rb1, b1)
        assert rw2.shape == (1, 4, 3, 3) and rb2.shape == (1,)
        # 顶层结构与原格式一致
        parsed = json.loads(res.new_weights_json)
        assert set(parsed) == {"layer1", "layer2"}
        assert set(parsed["layer1"]) == {"weight", "bias"}
        assert set(parsed["layer2"]) == {"weight", "bias"}

    def test_reload_full_inference_all_marked_points(self, case):
        # 模拟真实链路：写文件 -> 读文件 -> 整图推理 -> 逐点核对
        _, img, wts, _, out = case
        lows, highs = extreme_points(out, 2)
        pts = ([MarkedPoint(0, *rc, LABEL_CRACK) for rc in lows]
               + [MarkedPoint(0, *rc, LABEL_NONCRACK) for rc in highs]
               + [MarkedPoint(0, 3, 3, LABEL_KEEP),
                  MarkedPoint(0, 7, 9, LABEL_KEEP)])
        res = run_refine([img], wts, pts)
        assert res.success and len(res.points) == len(pts)
        for p in res.points:
            assert p.ok

    def test_change_only_layer2_params(self, case):
        _, img, wts, (w1, b1, w2, b2), out = case
        lows, highs = extreme_points(out, 1)
        pts = [MarkedPoint(0, *lows[0], LABEL_CRACK),
               MarkedPoint(0, *highs[0], LABEL_NONCRACK),
               MarkedPoint(0, 4, 4, LABEL_KEEP)]
        res = run_refine([img], wts, pts)
        rw1, rb1, rw2, rb2 = validate_weights(json.loads(res.new_weights_json))
        # layer2 至少有参数发生变化，layer1 完全不变
        assert not (np.allclose(rw2, w2) and np.allclose(rb2, b2))
        assert np.array_equal(rw1, w1) and np.array_equal(rb1, b1)

    def test_flat_key_input_normalized_to_nested_format(self, case):
        # 兼容 layer1_weight 等扁平键，输出仍为标准 layer1/layer2 嵌套结构
        _, img, wts, (w1, b1, w2, b2), out = case
        flat = {
            "layer1_weight": np.asarray(wts["layer1"]["weight"]).tolist(),
            "layer1_bias": wts["layer1"]["bias"],
            "layer2_weight": wts["layer2"]["weight"],
            "layer2_bias": wts["layer2"]["bias"],
        }
        lows, highs = extreme_points(out, 1)
        pts = [MarkedPoint(0, *lows[0], LABEL_CRACK),
               MarkedPoint(0, *highs[0], LABEL_NONCRACK),
               MarkedPoint(0, 4, 4, LABEL_KEEP)]
        res = run_refine([img], flat, pts)
        assert res.success
        parsed = json.loads(res.new_weights_json)
        assert set(parsed) == {"layer1", "layer2"}
        rw1, rb1, _, _ = validate_weights(parsed)
        assert np.array_equal(rw1, w1) and np.array_equal(rb1, b1)


# ----------------------------------------------------------- 5. 产物管理
class TestArtifactStore:
    def test_publish_revoke_atomicity(self, tmp_path):
        store = ArtifactStore(tmp_path / "data")
        assert store.current() is None
        store.publish(b'{"v":1}')
        assert store.current() == b'{"v":1}'
        store.publish(b'{"v":2}')  # 原子替换
        assert store.current() == b'{"v":2}'
        assert store.revoke() is True
        assert store.current() is None
        assert store.revoke() is False  # 再次撤下不报错


# ----------------------------------------------------------- 6. HTTP 接口
class TestHTTP:
    def _client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_index_links_to_refine_page(self):
        client = self._client()
        r = client.get("/")
        assert r.status_code == 200
        assert "/refine" in r.text

    def test_original_infer_endpoint_still_works(self):
        client = self._client()
        rng = np.random.default_rng(41)
        img = (rng.random((64, 64)) * 255).astype(np.uint8)
        r = client.post(
            "/infer",
            files={"image": ("a.png", png_bytes(img), "image/png"),
                   "weights": ("w.json", json.dumps(make_weights(rng)),
                               "application/json")},
            data={"budget": "1000", "as_json": "1"})
        assert r.status_code == 200 and r.json()["metrics"]["passed"]

    def _form(self, rng, images, points=None):
        files = []
        for i, img in enumerate(images):
            files.append(
                (f"image{i + 1}", (f"a{i}.png", png_bytes(
                    (img * 255).astype(np.uint8)), "image/png")))
        files.append(("weights", ("w.json", json.dumps(make_weights(rng)),
                                  "application/json")))
        data = {}
        if points is not None:
            data["points"] = json.dumps(
                [{"image_index": p.image_index, "row": p.row, "col": p.col,
                  "label": p.label} for p in points])
        return files, data

    def test_page_and_prepare(self):
        client = self._client()
        r = client.get("/refine")
        assert r.status_code == 200 and "复核" in r.text
        rng = np.random.default_rng(31)
        img = rng.random((64, 64))
        files, data = self._form(rng, [img])
        r = client.post("/refine/prepare", files=files, data=data)
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["images"]) == 1
        assert body["images"][0]["output_size"] == [32, 32]
        assert len(body["images"][0]["score_grid"]) == 32

    def test_solve_publish_and_download(self, tmp_path, monkeypatch):
        import app.refine_web as rw
        monkeypatch.setattr(rw, "store", ArtifactStore(tmp_path / "art"))
        client = self._client()
        rng = np.random.default_rng(32)
        img = rng.random((64, 64))
        wts = make_weights(rng)
        w1, b1, w2, b2 = validate_weights(wts)
        out = run_full_baseline(img, w1, b1, w2, b2)[0]
        lows, highs = extreme_points(out, 1)
        points = [MarkedPoint(0, *lows[0], LABEL_CRACK),
                  MarkedPoint(0, *highs[0], LABEL_NONCRACK),
                  MarkedPoint(0, 5, 5, LABEL_KEEP)]
        files, data = self._form(rng, [img], points)
        # 让 _form 使用同一套权重
        files[-1] = ("weights", ("w.json", json.dumps(wts),
                                 "application/json"))
        r = client.post("/refine/solve", files=files, data=data)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] and body["download_available"]
        d = client.get("/refine/download")
        assert d.status_code == 200
        dl = json.loads(d.content)
        validate_weights(dl)

    def test_failure_revokes_old_download(self, tmp_path, monkeypatch):
        import app.refine_web as rw
        monkeypatch.setattr(rw, "store", ArtifactStore(tmp_path / "art"))
        client = self._client()
        # 先发布一份
        rw.store.publish(b'{"old": true}')
        rng = np.random.default_rng(33)
        img = np.zeros((32, 32))
        points = [MarkedPoint(0, 8, 8, LABEL_CRACK),
                  MarkedPoint(0, 9, 9, LABEL_NONCRACK),
                  MarkedPoint(0, 5, 5, LABEL_KEEP)]
        files, data = self._form(rng, [img], points)
        r = client.post("/refine/solve", files=files, data=data)
        body = r.json()
        assert r.status_code == 200 and body["success"] is False
        assert "不可行" in body["reason"]
        assert client.get("/refine/download").status_code == 404

    def test_download_404_when_empty(self, tmp_path, monkeypatch):
        import app.refine_web as rw
        monkeypatch.setattr(rw, "store", ArtifactStore(tmp_path / "art"))
        client = self._client()
        assert client.get("/refine/download").status_code == 404

    def test_bad_requests(self, tmp_path, monkeypatch):
        import app.refine_web as rw
        monkeypatch.setattr(rw, "store", ArtifactStore(tmp_path / "art"))
        client = self._client()
        rng = np.random.default_rng(34)

        # 超过 128 边长
        big = (rng.random((129, 64)) * 255).astype(np.uint8)
        r = client.post(
            "/refine/prepare",
            files=[("image1", ("b.png", png_bytes(big), "image/png")),
                   ("weights", ("w.json", json.dumps(make_weights(rng)),
                                "application/json"))])
        assert r.status_code == 400 and "128" in r.json()["error"]

        # 缺修正点/保留点
        img = rng.random((32, 32))
        points = [MarkedPoint(0, 1, 1, LABEL_KEEP)]
        files, data = self._form(rng, [img], points)
        r = client.post("/refine/solve", files=files, data=data)
        assert r.status_code == 400 and "修正点" in r.json()["error"]

        # 标点超网格
        points = [MarkedPoint(0, 99, 99, LABEL_CRACK),
                  MarkedPoint(0, 0, 0, LABEL_KEEP)]
        files, data = self._form(rng, [img], points)
        r = client.post("/refine/solve", files=files, data=data)
        assert r.status_code == 400
