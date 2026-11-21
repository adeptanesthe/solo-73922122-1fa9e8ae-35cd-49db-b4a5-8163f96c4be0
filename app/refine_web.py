"""巡检复核修正网页与接口（APIRouter，挂载于 /refine）。

流程：
  GET  /refine            页面（上传 1～2 张 ≤128 灰度 PNG + 权重 JSON）
  POST /refine/prepare    返回每张图的输出网格尺寸与原分数网格，供页面填坐标
  POST /refine/solve      提交标点（裂缝/非裂缝/保留），求解并复查：
                          通过 -> 原子发布新权重并开放下载；
                          未通过/不可行 -> 说明原因，保留原上传权重，撤下旧下载
  GET  /refine/download   下载当前通过复查的权重；无产物时 404
"""

from __future__ import annotations

import json
import os
from tempfile import gettempdir

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
)

from .artifact import ArtifactStore
from .conv import layer1_out_size, run_full_baseline, validate_weights
from .io_utils import REFINE_MAX_SIDE, load_grayscale_png, load_weights_json
from .refine import (
    LABELS,
    MAX_POINTS,
    RECHECK_TOL,
    MarkedPoint,
    run_refine,
)

router = APIRouter(prefix="/refine")

DATA_DIR = os.environ.get("REFINE_DATA_DIR", os.path.join(gettempdir(),
                                                          "crack_refine_data"))
store = ArtifactStore(DATA_DIR)

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>裂缝判定复核修正</title>
<style>
 body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
   margin:2rem auto;max-width:1080px;padding:0 1rem;color:#1f2933}
 h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:1.6rem}
 form{border:1px solid #cbd2d9;border-radius:8px;padding:1rem 1.2rem;
   background:#f8fafc;margin-bottom:1rem}
 label{display:block;margin:.7rem 0 .25rem;font-weight:600}
 button{margin-top:.8rem;padding:.5rem 1.4rem;background:#1d4ed8;color:#fff;
   border:0;border-radius:6px;cursor:pointer}
 button.secondary{background:#475569}
 table{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}
 th,td{border:1px solid #cbd2d9;padding:.3rem .6rem;text-align:right}
 th{background:#eef2f7} .ok{color:#15803d;font-weight:700}
 .bad{color:#b91c1c;font-weight:700} .note{color:#52606d;font-size:.85rem}
 .err{background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;
   padding:.7rem 1rem;border-radius:6px;white-space:pre-wrap;margin:.6rem 0}
 .dl{background:#ecfdf5;border:1px solid #6ee7b7;color:#065f46;
   padding:.7rem 1rem;border-radius:6px;margin:.6rem 0}
 input[type=number]{width:5.5rem}
 .ptrow input,.ptrow select{margin:0 .3rem}
 code{background:#eef2f7;padding:.1rem .35rem;border-radius:4px}
 .tag-crack{color:#b91c1c;font-weight:600}
 .tag-noncrack{color:#1d4ed8;font-weight:600}
 .tag-keep{color:#52606d;font-weight:600}
</style></head><body>
<h1>裂缝误报/漏报复核 · 第二层卷积定点修正（NumPy float64）</h1>
<p class="note">上传 <b>1～2 张每边 ≤128 像素</b>的灰度 PNG（mode=L）和现有两层
权重 JSON。冻结首层（layer1 权重/偏置不变），只微调 layer2 的 36 个权重与 1 个
偏置。在每张图的模型输出网格上填坐标并标注：
<span class="tag-crack">裂缝（分数 ≥ 0.1）</span>、
<span class="tag-noncrack">非裂缝（分数 ≤ −0.1）</span>、
<span class="tag-keep">保留原分数（严格不变）</span>。
合计最多 32 点，修正点与保留点各至少 1 个。取满足全部约束且参数改变量平方和
最小的解；新权重写回原格式 JSON，重新载入后用整图推理复查全部标点，
约束误差 ≤ 1e-8 才开放下载。</p>

<form id="uploadForm">
 <label>灰度 PNG 图像 1（必填，每边 ≤128）</label>
 <input type="file" name="image1" accept="image/png" required>
 <label>灰度 PNG 图像 2（可选，每边 ≤128）</label>
 <input type="file" name="image2" accept="image/png">
 <label>现有两层权重/偏置 JSON</label>
 <input type="file" name="weights" accept=".json,application/json" required>
 <button type="button" onclick="prepare()">第一步：读取网格与原分数</button>
</form>

<div id="prepareInfo"></div>

<form id="pointsForm" style="display:none">
 <h2>第二步：填写标点（输出网格坐标，从 0 起）</h2>
 <p class="note">同一图内坐标不可重复；裂缝/非裂缝为修正点（至少 1 个），
 保留点至少 1 个，合计 ≤ 32。</p>
 <table id="ptTable"><thead><tr>
   <th>图序号</th><th>行 r</th><th>列 c</th><th>标注</th><th></th>
 </tr></thead><tbody></tbody></table>
 <button type="button" class="secondary" onclick="addRow()">添加标点</button>
 <div style="margin-top:1rem">
   <button type="button" onclick="solve()">第三步：求解并整图复查</button>
 </div>
</form>

<div id="result"></div>

<script>
let grids = null;
const LABELS = ["crack","noncrack","keep"];
const LABEL_CN = {crack:"裂缝", noncrack:"非裂缝", keep:"保留原分数"};

function uploadData(){
  const fd = new FormData(document.getElementById("uploadForm"));
  if(!fd.get("weights") || fd.get("weights").size === 0){
    throw new Error("请选择权重 JSON 文件");
  }
  return fd;
}

async function prepare(){
  const info = document.getElementById("prepareInfo");
  const result = document.getElementById("result");
  result.innerHTML = "";
  try{
    const fd = uploadData();
    const r = await fetch("/refine/prepare", {method:"POST", body:fd});
    const body = await r.json();
    if(!r.ok){ info.innerHTML = errBox(body.error || "读取失败"); return; }
    grids = body.images;
    info.innerHTML = "<h2>图像与输出网格</h2>" +
      "<table><tr><th>图序号</th><th>图像尺寸 (行,列)</th>" +
      "<th>输出网格 (行,列)</th><th>原分数范围 [min, max]</th></tr>" +
      body.images.map(g => `<tr><td>${g.image_index}</td><td>${g.image_size.join(" × ")}</td>`+
        `<td>${g.output_size.join(" × ")}</td>`+
        `<td>${g.score_min.toFixed(6)} , ${g.score_max.toFixed(6)}</td></tr>`).join("") +
      "</table><p class='note'>原分数网格（整图推理，行优先）：</p>" +
      body.images.map(g => `<details><summary>图 ${g.image_index} 分数网格</summary>`+
        `<pre style='font-size:.72rem;max-height:260px;overflow:auto'>`+
        g.score_grid.map(row => row.map(v => v.toFixed(4)).join("  ")).join("\\n")+
        "</pre></details>").join("");
    document.getElementById("pointsForm").style.display = "";
    document.querySelector("#ptTable tbody").innerHTML = "";
    addRow();
  }catch(e){ info.innerHTML = errBox(e.message); }
}

function addRow(){
  const tbody = document.querySelector("#ptTable tbody");
  const tr = document.createElement("tr");
  tr.className = "ptrow";
  const imgOpts = grids.map((_,i)=>`<option value="${i}">${i}</option>`).join("");
  tr.innerHTML = `<td><select name="image_index">${imgOpts}</select></td>`+
    `<td><input type="number" name="row" min="0" step="1" required></td>`+
    `<td><input type="number" name="col" min="0" step="1" required></td>`+
    `<td><select name="label">`+
    LABELS.map(l=>`<option value="${l}">${LABEL_CN[l]}</option>`).join("")+
    `</select></td>`+
    `<td><button type="button" class="secondary" `+
    `onclick="this.closest('tr').remove()">删除</button></td>`;
  tbody.appendChild(tr);
}

function collectPoints(){
  const rows = [...document.querySelectorAll("#ptTable tbody tr")];
  if(rows.length === 0) throw new Error("至少填写 1 个标点");
  if(rows.length > 32) throw new Error("标点总数不能超过 32");
  return rows.map(tr => {
    const q = tr.querySelectorAll("input,select");
    const image_index = parseInt(q[0].value, 10);
    const row = parseInt(q[1].value, 10);
    const col = parseInt(q[2].value, 10);
    const label = q[3].value;
    if(!Number.isInteger(row) || !Number.isInteger(col))
      throw new Error("坐标必须是整数");
    const g = grids[image_index];
    if(!(0 <= row && row < g.output_size[0]) ||
       !(0 <= col && col < g.output_size[1]))
      throw new Error(`图 ${image_index} 坐标 (${row},${col}) 超出输出网格 `+
        g.output_size.join("x"));
    return {image_index, row, col, label};
  });
}

async function solve(){
  const box = document.getElementById("result");
  try{
    const points = collectPoints();
    const seen = new Set();
    for(const p of points){
      const k = `${p.image_index}:${p.row}:${p.col}`;
      if(seen.has(k)) throw new Error("存在重复标点：" + k);
      seen.add(k);
    }
    const fd = uploadData();
    fd.append("points", JSON.stringify(points));
    const r = await fetch("/refine/solve", {method:"POST", body:fd});
    const body = await r.json();
    box.innerHTML = renderResult(body, r.ok);
  }catch(e){ box.innerHTML = errBox(e.message); }
}

function renderResult(b, ok){
  if(!ok) return errBox(b.error || "求解失败");
  if(!b.success){
    return "<div class='err'><b>本次未生成通过复查的权重。</b>\\n\\n" +
      escapeHtml(b.reason) + "\\n\\n已保留你上传的原权重文件不变，" +
      "并撤下网页上的旧下载。</div>" + pointTable(b);
  }
  return "<div class='dl'><b>修正完成并通过整图复查。</b><br>"+
    `参数改变量平方和 ‖d‖² = ${b.objective.toExponential(6)}；`+
    `复查最大约束违例 = ${b.max_violation.toExponential(3)}（容差 1e-8）。<br>`+
    `裂缝点 ${b.n_crack} 个，非裂缝点 ${b.n_noncrack} 个，`+
    `保留点 ${b.n_keep} 个。<br>`+
    `<a href='/refine/download'><b>下载新权重 JSON</b></a>`+
    " （已替换为当前唯一发布版本）</div>" + pointTable(b);
}

function pointTable(b){
  if(!b.points || b.points.length === 0) return "";
  return "<h2>全部标点复查明细</h2><table><thead><tr>"+
    "<th>图</th><th>行</th><th>列</th><th>标注</th>"+
    "<th>原分数</th><th>复查分数</th><th>违例量</th><th>判定</th></tr></thead><tbody>"+
    b.points.map(p => `<tr><td>${p.image_index}</td><td>${p.row}</td>`+
      `<td>${p.col}</td><td class="tag-${p.label}">${LABEL_CN[p.label]}</td>`+
      `<td>${p.score_before.toFixed(6)}</td>`+
      `<td>${p.score_after.toFixed(6)}</td>`+
      `<td>${p.violation.toExponential(2)}</td>`+
      `<td class="${p.ok ? "ok" : "bad"}">${p.ok ? "通过" : "失败"}</td></tr>`
    ).join("") + "</tbody></table>";
}

function errBox(msg){ return "<div class='err'>" + escapeHtml(msg) + "</div>"; }
function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => (
    {"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
}
</script>
</body></html>"""


def _load_request(image1: UploadFile | None, image2: UploadFile | None,
                  weights: UploadFile) -> tuple[list, dict]:
    images = []
    for up in (image1, image2):
        if up is not None and up.filename:
            images.append(load_grayscale_png(
                up.file.read(), max_side=REFINE_MAX_SIDE))
    if not 1 <= len(images) <= 2:
        raise ValueError("需要上传 1～2 张灰度 PNG")
    wts = load_weights_json(weights.file.read())
    validate_weights(wts)  # 提前校验形状，失败直接 400
    return images, wts


def _parse_points(raw: str, n_images: int) -> list[MarkedPoint]:
    if not raw:
        raise ValueError("缺少标点数据 points")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"标点数据不是合法 JSON：{exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("标点必须是非空数组")
    if len(data) > MAX_POINTS:
        raise ValueError(f"标点总数至多 {MAX_POINTS} 个，收到 {len(data)} 个")
    pts: list[MarkedPoint] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("每个标点必须是对象")
        try:
            ii = int(item["image_index"])
            row = int(item["row"])
            col = int(item["col"])
            label = str(item["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"标点字段不完整或类型错误：{item}") from exc
        if label not in LABELS:
            raise ValueError(f"未知标注类型：{label}（允许 {LABELS}）")
        if not 0 <= ii < n_images:
            raise ValueError(f"图序号 {ii} 超出 0..{n_images - 1}")
        pts.append(MarkedPoint(ii, row, col, label))
    return pts


def _grid_payload(images, w1, b1, w2, b2) -> list[dict]:
    out = []
    for idx, img in enumerate(images):
        scores = run_full_baseline(img, w1, b1, w2, b2)[0]
        ho, wo = layer1_out_size(*img.shape)
        assert scores.shape == (ho, wo)
        out.append({
            "image_index": idx,
            "image_size": [int(img.shape[0]), int(img.shape[1])],
            "output_size": [int(ho), int(wo)],
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_grid": scores.tolist(),
        })
    return out


@router.get("", response_class=HTMLResponse)
def refine_page() -> str:
    return PAGE


@router.post("/prepare")
async def prepare(image1: UploadFile | None = File(None),
                  image2: UploadFile | None = File(None),
                  weights: UploadFile | None = File(None)) -> JSONResponse:
    try:
        if weights is None or not weights.filename:
            raise ValueError("缺少权重 JSON 文件")
        images, wts = _load_request(image1, image2, weights)
        w1, b1, w2, b2 = validate_weights(wts)
        payload = {"images": _grid_payload(images, w1, b1, w2, b2),
                   "download_available": store.exists()}
    except (ValueError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(payload)


@router.post("/solve")
async def solve(image1: UploadFile | None = File(None),
                image2: UploadFile | None = File(None),
                weights: UploadFile | None = File(None),
                points: str = Form("")) -> JSONResponse:
    try:
        if weights is None or not weights.filename:
            raise ValueError("缺少权重 JSON 文件")
        images, wts = _load_request(image1, image2, weights)
        marked = _parse_points(points, len(images))
        result = run_refine(images, wts, marked)
    except (ValueError, RuntimeError) as exc:
        # 请求本身非法：不动已发布产物
        return JSONResponse({"error": str(exc)}, status_code=400)

    if result.success:
        # 通过复查才发布（原子替换旧产物），上传的原权重从未被改写
        store.publish(result.new_weights_json.encode("utf-8"))
    else:
        # 未生成通过复查的权重：撤下旧下载
        store.revoke()

    return JSONResponse({
        "success": result.success,
        "reason": result.reason,
        "objective": result.objective,
        "max_violation": result.max_violation,
        "recheck_tolerance": RECHECK_TOL,
        "n_crack": result.n_crack,
        "n_noncrack": result.n_noncrack,
        "n_keep": result.n_keep,
        "download_available": result.success and store.exists(),
        "points": [p.__dict__ for p in result.points],
    })


@router.get("/download")
def download() -> Response:
    data = store.current()
    if data is None:
        return JSONResponse(
            {"error": "当前没有通过整图复查的权重可下载；"
                      "成功提交一次修正后才会生成。"},
            status_code=404)
    return Response(
        content=data, media_type="application/json",
        headers={"Content-Disposition":
                 'attachment; filename="refined_weights.json"'})


# 供测试使用的纯函数入口
def _store_path() -> str:
    return str(store.path)
