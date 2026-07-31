from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

from pipeline import ELISpotPipeline  # noqa: E402


st.set_page_config(
    page_title="ELISpot Pipeline Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
    :root {
        --ink: #102a43;
        --muted: #627d98;
        --navy: #123b5d;
        --teal: #0f8b8d;
        --mint: #dff5f1;
        --coral: #f26b5b;
        --paper: #f5f8fa;
        --line: #d9e2ec;
    }
    html, body, [class*="css"] { font-family: "DM Sans", "Noto Sans SC", sans-serif; }
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 92% 4%, rgba(15,139,141,.10), transparent 24rem),
          linear-gradient(180deg, #f8fbfc 0%, #ffffff 38%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2.2rem; }
    [data-testid="stSidebar"] { background: #0f2f49; border-right: 0; }
    [data-testid="stSidebar"] * { color: #f3f8fb; }
    [data-testid="stSidebar"] input { color: #102a43 !important; }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.14);
    }
    .hero {
        position: relative; overflow: hidden; padding: 2.2rem 2.35rem;
        border: 1px solid rgba(18,59,93,.10); border-radius: 24px;
        background: linear-gradient(125deg, #102f49 0%, #164d65 55%, #0f8b8d 140%);
        box-shadow: 0 20px 55px rgba(16,42,67,.13); color: white; margin-bottom: 1.25rem;
    }
    .hero:after {
        content: ""; position: absolute; width: 270px; height: 270px; right: -70px; top: -120px;
        border: 38px solid rgba(255,255,255,.06); border-radius: 50%;
    }
    .eyebrow { font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; color: #8fe0d4; }
    .hero h1 { margin: .45rem 0 .55rem; color: white; font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.05; }
    .hero p { max-width: 760px; color: #d9edf2; font-size: 1.02rem; margin: 0; }
    .hero-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.25rem; }
    .hero-badges span { border: 1px solid rgba(255,255,255,.2); background: rgba(255,255,255,.08); padding: .38rem .7rem; border-radius: 999px; font-size: .78rem; }
    .section-kicker { color: var(--teal); font-size: .76rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; margin-bottom: .15rem; }
    .section-title { font-size: 1.45rem; font-weight: 700; color: var(--ink); margin-bottom: .25rem; }
    .section-copy { color: var(--muted); margin-bottom: 1rem; }
    .stage-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: .65rem; margin: .8rem 0 1.7rem; }
    .stage-card { background: rgba(255,255,255,.86); border: 1px solid var(--line); border-radius: 14px; padding: .8rem; min-height: 94px; }
    .stage-no { width: 26px; height: 26px; display: grid; place-items: center; border-radius: 8px; background: var(--mint); color: #087477; font-size: .72rem; font-weight: 700; }
    .stage-card strong { display: block; color: var(--ink); font-size: .83rem; margin-top: .55rem; }
    .stage-card small { color: var(--muted); font-size: .7rem; }
    [data-testid="stFileUploader"] { background: white; border: 1px solid var(--line); border-radius: 18px; padding: .65rem .85rem .9rem; box-shadow: 0 8px 25px rgba(16,42,67,.045); }
    [data-testid="stFileUploaderDropzone"] { background: var(--paper); border: 1px dashed #9fb3c8; border-radius: 12px; }
    [data-testid="stButton"] button[kind="primary"] { background: linear-gradient(100deg, #0f8b8d, #087477); border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700; box-shadow: 0 9px 20px rgba(15,139,141,.22); }
    [data-testid="stButton"] button[kind="primary"]:disabled { background: #d9e2ec; color: #829ab1; box-shadow: none; }
    .ready-box { border: 1px solid var(--line); border-radius: 16px; background: white; padding: 1rem 1.15rem; margin: .5rem 0 1rem; }
    .ready-row { display: flex; gap: .45rem; align-items: center; color: var(--muted); font-size: .85rem; margin: .3rem 0; }
    .dot-ok, .dot-wait { width: 8px; height: 8px; border-radius: 50%; flex: none; }
    .dot-ok { background: #20a779; box-shadow: 0 0 0 4px rgba(32,167,121,.12); }
    .dot-wait { background: #bcccdc; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: .8rem 1rem; }
    footer { visibility: hidden; }
    @media (max-width: 900px) { .stage-grid { grid-template-columns: repeat(3, 1fr); } .hero { padding: 1.6rem; } }
    @media (max-width: 600px) { .stage-grid { grid-template-columns: repeat(2, 1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_pages(text: str) -> list[int] | None:
    text = text.strip()
    if not text:
        return None
    pages: set[int] = set()
    for part in re.split(r"[,，\s]+", text):
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if start < 1 or end < start:
                raise ValueError(f"无效页码范围：{part}")
            pages.update(range(start, end + 1))
        else:
            value = int(part)
            if value < 1:
                raise ValueError("页码必须从 1 开始")
            pages.add(value)
    return sorted(pages)


def save_upload(upload, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(upload.getbuffer())
    return destination


def job_id_for(paper, supplement, pages: list[int] | None) -> str:
    digest = hashlib.sha256()
    digest.update(paper.getvalue())
    digest.update(supplement.getvalue())
    digest.update(json.dumps(pages).encode())
    return digest.hexdigest()[:12]


def read_csv_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def zip_directory(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return buffer.getvalue()


def show_stage1(run_root: Path) -> None:
    images = sorted((run_root / "stage1").glob("page_*/page.png"))
    if not images:
        st.info("没有找到 Stage 1 页面图像。")
        return
    st.markdown("#### Stage 1 · 主论文页面")
    columns = st.columns(min(3, len(images)))
    for index, image_path in enumerate(images):
        columns[index % len(columns)].image(str(image_path), caption=image_path.parent.name, use_container_width=True)


def show_stage3(run_root: Path) -> None:
    manifest = read_csv_if_present(run_root / "stage3" / "stage3_selected_heatmap_context_manifest.csv")
    if manifest is None:
        st.info("没有找到 Stage 3 热图识别结果。")
        return
    st.markdown("#### Stage 3 · 选中的热图区域")
    display_cols = [c for c in ["figure_unit_id", "candidate_id", "ai_detected_panel_id", "ai_confidence", "ai_reason"] if c in manifest]
    st.dataframe(manifest[display_cols], use_container_width=True, hide_index=True)
    image_cols = st.columns(2)
    for index, row in manifest.iterrows():
        path = Path(str(row.get("selected_context_path", "")))
        if path.exists():
            caption = f"{row.get('figure_unit_id')} · panel {row.get('ai_detected_panel_id', 'Not specified')}"
            image_cols[index % 2].image(str(path), caption=caption, use_container_width=True)


def show_stage4(run_root: Path) -> None:
    summary = read_csv_if_present(run_root / "stage4" / "Stage4_Panel_Summary.csv")
    records = read_csv_if_present(run_root / "stage4" / "Stage4_Cell_Records.csv")
    if summary is None or records is None:
        st.info("没有找到 Stage 4 热图数值结果。")
        return
    st.markdown("#### Stage 4 · 热图矩阵与数值")
    summary_cols = [c for c in ["Figure_Unit_ID", "Panel_ID", "Heatmap_Subtype", "Axis_X_Values", "Axis_Y_Values", "Axis_Alignment_Confidence"] if c in summary]
    st.dataframe(summary[summary_cols], use_container_width=True, hide_index=True)
    panel_values = records["Panel_ID"].dropna().astype(str).unique().tolist() if "Panel_ID" in records else []
    selected_panel = st.selectbox("查看面板", panel_values, key="stage4_panel") if panel_values else None
    panel_df = records[records["Panel_ID"].astype(str).eq(selected_panel)] if selected_panel else records
    value_cols = [c for c in ["Panel_ID", "Row_Index", "Col_Index", "X_Axis_Value", "Y_Axis_Value", "Epitope_ID", "ELISpot_Value", "ELISpot_Value_Text"] if c in panel_df]
    st.dataframe(panel_df[value_cols], use_container_width=True, hide_index=True, height=360)


def show_stage5(run_root: Path) -> None:
    reference = read_csv_if_present(run_root / "stage5" / "Epitope_Reference_Table_FINAL.csv")
    audit = read_csv_if_present(run_root / "stage5" / "Stage5_Row_Audit.csv")
    if reference is None:
        st.info("没有找到 Stage 5 补充表格结果。")
        return
    st.markdown("#### Stage 5 · 补充材料表格提取")
    core = [c for c in ["Epitope_ID", "Epitope_Gene", "Epitope_Mutation", "Epitope_Sequence", "MHC_Class", "Vaccine_or_Antigen_Set", "Source_Sheet", "Source_Page", "Source_Row"] if c in reference]
    st.dataframe(reference[core], use_container_width=True, hide_index=True, height=360)
    if audit is not None and "Status" in audit:
        counts = audit["Status"].value_counts().rename_axis("Status").reset_index(name="Rows")
        st.caption("每个来源行都有接受或跳过记录；跳过原因保存在审计文件中。")
        st.dataframe(counts, hide_index=True)


def show_stage6(run_root: Path) -> None:
    final = read_csv_if_present(run_root / "stage6" / "ELISpot_Data.csv")
    qc_path = run_root / "stage6" / "Stage6_QC_Summary.json"
    if final is None:
        st.info("没有找到 Stage 6 最终结果。")
        return
    st.markdown("#### Stage 6 · 最终 ELISpot 数据")
    metrics = st.columns(4)
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    metrics[0].metric("最终记录", len(final))
    metrics[1].metric("参考表匹配", qc.get("reference_matched_rows", "—"))
    metrics[2].metric("未匹配", qc.get("reference_unmatched_rows", "—"))
    metrics[3].metric("重复记录", qc.get("duplicate_rows_by_main_key", "—"))
    st.dataframe(final, use_container_width=True, hide_index=True, height=420)


def run_pipeline(paper_upload, supplement_upload, api_key: str, pages: list[int] | None) -> Path:
    job_id = job_id_for(paper_upload, supplement_upload, pages)
    workspace = APP_ROOT / "tool_runs" / job_id
    input_dir = workspace / "inputs"
    paper_path = save_upload(paper_upload, input_dir / f"main{Path(paper_upload.name).suffix.lower()}")
    supplement_path = save_upload(supplement_upload, input_dir / f"supplement{Path(supplement_upload.name).suffix.lower()}")
    runner = ELISpotPipeline(paper_path, supplement_path, output_root=workspace / "pipeline_runs", pages=pages, resume=True)
    steps = [
        ("Stage 1", "解析主论文页面", runner._run_stage1),
        ("Stage 2", "配对图与图注", runner._run_stage2),
        ("Stage 3", "识别 ELISpot 热图", runner._run_stage3),
        ("Stage 3B/3C", "恢复面板并提取图注字段", runner._run_stage3bc),
        ("Stage 4", "提取热图矩阵数值", runner._run_stage4),
        ("Stage 5", "提取补充材料表格", runner._run_stage5),
        ("Stage 6", "合并并生成最终结果", runner._run_stage6),
    ]
    progress = st.progress(0, text="准备运行")
    status = st.status("ELISpot pipeline 正在运行", expanded=True)
    old_key = os.environ.get("DASHSCOPE_API_KEY")
    os.environ["DASHSCOPE_API_KEY"] = api_key.strip()
    try:
        for index, (stage, description, function) in enumerate(steps, 1):
            status.write(f"**{stage}** · {description}")
            function()
            progress.progress(index / len(steps), text=f"{stage} 完成")
        runner.manifest["status"] = "complete"
        runner.manifest_path.write_text(json.dumps(runner.manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        status.update(label="全部阶段完成", state="complete", expanded=False)
    finally:
        if old_key is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = old_key
    return runner.run_root


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Reproducible immunology workflow</div>
      <h1>ELISpot Pipeline Studio</h1>
      <p>上传论文与补充材料，从图像识别、热图数值提取到 epitope 表格匹配，一次运行生成可追溯的 cell-level 数据。</p>
      <div class="hero-badges"><span>6 个处理阶段</span><span>PDF / Excel / CSV</span><span>全过程可审计</span><span>结果可下载</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Workflow</div><div class="section-title">从原始文献到结构化数据</div><div class="section-copy">关键中间结果会在运行后按阶段展示，便于人工核验。</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="stage-grid">
      <div class="stage-card"><div class="stage-no">01</div><strong>页面解析</strong><small>渲染指定论文页</small></div>
      <div class="stage-card"><div class="stage-no">02</div><strong>图文配对</strong><small>定位图与图注</small></div>
      <div class="stage-card"><div class="stage-no">03</div><strong>热图识别</strong><small>筛选 ELISpot 面板</small></div>
      <div class="stage-card"><div class="stage-no">04</div><strong>数值提取</strong><small>恢复矩阵与坐标</small></div>
      <div class="stage-card"><div class="stage-no">05</div><strong>表格解析</strong><small>统一 epitope 字段</small></div>
      <div class="stage-card"><div class="stage-no">06</div><strong>合并质控</strong><small>生成最终数据</small></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 运行控制台")
    st.caption("密钥仅保存在当前运行进程的内存中，不会写入 Notebook 或输出文件。")
    api_key = st.text_input("DashScope API Key", type="password", placeholder="sk-…", help="用于图像与表格的 AI 识别步骤")
    page_mode = st.selectbox(
        "主论文处理范围",
        ["All（整篇论文）", "Other（指定页码）"],
        index=1,
        help="选择 All 处理全文；选择 Other 后可填写单页、多个页码或连续范围。",
    )
    if page_mode.startswith("Other"):
        page_text = st.text_input(
            "指定页码",
            value="3,4",
            placeholder="例如：3,4 或 3-5",
            help="多个页码用逗号分隔；连续页码可写成 3-5。",
        )
    else:
        page_text = ""
        st.caption("将处理主论文的全部页面。")
    st.divider()
    st.markdown("**快速测试建议**")
    st.info("先选择包含 ELISpot 图的 2–3 页。补充材料仍会完整检查，以定位参考表。")
    st.caption("本工具适合本地或受信任的单用户环境。")

st.markdown('<div class="section-kicker">Inputs</div><div class="section-title">准备本次分析</div><div class="section-copy">主论文负责图像数据，补充材料负责 epitope 参考信息。</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    paper_upload = st.file_uploader("① 上传主论文 PDF", type=["pdf"], help="建议先上传只含目标页的 PDF 进行快速测试")
with right:
    supplement_upload = st.file_uploader("② 上传补充材料", type=["pdf", "xlsx", "xls", "csv", "tsv"], help="支持文字型 PDF 与常见表格格式")

try:
    pages = parse_pages(page_text)
    page_error = None
except Exception as exc:
    pages = None
    page_error = str(exc)
    st.error(page_error)

paper_ok = paper_upload is not None
supplement_ok = supplement_upload is not None
key_ok = bool(api_key.strip())
ready = paper_ok and supplement_ok and key_ok and page_error is None

def ready_row(done: bool, text: str) -> str:
    dot = "dot-ok" if done else "dot-wait"
    return f'<div class="ready-row"><span class="{dot}"></span>{text}</div>'

st.markdown(
    '<div class="ready-box"><strong>运行前检查</strong>'
    + ready_row(paper_ok, f"主论文：{paper_upload.name}" if paper_ok else "等待上传主论文 PDF")
    + ready_row(supplement_ok, f"补充材料：{supplement_upload.name}" if supplement_ok else "等待上传补充材料")
    + ready_row(key_ok, "API Key 已就绪" if key_ok else "等待在左侧填写 API Key")
    + ready_row(page_error is None, f"处理页码：{', '.join(map(str, pages)) if pages else '整篇论文'}")
    + "</div>",
    unsafe_allow_html=True,
)

if st.button("开始运行 Stage 1–6", type="primary", disabled=not ready, use_container_width=True):
    try:
        st.session_state["last_run_root"] = str(run_pipeline(paper_upload, supplement_upload, api_key, pages))
    except Exception as exc:
        st.exception(exc)

run_root_text = st.session_state.get("last_run_root")
if run_root_text:
    run_root = Path(run_root_text)
    if run_root.exists():
        st.divider()
        st.markdown('<div class="section-kicker">Review</div><div class="section-title">过程核验与最终结果</div><div class="section-copy">逐步检查论文页面、热图提取值、补充表格与合并质控。</div>', unsafe_allow_html=True)
        tabs = st.tabs(["论文页面", "热图识别", "热图数值", "补充表格", "最终结果", "下载"])
        with tabs[0]:
            show_stage1(run_root)
        with tabs[1]:
            show_stage3(run_root)
        with tabs[2]:
            show_stage4(run_root)
        with tabs[3]:
            show_stage5(run_root)
        with tabs[4]:
            show_stage6(run_root)
        with tabs[5]:
            final_xlsx = run_root / "stage6" / "Stage6_Final_ELISpot_Output.xlsx"
            if final_xlsx.exists():
                st.download_button("下载最终 Excel", final_xlsx.read_bytes(), final_xlsx.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            st.download_button("下载完整审计包 ZIP", zip_directory(run_root), "ELISpot_pipeline_results.zip", mime="application/zip", use_container_width=True)
