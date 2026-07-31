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
    page_icon="馃敩",
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
    for part in re.split(r"[,锛孿s]+", text):
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if start < 1 or end < start:
                raise ValueError(f"鏃犳晥椤电爜鑼冨洿锛歿part}")
            pages.update(range(start, end + 1))
        else:
            value = int(part)
            if value < 1:
                raise ValueError("椤电爜蹇呴』浠?1 寮€濮?)
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
        st.info("娌℃湁鎵惧埌 Stage 1 椤甸潰鍥惧儚銆?)
        return
    st.markdown("#### Stage 1 路 涓昏鏂囬〉闈?)
    columns = st.columns(min(3, len(images)))
    for index, image_path in enumerate(images):
        columns[index % len(columns)].image(str(image_path), caption=image_path.parent.name, use_container_width=True)


def show_stage3(run_root: Path) -> None:
    manifest = read_csv_if_present(run_root / "stage3" / "stage3_selected_heatmap_context_manifest.csv")
    if manifest is None:
        st.info("娌℃湁鎵惧埌 Stage 3 鐑浘璇嗗埆缁撴灉銆?)
        return
    st.markdown("#### Stage 3 路 閫変腑鐨勭儹鍥惧尯鍩?)
    display_cols = [c for c in ["figure_unit_id", "candidate_id", "ai_detected_panel_id", "ai_confidence", "ai_reason"] if c in manifest]
    st.dataframe(manifest[display_cols], use_container_width=True, hide_index=True)
    image_cols = st.columns(2)
    for index, row in manifest.iterrows():
        path = Path(str(row.get("selected_context_path", "")))
        if path.exists():
            caption = f"{row.get('figure_unit_id')} 路 panel {row.get('ai_detected_panel_id', 'Not specified')}"
            image_cols[index % 2].image(str(path), caption=caption, use_container_width=True)


def show_stage4(run_root: Path) -> None:
    summary = read_csv_if_present(run_root / "stage4" / "Stage4_Panel_Summary.csv")
    records = read_csv_if_present(run_root / "stage4" / "Stage4_Cell_Records.csv")
    if summary is None or records is None:
        st.info("娌℃湁鎵惧埌 Stage 4 鐑浘鏁板€肩粨鏋溿€?)
        return
    st.markdown("#### Stage 4 路 鐑浘鐭╅樀涓庢暟鍊?)
    summary_cols = [c for c in ["Figure_Unit_ID", "Panel_ID", "Heatmap_Subtype", "Axis_X_Values", "Axis_Y_Values", "Axis_Alignment_Confidence"] if c in summary]
    st.dataframe(summary[summary_cols], use_container_width=True, hide_index=True)
    panel_values = records["Panel_ID"].dropna().astype(str).unique().tolist() if "Panel_ID" in records else []
    selected_panel = st.selectbox("鏌ョ湅闈㈡澘", panel_values, key="stage4_panel") if panel_values else None
    panel_df = records[records["Panel_ID"].astype(str).eq(selected_panel)] if selected_panel else records
    value_cols = [c for c in ["Panel_ID", "Row_Index", "Col_Index", "X_Axis_Value", "Y_Axis_Value", "Epitope_ID", "ELISpot_Value", "ELISpot_Value_Text"] if c in panel_df]
    st.dataframe(panel_df[value_cols], use_container_width=True, hide_index=True, height=360)


def show_stage5(run_root: Path) -> None:
    reference = read_csv_if_present(run_root / "stage5" / "Epitope_Reference_Table_FINAL.csv")
    audit = read_csv_if_present(run_root / "stage5" / "Stage5_Row_Audit.csv")
    if reference is None:
        st.info("娌℃湁鎵惧埌 Stage 5 琛ュ厖琛ㄦ牸缁撴灉銆?)
        return
    st.markdown("#### Stage 5 路 琛ュ厖鏉愭枡琛ㄦ牸鎻愬彇")
    core = [c for c in ["Epitope_ID", "Epitope_Gene", "Epitope_Mutation", "Epitope_Sequence", "MHC_Class", "Vaccine_or_Antigen_Set", "Source_Sheet", "Source_Page", "Source_Row"] if c in reference]
    st.dataframe(reference[core], use_container_width=True, hide_index=True, height=360)
    if audit is not None and "Status" in audit:
        counts = audit["Status"].value_counts().rename_axis("Status").reset_index(name="Rows")
        st.caption("姣忎釜鏉ユ簮琛岄兘鏈夋帴鍙楁垨璺宠繃璁板綍锛涜烦杩囧師鍥犱繚瀛樺湪瀹¤鏂囦欢涓€?)
        st.dataframe(counts, hide_index=True)


def show_stage6(run_root: Path) -> None:
    final = read_csv_if_present(run_root / "stage6" / "ELISpot_Data.csv")
    qc_path = run_root / "stage6" / "Stage6_QC_Summary.json"
    if final is None:
        st.info("娌℃湁鎵惧埌 Stage 6 鏈€缁堢粨鏋溿€?)
        return
    st.markdown("#### Stage 6 路 鏈€缁?ELISpot 鏁版嵁")
    metrics = st.columns(4)
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    metrics[0].metric("鏈€缁堣褰?, len(final))
    metrics[1].metric("鍙傝€冭〃鍖归厤", qc.get("reference_matched_rows", "鈥?))
    metrics[2].metric("鏈尮閰?, qc.get("reference_unmatched_rows", "鈥?))
    metrics[3].metric("閲嶅璁板綍", qc.get("duplicate_rows_by_main_key", "鈥?))
    st.dataframe(final, use_container_width=True, hide_index=True, height=420)


def run_pipeline(paper_upload, supplement_upload, api_key: str, pages: list[int] | None) -> Path:
    job_id = job_id_for(paper_upload, supplement_upload, pages)
    workspace = APP_ROOT / "tool_runs" / job_id
    input_dir = workspace / "inputs"
    paper_path = save_upload(paper_upload, input_dir / f"main{Path(paper_upload.name).suffix.lower()}")
    supplement_path = save_upload(supplement_upload, input_dir / f"supplement{Path(supplement_upload.name).suffix.lower()}")
    runner = ELISpotPipeline(paper_path, supplement_path, output_root=workspace / "pipeline_runs", pages=pages, resume=True)
    steps = [
        ("Stage 1", "瑙ｆ瀽涓昏鏂囬〉闈?, runner._run_stage1),
        ("Stage 2", "閰嶅鍥句笌鍥炬敞", runner._run_stage2),
        ("Stage 3", "璇嗗埆 ELISpot 鐑浘", runner._run_stage3),
        ("Stage 3B/3C", "鎭㈠闈㈡澘骞舵彁鍙栧浘娉ㄥ瓧娈?, runner._run_stage3bc),
        ("Stage 4", "鎻愬彇鐑浘鐭╅樀鏁板€?, runner._run_stage4),
        ("Stage 5", "鎻愬彇琛ュ厖鏉愭枡琛ㄦ牸", runner._run_stage5),
        ("Stage 6", "鍚堝苟骞剁敓鎴愭渶缁堢粨鏋?, runner._run_stage6),
    ]
    progress = st.progress(0, text="鍑嗗杩愯")
    status = st.status("ELISpot pipeline 姝ｅ湪杩愯", expanded=True)
    old_key = os.environ.get("DASHSCOPE_API_KEY")
    os.environ["DASHSCOPE_API_KEY"] = api_key.strip()
    try:
        for index, (stage, description, function) in enumerate(steps, 1):
            status.write(f"**{stage}** 路 {description}")
            function()
            progress.progress(index / len(steps), text=f"{stage} 瀹屾垚")
        runner.manifest["status"] = "complete"
        runner.manifest_path.write_text(json.dumps(runner.manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        status.update(label="鍏ㄩ儴闃舵瀹屾垚", state="complete", expanded=False)
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
      <p>涓婁紶璁烘枃涓庤ˉ鍏呮潗鏂欙紝浠庡浘鍍忚瘑鍒€佺儹鍥炬暟鍊兼彁鍙栧埌 epitope 琛ㄦ牸鍖归厤锛屼竴娆¤繍琛岀敓鎴愬彲杩芥函鐨?cell-level 鏁版嵁銆?/p>
      <div class="hero-badges"><span>6 涓鐞嗛樁娈?/span><span>PDF / Excel / CSV</span><span>鍏ㄨ繃绋嬪彲瀹¤</span><span>缁撴灉鍙笅杞?/span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Workflow</div><div class="section-title">浠庡師濮嬫枃鐚埌缁撴瀯鍖栨暟鎹?/div><div class="section-copy">鍏抽敭涓棿缁撴灉浼氬湪杩愯鍚庢寜闃舵灞曠ず锛屼究浜庝汉宸ユ牳楠屻€?/div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="stage-grid">
      <div class="stage-card"><div class="stage-no">01</div><strong>椤甸潰瑙ｆ瀽</strong><small>娓叉煋鎸囧畾璁烘枃椤?/small></div>
      <div class="stage-card"><div class="stage-no">02</div><strong>鍥炬枃閰嶅</strong><small>瀹氫綅鍥句笌鍥炬敞</small></div>
      <div class="stage-card"><div class="stage-no">03</div><strong>鐑浘璇嗗埆</strong><small>绛涢€?ELISpot 闈㈡澘</small></div>
      <div class="stage-card"><div class="stage-no">04</div><strong>鏁板€兼彁鍙?/strong><small>鎭㈠鐭╅樀涓庡潗鏍?/small></div>
      <div class="stage-card"><div class="stage-no">05</div><strong>琛ㄦ牸瑙ｆ瀽</strong><small>缁熶竴 epitope 瀛楁</small></div>
      <div class="stage-card"><div class="stage-no">06</div><strong>鍚堝苟璐ㄦ帶</strong><small>鐢熸垚鏈€缁堟暟鎹?/small></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 杩愯鎺у埗鍙?)
    st.caption("瀵嗛挜浠呬繚瀛樺湪褰撳墠杩愯杩涚▼鐨勫唴瀛樹腑锛屼笉浼氬啓鍏?Notebook 鎴栬緭鍑烘枃浠躲€?)
    api_key = st.text_input("DashScope API Key", type="password", placeholder="sk-鈥?, help="鐢ㄤ簬鍥惧儚涓庤〃鏍肩殑 AI 璇嗗埆姝ラ")
    page_mode = st.selectbox(
        "涓昏鏂囧鐞嗚寖鍥?,
        ["All锛堟暣绡囪鏂囷級", "Other锛堟寚瀹氶〉鐮侊級"],
        index=1,
        help="閫夋嫨 All 澶勭悊鍏ㄦ枃锛涢€夋嫨 Other 鍚庡彲濉啓鍗曢〉銆佸涓〉鐮佹垨杩炵画鑼冨洿銆?,
    )
    if page_mode.startswith("Other"):
        page_text = st.text_input(
            "鎸囧畾椤电爜",
            value="3,4",
            placeholder="渚嬪锛?,4 鎴?3-5",
            help="澶氫釜椤电爜鐢ㄩ€楀彿鍒嗛殧锛涜繛缁〉鐮佸彲鍐欐垚 3-5銆?,
        )
    else:
        page_text = ""
        st.caption("灏嗗鐞嗕富璁烘枃鐨勫叏閮ㄩ〉闈€?)
    st.divider()
    st.markdown("**蹇€熸祴璇曞缓璁?*")
    st.info("鍏堥€夋嫨鍖呭惈 ELISpot 鍥剧殑 2鈥? 椤点€傝ˉ鍏呮潗鏂欎粛浼氬畬鏁存鏌ワ紝浠ュ畾浣嶅弬鑰冭〃銆?)
    st.caption("鏈伐鍏烽€傚悎鏈湴鎴栧彈淇′换鐨勫崟鐢ㄦ埛鐜銆?)

st.markdown('<div class="section-kicker">Inputs</div><div class="section-title">鍑嗗鏈鍒嗘瀽</div><div class="section-copy">涓昏鏂囪礋璐ｅ浘鍍忔暟鎹紝琛ュ厖鏉愭枡璐熻矗 epitope 鍙傝€冧俊鎭€?/div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    paper_upload = st.file_uploader("鈶?涓婁紶涓昏鏂?PDF", type=["pdf"], help="寤鸿鍏堜笂浼犲彧鍚洰鏍囬〉鐨?PDF 杩涜蹇€熸祴璇?)
with right:
    supplement_upload = st.file_uploader("鈶?涓婁紶琛ュ厖鏉愭枡", type=["pdf", "xlsx", "xls", "csv", "tsv"], help="鏀寔鏂囧瓧鍨?PDF 涓庡父瑙佽〃鏍兼牸寮?)

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
    '<div class="ready-box"><strong>杩愯鍓嶆鏌?/strong>'
    + ready_row(paper_ok, f"涓昏鏂囷細{paper_upload.name}" if paper_ok else "绛夊緟涓婁紶涓昏鏂?PDF")
    + ready_row(supplement_ok, f"琛ュ厖鏉愭枡锛歿supplement_upload.name}" if supplement_ok else "绛夊緟涓婁紶琛ュ厖鏉愭枡")
    + ready_row(key_ok, "API Key 宸插氨缁? if key_ok else "绛夊緟鍦ㄥ乏渚у～鍐?API Key")
    + ready_row(page_error is None, f"澶勭悊椤电爜锛歿', '.join(map(str, pages)) if pages else '鏁寸瘒璁烘枃'}")
    + "</div>",
    unsafe_allow_html=True,
)

if st.button("寮€濮嬭繍琛?Stage 1鈥?", type="primary", disabled=not ready, use_container_width=True):
    try:
        st.session_state["last_run_root"] = str(run_pipeline(paper_upload, supplement_upload, api_key, pages))
    except Exception as exc:
        st.exception(exc)

run_root_text = st.session_state.get("last_run_root")
if run_root_text:
    run_root = Path(run_root_text)
    if run_root.exists():
        st.divider()
        st.markdown('<div class="section-kicker">Review</div><div class="section-title">杩囩▼鏍搁獙涓庢渶缁堢粨鏋?/div><div class="section-copy">閫愭妫€鏌ヨ鏂囬〉闈€佺儹鍥炬彁鍙栧€笺€佽ˉ鍏呰〃鏍间笌鍚堝苟璐ㄦ帶銆?/div>', unsafe_allow_html=True)
        tabs = st.tabs(["璁烘枃椤甸潰", "鐑浘璇嗗埆", "鐑浘鏁板€?, "琛ュ厖琛ㄦ牸", "鏈€缁堢粨鏋?, "涓嬭浇"])
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
                st.download_button("涓嬭浇鏈€缁?Excel", final_xlsx.read_bytes(), final_xlsx.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            st.download_button("涓嬭浇瀹屾暣瀹¤鍖?ZIP", zip_directory(run_root), "ELISpot_pipeline_results.zip", mime="application/zip", use_container_width=True)
