"""
Streamlit UI for the RNA-seq deepagent.

Features:
  1. Chat window wired to the deepagents agent in ../main.py
  2. Fully custom sidebar: mount/unmount a folder, browse files, preview content
     — the mounted folder also becomes the agent's real filesystem backend root
  3. Chat history logging to a local SQLite file

Launch:
(deepagents) joan@cjoan90417-pc02 streamlit_ui % streamlit run app.py
"""

import gzip
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import build_agent  # noqa: E402
from agent_trace import format_agent_trace  # noqa: E402

st.set_page_config(page_title="RNAseq Agent UI Demo", layout="wide")

DB_PATH = Path(__file__).parent / "chat_history.db"
CHECKPOINT_DB_PATH = Path(__file__).parent / "agent_checkpoints.db"
LOG_PATH = Path(__file__).parent / "chat_history.log"
AGENTS_MD_PATH = Path(__file__).parent.parent / "AGENTS.md"
PREVIEW_LINES = 4

DEFAULT_MODEL_CONFIG = {
    "model": "",
    "model_provider": "ollama",
    "base_url": "",
}

# --- Local chat-history persistence (plain SQLite) --------------------------


def init_db() -> None:

    
    """建兩張表：messages 和 sessions"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            root_dir TEXT,
            output_dir TEXT
        )
        """
    )
    # 舊資料庫可能還沒有這兩欄，補上去才能存「執行歷程」跟「完整訊息紀錄」
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    if "progress_log" not in existing_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN progress_log TEXT DEFAULT ''")
    if "debug_output" not in existing_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN debug_output TEXT DEFAULT ''")
    conn.commit()
    conn.close()


def create_session(session_id: str) -> None:
    """Register a brand-new conversation so it immediately shows up in the
    sidebar list, even before any message has been logged for it."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at, root_dir, output_dir) "
        "VALUES (?, NULL, ?, ?, '', '')",
        (session_id, timestamp, timestamp),
    )
    conn.commit()
    conn.close()


def list_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT session_id, title, created_at, updated_at, root_dir, output_dir "
        "FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return rows


def delete_session(session_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    get_checkpointer().delete_thread(session_id)


def log_message(
    session_id: str,
    role: str,
    content: str,
    root_dir: str = "",
    output_dir: str = "",
    progress_log: str = "",
    debug_output: str = "",
) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at, progress_log, debug_output) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, timestamp, progress_log, debug_output),
    )
    if role == "user":
        row = conn.execute(
            "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row and not row[0]:
            title = content.replace("\n", " ").strip()[:24] or "(空白訊息)"
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id)
            )
    conn.execute(
        "UPDATE sessions SET updated_at = ?, root_dir = ?, output_dir = ? WHERE session_id = ?",
        (timestamp, root_dir, output_dir, session_id),
    )
    conn.commit()
    conn.close()

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [session:{session_id[:8]}] {role}: {content}\n")


def fetch_history(session_id: str, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT session_id, role, content, created_at, progress_log, debug_output "
        "FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return list(reversed(rows))


# --- Data folder viewer ------------------------------------------------------


def human_size(num_bytes: float) -> str:
    """Convert a byte count into a human-readable string with units(KB/MB/GB/TB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def list_data_files(data_dir: str):
    if not data_dir:
        return []
    path = Path(data_dir)
    if not path.is_dir():
        return []
    return sorted(f for f in path.iterdir() if f.is_file())


def preview_gz_text(path: Path, n_lines: int = PREVIEW_LINES) -> str:
    """Preview the first n_lines of a gzipped text file, returning a string."""
    lines = []
    with gzip.open(path, "rt", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def preview_text_file(path: Path, n_lines: int = PREVIEW_LINES) -> str:
    """Preview the first n_lines of a text file, returning a string."""
    lines = []
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


# --- Session init -------------------------------------------------------------
#
# st.session_state keys used in this app (persist across Streamlit reruns,
# scoped to one browser session; each key is set once on first access below):
#
#   session_id    str            unique id for this browser session; doubles as
#                                 the SQLite grouping key (log_message/fetch_history)
#                                 and the LangGraph thread_id (agent memory)
#   messages      list[dict]     chat bubbles shown in the UI, {"role", "content",
#                                 "progress_log", "debug_output"} — the latter two
#                                 back the two collapsible expanders per turn
#   data_dir      str            mounted input folder's absolute path, "" = unmounted
#   output_dir    str            mounted output/workspace folder's absolute path, "" = unmounted
#                                 — must be a folder separate from data_dir (see mount_clicked below);
#                                 this is where the QC/DEA pipeline writes its reports
#   preview_file  str | None     path of the file currently shown in the sidebar preview
#   model_config  dict           active {model, model_provider, base_url}
#   agent         Agent | None   cached deepagents agent instance
#   agent_key     tuple | object fingerprint of (data_dir, output_dir, model_config)
#                                 used to detect staleness and rebuild `agent` when any change
#
# The LangGraph checkpointer itself is NOT session-scoped — it's a single
# SqliteSaver shared across all browser sessions (see get_checkpointer() below),
# so switching to a past session_id/thread_id here can actually read back its
# agent memory instead of starting fresh.


@st.cache_resource
def get_checkpointer() -> SqliteSaver:
    """Return a single SqliteSaver shared across all browser sessions, 
    so switching to a past session_id/thread_id can read back its agent memory."""
    
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    return SqliteSaver(conn)


def start_new_session() -> None:
    new_id = str(uuid.uuid4())
    create_session(new_id)
    st.session_state.session_id = new_id
    st.session_state.messages = []
    st.session_state.data_dir = ""
    st.session_state.output_dir = ""
    st.session_state.preview_file = None
    st.session_state.agent_key = object()  # force get_agent() to rebuild for the new dirs


def switch_to_session(session_id: str, root_dir: str, output_dir: str) -> None:
    st.session_state.session_id = session_id
    st.session_state.messages = [
        {
            "role": role,
            "content": content,
            "progress_log": progress_log or "",
            "debug_output": debug_output or "",
        }
        for _sid, role, content, _ts, progress_log, debug_output in fetch_history(session_id, limit=200)
    ]
    st.session_state.data_dir = root_dir or ""
    st.session_state.output_dir = output_dir or ""
    st.session_state.preview_file = None
    st.session_state.agent_key = object()  # force get_agent() to rebuild for the new dirs


if "session_id" not in st.session_state:
    # first visit in this browser session: mint a new id and create the DB tables
    init_db()
    st.session_state.session_id = str(uuid.uuid4())
    create_session(st.session_state.session_id)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_dir" not in st.session_state:
    st.session_state.data_dir = ""
if "output_dir" not in st.session_state:
    st.session_state.output_dir = ""
if "preview_file" not in st.session_state:
    st.session_state.preview_file = None
if "model_config" not in st.session_state:
    st.session_state.model_config = dict(DEFAULT_MODEL_CONFIG)
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.agent_key = object()  # sentinel: never equals a real key


def get_agent():
    """(Re)build the agent whenever the mounted folders or model config
    change, so its system prompt always matches what the sidebar shows."""
    current_root = st.session_state.data_dir or None
    current_output = st.session_state.output_dir or None
    agent_key = (current_root, current_output, tuple(st.session_state.model_config.values()))
    if st.session_state.agent is None or st.session_state.agent_key != agent_key:
        st.session_state.agent = build_agent(
            root_dir=current_root,
            output_dir=current_output,
            checkpointer=get_checkpointer(),        # SQLite 連線
            **st.session_state.model_config,
        )
        st.session_state.agent_key = agent_key
    return st.session_state.agent


# --- Progress chunk rendering --------------------------------------------------
#
# render_progress_chunk()（322-333 行）負責把 nodes.py 裡用 get_stream_writer() 
# 推出來的「自訂 stream chunk」（例如 FastQC/MultiQC 開始/完成、reads 驗證中）
# 轉成中文進度字串（用到 _QC_STAGE_LABELS 對照表）。
# 它在第 580 行被呼叫，是主聊天迴圈裡 stream_mode="custom" 分支的處理函式，
# 讓 UI 在 agent 還沒跑完長任務時就能顯示中間進度。

_QC_STAGE_LABELS = {
    "fastqc_start": "▶️ 開始執行 FastQC",
    "fastqc_done": "✅ FastQC 完成",
    "multiqc_start": "▶️ 開始彙整 MultiQC 報告",
    "multiqc_done": "✅ QC 完成",
}

# 兩個持久化摺疊區塊的固定標題：不管這一輪成功/失敗都用同一個字樣，
# 這樣歷史訊息重新渲染時可以直接依標題判斷要不要顯示。
PROGRESS_EXPANDER_TITLE = "⏱️ 執行歷程"
DEBUG_EXPANDER_TITLE = "🔍 這一輪做了什麼"


def render_progress_chunk(chunk: dict) -> str:
    stage = chunk.get("stage")
    status = chunk.get("status")
    if stage == "verify_reads":
        if status == "stage_done":
            return "✅ 讀取驗證完成"
        icon = "🔬" if status == "start" else "✅"
        verb = "驗證中..." if status == "start" else "驗證完成"
        return f"{icon} {chunk.get('sample_id')} 的 {chunk.get('detail')} {verb}"
    if stage == "qc":
        return _QC_STAGE_LABELS.get(status, f"{stage}: {status}")
    return str(chunk)

# --- Sidebar: fully custom layout --------------------------------------------

with st.sidebar:
    st.header("💬 對話")

    if st.button("+ 新對話", use_container_width=True):
        start_new_session()
        st.rerun()

    for sid, title, created_at, updated_at, root_dir, output_dir in list_sessions():
        is_active = sid == st.session_state.session_id
        label = ("▶ " if is_active else "") + (title or "(尚無訊息)")
        row_switch, row_delete = st.columns([5, 1])
        with row_switch:
            if st.button(
                label, key=f"switch_{sid}", use_container_width=True, disabled=is_active
            ):
                switch_to_session(sid, root_dir, output_dir)
                st.rerun()
            st.caption(updated_at)
        with row_delete:
            if st.button("🗑️", key=f"delete_{sid}", use_container_width=True):
                delete_session(sid)
                if is_active:
                    start_new_session()
                st.rerun()

    st.divider()

    st.header("⚙️ 模型設定")

    model_input = st.text_input("model", value=st.session_state.model_config["model"])
    provider_input = st.text_input(
        "model_provider", value=st.session_state.model_config["model_provider"]
    )
    base_url_input = st.text_input(
        "base_url", value=st.session_state.model_config["base_url"]
    )

    if st.button("套用模型設定", use_container_width=True):
        st.session_state.model_config = {
            "model": model_input.strip(),
            "model_provider": provider_input.strip(),
            "base_url": base_url_input.strip(),
        }
        st.success("已套用，下一則訊息會用新設定重建 agent。")

    st.caption(
        f"目前使用：`{st.session_state.model_config['model']}` "
        f"@ `{st.session_state.model_config['model_provider']}` "
        f"（{st.session_state.model_config['base_url']}）"
    )

    st.divider()

    st.header("📁 輸入資料集")

    new_path = st.text_input(
        "資料夾絕對路徑",
        value=st.session_state.data_dir,
        placeholder="/absolute/path/to/folder",
    )
    col_mount, col_unmount = st.columns(2)
    with col_mount:
        mount_clicked = st.button("掛載", use_container_width=True)
    with col_unmount:
        unmount_clicked = st.button("卸載", use_container_width=True)

    if mount_clicked:
        resolved = Path(new_path).expanduser()
        if resolved.is_dir():
            st.session_state.data_dir = str(resolved)
            st.session_state.preview_file = None
        else:
            st.error(f"路徑不存在或不是資料夾：`{resolved}`")

    if unmount_clicked:
        st.session_state.data_dir = ""
        st.session_state.preview_file = None

    st.divider()

    if st.session_state.data_dir:
        st.caption(f"目前掛載：`{st.session_state.data_dir}`")
        files = list_data_files(st.session_state.data_dir)
        if not files:
            st.info("資料夾是空的，或路徑已經失效。")
        else:
            for f in files:
                stat = f.stat()
                size_str = human_size(stat.st_size)
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
                row_left, row_right = st.columns([4, 1])
                with row_left:
                    st.markdown(f"**{f.name}**  \n{size_str} · {mtime}")
                with row_right:
                    if st.button("👀", key=f"preview_{f.name}"):
                        st.session_state.preview_file = str(f)
    else:
        st.info("尚未掛載任何資料夾。")

    if st.session_state.preview_file:
        st.divider()
        preview_path = Path(st.session_state.preview_file)
        st.subheader(f"預覽：{preview_path.name}")
        if not preview_path.exists():
            st.error("檔案已經不存在了。")
        else:
            try:
                if preview_path.suffix == ".gz":
                    snippet = preview_gz_text(preview_path)
                else:
                    snippet = preview_text_file(preview_path)
                st.code(snippet or "(空白)")
            except Exception as e:
                st.error(f"預覽失敗：{e}")

    st.divider()

    st.header("📤 輸出資料夾")
    st.caption("檔案、分析報告將儲存於此，必須是跟輸入資料夾不同的資料夾。")

    new_output_path = st.text_input(
        "輸出資料夾絕對路徑",
        value=st.session_state.output_dir,
        placeholder="/absolute/path/to/output_folder",
    )
    col_out_mount, col_out_unmount = st.columns(2)
    with col_out_mount:
        output_mount_clicked = st.button("設定輸出資料夾", use_container_width=True)
    with col_out_unmount:
        output_unmount_clicked = st.button("清除", use_container_width=True)

    if output_mount_clicked:
        resolved_output = Path(new_output_path).expanduser()
        if st.session_state.data_dir and resolved_output == Path(st.session_state.data_dir).expanduser():
            st.error("輸出資料夾不能跟輸入資料夾相同，請另外選一個資料夾。")
        else:
            resolved_output.mkdir(parents=True, exist_ok=True)
            st.session_state.output_dir = str(resolved_output)

    if output_unmount_clicked:
        st.session_state.output_dir = ""

    if st.session_state.output_dir:
        st.caption(f"目前輸出資料夾：`{st.session_state.output_dir}`")

    st.divider()

    with st.expander("📝 編輯 agent 記憶"):
        agents_md_text = st.text_area(
            "AGENTS.md 內容",
            value=AGENTS_MD_PATH.read_text(encoding="utf-8") if AGENTS_MD_PATH.exists() else "",
            height=300,
            label_visibility="collapsed",
            key="agents_md_editor",
        )
        if st.button("儲存 AGENTS.md", use_container_width=True):
            AGENTS_MD_PATH.write_text(agents_md_text, encoding="utf-8")
            st.session_state.agent = None  # force rebuild so the agent picks up the new memory
            st.success("已儲存，下一則訊息會用新的 AGENTS.md 重建 agent。")

# --- Main chat area -----------------------------------------------------------

# --- Title & status caption ---
st.title("RNAseq Agent")
# ststus caption
st.caption(
    f"已連接 deepagents（thread_id = `{st.session_state.session_id[:8]}`）"
    + (f" · 檔案系統根目錄：`{st.session_state.data_dir}`" if st.session_state.data_dir else " · 尚未掛載輸入資料夾，agent 目前看不到真實檔案")
    + (f" · 輸出資料夾：`{st.session_state.output_dir}`" if st.session_state.output_dir else " · 請設定輸出資料夾")
)

# --- Render chat history (re-runs every rerun) ---
# 每次 rerun 都把 st.session_state.messages重新渲染一次
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 執行歷程跟完整訊息紀錄各自放進獨立的摺疊區，並存進 session_state/DB
        if msg.get("progress_log"):
            with st.expander(PROGRESS_EXPANDER_TITLE):
                # 渲染歷程訊息
                st.markdown(msg["progress_log"])
        if msg.get("debug_output"):
            with st.expander(DEBUG_EXPANDER_TITLE):
                # 渲染完整訊息紀錄: 這一輪做了什麼windows
                st.markdown(msg["debug_output"])
        # 渲染 agent/user 的訊息泡泡
        st.markdown(msg["content"])

# --- Sticky chat-input CSS ---
# chatting area
st.markdown(
    """
    <style>
    div[data-testid="stForm"] {
        position: sticky;
        bottom: 0;
        z-index: 999;
        background-color: var(--background-color, #ffffff);
        padding-top: 0.75rem;
        padding-bottom: 0.5rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* 列印時隱藏側邊欄，讓主內容撐滿整頁 */
    @media print {
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        section[data-testid="stMain"] {
            width: 100% !important;
            margin-left: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Chat input form ---
# chat input form: text area + submit button
with st.form("chat_form", clear_on_submit=True):
    input_col, button_col = st.columns([6, 1])
    with input_col:
        prompt = st.text_area(
            "輸入訊息",
            placeholder="輸入訊息...（Enter 換行，按「傳送」送出）",
            height=68,
            label_visibility="collapsed",
        )
    with button_col:
        submitted = st.form_submit_button("傳送", use_container_width=True)

# --- Handle new message: stream agent response, persist, rerun ---
# when the user submits a message, append it to the chat history and invoke the agent
if submitted and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    log_message(
        st.session_state.session_id, "user", prompt,
        st.session_state.data_dir, st.session_state.output_dir,
    )
    
    # user聊天泡泡渲染
    with st.chat_message("user"):
        st.markdown(prompt)

    # -- Stream agent response --
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    with st.chat_message("assistant"):
        status_box = st.status("Agent 執行中...", expanded=True)
        debug_output = ""
        progress_lines = []  # 中途步驟訊息，跑完後要存進 session_state/DB 才不會在 rerun 後消失
        result = None
        try:
            agent = get_agent()
            for mode, chunk in agent.stream(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "root_dir": st.session_state.data_dir,
                    "output_dir": st.session_state.output_dir,
                },
                config=config,
                stream_mode=["values", "custom"],
            ):
                if mode == "custom":
                    line = render_progress_chunk(chunk)
                    status_box.write(line)
                    progress_lines.append(line)
                else:
                    result = chunk
            status_box.update(label="Agent 執行完成", state="complete", expanded=False)

            reply = result["messages"][-1].content

            debug_output = format_agent_trace(result)
            print(f"\n===== Agent trace =====\n{debug_output}\n========================\n")

        except Exception as e:
            status_box.update(label="Agent 執行失敗", state="error", expanded=False)
            reply = f"Agent 執行失敗：{e}"

        if not reply or not reply.strip():
            reply = (
                "⚠️ 這一輪 agent 沒有產生文字回覆（模型回傳了空白內容）。"
                "請展開下方「這一輪做了什麼」查看實際執行紀錄，"
                "或直接重新輸入一次訊息再試一次。"
            )
        progress_log = "\n\n".join(progress_lines)

        # 執行歷程跟完整訊息紀錄各自放進獨立的摺疊區，並存進 session_state/DB
        # 執行歷程跟完整訊息紀錄渲染（見下方 append/log_message）— 放在 reply 前面
        if progress_log:
            with st.expander(PROGRESS_EXPANDER_TITLE):
                st.markdown(progress_log)
        if debug_output:
            with st.expander(DEBUG_EXPANDER_TITLE):
                st.markdown(debug_output)

        # agent聊天泡泡渲染
        st.markdown(reply)
        

    # -- Persist assistant turn & rerun --
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "progress_log": progress_log,
        "debug_output": debug_output,
    })
    log_message(
        st.session_state.session_id, "assistant", reply,
        st.session_state.data_dir, st.session_state.output_dir,
        progress_log=progress_log, debug_output=debug_output,
    )
    # the "💬 對話" sidebar section renders before this point in the script,
    # so without a rerun it would keep showing the pre-turn title/timestamp
    # until some unrelated interaction happened to trigger the next one
    st.rerun()


