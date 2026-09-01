# RNAseq Agent — Streamlit UI

`app.py` 是一個聊天介面不含 agent 系統，只透過幾個固定的呼叫點跟「後端」溝通。這份文件說明它現在對接的介面長什麼樣、以及 UI 本身內部比較重要的幾塊邏輯。

## Launch

```bash
pip install streamlit
```

```bash
cd streamlit_ui/
streamlit run app.py
```


## 這個 UI 對後端的介面

app.py 只靠這兩行 import 接後端（[app.py:24-26](app.py#L24-L26)）：

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import build_agent
from agent_trace import format_agent_trace
```

換句話說：**若要接自己的後端，只要在上層目錄提供一個 `main.py`，裡面有一個滿足以下 signature 的 `build_agent()` 就可以**，不必是 deepagents，也不必用 LangGraph（但下面的 `.stream()` 介面是照 LangGraph 的慣例寫的，換掉的話這段呼叫碼也要跟著改）。

### 1. `build_agent(...)` — 建立/重建 agent 的入口

呼叫點在 `get_agent()`（[app.py:289-303](app.py#L289-L303)）：

```python
build_agent(
    root_dir=current_root,            # str | None — 側邊欄掛載的輸入資料夾絕對路徑
    output_dir=current_output,        # str | None — 側邊欄掛載的輸出資料夾絕對路徑
    checkpointer=get_checkpointer(),  # 見下方「checkpointer」
    model=...,                        # 側邊欄「模型設定」三個文字框的值，原封不動傳入
    model_provider=...,
    base_url=...,
) -> agent
```

UI 會在 `(root_dir, output_dir, model/model_provider/base_url)` 任一項改變後、下一次送出訊息時重新呼叫這個函式重建 agent（不是改完側邊欄就立刻重建）。  


### 2. checkpointer — 對話記憶的存取層

 `SqliteSaver`（`get_checkpointer()`存取 `agent_checkpoints.db`：

要接的後端如果要用這個 checkpointer，物件至少要支援：

- `.delete_thread(thread_id)`：`delete_session()`（[app.py:105-111](app.py#L105-L111)）刪除對話時會呼叫。如果後端不需要「刪除對話」這個功能，可以不實作，但按下 UI 的垃圾桶按鈕會噴例外。

### 3. `format_agent_trace(result)` — UI 自帶的 debug 格式化（非後端要實作的介面）

`streamlit_ui/agent_trace.py` 為額外優化功能不影響整體，用途為拿後端回傳的最後一包 result 額外加工顯示用，以抓出agent tool calls, skills calls 等行為。但它對 `result` 的 shape 有隱性假設，換後端時要注意。

### 4. `../AGENTS.md`（純文字檔案，不是程式介面）

側邊欄有個文字框直接讀寫 `Path(__file__).parent.parent / "AGENTS.md"`（[app.py:33](app.py#L33), [app.py:489-500](app.py#L489-L500)）。UI 只負責讀寫這個檔案，**不會**把內容傳給 `build_agent` 或塞進 `.stream()` 的輸入 —— 存檔後只是把 `session_state.agent` 設成 `None` 逼下一輪重建 agent。這個檔案有沒有意義、後端要不要讀它，完全由後端自己決定；對 UI 來說就是一個純文字編輯框。

## 兩個資料庫

`streamlit_ui/` 資料夾下會產生兩個 `.db` 檔（路徑定義在 [app.py:30-31](app.py#L30-L31)）：

| | `chat_history.db` | `agent_checkpoints.db` |
|---|---|---|
| 怎麼建 | 自訂 `sqlite3`（`init_db()`，[app.py:45-78](app.py#L45-L78)） | LangGraph 提供的 `SqliteSaver`（`get_checkpointer()`，[app.py:232-238](app.py#L232-L238)） |
| 用途 | 純顯示層：讓側邊欄「對話列表」能列出/切換/重新渲染過去的聊天記錄。| agent層：讓 agent 用 thread_id 記錄多輪聊天記憶，包含agent狀態和聊天內容|
| Key | `session_id`（`start_new_session()` 產生，[app.py:242](app.py#L242)） | `thread_id`（`= session_id`）[app.py:588](app.py#L588) |

兩者的關聯：**`session_id` 被直接拿來當 `thread_id` 用**（[app.py:588](app.py#L588)），所以這兩個資料庫其實是用同一把 key，各自記錄性質完全不同的東西——一個是給人看的聊天紀錄，一個是給 agent 用的記憶。

會噴出來的坑：
- 兩個 db 目前**只有透過 UI 的刪除鈕才會同步清掉**（`delete_session()` 同時刪 `chat_history.db` 的列 + 呼叫 `get_checkpointer().delete_thread(...)`，[app.py:105-111](app.py#L105-L111)）。如果手動去砍其中一個檔案（例如清空 `chat_history.db` 但沒動 `agent_checkpoints.db`），兩邊就會不同步：UI 側邊欄看起來像全新對話，但只要之後又用到同一個 `session_id`/`thread_id`，agent 那邊其實還記得舊的內容。
- 反過來，砍掉/搬走 `agent_checkpoints.db` 只會讓 agent 失憶（下次同一個 thread_id 變成全新對話），但 `chat_history.db` 裡的聊天泡泡紀錄不會受影響，UI 還是會照常重新渲染出舊的對話內容。

## 側邊欄的其他變數

對應 `st.session_state`（[app.py:207-229](app.py#L207-L229) 有完整註解）：

| 欄位 | key | 說明 |
|---|---|---|
| 輸入資料夾絕對路徑 | `data_dir` | 必須是絕對路徑、存在的資料夾，才會被 UI 接受並傳進 `build_agent(root_dir=...)`。 |
| 輸出資料夾絕對路徑 | `output_dir` | 必須跟 `data_dir` 不同（UI 主動擋，[app.py:475-476](app.py#L475-L476)），不存在會自動 `mkdir`，然後傳進 `build_agent(output_dir=...)`。 |
| 模型設定（model/model_provider/base_url） | `model_config` | 目前只支援ollama |
| `session_id` | — | SQLite 聊天紀錄的分組 key，同時當作 `.stream()` 呼叫的 `thread_id`。 |
| `agent_key` | — | `(data_dir, output_dir, model_config)` 的指紋，`get_agent()` 用它判斷要不要重建 agent。 |


## 目前已知限制 

- 切換對話中途若中斷目前這輪回覆，會收不到agent回傳訊息。
- `output_dir` 目前必須跟 `data_dir` 不同資料夾。
