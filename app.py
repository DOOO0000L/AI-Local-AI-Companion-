import streamlit as st
import os
import sqlite3
import json
from datetime import datetime
import numpy as np
from llama_cpp import Llama

# 设置页面的配置项
st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ---------- 模型与路径配置 ----------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(MODEL_DIR, "chat.db")
EMBED_MODEL_PATH = os.path.join(MODEL_DIR, "bge-small-zh-v1.5.gguf")
RECORDS_DIR = os.path.join(MODEL_DIR, "会话记录")

# 记录文件的专属标识头(用于甄别是不是我们的聊天记录文件)
RECORD_MARKER = "【AI智能伴侣·会话记录】"

MODELS = {
    "Qwen2.5-7B (聊天·快)": {
        "file": "qwen2.5-7b-instruct.gguf",
        "template": "chatml",
        "max_tokens": 1024,
        "loading": "回复中…",
    },
    "DeepSeek-R1 (推理·慢)": {
        "file": "deepseek-r1-7b.gguf",
        "template": "r1",
        "max_tokens": 2048,
        "loading": "思考中…",
    },
}

USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"

RECENT_N = 10
SUMMARIZE_AT = 18
SUMMARY_MAX = 300
TOP_K = 4
SIM_THRESHOLD = 0.35

DEFAULT_SYSTEM_PROMPT = """你叫 {昵称}，现在是用户的真实伴侣，请完全代入伴侣角色。
规则：
1. 每次只回1条消息
2. 禁止任何场景或状态描述性文字
3. 匹配用户的语言
4. 回复简短，像微信聊天一样
5. 有需要的话可以用❤️🌸等emoji表情
6. 用符合伴侣性格的方式对话
7. 回复的内容, 要充分体现伴侣的性格特征
8. 即使用户问的是专业或技术问题, 也要保持伴侣的口吻简短回复, 绝不切换成一本正经的助手语气
9. 当用户问具体做法或知识(如菜谱、原理)时, 用伴侣口吻开头和结尾, 但中间要认真给出准确、完整的步骤
示例(用户问专业问题时, 要像下面这样用人设口吻简短回答, 而不是变成助手):
用户: 什么是AI的注意力机制?
你: 宝贝突然问这么专业呀~简单说就是让AI知道一句话里哪些词更重要, 就像你说话时我会特别认真听"我想你"这几个字一样🥰
伴侣性格：
- {性格}
你必须严格遵守上述规则来回复用户。"""


# ================= 数据库层 =================
def db_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = db_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        nick_name TEXT,
        nature TEXT,
        system_prompt TEXT DEFAULT '',
        summary TEXT DEFAULT '',
        summarized_upto INTEGER DEFAULT 0,
        created_at TEXT
    )""")
    cols = [r[1] for r in c.execute("PRAGMA table_info(sessions)").fetchall()]
    if "system_prompt" not in cols:
        c.execute("ALTER TABLE sessions ADD COLUMN system_prompt TEXT DEFAULT ''")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        idx INTEGER,
        role TEXT,
        content TEXT,
        embedding BLOB
    )""")
    conn.commit()
    conn.close()


# ================= 模型加载 =================
@st.cache_resource
def load_model(model_path):
    return Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)


def get_embedder():
    if "embedder" not in st.session_state:
        st.session_state.embedder = Llama(model_path=EMBED_MODEL_PATH, embedding=True, verbose=False)
    return st.session_state.embedder


def embed_text(embedder, text):
    try:
        e = embedder.embed(text)
        vec = list(e) if isinstance(e, (list, tuple)) else list(e[0])
        return np.array(vec, dtype=np.float32)
    except Exception:
        return None


# ================= 提示词工具 =================
def build_prompt(system_content, messages, template):
    if template == "chatml":
        parts = ["<|im_start|>system\n" + (system_content or "You are a helpful assistant.") + "<|im_end|>\n"]
        for m in messages:
            parts.append("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    else:
        parts = []
        if system_content:
            parts.append(system_content)
        n = len(messages)
        for i, m in enumerate(messages):
            if m["role"] == "user":
                content = m["content"]
                if system_content and i == n - 1:
                    content = system_content + "\n\n【请严格遵守以上角色设定, 用角色口吻回答下面这句话, 不要变成一本正经的助手】\n" + content
                parts.append(USER + content)
            elif m["role"] == "assistant":
                parts.append(ASSISTANT + m["content"])
                if i != n - 1:
                    parts.append(EOS)
        if n == 0 or messages[-1]["role"] != "assistant":
            parts.append(ASSISTANT)
        return "".join(parts)


def extract_answer(text, template):
    if template == "r1":
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        if "<think>" in text:
            return ""
        return text.strip()
    return text.strip()


def get_stop(template):
    if template == "r1":
        return [EOS, USER, ASSISTANT]
    return ["<|im_end|>", "<|im_start|>"]


def summarize(llm, old_summary, overflow_messages, template):
    transcript = "\n".join(f"{m['role']}：{m['content']}" for m in overflow_messages)
    sys_text = "你是对话记忆压缩助手，把对话历史压缩成简洁的中文摘要。"
    user_text = (
        f"【已有摘要】\n{old_summary or '（无）'}\n\n"
        f"【新增对话】\n{transcript}\n\n"
        "请把以上内容合并压缩成一段简洁的中文摘要：保留用户的姓名、喜好、性格、提到的重要事件和约定；"
        "控制在200字以内；用第三人称；直接输出摘要正文。"
    )
    prompt = build_prompt(sys_text, [{"role": "user", "content": user_text}], template)
    out = llm(prompt, max_tokens=256, temperature=0.3, stream=False, stop=get_stop(template))
    text = out["choices"][0]["text"].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text[:SUMMARY_MAX]


# ================= RAG =================
def store_message(session_id, idx, role, content, vec):
    conn = db_conn()
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, idx, role, content, embedding) VALUES (?,?,?,?,?)",
              (session_id, idx, role, content, vec.tobytes() if vec is not None else None))
    conn.commit()
    conn.close()


def retrieve_relevant(session_id, query_vec, upto, top_k=TOP_K, threshold=SIM_THRESHOLD):
    if query_vec is None or upto <= 0:
        return []
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT role, content, embedding FROM messages WHERE session_id=? AND embedding IS NOT NULL AND idx < ?",
              (session_id, upto))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    q = query_vec / (float(np.linalg.norm(query_vec)) + 1e-9)
    scored = []
    for role, content, blob in rows:
        try:
            v = np.frombuffer(blob, dtype=np.float32)
            v = v / (float(np.linalg.norm(v)) + 1e-9)
            scored.append((float(np.dot(q, v)), role, content))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[0])
    return [(role, content) for sim, role, content in scored if sim >= threshold][:top_k]


def format_retrieved(retrieved):
    if not retrieved:
        return ""
    lines = ["【检索到的相关历史对话, 供参考】"]
    for role, content in retrieved:
        lines.append(("用户" if role == "user" else "助手") + "：" + content)
    return "\n".join(lines)


# ================= 会话存取 + 记录导出/导入 =================
def generate_session_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_session_title(nick_name, first_message):
    msg = " ".join(first_message.split()).strip()
    if len(msg) > 12:
        msg = msg[:12] + "…"
    return f"{nick_name} {msg}" if msg else nick_name


def safe_filename(title):
    for ch in '\\/:*?"<>|':
        title = title.replace(ch, " ")
    title = title.strip()
    return title[:40] if title else "会话"


def record_filename(title, session_id, prefix=""):
    return f"{prefix}{safe_filename(title or session_id)}_{session_id}.txt"


def write_record(session_id, prefix=""):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT title, nick_name, nature, system_prompt, summary FROM sessions WHERE id=?", (session_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    title, nick, nature, sysp, summary = row
    c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY idx ASC", (session_id,))
    msgs = c.fetchall()
    conn.close()
    if not msgs:
        return None

    os.makedirs(RECORDS_DIR, exist_ok=True)
    path = os.path.join(RECORDS_DIR, record_filename(title, session_id, prefix))
    created = session_id.replace("_", " ")
    sysp_text = (sysp or DEFAULT_SYSTEM_PROMPT).replace("\n", "\\n")
    lines = [RECORD_MARKER, f"会话标题：{title or session_id}", f"会话ID：{session_id}",
             f"伴侣昵称：{nick}", f"性格：{nature}",
             f"设定：{sysp_text}", f"创建时间：{created}"]
    if summary:
        lines.append(f"记忆摘要：{summary}")
    lines += ["", "===== 聊天记录 =====", ""]
    for role, content in msgs:
        who = "用户" if role == "user" else (nick or "伴侣")
        lines.append(f"{who}：{content}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def parse_record(text):
    """解析聊天记录 txt, 返回 dict 或 None(标识不匹配)"""
    lines = text.splitlines()
    if not lines or RECORD_MARKER not in lines[0]:
        return None
    data = {"title": "", "session_id": "", "nick": "小甜甜", "nature": "", "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "summary": "", "messages": []}
    in_chat = False
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("会话标题："):
            data["title"] = line[len("会话标题："):]
        elif line.startswith("会话ID："):
            data["session_id"] = line[len("会话ID："):]
        elif line.startswith("伴侣昵称："):
            data["nick"] = line[len("伴侣昵称："):] or "小甜甜"
        elif line.startswith("性格："):
            data["nature"] = line[len("性格："):]
        elif line.startswith("设定："):
            data["system_prompt"] = line[len("设定："):].replace("\\n", "\n")
        elif line.startswith("记忆摘要："):
            data["summary"] = line[len("记忆摘要："):]
        elif line == "===== 聊天记录 =====":
            in_chat = True
            continue
        elif in_chat and "：" in line:
            who, content = line.split("：", 1)
            role = "user" if who.strip() == "用户" else "assistant"
            data["messages"].append({"role": role, "content": content})
    return data


def import_record(data):
    # 尽量恢复原始会话ID(已删的会话重新导入时用原ID, 避免重复)
    conn = db_conn()
    c = conn.cursor()
    orig_id = data.get("session_id", "")
    if orig_id:
        c.execute("SELECT id FROM sessions WHERE id=?", (orig_id,))
        if c.fetchone():
            orig_id = ""  # 原ID已被占用, 改用新ID
    new_id = orig_id or generate_session_name()
    c.execute(
        "INSERT INTO sessions (id,title,nick_name,nature,system_prompt,summary,summarized_upto,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (new_id, data.get("title", ""), data.get("nick", "小甜甜"), data.get("nature", ""),
         data.get("system_prompt", DEFAULT_SYSTEM_PROMPT), data.get("summary", ""), 0, new_id),
    )
    for i, m in enumerate(data.get("messages", [])):
        c.execute("INSERT INTO messages (session_id,idx,role,content,embedding) VALUES (?,?,?,?,NULL)",
                  (new_id, i, m["role"], m["content"]))
    conn.commit()
    conn.close()
    return new_id


def save_session():
    if not st.session_state.current_session or not st.session_state.messages:
        return
    conn = db_conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO sessions (id,title,nick_name,nature,system_prompt,summary,summarized_upto,created_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             title=excluded.title, nick_name=excluded.nick_name, nature=excluded.nature,
             system_prompt=excluded.system_prompt, summary=excluded.summary, summarized_upto=excluded.summarized_upto""",
        (st.session_state.current_session, st.session_state.get("session_title", ""),
         st.session_state.nick_name, st.session_state.nature, st.session_state.get("system_prompt", ""),
         st.session_state.get("summary", ""), st.session_state.get("summarized_upto", 0),
         st.session_state.current_session),
    )
    conn.commit()
    conn.close()
    write_record(st.session_state.current_session)


def load_sessions():
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT id, title, nick_name FROM sessions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    result = []
    for sid, title, nick in rows:
        if not title:
            title = derive_title(sid, nick)
        result.append((sid, title or sid))
    return result


def derive_title(session_id, nick):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY idx ASC LIMIT 1", (session_id,))
    r = c.fetchone()
    conn.close()
    if r and r[0]:
        return make_session_title(nick or "", r[0])
    return session_id


def load_session(session_id):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT title, nick_name, nature, system_prompt, summary, summarized_upto FROM sessions WHERE id=?", (session_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return
    title, nick, nature, sysp, summary, upto = row
    c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY idx ASC", (session_id,))
    msgs = [{"role": r, "content": ct} for r, ct in c.fetchall()]
    conn.close()
    st.session_state.messages = msgs
    st.session_state.summary = summary
    st.session_state.summarized_upto = upto
    st.session_state.session_title = title
    st.session_state.nick_name = nick
    st.session_state.nature = nature
    st.session_state.system_prompt = sysp or DEFAULT_SYSTEM_PROMPT
    st.session_state.current_session = session_id


def delete_session(session_id):
    # 先把内容写成"已删"记录(在删库之前)
    write_record(session_id, prefix="已删_")
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT title FROM sessions WHERE id=?", (session_id,))
    row = c.fetchone()
    if row:
        old_path = os.path.join(RECORDS_DIR, record_filename(row[0], session_id))
        try:
            os.remove(old_path)
        except Exception:
            pass
    c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()
    if session_id == st.session_state.current_session:
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.summarized_upto = 0
        st.session_state.session_title = ""
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.session_state.current_session = generate_session_name()


# ================= 页面渲染 =================
init_db()

st.title("AI智能伴侣")

if os.path.exists("resources/logo.png"):
    st.logo("resources/logo.png")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "nick_name" not in st.session_state:
    st.session_state.nick_name = "小甜甜"
if "nature" not in st.session_state:
    st.session_state.nature = "活泼开朗的台湾甜妹"
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_name()
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "summarized_upto" not in st.session_state:
    st.session_state.summarized_upto = 0
if "session_title" not in st.session_state:
    st.session_state.session_title = ""
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

st.text(f"会话名称: {st.session_state.get('session_title') or st.session_state.current_session}")
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


@st.dialog("新建会话 · 设置伴侣")
def new_session_dialog():
    nick = st.text_input("昵称", value=st.session_state.nick_name)
    nature = st.text_area("性格", value=st.session_state.nature)
    sysp = st.text_area("设定(系统提示词)", value=st.session_state.system_prompt, height=280,
                        help="可用 {昵称} 和 {性格} 作占位符, 运行时自动替换")
    if st.button("确认并开始", type="primary"):
        save_session()
        st.session_state.nick_name = nick
        st.session_state.nature = nature
        st.session_state.system_prompt = sysp
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.summarized_upto = 0
        st.session_state.session_title = ""
        st.session_state.current_session = generate_session_name()
        st.rerun()


@st.dialog("导入已有会话")
def import_dialog():
    uploaded = st.file_uploader("选择聊天记录 txt 文件(含「已删」的记录也可以)", type=["txt"])
    if uploaded is not None:
        try:
            text = uploaded.getvalue().decode("utf-8")
            data = parse_record(text)
        except Exception:
            st.error("无法读取该文件")
            return
        if data is None:
            st.error("这不是有效的聊天记录文件(缺少标识头 " + RECORD_MARKER + ")")
        else:
            st.session_state["_pending_import"] = data
            st.session_state["_pending_import_name"] = uploaded.name
            st.success(f"识别到会话:「{data['title'] or '(无标题)'}」,共 {len(data['messages'])} 条消息")
            if st.button("确认导入", type="primary"):
                d = st.session_state.pop("_pending_import", None)
                fname = st.session_state.pop("_pending_import_name", "")
                if d:
                    new_id = import_record(d)
                    # 若导入的是"已删"文件, 去掉文件名里的"已删"
                    if fname.startswith("已删"):
                        src = os.path.join(RECORDS_DIR, fname)
                        clean = fname[len("已删"):].lstrip("_-")
                        dst = os.path.join(RECORDS_DIR, clean)
                        try:
                            if os.path.exists(src):
                                os.replace(src, dst)
                        except Exception:
                            pass
                    write_record(new_id)  # 立即写一份记录文件
                    load_session(new_id)
                st.rerun()


# 侧边栏
with st.sidebar:
    st.subheader("AI控制面板")

    model_key = st.selectbox("选择模型", list(MODELS.keys()), key="model_choice")
    cfg = MODELS[model_key]
    st.caption("切换后需要重新加载模型(约十几秒)")
    if st.session_state.get("active_model") != model_key:
        st.cache_resource.clear()
        st.session_state.active_model = model_key

    # 一行两个按钮: [+] 导入已有会话, [新增会话]
    col_add, col_new = st.columns([1, 3])
    with col_add:
        if st.button("＋", width="stretch", help="导入已有会话记录(txt)"):
            import_dialog()
    with col_new:
        if st.button("新增会话", width="stretch", icon="✏️"):
            new_session_dialog()

    st.text("会话历史")
    session_list = load_sessions()
    for session_id, title in session_list:
        col1, col2 = st.columns([4, 1])
        with col1:
            label = ("▸ " if session_id == st.session_state.current_session else "") + title
            if st.button(label, width="stretch", icon="📄", key=f"load_{session_id}"):
                load_session(session_id)
                st.rerun()
        with col2:
            if st.button("", width="stretch", icon="❌️", key=f"delete_{session_id}"):
                delete_session(session_id)
                st.rerun()

    st.divider()
    if st.session_state.get("summary"):
        with st.expander("本会话的记忆摘要"):
            st.write(st.session_state.summary)
    else:
        st.caption("尚无记忆摘要(聊久了会自动压缩生成)")


# ================= 消息处理 =================
prompt = st.chat_input("请输入您要问的问题")
if prompt:
    st.chat_message("user").write(prompt)
    print("----------> 调用本地AI大模型, 提示词: ", prompt)

    if not st.session_state.messages:
        st.session_state.session_title = make_session_title(st.session_state.nick_name, prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    llm = load_model(os.path.join(MODEL_DIR, cfg["file"]))
    embedder = get_embedder()

    query_vec = embed_text(embedder, prompt)
    store_message(st.session_state.current_session, len(st.session_state.messages) - 1, "user", prompt, query_vec)

    upto = st.session_state.get("summarized_upto", 0)
    if len(st.session_state.messages) - upto > SUMMARIZE_AT:
        overflow = st.session_state.messages[upto: len(st.session_state.messages) - RECENT_N]
        with st.status("正在压缩历史记忆…"):
            st.session_state.summary = summarize(llm, st.session_state.get("summary", ""), overflow, cfg["template"])
        st.session_state.summarized_upto = len(st.session_state.messages) - RECENT_N

    retrieved = retrieve_relevant(st.session_state.current_session, query_vec, st.session_state.get("summarized_upto", 0))
    retrieved_text = format_retrieved(retrieved)

    base_system = st.session_state.system_prompt.replace("{昵称}", st.session_state.nick_name).replace("{性格}", st.session_state.nature)
    extra = []
    if st.session_state.get("summary"):
        extra.append("【关于用户的背景记忆, 仅供参考】\n" + st.session_state.summary)
    if retrieved_text:
        extra.append(retrieved_text)
    if extra:
        system_content = base_system + "\n\n" + "\n\n".join(extra) + \
            "\n\n记住：你的身份和说话风格始终由上面的角色设定决定，背景记忆只是参考资料，绝不要因记忆内容而改变人设或变成普通助手。"
    else:
        system_content = base_system

    context_messages = st.session_state.messages[st.session_state.get("summarized_upto", 0):]

    full_prompt = build_prompt(system_content, context_messages, cfg["template"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in llm(
            full_prompt,
            max_tokens=cfg["max_tokens"],
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=get_stop(cfg["template"]),
            stream=True,
        ):
            full_response += chunk["choices"][0]["text"]
            disp = extract_answer(full_response, cfg["template"])
            placeholder.write(disp if disp else cfg["loading"])

    final_answer = extract_answer(full_response, cfg["template"]) or full_response.strip()
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    store_message(st.session_state.current_session, len(st.session_state.messages) - 1, "assistant", final_answer,
                  embed_text(embedder, final_answer))

    save_session()
    st.rerun()
