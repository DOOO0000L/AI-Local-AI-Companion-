# 🤖 AI智能伴侣 (Local AI Companion)

**完全本地部署的 AI 聊天伴侣 —— 断网可用 · 隐私数据永不出本机 · 一键安装 · 带长期记忆**

> 不是又一个"套壳云 API"的聊天工具。它 100% 在你自己的电脑上运行:模型在本地、推理在本地、数据也永远在本地。

---

## ✨ 为什么与众不同

| 特性 | 说明 |
|---|---|
| 🔒 **隐私安全** | 聊天记录、个人数据**永不出本机**,不经过任何服务器 |
| 📴 **断网可用** | 完全离线运行,没网也能聊 |
| 🖱️ **一键安装** | 双击安装脚本,自动装环境、自动下模型,小白也能用 |
| 🧠 **长期记忆** | 三层记忆(滑动窗口 + 滚动摘要 + RAG 向量检索),聊再久也不失忆 |
| 🎭 **双模型切换** | Qwen2.5-7B(快) / DeepSeek-R1(慢但能推理)一键切换 |
| 💾 **数据可带走** | 每个会话自动导出成可读 txt,可备份、可转移、删除可恢复 |

---

## 一、环境要求

- **Windows 10/11**
- **NVIDIA 显卡**(显存 8GB 以上,RTX 3060/4060 更佳)
- **Python 3.11+**(安装时勾选 "Add python.exe to PATH")
- 磁盘约 **15GB**(模型约 9GB)

---

## 二、一键安装(推荐)

```text
1. 下载本项目
2. 双击「一键安装.bat」   ← 自动建环境、装依赖、下模型(9GB,支持断点续传)
3. 双击「启动AI伴侣.bat」 ← 开始使用
```

> 安装只需一次,以后每次直接双击「启动AI伴侣.bat」。

---

## 三、手动安装(备选)

```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit numpy
pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu132/llama_cpp_python-0.3.35-py3-none-win_amd64.whl"
python download_models.py
```

---

## 四、使用

- 左侧栏选模型、新建会话、删除会话、导入记录
- 底部输入框聊天
- 聊天记录自动导出到 `会话记录/` 文件夹(可读 txt)

---

## 五、文件结构

```
├── app.py                 # 主程序
├── 一键安装.bat            # 一键安装
├── 启动AI伴侣.bat          # 双击启动
├── download_models.py      # 模型下载脚本
├── requirements.txt        # 依赖清单
├── chat.db                 # 数据库(自动生成, 含隐私, 勿上传)
├── 会话记录/                # 导出的聊天记录(自动生成)
└── *.gguf                  # 模型文件(下载脚本生成)
```

---

## 六、常见问题

**Q: 启动报 "Could not find module ... llama.dll (or one of its dependencies)"**

缺 CUDA 运行库 DLL(`cublas64_13.dll`、`cublasLt64_13.dll`)。解决:安装 [CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-toolkit),或把这几个 DLL 复制到 `venv\Lib\site-packages\llama_cpp\lib\`。

**Q: 没有 NVIDIA 显卡,想用 CPU 跑**

把 `app.py` 里的 `n_gpu_layers=-1` 改成 `n_gpu_layers=0`,并换装 CPU 版 llama-cpp-python。

**Q: 模型下载慢/失败**

`download_models.py` 走 hf-mirror 国内镜像,支持断点续传,中断重跑即可。

---

## 七、隐私提示

`chat.db` 和 `会话记录/` 含真实聊天内容,请勿提交到公开仓库(已在 `.gitignore` 排除)。
