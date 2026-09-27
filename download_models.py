import urllib.request, os, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (本地文件名, 下载地址)
MODELS = [
    ("qwen2.5-7b-instruct.gguf",
     "https://hf-mirror.com/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("deepseek-r1-7b.gguf",
     "https://hf-mirror.com/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
    ("bge-small-zh-v1.5.gguf",
     "https://hf-mirror.com/CompendiumLabs/bge-small-zh-v1.5-gguf/resolve/main/bge-small-zh-v1.5-f16.gguf"),
]


def download_one(url, dest):
    """带断点续传的下载, 返回是否完成"""
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-", "User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60)
    total = done
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        total = int(cr.rsplit("/", 1)[1])
    if done >= total:
        return True
    t0 = time.time()
    with open(dest, "ab") as f:
        while True:
            try:
                chunk = resp.read(4 * 1024 * 1024)
            except Exception:
                break
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = done / total * 100
            spd = done / (time.time() - t0) / 1e6
            print(f"\r    {pct:.1f}%  {done/1e9:.2f}/{total/1e9:.2f} GB  {spd:.1f}MB/s", end="", flush=True)
    print()
    return done >= total


def main():
    for name, url in MODELS:
        print(f"下载 {name} ...")
        for _ in range(50):  # 最多重试 50 次(断点续传, 不会重复下载)
            try:
                if download_one(url, name):
                    print(f"  {name} 完成")
                    break
            except Exception as e:
                print(f"\r  中断({type(e).__name__}), 2 秒后重试...", end="", flush=True)
                time.sleep(2)
        else:
            print(f"  [警告] {name} 未下载完整, 重新运行本脚本可继续")
    print("\n全部模型处理完毕")


if __name__ == "__main__":
    main()
