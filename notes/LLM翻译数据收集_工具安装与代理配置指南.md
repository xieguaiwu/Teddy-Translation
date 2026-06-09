# LLM 翻译数据收集 — 工具安装与代理配置

> **依赖**：Python 3.9+，仅需安装 `openai` 包
> **模型**：OpenCode Go 全部 6 个主流模型（禁用推理模式）
> **运行**：`python3 llm_batch_translate.py --models all --api-key "sk-..."`
> **输出**：`translations/{model_slug}/{stem}_{direction}.txt` + `.meta.json` 侧边元数据

---

## 1. 安装

```bash
pip3 install openai
```

如需代理：

```bash
pip3 install openai --proxy http://127.0.0.1:7890
# 或国内镜像
pip3 install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 脚本

`llm_batch_translate.py` 由组长提供。无需其他文件。

## 3. 运行

```bash
# 全部 6 个模型（推荐）
python3 llm_batch_translate.py --models all --api-key "sk-..."

# 选择性运行
python3 llm_batch_translate.py --models deepseek-v4-flash,glm-5.1 --api-key "sk-..."

# 从环境变量读 key
export OPENCODE_API_KEY="sk-..."
python3 llm_batch_translate.py --models all
```

## 4. 输出结构

```
translations/
├── checkpoint.json                               ← 全局 checkpoint
├── deepseek-v4-flash/
│   ├── zh_news_001_zh2en.txt                     ← 译文
│   └── zh_news_001_zh2en.meta.json               ← 元数据
├── glm-5.1/
├── glm-5/
├── deepseek-v4-pro/
├── kimi-k2.6/
└── qwen3.6-plus/
```

每条译文附带 `.meta.json`（含 model_served, token_usage, finish_reason, seed, timestamp 等）。

## 5. 代理配置（如需要）

```bash
# 方案 A：HTTP 代理
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 方案 B：SOCKS5 → proxychains-ng
brew install proxychains-ng
proxychains4 python3 llm_batch_translate.py --models all --api-key "sk-..."
```

验证网络：

```bash
curl -s -o /dev/null -w "%{http_code}" https://opencode.ai/zen/go/v1/models
# 预期：200 或 401
```

## 6. 功能特性

| 特性 | 说明 |
|:----|:-----|
| 多模型 | 6 个模型一次跑完，或选子集 |
| 无推理模式 | 仅翻译输出，无 chain-of-thought |
| 质量门控 | 自动检查语言正确性、截断、拒绝回复、长度比 |
| 并行 | 4 线程并发（可调 `--workers`） |
| Checkpoint | 每翻译一条原子写入，断点续传 |
| 元数据 | 每文件 JSON 侧边记录可复现参数 |
| API Key 安全 | 支持环境变量 `OPENCODE_API_KEY` |

## 7. 费用估算（禁用推理模式后）

| 模型 | 约 ¥ |
|:----|:----:|
| deepseek-v4-flash | 1-2 |
| deepseek-v4-pro | 8-12 |
| glm-5.1 | 5-8 |
| glm-5 | 3-5 |
| kimi-k2.6 | 4-6 |
| qwen3.6-plus | 5-8 |
| **全部 6 个** | **~25-40** |

---

*有问题直接截图终端输出发过来。*
