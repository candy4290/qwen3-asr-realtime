# Qwen3-ASR Realtime WebSocket 服务

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) vLLM 流式 API 的 WebSocket 实时语音识别服务。

## 功能

- WebSocket 端点：`ws://host:port/v1/realtime`
- 客户端推送 PCM16 base64 音频包，服务端返回流式/最终识别结果
- 内置浏览器测试页：`http://host:port/demo.html`
- 健康检查：`GET /health`

## 环境要求

- Python 3.12（推荐通过 Conda 创建环境）
- NVIDIA GPU + CUDA
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda

## 安装

### 1. 创建 Conda 环境并安装 ASR 推理栈

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr[vllm]
```

> `qwen-asr[vllm]` 会拉取 vLLM、PyTorch 等大体积依赖，首次安装耗时较长，请确保 GPU 驱动与 CUDA 可用（`nvidia-smi`）。
>
> **版本注意**：若启动时在 `vllm/renderers/hf.py` 报错，或日志出现 `BaseMultiModalProcessor._get_data_parser`，说明 vLLM 版本过高，请执行上面的版本锁定命令后重启。

### 2. 安装本项目 Web 服务依赖

```bash
cd qwen-asr-realtime
pip install -r requirements.txt
```

### 3. 预下载模型（推荐）

**建议在联网环境下先把模型下载到本地缓存**，后续启动可离线、也避免首次启动长时间等待。

```bash
conda activate qwen3-asr

# 安装 HuggingFace CLI（若尚未安装）
pip install -U huggingface_hub

# 下载模型（与 .env 中 ASR_MODEL_PATH 保持一致）
huggingface-cli download Qwen/Qwen3-ASR-0.6B
# 或
# huggingface-cli download Qwen/Qwen3-ASR-1.7B
```

默认缓存目录：`~/.cache/huggingface/hub/`

> **注意**：`ASR_MODEL_PATH` 请填写 **HuggingFace 模型 ID**（如 `Qwen/Qwen3-ASR-0.6B`），不要填写 `snapshots/` 缓存路径；离线时会自动从上述缓存目录加载。

### 4. 配置环境变量

```bash
# 按需编辑 .env
```

可选环境变量（或 `.env`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_MODEL_PATH` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace 模型 ID（与预下载时一致） |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM 显存占用（相对单卡总显存） |
| `MAX_MODEL_LEN` | `8192` | vLLM 最大上下文；模型默认 65536 会占用大量 KV cache |
| `MAX_NEW_TOKENS` | `32` | 流式 max tokens |
| `HF_HUB_OFFLINE` | `false` | 设为 `true` 时仅从本地缓存加载模型，不访问网络 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `9800` | 监听端口 |

## 启动

确保已激活 Conda 环境，且 **步骤 3 已预下载与 `ASR_MODEL_PATH` 对应的模型**。

### 离线启动（推荐）

在 `.env` 中开启离线模式：

```env
HF_HUB_OFFLINE=true
ASR_MODEL_PATH=Qwen/Qwen3-ASR-0.6B
```

然后启动：

```bash
conda activate qwen3-asr
cd qwen-asr-realtime
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 9800
```

也可临时指定（不写进 `.env`）：

```bash
HF_HUB_OFFLINE=1 python -m app.main
```

### 联网启动

若未预下载，首次启动会自动从 HuggingFace 拉取模型（需网络畅通）。`.env` 中保持 `HF_HUB_OFFLINE=false` 或不设置即可。

浏览器打开（测试页）：`http://localhost:9800/demo.html`

## 常见问题

### 测试页报错：`Cannot read properties of undefined (reading 'getUserMedia')`

通过 **HTTP + IP 地址**（如 `http://10.0.2.101:9800`）打开 `demo.html` 时会出现此错误。浏览器要求麦克风 API 运行在安全上下文中。

**解决办法（任选其一）：**

1. **本机调试**：使用 `http://localhost:9800/demo.html`（不要用 IP）
2. 使用https访问
3. **Chrome 临时调试**：地址栏输入 `chrome://flags`，搜索 `Insecure origins treated as secure`，将你的 `http://IP:端口` 加入列表后重启浏览器（仅开发用）

### 启动报错：`Repo id must be in the form 'repo_name'...`

日志里出现 `Error retrieving safetensors: Repo id must be in the form...` 且路径为 HuggingFace 缓存目录（如 `~/.cache/huggingface/hub/models--Qwen--...`）时，**通常只是 vLLM 的警告**，可忽略。若服务随后仍崩溃，请看下一条。

### 启动报错：`vllm/renderers/hf.py` / `_get_data_parser`

这是 **vLLM 0.16+ 与 Qwen3-ASR 不兼容** 导致的，与模型路径无关。修复方式：

```bash
conda activate qwen3-asr
pip install "vllm>=0.14.0,<0.16.0"
python -m app.main
```

可用 `pip show vllm` 确认版本在 0.14.x 或 0.15.x。

### 启动报错：显存不足（`Free memory on device cuda:0 ... is less than desired GPU memory utilization`）

vLLM 启动时会检查 **当前空闲显存** 是否 ≥ `GPU_MEMORY_UTILIZATION × 显卡总显存`。例如 16GB 显卡、`.env` 里 `GPU_MEMORY_UTILIZATION=0.8` 时，需要约 **12.5 GiB 空闲**；若 `nvidia-smi` 显示只剩 3 GiB，就会报此错。

**优先方案：释放被占用的显存**

```bash
nvidia-smi          # 查看占用 GPU 的进程 PID
kill <pid>          # 关闭不需要的推理/训练/桌面程序后重启服务
```

**仍不够时的配置调整（编辑 `.env`）**

1. 降低占用比例（按「当前空闲显存 ÷ 总显存」估算，略留余量）：

   ```env
   GPU_MEMORY_UTILIZATION=0.2
   ```

2. 换更小模型（空闲显存 < 6 GiB 时建议）：

   ```env
   ASR_MODEL_PATH=Qwen/Qwen3-ASR-0.6B
   GPU_MEMORY_UTILIZATION=0.4
   ```

> 1.7B 模型权重本身约需 3–4 GiB，仅调低 `GPU_MEMORY_UTILIZATION` 无法替代「先腾出足够空闲显存」；在 16GB 单卡上跑 1.7B，通常需要关闭其它 GPU 进程后再用默认 `0.8`。

### 启动报错：KV cache 不足（`65536 ... KV cache is needed ... available KV cache memory`）

模型 HuggingFace 配置里默认 `max_model_len=65536`，vLLM 会据此预分配 KV cache（约需 7+ GiB）。若 `GPU_MEMORY_UTILIZATION` 较低或显存已被占用，剩余 KV cache 预算可能只有 1 GiB 左右，就会报此错。

**推荐修复（实时 ASR 足够）：** 在 `.env` 中限制上下文长度：

```env
MAX_MODEL_LEN=8192
```

仍不够时可改为 `4096`，或换 `Qwen/Qwen3-ASR-0.6B`。若显存充足，也可提高 `GPU_MEMORY_UTILIZATION` 并关闭其它 GPU 进程。

### 模型路径建议

`ASR_MODEL_PATH` 推荐使用 HuggingFace repo id（如 `Qwen/Qwen3-ASR-0.6B`），不要手动填写 `snapshots/` 缓存路径。离线启动前请确认已执行 `huggingface-cli download <同一模型 ID>`。

## WebSocket 协议摘要

### 连接

```
ws://localhost:9800/v1/realtime
```

### 事件流

```
Server → session.created
Client → session.update { language?, hotwords?, chunk_size_sec?, ... }
Server → session.updated

Client → input_audio_buffer.append { audio: "<pcm16 base64>" }
Client → input_audio_buffer.commit { final: false }
Server → transcription.delta { text, language }   // text 为累计全文

Client → input_audio_buffer.commit { final: true }
Server → transcription.done { text, language }
```

### 音频格式

- 16kHz、单声道、PCM16 little-endian、base64 编码

### 错误

```json
{ "type": "error", "error": "...", "code": "invalid_audio" }
```

错误码：`invalid_message` | `invalid_audio` | `session_not_ready` | `inference_error` | `internal_error`

## 项目结构

```
app/
├── main.py           # FastAPI 入口、/health、静态页
├── config.py         # 环境变量配置
├── protocol/         # 事件模型与错误码
├── session/          # 会话状态机、音频缓冲
├── asr/              # Qwen3 适配器（to_thread + 模型锁）
└── ws/               # WebSocket 处理器
static/
└── demo.html         # 浏览器测试页
```

## License

Apache-2.0（与 Qwen3-ASR 一致）
