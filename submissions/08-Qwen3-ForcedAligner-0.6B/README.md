# Qwen3-ForcedAligner-0.6B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> **赛题归属：T04 语音（强制对齐，计件）**。验收口径：音频 + 文本 → 词级/字级时间戳，指标为时间戳偏移 AAS(ms)。

## 1. 简介

记录 `Qwen/Qwen3-ForcedAligner-0.6B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（bf16）上跑通
「音频 + 文本 → 词级/字级时间戳」的适配结果。

该模型与 Qwen3-ASR 共用音频塔与语言主干，把 LM head 换成 5000 类的时间格分类头：在 `<timestamp>` token
位置预测时间格索引，`时间(ms) = 索引 × timestamp_segment_time`，本模型 `timestamp_segment_time = 80`（每格 80 ms，
5000 格 ≈ 400 s，与「最长 5 分钟语音」对得上）。

与官方部署方式的核心差异：官方用 `qwen_asr` 包的 `Qwen3ForcedAligner.from_pretrained(...).align(...)`；
本仓库在没有该包的环境里，用 **vLLM 的 pooling runner** 复刻同一条路径（算法逐行对照
`qwen_asr/inference/qwen3_forced_aligner.py` 的 `encode_timestamp` / `fix_timestamp` / `parse_timestamp`）。

- 权重：https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 卡型 | K500SM_AI 64 GB × 1（计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM |
| Python / torch | 3.10.21 / 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| vllm / triton | 0.21.0+das / 3.6.0+das185 |
| 模型路径 | `/root/hygon/models/fa06` |
| 显存占用 | 引擎初始化 8.74 s，`gpu-memory-utilization 0.35` |

## 3. 适配优化方案

### 3.1 基础镜像
同上（vLLM 0.21 官方 HCU 镜像）。

### 3.2 软件包升级
无。**也不需要装 `qwen_asr` 包**——对齐算法用 vLLM 自带的 `Qwen3ASRForcedAlignerForTokenClassification`
实现复刻，这是本适配的关键点。

### 3.3 代码改动
无模型代码改动。驱动该模型需要三处运行时设置，缺一不可：

1. **`--runner pooling`**：该模型是 token classification 头，不是生成模型，用默认 runner 会按 CausalLM 加载失败。
2. **`--hf-overrides '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}'`**：
   权重里 `config.json` 的 `architectures` 写的是 `Qwen3ASRForConditionalGeneration`，不覆盖就会走生成路径。
3. **`--max-model-len 8192`**：模型 `max_position_embeddings` 只有 8192，写 32768 会直接报
   `User-specified max_model_len (32768) is greater than the derived max_model_len`。

另加两处环境装配（同其它包）：rootfs 内 `vllm_hcu` 的绝对符号链接改指向；`ROCM_PATH` 指向镜像内 DTK。

### 3.4 升级步骤
解包镜像 → `bash run.sh`（起 pooling 服务）→ `python fa_align.py <音频> "<文本>" <语言>`
（也可用 `bash smoke.sh` 跑仓库内两条样例）。

## 4. 服务启动脚本

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/hygon/models/fa06 \
  --served-model-name qwen3-forcedaligner-0.6b \
  --runner pooling \
  --hf-overrides '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}' \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.35 \
  --host 127.0.0.1 --port 8000
```

参数理由：`--runner pooling` + `architectures` 覆盖见 3.3；`--max-model-len 8192` 是模型上限；
`gpu-memory-utilization 0.35` 足够（0.6B 分类头 + 长音频 KV），留出余量便于与其它模型共用一张卡。

> 若用 `fa_align.py` 直接跑（in-process `vllm.LLM`），它内部同样用这三项参数，不需要先起 HTTP 服务。

## 5. Smoke 测试

`bash smoke.sh`，跑仓库内 `data/` 的两条样例。英文（`en.mp3`，7.18 s）：

```json
{"text": "The",       "start": 1.12, "end": 1.20}
{"text": "tribal",    "start": 1.20, "end": 1.76}
{"text": "chief",     "start": 1.76, "end": 2.08}
...
{"text": "fifty",     "start": 4.96, "end": 5.20}
{"text": "pieces",    "start": 5.52, "end": 5.92}
{"text": "of",        "start": 5.92, "end": 6.00}
{"text": "gold.",     "start": 6.96, "end": 7.04}
```

中文（`zh.mp3`，5.62 s，按官方 CJK 逐字切分）：

```json
{"text": "開", "start": 0.72, "end": 0.96}
{"text": "放", "start": 0.96, "end": 1.12}
{"text": "時", "start": 1.20, "end": 1.44}
...
{"text": "五", "start": 4.56, "end": 4.80}
{"text": "點", "start": 4.80, "end": 5.12}
```

两条都满足：时间戳严格单调、首尾落在音频时长之内（7.04 < 7.18、5.12 < 5.62）、词数与文本一致。
完整输出见 [`evidence/en.json`](evidence/en.json)、[`evidence/zh.json`](evidence/zh.json)。

```bash
python fa_align.py data/en.mp3 "The tribal chief then called for the boy and presented him with fifty pieces of gold." English
```

## 6. 精度测评

**未完成 —— 缺官方输入。** T04 强制对齐的验收指标是时间戳偏移 **AAS(ms)**，需要官方下发的音频测试集
与 GPU 参考输出才能算；`baselines/Qwen__Qwen3-ForcedAligner-0.6B.json` 与测试集随分题材料下发，本仓库未持有。
`verify.py` 已保留官方判分逻辑（容差 2%、`--selfcheck` 通过）。

本轮只做了**功能正确性**判断（单调性、落在音频时长内、词数一致），**没有 AAS 数字**，不臆造。

## 7. 性能参考

官方写明「生成类与语音类的测法随分题材料下发」，T04 的对齐吞吐测法未下发，故只记录实测的端到端耗时：

| 项 | 值 |
| --- | --- |
| 引擎初始化（含 profile + KV cache + warmup，`enforce_eager`） | 8.74 s |
| 单条 7.18 s 音频、16 词，进程冷启动到出结果 | 120 s（其中绝大部分是 Python/torch/vllm 导入） |
| 稳态推理 | 未单独计时（本轮未做重复压测） |

需要说明：这个 120 s 是**进程级冷启动**，不是稳态延迟；稳态延迟应在服务常驻后用多次请求取中位数，本轮没做。

## 8. 注意事项

- **三个参数缺一不可**：`--runner pooling`、`architectures` 覆盖、`max-model-len 8192`。少任何一个都会在
  初始化阶段报错（分别是：CausalLM 加载失败、走生成路径、`max_model_len` 校验失败）。
- **音频必须以数组传入**：`multi_modal_data={"audio": path}` 会抛
  `AssertionError: Expected code to be unreachable, but got: '/'`（vLLM 把字符串当成序列拆开了）。
  正确做法是先解码成 `(float32 ndarray @16k, 16000)`；镜像里 `torchaudio.load` 因缺 `torchcodec` 不可用，
  用 `ffmpeg -f f32le -ac 1 -ar 16000` 管道解码最稳。
- **时间戳掩码不能按纯文本 token 数取**：音频占位符会在引擎侧展开（本例 pred=148、文本 token=52），
  必须按「文本部分为整条序列后缀」对齐后再按 `<timestamp>` 位置取（`fa_align.py` 里有断言保护）。
  早期用 `ids[-n_ts:]` 的写法能出数但是错的，同一句音频的起点会从 1.12 s 漂到 2.96 s。
- 中文要按官方 CJK 逐字切分（传入时空格分隔每个字），否则会把整句当成一个词、只出 2 个时间戳。
- 与其它模型共用一张卡时注意显存未被占满；本模型 `gpu-memory-utilization` 只需 0.35。

---

### 仓库内容

| 文件 | 章节 | 说明 |
| --- | --- | --- |
| `env.sh` | 2 / 4 | 变量表 |
| `run.sh` | 4 | 起 pooling 服务（含三项关键参数） |
| `fa_align.py` | 4 / 5 | 对齐主程序：分词 → 构造 prompt → pooling 推理 → 掩码取时间格 → LIS 平滑 → 时间戳 |
| `smoke.sh` | 5 | 跑 `data/` 两条样例 → `evidence/*.json` |
| `env_check.sh` | 2 | → `evidence/env.log` |
| `verify.py` | 6 | 官方判分逻辑（缺 AAS 基线不臆造） |
| `data/` | 5 | 测试音频（en.mp3 / zh.mp3） |
| `evidence/` | — | 两条样例的 JSON 输出 |
