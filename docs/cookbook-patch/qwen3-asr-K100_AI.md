# 给 `inference-cookbook-das` 的 `docs/model-deployment/vllm/qwen3-asr.md` 追加 K100_AI

在现有文档里补 K100_AI 一列的内容（当前矩阵 `Qwen3-ASR / vLLM / K100_AI` 是 `-`）。
以下内容按 `.claude/skills/add-model/SKILL.md` 的表格与锚点规范撰写，可直接贴入对应位置。

## 1) 追加到「模型列表」表格的行

现有表格 `推荐硬件` 只有 BW1100 的行；K100_AI 的行按 `BW1100 → BW1000 → K100_AI` 顺序放在同框架版本之后。
模型权重列留空（沿用同模型已有链接）。

```markdown
|  | BF16 | 0.21 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-asr-17b-ifb-k100_ai-1x-vllm-021) |
```

对应到 `Qwen/Qwen3-ASR-1.7B`（非 `-hf` 版）那一行之后；`Qwen/Qwen3-ASR-0.6B` 同理加一行、
锚点改成 `#qwen3-asr-06b-ifb-k100_ai-1x-vllm-021`（0.6B 未在本轮实测，见文末说明）。

## 2) 追加到「启动命令」章节

```markdown
### Qwen3-ASR-1.7B IFB K100_AI 1x vLLM 0.21

```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@zh.mp3" -F "model=Qwen/Qwen3-ASR-1.7B"
```
```

## 3) 实测依据（提交 PR 时附在说明里）

在 `K500SM_AI 64GB`（按赛事口径计 K100-AI）、镜像
`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` 上：

- 服务启动成功，`/health` 200，`/v1/models` 正常
- 英文样例转写逐字正确：`The tribal chief then called for the boy, and presented him with fifty pieces of gold.`
- 中文样例：`開放時間：早上九點至下午五點。`
- 稳态显存 60.2 GB / 64 GB；吞吐 22× 实时（单并发）／208× 实时（32 并发）

## 4) 两个必须写进文档的坑

1. `Qwen/Qwen3-ASR-1.7B-hf`（`-hf` 后缀）权重全部带 `model.` 前缀，vLLM 0.21 加载失败：
   `ValueError: There is no module or parameter named 'model'`。必须用 `Qwen/Qwen3-ASR-1.7B`。
2. `ROCM_PATH` 必须是镜像内的 DTK（含 `aillvm`），否则 triton 编译报
   `HSACOError: File operation failed: ... .amdgcn`，表现为 EngineCore 启动失败。

## 未做

- `Qwen/Qwen3-ASR-0.6B` 同样只差 K100_AI 一格，本轮未实测（权重未下载），不要顺手补。
- 精度（WER/CER）未测：官方 `baselines/*.json` 与 `data/` 未持有，按赛事红线不编造。
