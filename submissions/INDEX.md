# 提交物总览（国芯国模黑客松 · K100-AI）

全部实测在 SCNet `ksai.scnet.cn` 的 **K500SM_AI 64GB** 单卡实例上完成，按赛事口径计 `K100-AI`。
镜像 `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`，
vLLM `0.21.0+das`，DTK 6.3.26113，torch `2.10.0+das.opt1.dtk2604`。

## 已完成（8 个包）

| # | 模型 | 框架 | 卡型 | 赛题 | 部署 | Smoke | 性能 | 精度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Qwen3-ASR-1.7B | vLLM 0.21.0 | K100-AI | **T04 计件** | ✅ | ✅ 中英转写正确 | ✅ 3 档并发，22×→208× 实时 | ⛔ 缺基线 |
| 02 | Qwen3-ASR-0.6B | vLLM 0.21.0 | K100-AI | **T04 计件** | ✅ | ✅ 中英转写正确 | ✅ 3 档并发，37×→225× 实时 | ⛔ 缺基线 |
| 03 | Qwen3-0.6B | vLLM 0.21.0 | K100-AI | 非赛题 | ✅ | ✅ 单轮/多轮/算术 | ✅ 官方 6 档（16K×64 超时） | ⛔ 缺基线 |
| 04 | Qwen3-1.7B | vLLM 0.21.0 | K100-AI | 非赛题 | ✅ | ✅ 算术 221 | ✅ 4 档（4K/16K×1/8） | ⛔ 缺基线 |
| 05 | Qwen3-4B | vLLM 0.21.0 | K100-AI | 非赛题 | ✅ | ✅ 见 evidence | ✅ 4 档（4K/16K×1/8） | ⛔ 缺基线 |
| 06 | Qwen3-8B | vLLM 0.21.0 | K100-AI | 非赛题 | ✅ | ✅ 见 evidence | ⚠️ 仅 4K×1 / 4K×8 | ⛔ 缺基线 |
| 07 | MiMo-V2.6-Distill-Qwen-9B | vLLM 0.21.0 | K100-AI | **T01 计件** | ✅ | ✅ 单轮/多轮/算术 | ✅ 4K 两档（20.9→90.4 tok/s） | ⛔ 缺基线 |
| 08 | Qwen3-ForcedAligner-0.6B | vLLM 0.21.0 | K100-AI | **T04 计件** | ✅ | ✅ 中英词/字级时间戳 | ⚠️ 仅端到端计时 | ⛔ 缺 AAS 基线 |

## 关键数字

**语音（T04，计件）**

| 模型 | 单并发 | 8 并发 | 32 并发 |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B | 21.97 audio-s/s（RTF 0.046×） | 119.53 | 208.23 |
| Qwen3-ASR-0.6B | 37.04 audio-s/s（RTF 0.027×） | 127.26 | 225.05 |

**文本（4K 输入 / 1K 输出，输出吞吐 tok/s）**

| 模型 | c=1 | c=8 | 备注 |
| --- | --- | --- | --- |
| Qwen3-0.6B | 97.59 | 329.99 | 另有 c=64：482.51 |
| Qwen3-1.7B | 64.56 | 269.29 | 16K：51.14 / 138.77 |
| Qwen3-4B | 40.26 | 170.28 | 16K：32.12 / 92.16 |
| Qwen3-8B | 21.28 | 97.29 | 仅 4K 档；单轮耗时 349 s / 162 s |
| MiMo-V2.6-Distill-Qwen-9B | 20.92 | 90.37 | Mamba+Attention 混合；TTFT 1.96 s / 5.75 s |

规模越大单请求越慢，符合预期；并发从 1 到 8 吞吐提升约 4 倍。

## 每个包的结构

```
NN-<模型>/
├── README.md          技术报告，按官方 8 章节
├── env.sh             变量表（模型/镜像/卡型/并行/启动参数）
├── run.sh             装配环境 + 起服务 + 轮询 /health
├── env_check.sh       采集版本 → evidence/env.log
├── smoke.sh           健康检查 + 业务请求 → evidence/smoke.log
├── verify.py          官方判分逻辑（--selfcheck 可用）
├── bench.sh           性能脚本
├── asr_bench.py       （T04 两个包）ASR 并发吞吐
├── data/              （T04 两个包）测试音频
└── evidence/          原始产物，报告里引用的数字全部出自这里
```

## 唯一系统性缺口：精度（第 6 节）

所有包的精度测评都空着，原因同一个：赛事要求 `verify.py` 按 `baselines/<模型>.json` 里写死的子集题数、
max_tokens、思考开关和**官方 GPU 基线分**现场跑分。这套文件随分题材料下发，拿不到：

- 飞书知识空间 `131006 permission_denied`（该节点所属空间未授权）
- 文档搜索 `分题材料 / baselines / gsm8k.jsonl` 全部 0 命中
- 机器与共享盘全盘 find 无 `baselines/`、无评测子集

按赛事红线（伪造日志取消资格）未编造任何分数。补齐方式：放入官方 `baselines/` 与 `data/` 后
`source env.sh && python verify.py` 即可产出 `verify_result.json`。

## 试过但没出包的

- **Qwen3-ForcedAligner-0.6B（T04 计件候选）**：权重下载并上传，`--runner pooling` +
  `architectures` 覆盖后**能起服务、权重加载 2.24 s、`/pooling` 路由可用**。但时间戳语义
  （5000 个时间槽 → 毫秒的映射、prompt 里 `<timestamp>` 的插入位置）定义在 `qwen_asr` 包内，
  镜像没有该包、官方分题材料也未下发。盲猜映射产出的 AAS 必然是错的，故不出包。
  **这是目前最接近成功的一个，只差一份官方 prompt 规范。**
- **MiMo-V2.6-Distill-Qwen-9B**：已出包（见上表 07）。办法是边下边传，不经本机磁盘中转。
- **Kimi-Linear-48B-A3B**：bf16 需 96 GB > 64 GB，必须量化，而官方只认海光官方 channelwise 权重，本地没有。
  **MiniMax-M3.1-Flash** 235 GB、**Kimi-VL-A3B** 30.6 GB + 需额外依赖，均不可行。
- **QwQ-32B / DeepSeek-R1-Distill-32B**：bf16 权重就占满 64 GB，加 KV cache 放不下；AWQ 版未做。

## 关于远程安全

SCNet 是公共环境，本轮所有文件按以下方式处理（详见 `FINDINGS.md` 第十三节）：

- 工作目录 `/root/hygon` 权限 `700`（仅 root 可读），`/root` 本身亦为 `700`
- 待机/结束时用 `lock.sh` 把整棵树 **AES-256 加密**成 `/root/hygon.enc`，明文以 `shred -u -z` 粉碎
- 最终态：连加密包一起删除，远端不留任何产物
- 密钥只存在本地 `D:/tmp/keys/hygon.key`，不写进远程任何文件
