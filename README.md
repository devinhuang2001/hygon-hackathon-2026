# 国芯国模黑客松 · 国产加速卡（K100-AI / K500SM_AI）模型适配与优化

本仓库是本次黑客松的技术报告与可复现代码。所有实测均在 **SCNet 超算互联网的单卡 K500SM_AI 实例**上完成，
按赛事口径计入 **K100-AI**。

- 运行镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`
- 推理框架：vLLM `0.21.0+das.dtk2604`（vllm-hcu `0.21.0+das`）
- 运行时：DTK 6.3.26113（HIP 6.3.26113）/ torch `2.10.0+das.opt1.dtk2604` / triton `3.6.0+das185`
- 精度：bf16；并行：单卡（TP=1, PP=1）

## 提交内容

| # | 模型 | 赛题 | 部署 | Smoke | 性能 | 精度 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Qwen3-ASR-1.7B | **T04 语音识别** | ✅ | ✅ 中英转写正确 | ✅ 22×→208× 实时 | ⛔ 缺官方基线 |
| 02 | Qwen3-ASR-0.6B | **T04 语音识别** | ✅ | ✅ 中英转写正确 | ✅ 37×→225× 实时 | ⛔ 缺官方基线 |
| 03 | Qwen3-0.6B | 非赛题 | ✅ | ✅ | ✅ 官方 6 档口径 | ⛔ 缺官方基线 |
| 04 | Qwen3-1.7B | 非赛题 | ✅ | ✅ | ✅ 4 档 | ⛔ 缺官方基线 |
| 05 | Qwen3-4B | 非赛题 | ✅ | ✅ | ✅ 4 档 | ⛔ 缺官方基线 |
| 06 | Qwen3-8B | 非赛题 | ✅ | ✅ | ⚠️ 仅 4K 两档 | ⛔ 缺官方基线 |
| 07 | MiMo-V2.6-Distill-Qwen-9B | **T01 文本生成** | ✅ | ✅ 单轮/多轮/算术 | ✅ 4K 两档 | ⛔ 缺官方基线 |
| 08 | Qwen3-ForcedAligner-0.6B | **T04 强制对齐** | ✅ | ✅ 中英词/字级时间戳 | ⚠️ 仅端到端计时 | ⛔ 缺 AAS 基线 |

每个目录都是完整的自包含提交包：8 章节技术报告 `README.md` + `env.sh` / `run.sh` / `env_check.sh` /
`smoke.sh` / `verify.py` / `bench.sh` + `evidence/`（原始日志与压测 JSON）+（部分）`data/` 测试素材。

## 快速开始（以 01 为例）

```bash
cd submissions/01-Qwen3-ASR-1.7B
bash run.sh            # 装配环境 + 起服务 + 轮询 /health
bash env_check.sh      # 采集版本 → evidence/env.log
bash smoke.sh          # 健康检查 + 中英转写 → evidence/smoke.log
bash bench.sh          # 并发吞吐 → evidence/bench.txt
```

前提：机器上已有该镜像解包出的 rootfs（本仓库的 `run.sh` 说明了解包与三处运行时装配）。

## 环境适配的关键点（踩过的坑，都在报告第 3 / 8 节）

本实例没有容器运行时，镜像层解包后直接以宿主内核运行，过程中有三个必须处理的点：

1. **`vllm_hcu` 用绝对符号链接覆盖 `vllm` 的部分模块**，解包后指向失效，报
   `ModuleNotFoundError: ...modular_kernel`；需把 link 重指向 rootfs。
2. **镜像不含 `/opt/hyhal`**（驱动用户态由平台注入，等同 nvidia-container-toolkit），需补宿主驱动库。
3. **triton 的 HCU 后端要求 `ROCM_PATH` 下有 `aillvm`**，宿主 DTK 没有该目录，否则 kernel 编译报
   `HSACOError: File operation failed: ... .amdgcn`，表现为 EngineCore 启动失败。

另有两个容易误判为「模型不支持」的坑：

- **`Qwen3-ASR-*-hf` 权重不能用**：其 707 个张量全带 `model.` 前缀，vLLM 0.21 加载报
  `no module or parameter named 'model'`。必须用非 `-hf` 版（`thinker.` 前缀）。
- **强制对齐模型必须用 pooling runner**：`--runner pooling` +
  `--hf-overrides '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}'` + `--max-model-len 8192`。

## 性能数据（原始 JSON 在各包的 `evidence/bench/`）

**语音识别（T04，RTF = 墙钟/音频时长，越小越好）**

| 模型 | 单并发 | 8 并发 | 32 并发 |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B | 21.97 audio-s/s（RTF 0.046×） | 119.53 | 208.23 |
| Qwen3-ASR-0.6B | 37.04 audio-s/s（RTF 0.027×） | 127.26 | 225.05 |

**文本（4K 输入 / 1K 输出，输出吞吐 tok/s）**

| 模型 | c=1 | c=8 |
| --- | --- | --- |
| Qwen3-0.6B | 97.59 | 329.99（c=64：482.51） |
| Qwen3-1.7B | 64.56 | 269.29 |
| Qwen3-4B | 40.26 | 170.28 |
| Qwen3-8B | 21.28 | 97.29 |
| MiMo-V2.6-Distill-Qwen-9B | 20.92 | 90.37 |

## 关于精度测评（必须说明）

**本仓库没有提交精度分数。** 赛事要求 `verify.py` 按官方 `baselines/<模型>.json` 里写死的子集题数、
max_tokens、思考开关与**官方 GPU 基线分**现场跑分；该基线文件与评测子集随分题材料下发，本团队未获取到
（知识空间无访问权限、文档搜索 0 命中、机器与共享盘均无该目录）。

按赛事红线（伪造日志取消资格），我们选择留空并写明补齐方式，而不是给不可复现的数字。
`verify.py` 已实现官方判分逻辑（容差 2%、只罚劣化不罚提升），放入官方 `baselines/` 与 `data/` 后
`source env.sh && python verify.py` 即可产出 `verify_result.json`。

## 目录

```
submissions/
├── INDEX.md                        总览与缺口说明
├── 01-Qwen3-ASR-1.7B/              T04 计件
├── 02-Qwen3-ASR-0.6B/              T04 计件
├── 03-Qwen3-0.6B/                  同栈文本样例
├── 04-Qwen3-1.7B/
├── 05-Qwen3-4B/
├── 06-Qwen3-8B/
├── 07-MiMo-V2.6-Distill-Qwen-9B/   T01 计件
└── 08-Qwen3-ForcedAligner-0.6B/    T04 计件
```

`docs/cookbook-patch/` 另附给 `inference-cookbook-das` 补 K100_AI 一列的现成内容（Qwen3-ASR 行的
`-` → ✅），可直接提 PR。
