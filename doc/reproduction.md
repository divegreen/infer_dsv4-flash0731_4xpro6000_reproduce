# 原生源码项目复现指南

> 原生复现文档：命令与 `tools/reproduce.py` 和 catalog 对齐。源码发布不附带历史 GPU 原始结果、已运行报告或旧虚拟环境；需要按本指南建立本次运行的证据。
>
> 文件性质：人工维护的复现说明，属于项目原生文档。初始文件和历史证据在项目外本地保留，不是新复现的输入依赖，也不随源码上传。

本文的目标是：把整个项目复制或解压到任意目录后，使用固定依赖和固定模型，按研究阶段重新执行实验、校验结果并生成报告。项目会自动建立运行所需的目录，不需要预先手动建立 `results/`、`logs/`、`summary/` 等目录，也不需要逐个修改源码中的路径。

所有示例 Shell 命令都在项目根目录执行，以便 Shell 找到 `tools/reproduce.py`。程序内部根据脚本自身的位置解析项目路径，不依赖当前工作目录；从其他目录用脚本的真实完整路径调用，也会定位到该项目。模型是外部输入，默认使用 `~/models/dsv4-flash0731`。推荐 Python 3.12。

## 1. 先区分原生输入、生成物与历史参考

| 位置 | 内容 | 从零复现是否需要 |
| --- | --- | --- |
| `tools/` | 统一复现入口、路径解析和辅助工具 | 必须保留 |
| `research/dsv4-flash0731/experiments/architecture/` | 原生配置与自动生成的process索引 | 原生配置保留，索引可由prepare重建 |
| `research/dsv4-flash0731/experiments/scenarios/` | 场景、容量、长上下文和 Session 的配置、负载、任务语料 | 必须保留 |
| `research/dsv4-flash0731/experiments/scheduler/` | 原生配置/负载与自动process索引 | 原生输入保留，索引可由prepare重建 |
| `research/dsv4-flash0731/scripts/` | 架构研究的运行与分析代码 | 必须保留 |
| `research/dsv4-flash0731/{scenarios,long-context,sessions,scheduler}/scripts/` | 各主题的运行、分析和检查代码 | 必须保留 |
| `requirements-vllm.lock.txt`、`activate-vllm.sh` | 推理依赖快照、环境激活脚本 | 必须保留 |
| `doc/` 及各研究目录的 `doc/` | 原生方法、协议和复现说明 | 保留；运行生成报告写到 results 下 |
| `results/runs/<语义实验ID>/` | 原始请求、服务日志、GPU 采样、配置及源码快照、退出状态 | 从零运行前可不存在；运行后是分析输入 |
| `results/analysis/<主题>/` | 汇总、审计、图表、已执行 Notebook 等派生结果 | 可以删除后从原始运行结果重建 |
| `results/setup/` | 安装清单、安装日志、实际依赖和环境身份 | 自动产生；不是源码 |
| `.venv-vllm/`、`.venv-analysis/`、`.cache/` | 本机环境与下载/编译缓存 | 迁移时重新建立，不复制旧虚拟环境 |

`AGENTS.md` 和 `progress.md` 按项目协议保留在根目录；根 `README.md` 是入口。原有归档、历史结果与迁移审计文件在项目外本地保留，不应放回原生源码树。任务语料中的 Markdown，例如模型要读取的 `public/README.md` 或 `public/docs/*.md`，是实验输入，不能因扩展名为 `.md` 而删除或移动出任务目录。

实验 ID 描述配置与用途，例如架构、缓存模式、筛选/确认/诊断用途。目录名不再使用日期或仅有字母的阶段名。随机种子、模型名称中的版本标识和原始历史记录仍有实验语义，不应因为看起来像日期而改写。

## 2. 查看目录与阶段，再自动准备工作区

目的：先看到本项目实际包含哪些实验、各阶段的前置条件和输出位置，然后让工具创建全部必要目录。

```bash
python3 tools/reproduce.py --help
python3 tools/reproduce.py list
python3 tools/reproduce.py prepare
```

预期：`list` 按主题和 `process1`、`process2` 等顺序显示实验；`prepare` 读取 catalog、解析项目根目录和模型路径，并创建缺失的运行、分析、安装日志目录。重复执行准备操作应安全，不要求先删除目录。当前 catalog 为 84 个定义、83 个可执行定义和 21 个阶段。

`prepare` 自动把固定依赖拆分到 `results/setup/core.requirements.txt`、`cubin.requirements.txt`、`jit.requirements.txt`，分别 199/1/7 条；同时记录调用它的实际解释器、包版本、计划运行解释器和模型文件身份。它不会安装依赖或下载模型；查看 `model_exists`、`runtime_environment_exists`，不能仅凭 PREPARED 宣称环境可运行。

这一步只准备文件和路径，不安装依赖、不下载模型、不启动 GPU，也不把缺失的实验结果标记为成功。`catalog.json` 是唯一原生执行登记，process 目录中的 stage.json/实验描述是 prepare 生成的索引视图，不需要维护第二份参数；configs/workloads/fixtures 仍是原生输入。

## 3. 安装固定的 Python 和推理依赖

目的：在新项目内建立独立环境，保持历史研究使用的软件版本。

原研究记录为 Linux x86_64、uv 0.13.6、Python 3.12.3、vLLM 0.33.0、PyTorch 2.14.0+cu130。推理依赖快照包含 207 个固定版本；它不是包含全部下载来源与文件哈希的完整锁文件，因此“版本一致”不等于“所有 wheel 字节完全一致”。

先查看 uv。已安装 0.13.6 的环境可直接跳到创建虚拟环境。

```bash
uv --version
```

尚未安装该版本时，使用固定版本的官方安装器。下面的下载目录由第 2 步自动创建。

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://astral.sh/uv/0.13.6/install.sh -o results/setup/uv-install.sh
env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh results/setup/uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

创建新环境。虚拟环境的生成是命令自身的工作，不需要手动建目录。

```bash
uv python install 3.12.3
uv venv --python 3.12.3 --managed-python .venv-vllm
.venv-vllm/bin/python -c 'import sys; print(sys.version); print(sys.executable); print(sys.prefix)'
```

预期：Python 3.12.3，解释器和 `prefix` 指向当前项目下的 `.venv-vllm`。项目迁移后重建环境，不复用指向旧绝对路径的虚拟环境。

三个依赖来源分别安装，顺序固定。`cu130` 保持显式指定，不能换成 `auto`；原研究曾观察到自动选择其他 CUDA 构建。

```bash
uv pip install --no-config --cache-dir .cache/uv --python .venv-vllm/bin/python --default-index https://pypi.org/simple --torch-backend=cu130 --strict -r results/setup/core.requirements.txt > results/setup/install-core.log 2>&1
uv pip install --no-config --cache-dir .cache/uv --python .venv-vllm/bin/python --default-index https://flashinfer.ai/whl --no-deps -r results/setup/cubin.requirements.txt > results/setup/install-cubin.log 2>&1
uv pip install --no-config --cache-dir .cache/uv --python .venv-vllm/bin/python --default-index https://flashinfer.ai/whl/cu130 --no-deps -r results/setup/jit.requirements.txt > results/setup/install-jit.log 2>&1
uv pip check --python .venv-vllm/bin/python
uv pip freeze --python .venv-vllm/bin/python > results/setup/installed-freeze.txt
```

预期：三份清单分别为 199、1、7 条固定依赖；三次安装均成功，`pip check` 没有依赖冲突。任何一步失败都先检查对应日志，不能通过安装最新版绕过固定版本。

比较实际依赖和原始版本快照。`pip check` 只检查依赖冲突，不证明版本与项目一致：

```bash
.venv-vllm/bin/python - <<'PY'
from pathlib import Path
def pins(path):
    return {line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')}
expected = pins('requirements-vllm.lock.txt')
actual = pins('results/setup/installed-freeze.txt')
print('Missing or different:', sorted(expected - actual))
print('Unexpected or different:', sorted(actual - expected))
assert expected == actual, 'Installed package versions differ'
print('PASS:', len(expected), 'pinned package versions')
PY
.venv-vllm/bin/python tools/reproduce.py prepare
```

预期差集均为空。用新推理解释器再执行 `prepare`，保存本次实际环境身份到 `results/setup/environment.json`。

图表使用单独分析环境，避免改变推理环境的依赖。

```bash
uv venv --python 3.12.3 --managed-python .venv-analysis
uv pip install --no-config --cache-dir .cache/uv --python .venv-analysis/bin/python --default-index https://pypi.org/simple 'numpy==2.3.5' 'matplotlib==3.10.8'
uv pip freeze --python .venv-analysis/bin/python > results/setup/analysis-freeze.txt
```

已有 Notebook 使用顺序 Python cell 执行器，不要求安装 Jupyter。运行器应选择 `.venv-vllm/bin/python`，绘图步骤选择 `.venv-analysis/bin/python`；文档中的直接 Python 路径均相对于当前项目。

## 4. 准备模型，明确外部路径与硬件条件

目的：让所有服务、客户端和审计读取同一 checkpoint；只改变模型所在位置，不改变实验输入身份。

默认路径为 `~/models/dsv4-flash0731`，无需额外设置。模型放在其他位置时，仅配置环境变量，代码和 JSON 不需要手工修改。

```bash
export DSV4_MODEL_PATH="$HOME/models/dsv4-flash0731"
python3 tools/reproduce.py prepare
```

上面的外部模型路径是用户配置示例，不是源码中的固定路径。重构工具会展开 `~`；在 Shell 中使用变量时，推荐传入已展开的 `$HOME/models/...` 或实际完整目录。

已有模型可以只读复用。若要验证从下载开始的流程，使用原记录中的固定仓库和 revision。`hf download` 会创建缺失的目标目录。

```bash
export DSV4_MODEL_PATH="${DSV4_MODEL_PATH:-$HOME/models/dsv4-flash0731}"
.venv-vllm/bin/hf download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 --local-dir "$DSV4_MODEL_PATH"
```

原本地记录为 48 个权重分片，总计 166,886,535,336 字节，约 155.43 GiB；tokenizer SHA256 为 `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`。原始调查只核对元数据和 safetensors 分片头，不是全部权重内容的 SHA 验证。远端仓库当前可获取性需以下载实际结果为准；下载失败时保留错误，不自动换 revision。

参考硬件是四张 RTX PRO 6000 Blackwell Server Edition，每卡 `nvidia-smi` 记录 97887 MiB，计算能力 12.0，驱动 580.173.02，PCIe 互联且无 NVLink。当前实验运行器限定物理 GPU 0–3；其他 GPU 不参与实验。硬件或驱动不同可以验证可运行性，但不能把新测性能直接称为历史数值的等价复现。

`CUDA_VISIBLE_DEVICES=0,1,2,3` 只限制可见 GPU；TP/PP/DP/EP 等配置才决定并行方式。显卡容量、模型大小和 KV cache 共同限制可执行负载，入口预检不能替代真实模型加载。

## 5. 先执行不加载模型的校验

目的：检查目录、catalog、配置引用、依赖关系和代码语法，再校验服务启动输入，尽早发现迁移错误。

```bash
.venv-vllm/bin/python tools/reproduce.py validate
```

`validate` 只检查 ID、文件引用、阶段依赖无环、Python/JSON 语法与活跃代码中的旧机器路径，报告位于 `results/validation/layout.json`，不执行客户端测试或 GPU。安装好推理环境后执行 CPU 启动预检：

```bash
.venv-vllm/bin/python tools/reproduce.py preflight
```

`preflight` 给实际服务命令加 `--validate-only`，隐藏 GPU，不发送模型请求，通信实验的 torchrun 跳过。逐项输出在 `results/validation/preflight/`。也可以检查单项：

```bash
.venv-vllm/bin/python tools/reproduce.py preflight architecture-tp4-marlin-smoke-review
```

预期：验证输出清楚列出 `PASS`、`FAIL`、`NOT VERIFIED` 或缺少输入，日志和机器可读报告自动写入 `results/`。这里的预检通过只说明静态和启动输入检查满足预期，不表示全部客户端夹具、模型加载、四卡通信或性能已经通过。

源码发布和迁移可以独立做 CPU 检查，不需要模型或推理环境：

```bash
python3 tools/check_source_release.py
python3 tools/check_portability.py
```

`check_source_release.py` 会在临时目录生成真实源码包、解压到含空格的新路径，检查全部实验命令的 dry-run，并比较 `prepare` 前后的 Git 候选文件、包成员和文件字节。它还验证生成报告、缓存和自动索引不会混入发布；报告默认写入 `results/validation/source-release.json`。可用 `--output` 指定项目外的报告路径，保持待发布源码树不产生运行报告。

`check_portability.py` 检查自动建目录、阶段前置条件、恢复与覆盖，以及异名目录调用。默认只使用当前原生源码，不需要旧归档。只有需要额外核对重构前文件时，才使用可选的 `--reference-root` 指向项目外保留树；该目录须包含原始归档、manifest 和迁移映射，不能因此把它们重新放回源码仓库。上述检查均不启动 GPU、模型推理或网络请求。

启动保护本身可以独立做 CPU 回归：

```bash
python3 tools/check_launch_guard.py --output results/validation/launch-guard.json
```

该检查用模拟 GPU 查询、模拟推理子进程和临时文件锁验证占用拒绝、跨副本互斥、失败保留及缓存指纹。预期 38 项 PASS；它不会访问物理 GPU，不构成硬件空闲或推理成功的证据。

基础 GPU 安装验证使用重构后的 `tools/verify_install.py`，检查解释器、GPU UUID、小矩阵乘法和 vLLM RMSNorm，并显式同步 CUDA。该脚本通过后，也只能说明这些基础运算通过；NCCL、模型推理、长上下文和真实四卡性能仍需各自实验验证。

基础 GPU 检查是显式单独操作。确认 GPU 0–3 空闲并允许使用后执行：

```bash
flock -n results/.reproduction.lock env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 .venv-vllm/bin/python tools/verify_install.py > results/validation/gpu-install.log 2>&1
```

预期四卡小矩阵乘法和 RMSNorm 逐项 PASS；本轮文档编写未执行此 GPU 命令。这条独立安装检查仅持有项目级锁，不经过下一节的统一启动保护；执行前仍需确认四卡空闲。



## 6. 按 catalog 的 process 顺序运行

目的：保持先筛选、后正式验证的真实依赖。所有阶段命令由当前 catalog 自动取得实验、配置、负载、超时和GPU参数，用户不需要手改输入或状态。下面的命令列表由 catalog 生成；最新索引可用 `python3 tools/reproduce.py list --json` 查询。

正式 `run` 会先持有项目级锁，再在真正启动前取得主机临时目录中的 `portable-inference-gpu-0.lock` 至 `portable-inference-gpu-3.lock`，并检查 GPU 0–3 完整可见、每卡显存占用不超过 512 MiB、采样利用率为零且没有计算进程。任何检查失败都会阻止启动，保留 `FAILED_BEFORE_LAUNCH` 状态及日志；先排除占用或设备信息问题，再显式重试。

这些文件锁只约束使用同一主机临时目录、遵守相同协议的协作 CLI 副本。不同 `TMPDIR`、不使用该入口的程序和其他租户不受此协议保证；采样空闲也不能阻止外部程序随后占用设备。即使一个实验只选择其中部分设备，入口仍预约全部四卡，以保持当前研究协议的串行资源前提。锁在退出和异常后释放，不终止其他用户的进程。

每条命令的原始结果统一写入 `results/runs/<实验ID>/`，阶段回执为 `results/stage-state/architecture/process1-smoke.json` 等主题/阶段回执，监督日志在 `results/stage-state/logs/`。完成后查看终态、退出码与错误再进入下一条。

先预览一个阶段，确认将运行的命令：

```bash
python3 tools/reproduce.py run --stage architecture/process1-smoke --dry-run
```

dry-run只打印命令，不检查前置结果、完整模型或GPU空闲；正式运行才检查这些前提。阶段COMPLETED表示安排的尝试已经形成可核对终态或明确排除，不代表每项成功或SLO达标。失败终态仍保留；架构full/long候选额外要求对应筛选/前阶段成功。

阶段回执绑定代码/输入、项目路径、运行解释器和包安装元数据、模型小文件SHA/文件尺寸与mtime，以及原始证据身份。模型的生成缓存、Python字节码、临时下载及锁文件不纳入原生身份；子进程设置 `PYTHONDONTWRITEBYTECODE=1`。完整权重payload不会在每次恢复时重新做SHA，不能把尺寸/mtime检查称为逐字节验证。修改源码、负载或迁移项目后，旧回执不能充当新执行证据。缺少前置阶段时工具报告应先执行的阶段，不手动编辑state来通过检查。单项 `run <实验ID>` 不代替完整阶段回执。

### 通信与架构

| 阶段 | 目的 | 默认实验数 | 前置 |
| --- | --- | ---: | --- |
| `architecture/process1-smoke` | 四卡通信与最小模型推理 | 3 | 准备和检查 |
| `architecture/process2-screen` | 筛选17种架构/推理配置，保留失败 | 17 | `architecture/process1-smoke` |
| `architecture/process3-full` | 对筛选通过候选运行统一full负载 | 6 | `architecture/process2-screen` |
| `architecture/process4-long` | 对full候选进一步验证长输入 | 4 | `architecture/process3-full` |

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage architecture/process1-smoke
.venv-vllm/bin/python tools/reproduce.py run --stage architecture/process2-screen
.venv-vllm/bin/python tools/reproduce.py run --stage architecture/process3-full
.venv-vllm/bin/python tools/reproduce.py run --stage architecture/process4-long
.venv-vllm/bin/python tools/reproduce.py analyze architecture
```

### 场景研究

| 阶段 | 目的 | 默认实验数 | 前置 |
| --- | --- | ---: | --- |
| `scenarios/process1-scenarios` | 长输出、64K、到达、缓存与初期Agent对照 | 5 | `architecture/process4-long` |
| `scenarios/process2-profile` | 独立采样profile，不混入正常性能 | 1 | `scenarios/process1-scenarios` |
| `scenarios/process3-optimization` | 有限batch预算、DP/EP和缓存优化对照 | 4 | `scenarios/process2-profile` |

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage scenarios/process1-scenarios
.venv-vllm/bin/python tools/reproduce.py run --stage scenarios/process2-profile
.venv-vllm/bin/python tools/reproduce.py run --stage scenarios/process3-optimization
.venv-vllm/bin/python tools/reproduce.py analyze scenarios
```

### 服务容量

| 阶段 | 目的 | 默认实验数 | 前置 |
| --- | --- | ---: | --- |
| `capacity/process1-screen` | 不同架构开放服务能力筛选 | 4 | `scenarios/process3-optimization` |
| `capacity/process2-boundary-confirm` | 较长窗口与容量边界确认 | 4 | `capacity/process1-screen` |
| `capacity/process3-lower-bound-confirm` | 可选保守下界确认；历史计划未启动 | 0 | `capacity/process2-boundary-confirm` |

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage capacity/process1-screen
.venv-vllm/bin/python tools/reproduce.py run --stage capacity/process2-boundary-confirm
.venv-vllm/bin/python tools/reproduce.py analyze capacity
```

额外下界确认是可选分支，不阻断long-context；主动选择才执行：

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage capacity/process3-lower-bound-confirm --include-optional
.venv-vllm/bin/python tools/reproduce.py analyze capacity --overwrite
```

`capacity-tp4-lower-bound`、`capacity-dspark3-prefill-bound` 对应历史a09/a10：有完整计划，历史为未启动，不是原已完成结果。

### 长上下文混合调度

| 阶段 | 目的 | 默认实验数 | 前置 |
| --- | --- | ---: | --- |
| `long-context/process1-screen` | 混合长短输入下架构与prefill阈值筛选 | 5 | `capacity/process2-boundary-confirm` |
| `long-context/process2-cadence-screen` | decode/prefill节奏的单因素筛选 | 1 | `long-context/process1-screen` |
| `long-context/process3-profile` | 默认、阈值和cadence独立profile | 3 | `long-context/process2-cadence-screen` |
| `long-context/process4-confirm` | 匹配长窗确认收益与代价 | 2 | `long-context/process3-profile` |

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage long-context/process1-screen
.venv-vllm/bin/python tools/reproduce.py run --stage long-context/process2-cadence-screen
.venv-vllm/bin/python tools/reproduce.py run --stage long-context/process3-profile
.venv-vllm/bin/python tools/reproduce.py run --stage long-context/process4-confirm
.venv-vllm/bin/python tools/reproduce.py analyze long-context
```

### Agent Session

| 阶段 | 目的 | 默认实验数 | 前置 |
| --- | --- | ---: | --- |
| `sessions/process1-pilot` | 回放、工具闭环与协议pilot | 2 | `long-context/process4-confirm` |
| `sessions/process2-matched-screen` | 匹配比较缓存、架构与任务结果 | 5 | `sessions/process1-pilot` |
| `sessions/process3-open-screen` | 以Session为到达单位的开放筛选 | 4 | `sessions/process2-matched-screen` |
| `sessions/process4-confirm` | 分别确认回放与真实Agent闭环 | 4 | `sessions/process3-open-screen` |
| `sessions/process5-continuation` | 单独保存后续continuation，不改写失败 | 1 | `sessions/process4-confirm` |

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage sessions/process1-pilot
.venv-vllm/bin/python tools/reproduce.py run --stage sessions/process2-matched-screen
.venv-vllm/bin/python tools/reproduce.py run --stage sessions/process3-open-screen
.venv-vllm/bin/python tools/reproduce.py run --stage sessions/process4-confirm
.venv-vllm/bin/python tools/reproduce.py run --stage sessions/process5-continuation
.venv-vllm/bin/python tools/reproduce.py analyze sessions
```

### 调度优化

| 阶段 | 目的 | 默认实验数 | 前置 |
| --- | --- | ---: | --- |
| `scheduler/process1-profile` | Observe/FairShare的dense诊断和调度证据 | 2 | `sessions/process4-confirm` |
| `scheduler/process2-matched-performance` | 交错顺序、无profile的性能与质量配对 | 4 | `scheduler/process1-profile` |

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage scheduler/process1-profile
.venv-vllm/bin/python tools/reproduce.py run --stage scheduler/process2-matched-performance
.venv-vllm/bin/python tools/reproduce.py analyze scheduler
```

历史d02对应 `scheduler-tp4-fairshare-dense-profile`：当前保留失败输入，缺少完整原监督器启动定义，标记为historical input only，不自动执行。默认阶段使用完整定义的 `scheduler-tp4-fairshare-dense-profile-retry`；不能伪造旧d02失败。scheduler前置为Session confirm，Session continuation是另一后续分支。

## 7. 读取结果，理解三种不同的“成功”

`analyze <主题>` 已提供独立入口，自动按 catalog 收集该主题的现有原始运行目录，生成 JSON/Markdown 汇总并记录实际命令和退出码。示例：

```bash
.venv-vllm/bin/python tools/reproduce.py analyze scheduler
```

正常性能与 profile 分开分析。分析执行记录在 `results/analysis/<主题>/analysis.json`，详细日志为 `analysis-*.log`，生成 Markdown 在该目录的 `doc/` 下。prepare会生成当前输入SHA绑定的 `results/setup/scheduler-pair-blocks.json`，运行快照捕获它，调度分析读取实际快照中的登记；原历史decision保持不改。当前统一入口不自动生成旧绘图和已执行 Notebook；这些原生工具/模板保留，但缺失旧 raw 时历史固定数字检查属于 NOT VERIFIED。历史 decision_provenance 的旧 hash 和 review_*decision/review_*confirm_plans 是历史回顾输入，不应手改 hash 来让新运行通过。

每个实验的统一原始目录为：

```text
results/runs/<语义实验ID>/
```

它包含本次实际使用的输入、源码快照、命令、服务日志、GPU 采样、逐请求结果及终态；不同客户端的报告文件可能为 `benchmark-*.json`、`capacity.json` 或 `sessions.json`。各研究主题的派生结果统一放在：

```text
results/analysis/<主题>/
```

| 结果 | 说明 | 不能据此推断 |
| --- | --- | --- |
| 监督器退出成功 | 进程按运行器要求结束 | 请求全部成功或任务答案正确 |
| 协议/审计 PASS | 保存的数据、请求计数、输入身份和指标满足检查 | 性能最优、任务质量通过或容量达标 |
| 研究指标达标 | 在已测硬件、负载和窗口下满足指定指标 | 数学最大容量或长期生产 SLA |

必须分别查看请求错误、任务评分、稳定性、SLO 与性能。固定 token 回放便于控制工作量，但不证明真实任务质量；真实 Agent 闭环的工具调用数和上下文会随输出变化，不能强行当作等工作量比较。TTFT 是首个非空文本到达时间；TPOT 是请求级平均间隔，不是每个 token 的 ITL。

生成的 Markdown 报告应放在对应主题的 `doc/` 中，并在页首注明“自动生成”、生成器、输入位置和重建命令。原生协议、任务文件和脚本不能当作派生结果删除。

## 8. 重跑、覆盖和失败保留

目的：在不手工改状态、不丢失失败证据的前提下重新执行同一实验。

重跑使用真实入口选项。无覆盖参数时，源码/输入和终态可核对的尝试会 `REUSED_TERMINAL`，失败仍是失败，也不算新增测量。需要重新测量时：

```bash
.venv-vllm/bin/python tools/reproduce.py run --stage architecture/process1-smoke --overwrite
```

旧运行自动移动到 `results/preserved/runs/<实验ID>/attempt-N/`，旧状态和外层启动日志也单独保留。即使失败发生在原始目录创建之前，已有状态/日志也要求显式覆盖并先保留。新状态或旧尝试不完整/过期时，程序要求显式覆盖，而非默默复用。派生分析重建使用：

```bash
.venv-vllm/bin/python tools/reproduce.py analyze scheduler --overwrite
```

旧分析自动保存到 `results/preserved/analysis/<主题>/attempt-N/`。工具不会自动删除失败记录。

- 汇总、图表、审计和已执行 Notebook 是派生结果；当前主入口自动重建汇总分析，旧图表/Notebook不在自动执行范围。失败或不完整结果必须按其真实状态解释。
- 同一语义实验可以重跑，但旧失败、特殊中断和曾可运行的尝试必须保留。统一入口自动保留旧 attempt 后建立新 attempt，不能静默删除原始结果。
- 确定不支持的配置保留原因与完整失败记录；偶发资源、超时、网络或 profile 导出问题不能被改记为“实验不可行”。
- 当前正在运行的实验不得被覆盖。旧结果被替换后，依赖它的摘要需要从新输入重新生成，不能沿用旧 PASS 标志。

首次复现不需要旧生成物。已经获得原始结果后，删除 `results/analysis/` 可以重新分析；删除 `results/runs/` 则必须重新执行相应 GPU 实验，不能仅靠报告恢复原始证据。

## 9. 原生发布、打包迁移与复现验收

目的：得到可搬到另一目录或机器的原生源码包，并保证运行过的工作区不会把生成物重新带入发布。

源码包输出路径必须显式指定，且只允许位于项目的 `results/` 内或项目外，避免源码目录捕获上一次生成的包。工具自动创建父目录。包或同名 manifest 已存在时拒绝覆盖，需要选择另一包名：

```bash
python3 tools/reproduce.py package --output results/packages/portable-inference-source.tar.gz
```

预期生成 tar.gz 及相邻 `.manifest.json`，记录成员 SHA 和整个包 SHA。包内顶层为 `portable-inference/`，解压时会创建该目录。源码包只包含原生工具、固定输入、依赖快照、Notebook 模板和必要文档；排除 `.git`、results、虚拟环境、缓存、历史归档、历史结果、验证报告和自动 process 索引。先执行 `prepare` 再打包，也不应改变原生文件集合。源码包和包的 manifest 自身是生成物，不提交到源码仓库。

发布时以 Git 实际跟踪清单为准，不能只凭目录里没有 results 就判断干净。检查 `git status --short` 和 `git ls-files`：不应列入环境、缓存、日志、已执行 Notebook、生成索引或历史结果。已经跟踪的文件不会因后来添加 `.gitignore` 而自动停止跟踪。源码软链接会明确拒绝；模型是项目外输入，不放入源码树。

需要移交历史证据或新运行结果时，单独选择项目外的归档或 results，并保留相应身份清单；它们不混入原生源码发布。

移动后的最小检查顺序：

```bash
python3 tools/reproduce.py list
python3 tools/reproduce.py prepare
```

然后按第 3 步建立本地环境、按第 4 步准备模型、按第 5–6 步进行验证和实验。无需修改项目内部相对路径。历史路径只作为来源标识保留在必要的输入元数据中，活跃执行链不能依赖项目外的历史归档。

复现验收应逐项记录：源码/配置身份、实际软件版本、模型来源和核查方法、硬件与拓扑、每个阶段的实际命令和终态、结果审计、失败或跳过原因。报告应区分：已实现、已执行、CPU 测试通过、GPU 实测完成、独立重算通过和未验证。

本指南描述固定 CLI/catalog、依赖、模型和运行协议。源码发布不包含既有 GPU raw，也不携带旧 PASS 报告作为新运行证明。实际完成的工程检查与未验证边界记录在 [progress.md](../progress.md)；完整 GPU 推理、性能与质量需要在目标机器按各阶段重新执行。
