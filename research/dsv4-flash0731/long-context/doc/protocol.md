> 原生研究方法与协议文档：保留固定研究问题、输入约束和计量口径，不表示本次实验已执行。阶段、命令和输出位置见[复现指南](../../../../doc/reproduction.md)；实际结论须由本次原始结果支持。

# 长上下文混合调度协议

本协议沿用容量研究的模型、软件、硬件与计量口径；按 long-context 的架构/阈值筛选、cadence 筛选、独立 profile、匹配确认四个 process 执行。固定输入集中在 `research/dsv4-flash0731/experiments/scenarios/workloads/long-context/` 和对应 configs，原始证据写入项目根 `results/runs/<实验ID>/`，分析写入 `results/analysis/long-context/`。每次运行保留实际源码快照，已有证据不改写。

## 问题与前提

在 GPU0–3（4×RTX PRO6000 Blackwell，NUMA0/PCIe、无NVLink）上，测试16K/32K/64K长输入与512 token短请求共存时的排队/尾延迟、KV驻留压力，以及架构/调度取舍。只读0731权重，vLLM0.33.0、Marlin、FP8 V4 KV、block256、maxlen131072、max-num-seqs64、GPU内存比例.90、APC off。若共享主机GPU4–7存在外部负载，应记录该限制，不使用或干预。

容量与场景中的总预算1024、4096、8192对照不在本主题重新排列；本阶段固定4096总预算，研究单请求long-prefill threshold。总batch token budget是每轮总和，不能直接称每请求固定chunk size。显式chunked prefill ON；关闭chunking需要budget≥maxlen，改变多个条件，作为技术限制记录，不列入主对照。

## 预注册负载

- `context_mix`：512/128占60%、16384/128占15%、32768/128占15%、65536/128占10%。固定种子随机抽shape、开放Poisson，筛选0.5和1 req/s各90秒，所有候选到达序列和精确token IDs一致。小样本长桶只描述，不声称可靠尾部分位数或容量。
- `interference64`：显式开放到达trace，短512/128每1秒一次，64K/128在8、20、32、44、56、68、80秒到达，horizon96秒；两个不同prompt seed。它用于重复观察长Prefill共存，不能标称Poisson容量。长请求不保证隔离，实际交叠由时间线分析验证。
- `kv_residency`：8个64K/128请求在0秒到达，16个512/128请求在0.5起每0.5秒到达，horizon12秒。是有界并发/排空压力实验；逐shape延迟、waiting/running、client outstanding、KV usage与抢占共同解释，不能将预留显存当有效KV占满。
- 每种shape先C1暖机1次，另以C8混合8请求暖机。暖机不纳入性能排名。重复shape也可能有新编译，查看服务日志后披露。

请求都以temperature0、ignore_eos、fresh TCP、无重试执行。timeout300秒、max_inflight128，触发保护/失败即停止后续phase并保留证据；运行只访问本任务localhost:8260，GPU0–3串行、锁、启动空闲检查、进程组清理、唯一目录。不会向其他服务施加负载。

## 架构与调度顺序

先匹配TP4、TP2×PP2、TP4+DSpark3、DP4+EP四架构。DP4+EP仍为完整部署比较，每engine4096预算不能与TP4视作相同全局预算；DSpark模型驻留也不同，报告实际KV池。

调度对照：TP4默认`long_prefill_token_threshold=0`对TP4阈值2048、adaptive off；保持总预算4096和其余参数不变。需源码与实际启动日志共同核实。若阈值对短请求有清楚收益，再在场景候选架构验证，不做全排列。

后续确认依据筛选：选择有意义的吞吐/短TTFT折衷点，用至少180秒、两个seed的匹配Poisson长窗；优先300–600秒，使64K桶的样本数可见。不随结果删去失败seed，不按数据挑选有利负载。未夹住容量边界只报已测点/下界，不以本阶段有限窗口声称长期生产SLA。

## 评价与验证

cadence 单因素筛选使用 `context-tp4-th2048-cad8-screen` 对照 `context-tp4-th2048-screen`，仅改 `prefill-schedule-interval=8`，其余配置、10阶段、输入/到达完全一致。假设有Decode时为其留出无Prefill步骤可降低TPOT；长Prefill延长、重负载capacity escape解除节流均是预期可能代价，必须报告，不预写成功。该候选在筛选后、独立Profile前运行；Profile始终独立于性能。

固定确认计划为 `context-tp2pp2-confirm` 与 `context-tp4-th2048-cad8-confirm`：TP2×PP2默认对TP4 threshold2048+cadence8，保持同context_mix，0.5 req/s、450秒×2，seed202611670/671，分别222/252请求，其中64K32/24、short129/160。两组按相同长窗检验，不重新排列四架构；不能从候选选择推断实测收益。三组独立Profile为 `context-tp4-profile`、`context-tp4-th2048-profile`、`context-tp4-th2048-cad8-profile`，分别默认/仅cap/cap+cadence，均相同7请求、19暖机，不进入正常排名。

沿用用户Research SLO：≤1K/4K/16K输入TTFT≤1/3/8秒、TPOT≤50ms，E2E≤对应TTFT限额+.05×(实际输出−1)。P95逐输入档判定；32K/64K无联合SLO，不把它们计为失败或成功。单独给短512请求SLO满足率和Goodput、每shape P50/P95/P99、TPOT与E2E、失败/超时。总吞吐、符合既有SLO子集的Goodput和稳定性分开，禁止把适用子集P95通过称整个长上下文混合SLO容量。旧统一SLO另报。

开放Poisson/trace从计划到达计时并报告dispatch lag；固定horizon与drain分开；吞吐使用cohort/(window+drain)，窗口完成量另报。容量稳定性沿用finite-capacity-v1，trace/压力只描述，不认证稳定容量。

证据：每请求JSONL、完整文本/usage/hash、prompt bank、源码/配置快照、每秒原始Prometheus与GPU采样。重点分析精确16/32/64K请求的pre-first-text及post-first-text时间线和短请求关联；pre-first-text含排队，不能等同纯GPU Prefill或kernel因果。KV usage按engine报告、不能误加成百分比。必要独立Profile排除于性能排名。

CPU先验证trace开放发送/顺序/同到达/计时/drain/异常拒绝，客户端和独立审计回归；核对当前配置、固定输入和运行快照的SHA。GPU终态后独立重算请求/usage/SLO/到达身份、资源指标与状态；不以仅写完代码标记COMPLETED。

固定定义及阶段依赖见[实验 catalog](../../experiments/catalog.json)，容量判据见[容量协议](../../scenarios/doc/capacity-protocol.md)。
