> 原生研究方法与协议文档：保留固定研究问题、输入约束和计量口径，不表示本次实验已执行。阶段、命令和输出位置见[复现指南](../../../../doc/reproduction.md)；实际结论须由本次原始结果支持。

# 在线服务容量协议

本协议承接场景研究，按 capacity 筛选、边界确认、可选下界确认三个 process 执行；已有尝试与评价保留。结果是候选容量参照，用户接受前不提升为稳定项目基线。

## 固定条件与目标

GPU0–3，四张RTX PRO 6000 Blackwell，PCIe，无NVLink；只读本地0731权重、现有vLLM0.33.0环境。固定比较四份配置：TP4、TP4+DSpark3、TP2PP2、DP4EP。maxlen131072、max-num-seqs64、batch-token4096、Marlin、FP8 V4 KV、APC关闭。DP4EP的预算是每engine/rank口径，aggregate预算不等同TP4，因此比较的是完整部署，不能把差异单独归因于EP。

目标是按负载区分SLO达标容量、最大稳定服务吞吐和无SLO峰值。旧48请求Poisson与固定并发数据仅作筛选参照，不作为本阶段长期容量证据。APC、Agent、真实64K和batch-budget由对应研究主题覆盖，不在容量协议内重复展开；模块异构分离与后续优化不在本阶段实施范围。

## 负载与执行

1. `short_mixed`：512/128占70%、4096/256占20%、16384/128占10%，观察长prefill对短交互的干扰。
2. `long_generation`：512/128占70%、1024/1024占30%，观察长decode与短交互共存。

比例用于有种子的随机抽样，实际各shape数量另报。所有架构同phase seed、同计划到达、同精确token IDs、同输出预算；temperature0、ignore_eos、fresh TCP、无POST重试。合成内容不代表真实业务分布；本阶段不新增质量结论。

Poisson为开放式到达：下一请求不等待上一请求结束；发送器不以semaphore限速。每阶段固定arrival horizon，随后独立记录drain。max_inflight256仅为本地研究保护，触发则逐项记censored并停止后续阶段，不把被截断的流标为稳定容量。单请求300秒超时，整个实验有总超时，所有失败保留。burst为同时提交64个mixed请求；它不是固定并发闭环，不用于认证稳定容量。

粗扫：每架构、每mix先用0.5/1/2/4 req/s × 60秒；若已显著过载不追加无意义高率点。根据旧数据和粗扫，仅在有价值的边界补中间点或更高速率。确认：候选边界使用至少180秒、两个不同seed的匹配重复；低rate或稀有shape样本不足时预先增加到300秒。每个窗口长度在生成计划前确定，不按观测结果或请求数提前结束。四架构串行，暖机与正式phase分离；burst仅在TP4及有数据支持的候选上做，不做全排列。

## Research SLO（用户确认，非业务SLA）

```text
TTFT limit = 1s for input <=1024
             3s for input <=4096
             8s for input <=16384
             undefined for input >16384
TPOT <=0.050s
E2E <= TTFT limit +0.050s*(actual completion tokens-1)
```

开放到达与burst的TTFT/E2E从计划到达计时，另报HTTP发送起算的时间和client dispatch lag；闭环只用HTTP发送起算，不把等待worker的时间说成服务端排队。TPOT=(E2E−TTFT)/(实际输出tokens−1)，不是逐token ITL；单token输出TPOT不适用。

P95为主判定。逐输入档位计算TTFT/对应阈值和E2E/逐请求阈值的P95，均须≤1，TPOT P95≤50ms；观察到的各适用档位须分别通过，避免占比小的长输入被总体分位数掩盖。各shape也单独报告。P50/P99、样本数和联合SLO实际满足率同时给出；不额外要求P99达标或99%联合满足率。失败/超时不填零进入成功延迟分位数，但计入请求满足率分母且阻断容量认证。小样本P95/P99是描述量，无统计置信保证。16K以上没有TTFT/E2E或联合SLO通过率，独立报告原始延迟与可评的TPOT。

旧统一TTFT≤1秒、TPOT≤50ms、E2E≤10秒保留为legacy口径，不能替代新口径。

## 稳定性与容量

稳定性独立于SLO。保留 `stable_in_window` / `growing_backlog` / `inconclusive` 三态，依据后半窗口到达与完成、client inflight和server waiting趋势、drain、失败和超时共同判断；窗口结束仍有在途请求本身不表示不稳定。具体预先固定的可执行判据见[分析器中的 RULES 与 classify 实现](../scripts/summarize_capacity.py)。60秒粗扫只能筛选，边界必须用更长匹配重复确认；有限窗口稳定不表示长期SLA。

所有吞吐同时报告两种分母：固定arrival window内完成的请求/输出token除窗口长度，以及完整到达cohort的成功请求/输出token除(window+drain)。前者输出tokens按请求完成时归账，不是窗口内逐token产生率；后者避免隐藏drain但包含启动边界损失。另报按arrival window归一的满足SLO cohort请求数，明确其归属口径。Goodput只计算同时满足逐请求SLO的成功请求，不把全部成功吞吐改名Goodput。

- SLO达标容量：稳定且逐档P95通过的最大已测试并重复确认offered rate，报告其Goodput、请求和token吞吐。
- 最大稳定服务吞吐：不要求SLO，但通过稳定性确认的最大已测试吞吐，并报告对应offered rate。
- 无SLO峰值：有限粗扫/确认/burst中观测到的最高成功吞吐，注明分母和负载；不冒充稳定容量。

未夹住边界时只能给已验证下界；相邻pass/fail点形成测试区间，不宣称数学上的精确最大值。不在不同mix之间混排名。

## 证据和复现

每个 catalog 中的语义实验ID保存命令、Git SHA/dirty状态、配置/计划/source snapshot、完整请求JSONL、usage/输出文本、精确prompt bank、服务日志、每秒GPU采样及原始Prometheus。每phase完成即checkpoint；失败不覆盖、不静默重试；只清理自身实验进程组。计划放 `research/dsv4-flash0731/experiments/scenarios/workloads/capacity/` 并进入快照，代码与计划先CPU验证、本地提交，再启动GPU。

CPU夹具覆盖分段SSE、非200、超时、无DONE/usage、真实输出SLO、Poisson无闭环节流、保护截断、burst和分位数。独立审计重算输入hash、输出usage、时序和summary；旧Profile可解释已见机制，其kernel累计占比有重叠，不能当E2E占比或PCIe带宽利用率。只有新差异需要证据时追加独立Profile，不将Profile时延放入容量排行。

原始结果位于项目根 `results/runs/<实验ID>/`，分析位于 `results/analysis/capacity/`。可执行定义和前置关系见[实验 catalog](../../experiments/catalog.json)。
