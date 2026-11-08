> 原生研究方法与协议文档：保留固定研究问题、输入约束和计量口径，不表示本次实验已执行。阶段、命令和输出位置见[复现指南](../../../../doc/reproduction.md)；实际结论须由本次原始结果支持。

# Agent Session 与任务级服务能力协议

本协议承接容量与长上下文研究，按 sessions 的 pilot、匹配筛选、开放筛选、确认、continuation 五个 process 执行；不改写已有证据。主线使用GPU0–3、4×RTX PRO6000 Blackwell PCIe、只读0731、vLLM0.33.0、Marlin/V4 sparse MLA、FP8 KV、maxlen131072、maxseq64、batch预算4096、memory .90。不升级依赖，不使用或干预GPU4–7。

## 研究假设与单位

- Request：一次流式模型调用。真实工具回合分别记录首text、首tool delta、首任意输出、完整合法tool-call和完成时间；tool-only不能伪造text TTFT。TPOT仍是平均间隔，不是ITL。
- Session：一个外部到达触发的顺序依赖模型/工具循环。不同Session并发，Session内部等待模型输出和工具结果。User Task在本阶段与一次独立Session一一对应，不跨失败session自动重试。
- 固定token轨迹：后续输入只由已保存token银行决定，忽略当前输出，因此能比较精确工作量但没有任务质量成功率。正常EOS闭环：后续消息使用真实模型tool_calls/结果/历史，工作量与成功率可以不同，不能强行对齐调用数。
- 成功任务/s（quality-qualified Session Goodput）=独立客观评分成功且自然结束的任务数/完整cohort wall。无Session时延阈值，不能称latency-SLO Goodput。旧request Research SLO只用于可比纯文本回合；>16K联合阈值仍未定义。

核心假设：APC的Session内共享历史降低重复Prefill；跨Session共享收益依赖确切前缀和部署路由；DSpark适合较长输出但不保证工具闭环质量；PP对冷长Prefill可能有优势；长上下文的cap+cadence不自动迁移到Agent。通过计数和同负载对照验证，不把cache标签当命中。

## 预定工作负载

固定回放三种人工构造shape，均4轮，精确token IDs、ignore_eos固定输出，包含可控工具等待：

| profile | 逐轮输入tokens | 逐轮输出tokens | 三次工具等待s |
| --- | --- | --- | --- |
| coding | 8192/12288/16384/20480 | 128/256/128/256 | .1/.2/.1 |
| research | 16384/24576/32768/40960 | 128/128/128/512 | .5/1/.5 |
| data | 4096/6144/8192/10240 | 128/256/128/256 | .1/.5/.1 |

每Session完整前缀链严格相等，前三轮模型生成不进入下一请求。所有Session共享2048token系统背景，之后为各Session固定唯一历史。等待是明确合成工具耗时，不是实测外部搜索/工具性能。三类只代表Agent形状，不称真实Coding/Research生产轨迹。真实任务另保存模型完整工具轨迹；如后续导出真实轨迹回放将使用新版本计划，不能改写当前负载。

闭环使用六个新增人工构造本地任务：两Coding多文件bug、两固定语料Research证据任务、两CSV Data Analysis任务。模型自主决定工具与停止，最多12轮、当前客户端每轮8tool、每session32tool。客户端缺省每轮768token自然EOS，`session-tp4-apc-pilot`计划显式采用1024token；后续以冻结计划值为准，不混合不同预算比较。独立评分代码与答案不进入模型上下文。任务状态各Session新建；有限纯函数测试执行器不是通用Python安全沙箱。本地人工任务不冒用公开SWE-bench/AgentBench/τ的数据或分数；新增外部语料须另外登记来源、版本、许可与采用范围。

## 缓存控制与候选

每phase在本服务没有在途请求时reset APC，并保存HTTP200/success=true响应；同phase不随意清缓存。`cache_salt` none模式每request新salt，session模式每Session固定独有salt，shared模式整个phase共享salt。实际token IDs在三个模式保持一致，差别仅命名空间；APC off部署做完整参照，V4命中受group状态/块对齐限制。

第一项 `session-tp4-apc-pilot`以TP4 APC启用、仅session cache scope，验证三种固定回放shape、六任务闭环和Session并发；三个salt scope的CPU语义检查不等于三模式GPU实测。`session-tp4-cache-screen`已登记none/session/shared三个scope的固定回放对照，按其冻结计划执行。CPU验证通过并冻结后启动。后续匹配架构优先TP4 off/on、TP2PP2 APC、DSpark3 APC，各配置保持同工具parser/reasoning none/基础预算。根据结果决定DP4EP默认/Session固定rank对照；源码支持`X-data-parallel-rank`，默认不根据cache_salt保证亲和，性能尚未验证。暂不预设cap+cadence或独立双副本最优。

## 并发、开放到达与稳定性

闭环C表示固定数Session workers，每个Session完成再领下一任务，不称开放容量。开放Poisson以Session为到达单位，预定seed/时长/offset/任务类型，在horizon结束后继续完成既有Session的全部后续模型调用和工具步骤。max_active_sessions仅保护上限，触发记录censored，不能悄悄以semaphore减速后声称稳定。

`finite-session-v1`继承容量研究的有限窗口思想，计量单位改为Session并明确命名，精确判据以[冻结数值规则](finite-session-v1.md)为准：后75%窗口分三窗，精确Session outstanding积分；epsilon=max(2, .05×后段Session arrivals)；小于180s仅SCREEN_ONLY，后段到达不足20则INCONCLUSIVE。确认同完整配置、相同任务/shape mix/rate至少2不同seed长窗，所有实际开始的同点长窗都必须通过，不挑seed。server request waiting与Session数量单位不同，单独报告其趋势/覆盖，不能用相同epsilon相减。

执行前应核对实现与冻结数值规则一致，并保留独立方法审查。服务稳定性不代替任务质量，快速失败或截断任务不能成为“更高成功任务容量”；传输失败/超时/保护截断均保留并阻止健康容量认证。边界未夹住仅报最高已测试有限确认点，不能称数学极限/生产SLA。

## 数据与验证

复用run_logged.py GPU锁、源快照、有限超时、GPU每秒采样与自身进程组清理；只访问localhost:8260、不重试请求。每phase保存reset、原始Prometheus；每Session保存开始/结束/计划到达、每模型payload/完整SSE/usage、工具参数/结果/耗时、任务工作区、评分与结束原因。启动预热单列，不混性能。模型时间/工具时间/评分/客户端调度与Session总时间分开；服务端排队只有聚合Prometheus均值，不伪造每request纯GPU时间。

CPU先覆盖流式解析、工具调用、usage、开放到达/drain/失败、前缀链和salt、任务初始失败/参考成功、隔离及超时。实际GPU终态之后独立核对source/hash/任务状态/计数和指标。重要优化若APC计数与固定负载已能解释，先用证据判断是否需要独立Profile；Profile始终不参与性能排名。保留失败配置/不可用任务、质量差异和共享主机噪声，不删慢样本。

运行代码位于本主题 `scripts/`，固定计划和任务语料位于 `research/dsv4-flash0731/experiments/scenarios/workloads/sessions/`。原始证据写入项目根 `results/runs/<实验ID>/`，分析和生成报告写入 `results/analysis/sessions/` 及其中的 `doc/`。Notebook 模板保留代码，执行结果另存；不将其他主题的KV硬极限或异构放置假设当成已验证能力。当前执行定义见[实验 catalog](../../experiments/catalog.json)。

## 固定输入版本与接口边界

当前客户端的每轮工具上限统一为8；`session-tp4-apc-pilot`使用1024输出预算，`session-tp4-agent-pilot2`使用4096输出预算，自然EOS、12轮和32总工具不变。两者输入预算不同，不能合并为相同条件的测量。严格评分与原始任务语料保持固定；任务质量和结束原因须由本次原始结果报告。

Data filter value明确为integer/string/null/array并明确items类型，按列严格验证，来源校验与任务fixture不变。缺少类型声明可能使ParserEngine将数字解释为string；接口schema改变也会改变tools prompt，pilot2与后续匹配筛选不能称完全相同任务输入的纯速度对照。数学答案与评分规则不改。通用union对数字样式string有类型优先级限制，当前固定语料列未包含此类字符串，不声称适用于任意数据集。