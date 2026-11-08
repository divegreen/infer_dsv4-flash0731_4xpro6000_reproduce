> 原生研究方法与协议文档：保留固定研究问题、输入约束和计量口径，不表示本次实验已执行。阶段、命令和输出位置见[复现指南](../../../../doc/reproduction.md)；实际结论须由本次原始结果支持。

# Profile 驱动的调度优化协议

本协议承接容量、长上下文与Session研究，按 scheduler 的独立profile、匹配正常性能两个 process 执行；已有基线不变。新结果须经实际运行和独立审查，不自动成为稳定基线，不追求参数全排列或为开发而开发Kernel。

## 研究问题与证据

研究从稀疏1×64K+6×512与多长请求混合的诊断负载出发，区分running/waiting顺序、cap2048与cadence8的取舍；更密集Poisson和8长突发用于检查边界。APC、DSpark和DP局部性由Session对照覆盖。本主题问题是多条running长Prefill竞争4096步预算时，能否通过可追踪的顺序/份额调整改善短请求尾部，同时限制长请求损失；收益、KV占用及抢占情况必须从本次trace与正常性能分别核实。

通信Kernel累计占比不是E2E或PCIe饱和证明；先解决有直接调度证据的问题。真实Agent的格式/引用/输出预算问题独立保留，不把它们归因于性能。现有adaptive cap在4096/2048和N>1下无有效变化，因此不安排开关测试。

## 干预与实现边界

外置ObserveResearchScheduler原样调用安装AsyncScheduler；FairShareResearchScheduler仅调整running/waiting顺序和一次schedule调用期间的cap，再恢复配置。KV分配、async placeholders、输出生成、原cadence/escape和资源检查保留在父类。策略：Decode与短Prefill优先，按上次进展轮转，waiting short总预算软预留≤512，按running Prefill数动态分配cap256..2048，64步无进展请求提升次序。

软预留和轮转不保证满slot/KV不可分配时立即admit，不构成不饥饿数学证明。running排序也改变FCFS末项preemption victim，这是明确行为差异；需计数preempt与正确性证据。首轮范围TP4、PP1、DP1、APC off、无spec/LoRA/connector、多模态/streaming输入；只测试普通text generation。配置与请求不兼容应明确拒绝，不静默降级。模块不写入venv，不修改权重。

## 首轮Profile诊断

默认诊断ID为 `scheduler-tp4-observe-dense-profile` 与 `scheduler-tp4-fairshare-dense-profile-retry`；`scheduler-tp4-fairshare-dense-profile`保留为历史输入，缺完整启动定义，不自动执行。TP4、maxlen131072、maxseq/active64、总预算4096、cap2048、cadence8、Marlin、FP8 DS MLA KV、SM120 sparse attention、APC off；scheduler类与已登记的profile导出deadline差异必须显式核对，其他服务与负载条件保持匹配。

固定8请求：两条65536/128在t=0到达，六条512/128在.25/.75/1.25/1.75/2.25/2.75到达，horizon4s，完整drain。各phase种子与prompt/offset在两组相同；预热含512/16K/32K/64K、C8及同形状dense暖机。精确Token IDs由本地tokenizer预生成。Profiler CPU+CUDA四rank，start ack在phase origin前，stop在drain后。逐请求scheduler trace开启，normal关闭。

调度客户端基于冻结capacity_client，仅增加明确wire request_id和实际SSE response IDs；原发送/计时/usage/SLO/错误/无重试逻辑不变。server request ID必须与客户端精确绑定，不能凭字符串包含或时间接近猜测。step computed位置是scheduler乐观计数，不是GPU完成；原CPU execution标签与request-level决策分别报告/对账。

预期可证伪观察：原策略short ready却无token时，FairShare应让short获得预算并较早Prefill完成；若只是拉长long/增加step而short没有改善，则无效。每rank预期context134144、generation1016、输出总1024（无spec/零抢占前提），不匹配必须解释或NOT VERIFIED。Profiler时延不能进入性能排行，不能以6short的P95称稳定容量。

## 正常性能与质量验收节点

先完成诊断及正确性/来源审计，再执行已登记的有限正常性能实验。复用长上下文context_mix负载的精确种子/构成，使用调度语义ID及相同客户端进行base与candidate匹配重复，至少两独立窗口并保留两轮各自结果。可增加有价值的8长突发或固定dense trace作边界，不为测试数量全排列。服务启动顺序交错/反序以缓解时间漂移；记录仍不能控制的共享主机差异。

固定fresh TCP、无自动重试、温度0、ignore_eos固定输出、输入Token数量与hash、请求到达、模型权重、GPU0–3、框架与预算。APC off以隔离cold调度；不能直接外推到热Session。性能给逐shape TTFT/TPOT/E2E P50/P95/P99、样本、实际SLO率/Goodput、完整cohort吞吐与drain。资源给GPU板卡能量/显存/device busy、KV/preempt/waiting。>16K联合SLO仍未定义；Research SLO沿用容量协议，未新增Session SLO。稳定性使用既有冻结数值规则且只在适用窗口评价，研究成功不等于最大容量认证。

输出质量独立保留：固定随机提示吞吐不能证明应用质量。正常EOS的受控任务或小型质量探针用同样工具/评分/预算比较，成功率、工作量和功能错误单列；不要求逐token相同。不修改Session评分或删除其失败来提升结果。严重新协议错误、异常usage、漏请求、崩溃应先定位；合理输出差异不能无条件阻断性能优化。

## 运行、追踪与验证

调度脚本位于本主题 `scripts/`，固定配置与计划集中在 `research/dsv4-flash0731/experiments/scheduler/`。统一入口执行项目锁、主机四卡协作锁及启动空闲检查；子监督器保留自身研究锁与进程组，严格CUDA0–3/PCI_BUS_ID。每项唯一ID、有限总timeout、源ZIP、安装关键文件SHA与RECORD核对及GPU采样均须保留在项目根 `results/runs/<实验ID>/`。源和ZIP由同一读入bytes生成，运行结束再次核对；主监督器exit和source_identity_unchanged是验收条件。分析脚本不纳入正在运行的runtime冻结集合，模型与安装源码只读。

操作状态按项目协议记录根 `progress.md`，实验定义见[实验 catalog](../../experiments/catalog.json)，原始日志随对应运行保存；分析与生成报告写入 `results/analysis/scheduler/` 及其中的 `doc/`。CPU验证覆盖正常/异常、source变更、GPU锁、取消/timeout、参数验证、调度request对象状态与fairness边界；GPU证据核验实际模型状态。Profile与性能分开，原始样本不裁剪、不换seed，失败另列。源码/测量审计与真实质量/性能分别验收；Notebook保留未执行模板，执行产物与证据包另存。

## 正常对照名称与包装开销（运行前明确）

四个正常匹配定义为 `scheduler-tp4-observe-matched-670`、`scheduler-tp4-fairshare-matched-670`、`scheduler-tp4-fairshare-matched-671`、`scheduler-tp4-observe-matched-671`。ObserveResearchScheduler是原调度策略的包装对照，trace关闭；它仍有每步counter、try/finally和恢复cap的开销，不是未经包装的原生AsyncScheduler。FairShare继承同一包装，额外排序/预算计算及增多step的Python成本属于实际实现的测量范围。当前配置不含原生Async替代对照；新对照不替换长上下文基线，跨主题比较须披露时间、客户端、超时和包装差异，不能直接归因。正常配对仅改变scheduler类，实际源码与输入快照分别记录。