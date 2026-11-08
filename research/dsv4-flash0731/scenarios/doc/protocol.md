> 原生研究方法与协议文档：保留固定研究问题、输入约束和计量口径，不表示本次实验已执行。阶段、命令和输出位置见[复现指南](../../../../doc/reproduction.md)；实际结论须由本次原始结果支持。

# 场景化研究协议 v1

本协议承接四卡 PCIe 架构筛选，模型、环境与拓扑按复现指南核查；新尝试保留独立原始证据，不覆盖旧尝试。研究按 scenarios 的场景、独立 profile、优化三个 process 顺序进行，不构成模型质量榜单或单一最优配置认证。

## 问题、优先级与参照

1. 长输出Decode：短生成参照为512/256；本项固定1024输入/1024输出，在C1/C8比较TP4与DSpark，查看投机收益随并发变化。
2. 长上下文：以32K参照为起点，统一maxlen131072，实际65536输入/128输出、C1/C4，比较TP4/TP2PP2，必要时DSpark，不能把配置上限当实测128K。
3. 历史/前缀复用：APC关闭/开启成对，在8192和32768公共前缀+128新后缀上测冷、完全重放、首次分叉、串行/并行复用。若默认检查点导致第一次分叉miss，再仅调整prefix retention做单因素验证。
4. 到达率和高并发：固定seed的48请求Poisson流，2和6 req/s；输入混合512(70%)/4096(20%)/16384(10%)，输出分别128/256/128；另512/128 C64N128闭环。根据TP4参照和profile再选择DP4EP或调度token budget对照，不遍历全部组合。
5. Agent：固定轨迹回放与真实工具闭环分开。三个人工小Python修复任务，工具read/search/replace/test受控、最多8轮；报告任务通过率、工具调用错误、任务时间、模型/工具耗时、上下文增长与命中。不是SWE-bench官方成绩；如用公开真实轨迹需保存来源许可/哈希，并标记只是固定回放。
6. 质量：独立正常EOS短题/格式/任务测试，保留完整输出。逐token一致不作优化门槛；严重退化、工具不可用、明显任务错误必须披露。固定任务与评分器作为原生输入保留。

## 共同控制与计量

- GPU0–3，PCI_BUS_ID，四卡实验串行，flock防本阶段竞争、启动前显存空闲检查。不改时钟/驱动/功耗上限，不动其他GPU或权重。
- vLLM0.33.0、Marlin、SM120 V4 FlashInfer、原量化、fp8_ds_mla、block256，maxlen131072、maxseq64、batch tokens4096、memory0.90；各单因素差异明确记录。FP4 MoE并行/通信能力须以当前环境预检和实际运行证据核验。
- temperature0、seed20261105、精确token输入；性能请求ignore_eos，输出完整保存并核对usage；Agent/质量自然EOS。客户端每请求新TCP连接、无自动重试。
- 每场景默认2轮，每case单独暖机（cache冷/热状态按专门阶段），不同round的prompt有独立标记以避免意外APC交叉命中。跨配置完全相同提示序列。
- TTFT到首非空文本，TPOT=(E2E-TTFT)/(actual output tokens-1)，不是ITL；记录文本chunk时间但不冒称逐token间隔。E2E含连接/服务队列，不含客户端semaphore等待；开放到达率另报告计划到达至首字/结束、dispatch lag。
- 输出吞吐为实际成功tokens/完整批次墙钟；所有失败保留。闭环与Poisson结果不混成同一排名。探索性goodput阈值为scheduled TTFT<=1s、TPOT<=50ms、scheduled E2E<=10s，仅用于说明延迟约束下取舍，不是用户生产SLA。
- P50/P95直接从逐请求计算，注明N；两轮范围非置信区间。记录固定执行顺序和共享主机干扰，重要结论在同协议matched配置复核，不从<2%差距宣称统计显著。
- GPU每1s采样memory/util/power/temperature。按正式阶段时间窗梯形积分估计整卡能量和tokens/J；不是板级仪器测量或CPU/整机能耗，短窗口误差必须注明。启动和编译单列。

## Profile与优化控制

原生torch profiler CPU+CUDA，限定约20个engine iteration，关stack/shape高开销，独立profile请求（16K prefill/128 token decode）保存trace和top kernels。以真实trace文件和CUDA事件为通过证据，HTTP200不足够。正常性能与profile时间分离；图执行下的采样kernel总时长不是端到端时间或绝对利用率。

优先解释Attention/MoE/communication与调度等待；据证据选择prefill chunk、APC retention或并行方案。若只参数优化已获实质收益，不为源码行数新增内核；A-only/E-only开发仍需要独立验证，不冒充本轮已实现。

## 工程、文件与验证

运行代码位于本主题 `scripts/`；配置与负载集中在 `research/dsv4-flash0731/experiments/scenarios/`。统一入口将原始结果、日志和源码快照写入项目根的 `results/runs/<实验ID>/`，分析写入 `results/analysis/scenarios/`，生成报告放其中的 `doc/`。`run_logged` 管理本任务进程组、锁和采样，`serve_scenarios` 负责服务与客户端顺序；Agent 独立客户端不改变性能请求含义。

CPU夹具实际检查SSE、usage、完整输出、fresh连接、错误POST一次尝试、提示确定性与缓存共享；GPU实验开始前提交源码冻结。每ID独立原始目录，不覆盖结果。跨轮不中断编译/修改正在执行的客户端；必要修复新ID重跑，原失败归档。

实现、CPU 检查与 GPU 终态分别记录；本协议本身不声明任何检查或性能结论已经通过。当前执行定义以[实验 catalog](../../experiments/catalog.json)为准。

## 测量细节与版本差异

- Profile suite包含64K/C4混合prefill/decode采样；该诊断独立于正常性能，不改变其他suite的负载、请求和计时口径。
- Agent模式统一每轮reset prefix cache，replay/closed-loop/quality输出上限分别256/768/128；完整收集自然EOS。服务在包含Agent任务时加`--enable-prompt-tokens-details`以返回逐请求cached_tokens，另保存每模式前后metrics。该开关只改变usage响应字段，`scenarios-tp4-nocache`的原始定义没有该辅助字段；端到端含完整响应这一口径不变，报告此测量元数据差异。
- 原始客户端闭环stage也计算scheduled/goodput字段，其中等待semaphore属于批次排队，不能当外生请求到达后的服务延迟。正式报告只对Poisson使用这些字段，闭环使用HTTP发起起点的TTFT/E2E。
- 固定48请求Poisson实际组合为30×512/128、10×4096/256、8×16384/128（概率配置70/20/10不等于小样本实际比例）。两种rate同round共用同一prompt bank以匹配负载；本轮arrival对照全部APC off。若将来在APC on下运行arrival，必须先改成逐case显式reset并另立协议，避免rate2缓存流入rate6。
