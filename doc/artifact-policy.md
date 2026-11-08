# 原生输入、生成结果与保留规则

> 原生维护文档。源码发布只包含重新执行所需的代码、固定输入和必要文档；历史结果和本地产物在项目外保留，清理发布树不等于丢弃证据。

| 分类 | 位置或例子 | 发布与保留规则 |
| --- | --- | --- |
| 原生源码 | tools、研究 scripts、project_layout.py、分析和报告生成器 | 保留并版本管理；生成报告的脚本本身仍是源码 |
| 原生执行登记 | experiments/catalog.json | 唯一执行定义，不手改多份索引 |
| 固定实验输入 | configs、workloads、任务语料、评分器、必要研究决策 | 保留；曾由工具构造也不等于可删的运行结果 |
| Notebook 模板 | notebooks 下无 outputs、无执行计数的文件 | 保留代码和分析步骤；已执行版本保存到 results |
| 自动索引 | experiments 各 process 的 stage.json 和实验描述 JSON | prepare 生成、标记 generated；Git 与源码包都排除 |
| 原生文档 | 协议、方法、复现指南、产物规则 | 保留；不把旧 PASS 或失效报告链接当新验证证据 |
| 原始运行证据 | results/runs 中的请求、SSE、工具轨迹、日志、源码/配置快照、终态 | 不发布到源码仓库；运行后是分析输入，失败也保留 |
| 派生结果 | results/analysis 中的 JSON、CSV、图、生成 Markdown、执行后 Notebook | 不发布到源码仓库；可从 raw 重建 |
| 本机产物 | results/setup、results/validation、虚拟环境、缓存、字节码和锁 | 自动重建，Git 与源码包都排除 |
| 覆盖前证据 | results/preserved | 不发布到源码仓库；保留失败、特殊中断和旧分析 |
| 初始历史 | 初始项目归档、历史结果/报告、迁移清单、历史命令 | 项目外本地保留；不作为新复现必须读取的数据 |

首次复现无需已有 results、process 索引、虚拟环境或缓存，入口会创建必要目录。已有结果以后，删除 analysis 只需重算；删除 runs 需要重新执行测量。任务中的 Markdown 是输入，Notebook 模板中的代码是分析输入，不能按后缀或“由工具生成过”一律删除。

## 覆盖和失败

`run --overwrite` 在新运行前将旧目录移到 `results/preserved/runs/<实验ID>/attempt-N/`，旧运行回执另存 `results/preserved/run-state/`，外层日志另存 `results/preserved/launcher-logs/`。结果目录创建前的失败也保存状态和日志，显式覆盖才继续。`analyze --overwrite` 将旧分析保存在 `results/preserved/analysis/<主题>/attempt-N/`。attempt 序号自动选择，无需创造日期或修改实验 ID。

重复阶段可能复用源码/输入签名和终态一致的尝试；REUSED_TERMINAL 不是新增独立测量，失败终态仍是失败。过期或不完整结果不静默复用。统一入口持项目锁，真正启动前还取得主机临时目录中的四卡协作锁并检查设备空闲。同一临时目录且遵守协议的 CLI 会互斥；不同 TMPDIR、外部程序或其他租户没有此保证。启动前拒绝留下 FAILED_BEFORE_LAUNCH 状态和日志。

不支持的配置、资源暂时不足、运行错误、profile 导出超时、协议失败和答案错误分别记录。失败可以作为一次终态尝试记账，不能改写为成功。可行实验遇到特殊失败仍保留，后续尝试单独保存。catalog 中缺完整启动定义的历史输入不自动执行；完整 retry 定义是另一项实验。

## 状态和基线

阶段 COMPLETED 表示安排的尝试均有终态或明确排除，不代表每个配置通过、SLO 达标或任务质量正确。后续阶段检查源码/输入、环境/模型身份和原始证据；修改 state 不能代替执行。raw 缺失时不补写虚构数据。

历史基线不因源码整理而替换。新运行即使更快，也须记录环境、配置和协议并由用户接受，才能成为新参考。随机种子、负载语义和失败样本不因发布整理改变。

## 原生发布与打包

`package --output ...` 生成源码包及 SHA manifest，只允许输出到 `results/` 内或项目外；这两者是发布产物，不是需要提交的原生文件。源码选择应与 Git 原生范围一致：排除环境、缓存、运行结果、自动索引、验证报告、历史归档及历史结果。执行 prepare 前后，源码包的原生成员集合应相同。`tools/check_source_release.py` 使用真实打包、临时 Git 候选清单和全部实验 dry-run 检查这一约定；它的报告是生成物。

源码包使用明确的原生目录/文件范围并拒绝软链接，避免带入不可迁移的外部依赖。模型是项目外输入，不随源码发布。需要移交旧归档或新结果时，另外选择证据及身份清单；不把它们混进源码包，也不因源码发布而丢弃本地保留副本。

历史专用 `package_evidence.py`、`package_capacity.py`、`package_scenarios.py`、`package_long_context.py` 和 `package_sessions.py` 是原生工具，仍予以保留。它们包含特定历史结果范围、计数或结论，缺少对应 raw 时不能作为新实验成功的证明；统一源码打包入口不替代历史结果归档工具。

`prepare` 为当前调度输入生成配对登记，监督器将它捕获到源码快照，分析绑定实际捕获的版本。必要历史决策中原有 hash 作为来源信息保留，不手改为新运行通过。旧绘图/Notebook 工具保留，主 analyze 产生新 JSON 和 Markdown，暂不自动执行全部展示流水线。
