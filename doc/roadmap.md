# 路线图与研究 backlog

> 原生维护文档。当前主线是纯原生源码发布和可迁移复现，不改变研究目标、测量口径或已接受基线。当前执行状态见根 [progress.md](../progress.md)。

| 主线阶段 | 目标 | 状态 | 验收依据 |
| --- | --- | --- | --- |
| 初始文件保护 | 在项目外保留初始源码、历史证据和本机环境 | COMPLETED | 44 项迁移逐项核对、初始源码归档，具体位置见当前进度 |
| 可迁移路径与自动目录 | 项目内部相对定位、外部模型变量、自动输出目录 | COMPLETED（已有 CPU 验证） | 现有路径模块、入口与迁移检查源码 |
| 集中实验与阶段依赖 | 84 定义 / 83 可执行 / 21 阶段，catalog 单源 | COMPLETED（静态定义） | catalog、固定输入、process 依赖 |
| 纯原生发布范围 | Git 和源码包都排除生成物，prepare 前后发布集合一致 | COMPLETED（CPU 验证） | 851 项发布检查、83 实验 dry-run、375 原生文件与 Git 候选一致 |
| 复现指南与产物分类 | 每个 process 的目的、命令、预期和失败处理 | COMPLETED | 顶层文档 93 项检查、研究方法文档 71 项检查；原文和报告另存项目外 |
| 完整 GPU 复现 | 在目标环境按固定协议运行和分析 | NOT VERIFIED | 需要本次真实执行证据，源码发布不附带旧 raw |
| 新结果接受 | 复核性能、质量、限制，决定是否接受新参考 | NOT STARTED | 用户最终接受，不能由源码整理自动提升基线 |

复现主线为通信/架构 smoke → screen → full → long → 场景 → 容量筛选/边界 → 长上下文筛选/诊断/确认 → Session pilot/筛选/开放/确认 → 调度诊断与匹配性能。容量额外下界是可选分支，Session continuation 与 scheduler 是 confirm 之后的不同分支。真实依赖以[复现指南](reproduction.md)和 catalog 为准。

## Backlog

| 分类 | 项目 | 与当前主线关系 | 启动条件 |
| --- | --- | --- | --- |
| Mainline | 纯原生树验证并建立本地提交（COMPLETED） | 直接满足 GitHub 发布范围要求 | 375 原生文件已验收并提交 main；GPU 运行另行验证 |
| Mainline | 目标机器新环境、模型与 GPU 全流程 | 验证运行能力和研究复现 | 按指南建立环境与模型，CPU 预检通过 |
| Mainline | 审计新 raw 与分析 | 防止将监督器成功当研究结论 | 原始结果完整，分别检查质量、稳定性和 SLO |
| Extension | 统一绘图和 Notebook 自动入口 | 当前 analyze 只汇总/分析 | 主复现稳定后另行确定范围 |
| Extension | 新结果证据包与完整依赖来源哈希 | 源码包不含 raw，版本锁不含全部 wheel 哈希 | 明确证据范围及安装来源，不改测量协议 |
| Extension | 分类历史专用检查与新数据审计 | 部分原生检查依赖旧 raw 和固定数字 | 不通过放宽断言伪造通过 |
| Exploratory | Decode 成本、准入/隔离、并发长上下文质量 | 后续研究方向，不是发布整理所需 | 用户确认目标、输入、基线和验证协议 |
| Exploratory | Attention/MoE 分离或新 kernel | 改变架构与资源条件 | 独立研究计划和用户决策 |

`capacity-tp4-lower-bound` 与 `capacity-dspark3-prefill-bound` 是可选实验；`scheduler-tp4-fairshare-dense-profile` 仅保留历史输入，完整 retry 定义单独存在。发布源码保留这些固定定义，运行结果由实际实验生成。
