# 当前任务：纯原生源码发布整理

> 原生维护记录，按 AGENTS.md 保留在项目根目录。验证报告和运行日志保存在项目外，不属于源码发布内容。

目标：可直接上传 GitHub 的完整原生源码树。保留代码、固定实验输入、任务语料、评分器、未执行 Notebook 模板和必要方法文档；移出历史结果、报告、环境、缓存和自动索引。范围不含联网安装、真实 GPU 复现或远端推送，不改变研究目标、测量协议和已接受基线。

| 阶段 | 状态 | 已核实的结果 |
| --- | --- | --- |
| 状态与依赖盘点 | COMPLETED | 本轮开始时无 .git；原生输入与运行产物已分类，必要 pair/retry 设计保留。 |
| 本地产物移出与初始保护 | COMPLETED | 44 项按原相对路径移到项目外，rename 的 inode 逐项核对；初始非环境树另有归档。 |
| 纯源码发布规则 | COMPLETED | Git/package 排除环境、缓存、结果、历史报告和 prepare 自动视图；拒绝意外生成 JSON、已执行 Notebook 和软链接。 |
| 干净副本与 CPU 验证 | COMPLETED | 851 项源码发布检查、83 实验 dry-run、12 项迁移、38 项启动守护、70 项 profile 控制检查通过。 |
| 文档闭包与最终审查 | COMPLETED | 顶层文档 93 项检查通过；7 份研究方法文档原文已外存，去除旧状态/路径/断链后 71 项检查通过。 |
| 原生 Git 起点与提交 | IN PROGRESS | main 已初始化，375 原生文件已暂存；保留 CRLF 的空白检查通过，尚未提交。 |
| 完整 GPU 复现 | NOT VERIFIED | 本轮无 GPU 推理、NCCL、真实 profile 或性能测量。 |

## 原始文件保护与证据位置

本机保留目录：`/home/liuweiming/playground-review-01-local/native-release-backup-03zsrq5u`。此绝对路径只定位本次本机证据，不被复现代码读取。

- `source-before-publication.tar.gz` 与 JSON 清单保存本轮实现修改前的 401 个非环境文件。
- `preserved/` 按原相对路径保留迁出的环境、缓存、初始 archive、历史报告、orders_history 和旧进度。
- `moves.json` 保存 44 项迁移前后位置及 inode 核对。未删除旧数据；移出的虚拟环境仅作保留，新环境按指南重建。
- `validation/` 保存下列完整报告和相关日志；它们不进入 Git 或源码包。

## 实现与必要决策

- `tools/reproduce.py`、`.gitignore`：统一原生发布范围；prepare 前后包内容不增加运行产物。打包只写 results 或项目外，先规范化路径、检查原生来源，再创建目标目录。
- `tools/check_source_release.py`：新增真实打包/解包、临时 Git 候选、全量 dry-run、污染排除和输出边界集成检查；检查在临时目录进行。
- `tools/check_portability.py`：旧归档比对改为可选 `--reference-root`；新复现不依赖已移出的历史。
- scheduler 的 `analyze_profiles.py` / `check_profiles.py`：相同源码下已登记的 profile 导出重试按输入、请求和运行身份严格比较；旧不同源码例外继续要求原始证据，不靠旧日志完成新复现。
- `README.md`、`doc/reproduction.md`、`doc/layout.md`、`doc/artifact-policy.md`、`doc/roadmap.md`：说明原生/生成边界、阶段命令和限制，不承诺源码中包含旧验证报告。
- 7 份研究方法文档保留方法、数值规则和研究约束，移除旧完成状态、实测结果总结与断链，原文另存 `preserved/pre-final-doc-review/`。Session 每轮工具上限按现有代码纠正为 8，未改代码或固定计划。
- 固定配置、工作负载、种子、评分器及协议决策不改；189 个受保护输入/依赖文件与初始字节一致。

## 已执行验证及限制

| 方法与预期 | 实际结果 | 项目外报告 |
| --- | --- | --- |
| 真包解压到带空格目录，从 / 调用 prepare/validate 和所有可执行定义 dry-run；prepare/污染注入前后 Git 与包成员及字节不变 | PASS，851 项，375 成员，83 dry-run，105 个自动视图仅在临时副本生成 | source-release-r2.json |
| 自动目录、恢复、覆盖、前置依赖及异名目录调用 | PASS，12 项，合成 CPU 生命周期 | portability-r1.json |
| 主机协作锁、设备守护、启动前失败留证 | PASS，38 项，CPU mocks/flock，不查询真实 GPU | launch-guard-r1.json |
| profile 账本与已登记重试的正/负控制 | PASS，70 项，CPU mocks | native-profile-comparison-r1.json |
| 顶层文档命令语法、链接、21 阶段与实际 CLI 对齐 | PASS，93 项；不代表安装命令或 GPU 已执行 | native-publication-docs-review-final.json |
| 固定输入字节、Python AST、Shell 语法、Notebook 未执行、无生成 JSON | PASS，189 固定文件不变，173 Python 文件，375 总文件 | frozen-inputs.json、native-tree.json |
| 独立复核打包输出路径边界与拒绝行为 | 初次发现问题，修正后独立复测 5 项 PASS | independent-review-initial.json、independent-review-final.json |
| 研究方法文档链接、catalog ID、当前预算、冻结判据和改动范围 | PASS，71 项；最终收尾仅改文档，已验 CPU 代码保持不变 | final-research-docs-review.json |
| 原始 CSV、工作树、Git index 三方字节比对，识别 CRLF 后检查空白 | PASS，两文件各 25 CRLF 行，未规范化数据 | git-crlf-review.json |

执行入口为 `python3 -B tools/check_source_release.py --output <外部报告>`、`check_portability.py --output <外部报告>`、`check_launch_guard.py --output <外部报告>`，以及 `python3 -B research/dsv4-flash0731/scheduler/scripts/check_profiles.py --output <外部报告>`。各命令的 stdout/报告保存在上述 validation 目录。具体复现命令、每阶段目的和输出见 [复现指南](doc/reproduction.md)。

## 失败、修复与边界

- 默认 sandbox 的 bwrap 初始化因 loopback 权限失败；之后由工具审批在受控路径执行，未联网或启动 GPU。
- 独立审查发现打包输出的 `..`/软链接可能绕过目的目录限制；已规范化目标路径并在任何目录创建前验证，新增正/负回归并通过独立复测。
- `git diff --cached --check` 首次将两份原始 CRLF CSV 的每行判断为尾随空白，日志保留为 `git-prepare-3.log`。数据与归档、index 字节一致；仅在本次命令用 `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check` 识别 CRLF，检查通过，不改数据或全局 Git 设置。
- 本轮新建 main，不能视为恢复此前不存在的 Git 历史。未设置或推送远端。源码/CPU 验证不等于 GPU 性能复现，未来需按协议实测并由用户接受新结果。

下一步：核对最终暂存内容，建立本地源码提交，完成 Git 干净状态与项目外源码包的一致性验收。
