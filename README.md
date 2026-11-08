# 推理优化研究与复现

本仓库发布原生源码、固定实验配置与负载、任务语料与评分器、无运行输出的 Notebook 模板，以及必要的协议和复现文档。运行结果、验证报告、虚拟环境、缓存和历史归档不属于源码发布内容。

完整步骤见[复现指南](doc/reproduction.md)。实验按主题和 process 组织，固定输入集中在 `research/dsv4-flash0731/experiments/`，新结果统一写入 `results/`。所需输出目录自动建立，模型默认位于 `~/models/dsv4-flash0731`。

在项目根目录先执行：

```bash
python3 tools/reproduce.py list
python3 tools/reproduce.py prepare
python3 tools/reproduce.py validate
```

之后按指南安装环境、准备模型、预检并依次运行阶段。`prepare` 会生成可删除的 process 索引、依赖安装清单和本次环境元数据；它们不提交到 Git。源码完整或 CPU 检查通过不等于完整 GPU 性能已复现。

- [目录与数据流](doc/layout.md)
- [原生输入、生成结果与失败保留](doc/artifact-policy.md)
- [路线图与研究 backlog](doc/roadmap.md)
- [当前工程进度](progress.md)

`AGENTS.md`、`progress.md` 和本入口按项目协议保留在根目录。重构前初始文件和历史证据另存于项目外的本地保留目录；新复现不依赖这些文件。
