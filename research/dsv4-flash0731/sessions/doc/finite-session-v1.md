> 原生研究方法与协议文档：保留固定研究问题、输入约束和计量口径，不表示本次实验已执行。阶段、命令和输出位置见[复现指南](../../../../doc/reproduction.md)；实际结论须由本次原始结果支持。

# finite-session-v1：有限会话稳定性规则

本文件保留确认实验的冻结数值规则，与容量/长上下文使用的 `finite-capacity-v1` 区分命名；发布整理不修改判据或历史结果。独立CPU夹具与实际终态审计仍须执行，冻结文档不等于实现或本次测量已验证。

## 范围与计量单位

只对 role=`confirm`、arrival=`poisson` 的开放 **session** 长窗作有限确认。pilot、screen、warmup、profile、closed 并发和固定 burst 不认证容量。动态 Agent 与固定回放分别汇总，task mix/profile、cache、routing、tool latency、预算和模型配置不同不并成一个容量曲线。

会话 arrival 是预先计划的用户到达；terminal 是 session 行 `end_offset_s`，包含工具、最终独立评分与有界清理。本规则认证的是**完整 benchmark session 的协议处理稳定性**，不把它说成纯GPU服务稳定性；另报告用户可见 `user_visible_end_offset_s`/`user_visible_latency_s`。任务评分成功率单独报告。HTTP请求与vLLM waiting均是 request 单位，不代替 session 计数。

## 终态和完整性前提

整个实验 supervisor 已成功终态、无总超时；report完成、全部预注册 phase 有对应完整记录。该 confirm phase 的全部计划session必须有且仅有一行，精确输入/计划身份、事件、usage和时序经独立审计通过。

该 phase 所有session须 admitted，非censored、非timeout/cancelled；全部实际模型请求 `ok=true`、无超时、取消或不完整流。固定回放须 `protocol_ok=true` 且 terminal_reason=`replay_complete`；动态Agent须 `protocol_ok=true` 且 terminal_reason=`natural_stop`，独立grade存在、`grade.success`为严格boolean，并与`task_success`一致。预算耗尽、输出截断、保护拒绝、缺grade、客户端错误或HTTP错误使健康容量确认不成立，保留为INCONCLUSIVE及明确失败原因，不能通过快速失败释放队列认证容量。

自然stop且grade.success=false属于协议完成、任务失败：计入协议完成C，但不计 successful tasks；可形成“协议处理窗内稳定”的结论，必须同时展示任务成功率和失败类别，不能称成功任务容量/SLO容量。一次执行失败的工具动作可以由后续模型恢复；是否阻断质量由最终评分决定，工具协议错误/恢复次数须完整保留。

## 固定分析窗口与观测前提

到达horizon为预注册T，必须T≥180秒。分析区间 `[T/4,T)` 分成三个等长半开区间 `[T/4,T/2)`、`[T/2,3T/4)`、`[3T/4,T)`。不根据观察到的完成数延长或提前停止；低率/长session可在计划生成前选更长T。horizon之后允许已到达session完成后续模型/工具轮次，独立drain不回填窗口内完成数。

分析到达A为计划arrival在该区间的session数，至少20。C为通过上述协议终态定义、terminal在该区间的session数；可能包含T/4之前到达的session，故C>A合法。不另添加独立最少C阈值；stable判据中的完成缺口已经约束C。未达到A≥20为INCONCLUSIVE。

每段session outstanding均值为所有该phase已计划session在段内的 `[planned_offset_s,end_offset_s)` 重叠时长之和除段长。这个量包含会话准入延迟、模型、工具、think time、评分和清理；其状态组成另报。仅在完整成功前提下解释稳定性，不将提前被保护取消的行带入并认证。

所有实际启动session的 `dispatch_lag_s=max(0,start_offset_s-planned_offset_s)` 的P95须≤0.100秒。各时刻须有限且顺序正确；缺值、负持续时间或任意NaN拒绝。P95沿用项目线性插值，无提前舍入。

服务waiting监控只取指标名精确等于 `vllm:num_requests_waiting` 的有限、非负值，按engine标签求和；不把 `num_requests_waiting_by_reason` 再叠加。按实际固定poll interval估算后75%有效覆盖，覆盖≥80%且三段都有有限观测均值。该覆盖是**可观测前提**；waiting的request单位增长单独报告，**不拿session的ε与request队列数比较**。短间隔重复样本不得虚增覆盖；原始Prometheus内容/hash由独立审计核对。

## 三态判定

定义 `ε=max(2,0.05*A)`、`D=A-C`、三个session outstanding均值为q1/q2/q3，`Δq=q3-q1`。所有上述前提成立后：

- **STABLE_IN_WINDOW**：D≤ε且Δq≤ε。
- **GROWING_BACKLOG**：D>ε、Δq>ε，并且q1<q2<q3严格逐段增加。
- **INCONCLUSIVE**：其余情况；任何前提失败也归此类，同时列原因。

此规则是有限观察的保守研究判据，不是无限时域队列稳定证明。稳定分类不要求session在T时全部结束；完成drain也不自动证明窗口稳定。服务waiting、模型在途、工具池等待、会话状态、资源与drain用于解释，不能事后按某架构表现调整ε或删除不利窗口。

## 重复确认和报告用语

对同配置、同workload条件和同名义session rate，至少两个不同预注册seed的confirm窗口；**全部已开始的confirm窗口**均终态完整且STABLE_IN_WINDOW，方可记 `finite_protocol_stable_confirmed=true`。不只选通过seed；失败/不确定/中断未终态的窗口阻断确认。输入实际shape组成随seed变化，保留分桶样本数，不把不同mix合并。

本阶段 `session_latency_slo=null`，没有用户批准的session deadline，不生成session SLO通过率或SLO容量。适用text-only模型请求可保留既有request Research SLO，但不能代替整个session的目标。successful task/s、协议完成session/s、请求/token吞吐和任务成功率分别列出。

“最大已测试稳定吞吐”只能在多个rate终态及完整重复规则核对后，在匹配条件中选已确认点；没有夹住边界则只给已测下界。只有一个rate无论通过与否，都不能称找到了精确最大容量。closed C下的高session/s和快速失败的观测峰值不代替开放稳定容量。

## 实现与独立验证要求

新客户端记录完整session与每次模型/工具事件、初始/最终任务状态和source SHA；正式分析器与独立审查程序分别从原始数据复算A/C、区间积分、dispatch P95、coverage和判定。CPU夹具至少覆盖窗口边界、C>A、精确ε、不足A、非单调增长、任务失败/grade缺失、budget/censor、waiting同名前缀去重、监控覆盖及confirm所有seed聚合。自生成夹具不能替代终态原始数据审计。
