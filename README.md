## 食刻推荐 (Xiaohongshu 全流程工具)

基于 LangChain + LangGraph 的一键发布工作流，自动完成餐次判断、菜谱生成、文案撰写、内容审核、图片生成以及小红书发布。

### 环境准备
- Python 3.12+
- 依赖通过 `uv` 管理：`uv sync`
- 配置 `.env`（未配置时使用 mock 数据）：
```
QWEN_API_KEY=your_qwen_key
DEEPSEEK_API_KEY=your_deepseek_key
LONGCAT_API_KEY=your_longcat_key
DOUBAO_API_KEY=your_doubao_key
GPT4_API_KEY=your_gpt4_key
XIAOHONGSHU_MCP_URL=your_mcp_endpoint           # 可选
HOWTOCOOK_MCP_URL=your_howtocook_endpoint       # 可选
QWEN_ENDPOINT=...
DEEPSEEK_ENDPOINT=...
LONGCAT_ENDPOINT=...
DOUBAO_ENDPOINT=...
GPT4_ENDPOINT=...
```

### 运行
```
uv run python -m today_eat_what
# 或
uv run python main.py
```

### 工作流步骤
1. 当前时间 -> 判断餐次。
2. 调用 Qwen 生成菜谱（若端点缺失则使用 HowToCook MCP 或内置示例）。
3. 调用 DeepSeek 生成小红书风格标题+正文。
4. 调用 LongCat 审核内容，未通过则自动重写文案。
5. 调用 豆包 生成封面与步骤图片。
6. 调用 GPT-4.5-flash / 小红书 MCP 发布，并返回发布结果。

### 技术要点
- LangGraph 管线在 `today_eat_what/workflow.py`。
- 所有模型调用带超时 + 重试，成本估算在 `CostTracker`。
- 端点缺失时会回退到 mock 数据，便于本地调试。
