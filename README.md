## 食刻推荐 (Xiaohongshu 全流程工具)

基于 LangChain + LangGraph 的一键发布工作流，自动完成餐次判断、菜谱生成、文案撰写、内容审核、图片生成以及小红书发布。

### 环境准备
- Python 3.12+
- 依赖通过 `uv` 管理：`uv sync`
- 配置 `.env`（未配置时使用 mock 数据，菜谱节点会回退到内置模板）：
```
QWEN_API_KEY=your_qwen_key                     # 或使用 Qwen_API_KEY
Qwen_API_KEY=your_siliconflow_key
Qwen_BASE_URL=https://api.siliconflow.cn/v1
QWEN_MODEL=<qwen-model-id>
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=<deepseek-model-id>
LONGCAT_API_KEY=your_longcat_key
LONGCAT_BASE_URL=https://api.longcat.chat/openai
LONGCAT_MODEL=<longcat-model-id>
DOUBAO_API_KEY=your_doubao_key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3        # 豆包文生图基址
DOUBAO_IMAGE_MODEL=<doubao-image-model-id>                      # 可选，图像模型
DOUBAO_IMAGE_SIZE=1080x1920                                     # 可选，保持 9:16 竖版
GLM_API_KEY=your_glm_key
GLM_BASE_URL=https://api.siliconflow.cn/v1
GLM_MODEL=<glm-model-id>
XIAOHONGSHU_MCP_URL=your_mcp_endpoint           # 可选
HOWTOCOOK_MCP_URL=your_howtocook_endpoint       # 可选
QWEN_ENDPOINT=...
DEEPSEEK_ENDPOINT=...
LONGCAT_ENDPOINT=...
DOUBAO_ENDPOINT=...
GLM_ENDPOINT=...
```

如果需要本地 HowToCook MCP，可在 IDE 配置：
```
{
  "mcpServers": {
    "howtocook-mcp": { "command": "npx", "args": ["-y", "howtocook-mcp"] }
  }
}
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
5. 调用 豆包 Ark 文生图生成 9:16 手绘/涂鸦风封面与步骤插画。
6. 调用 GLM（通过 `GLM_MODEL` 配置）或小红书 MCP 发布，并返回发布结果。

各步骤在 `today_eat_what/agents/` 下的独立智能体实现，LangGraph 在 `today_eat_what/workflow.py` 里编排。

### Qwen / SiliconFlow 接入
- 通过 `QWEN_MODEL`（或兼容的 `Qwen_MODEL`）指定模型；`QWEN_ENDPOINT` 可自定义 chat-completions 端点。
- 传入 `.env` 中的 `Qwen_API_KEY` 或 `QWEN_API_KEY` 即可。
- 菜谱生成在 `today_eat_what/workflow.py` 的 `fetch_recipe` 节点，使用 chat-completions，温度 0.4 保持稳定输出。

### 技术要点
- LangGraph 管线在 `today_eat_what/workflow.py`。
- 所有模型调用带超时 + 重试，成本估算在 `CostTracker`。
- 端点缺失时会回退到 mock 数据，便于本地调试。
- env 读取：`DEEPSEEK_BASE_URL`、`LONGCAT_BASE_URL` 会自动拼接 `/chat/completions` 兼容 OpenAI 风格接口；也可直接提供完整 `*_ENDPOINT`。
