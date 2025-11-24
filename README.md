## 食刻推荐 (Xiaohongshu 一键发布)

基于 LangChain + LangGraph 的自动化流水线：判断餐次 → 生成菜谱 → 写文案 → 审核/重写 → 并行生成图片 → 小红书 MCP 发布。

### 快速开始
- 环境：Python 3.12+，依赖用 `uv sync` 安装。
- 运行：`uv run python -m today_eat_what` 或 `uv run python main.py`。
- `.env` 示例（未配时部分节点走 mock）：
```
QWEN_API_KEY=your_qwen_key
QWEN_MODEL=Qwen/Qwen3-8B
QWEN_BASE_URL=https://api.siliconflow.cn/v1

DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

LONGCAT_API_KEY=your_longcat_key
LONGCAT_BASE_URL=https://api.longcat.chat/openai
LONGCAT_MODEL=LongCat-Flash-Chat

DOUBAO_API_KEY=your_doubao_key            # 文生图
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_IMAGE_MODEL=doubao-seedream-4-0-250828

GLM_API_KEY=your_glm_key                  # 发布 LLM 备用
GLM_BASE_URL=https://api.siliconflow.cn/v1
GLM_MODEL=THUDM/GLM-4-9B-0414

XIAOHONGSHU_MCP_URL=http://your-xhs-mcp   # 发布 MCP
HOWTOCOOK_MCP_URL=http://your-cook-mcp    # 可选，菜谱 MCP
```

### 流程与智能体
1) `determine_meal`：按当前时间判定餐次。  
2) `recipe_agent`：Qwen 生成菜谱（支持 MCP 工具；缺配置时报错提示）。  
3) `content_agent`：DeepSeek 写文案，输出结构 `title/content/tags`，自动抽取正文里的 #标签并补足到 6+ 条。  
4) `audit_agent`：LongCat 审核，不通过则进入安全重写再审。  
5) 图片并行：菜谱生成后立即异步触发豆包文生图，发布前收集结果，失败再同步兜底。  
6) `publish_agent`：小红书 MCP 发布，标题过长自动截断；发布失败会直接返回错误详情，需自行处理。

### 发布格式约束
- 发布时会剥离标题，正文按换行输出，正文内的 `#标签` 会移除并加入 `tags` 参数；始终附加默认标签“今天吃什么呢”。

### 本地/调试提示
- HowToCook MCP 不可用时会直接用 Qwen 生成；图片生成缺配置会返回占位 URL。
- MCP 500/登录失效：会直接返回错误详情，请检查 MCP 服务或手动处理登录。  

### Docker 部署 & 定时
- 构建镜像：`docker build -t today-eat-what .`
- 运行一次（读取宿主 `.env`）：`docker run --rm --env-file /path/to/.env today-eat-what`
- 如需时区对齐，可追加 `-e TZ=Asia/Shanghai`。
- 通过宿主机 cron 定时（示例：7/12/18 点执行）：
  - `0 7,12,18 * * * docker run --rm --env-file /path/to/.env -e TZ=Asia/Shanghai today-eat-what >> /var/log/today_eat_what.log 2>&1`
- DNS 访问受限时豆包图片可能无法保存本地，但会继续流程。

### 目录索引
- `main.py`：入口。  
- `today_eat_what/workflow.py`：LangGraph 编排与并行逻辑。  
- `today_eat_what/agents/`：recipe/content/audit/image/publish 智能体实现。  
- `today_eat_what/config.py`：环境变量解析与默认端点。  
- `scripts/`：单独模型/MCP 调试脚本。
