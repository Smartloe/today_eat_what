import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure package imports work when running as a script
ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
from langchain_core.tools import tool
from today_eat_what.clients import CostTracker, ModelClient
from today_eat_what.models import Recipe

try:  # 优先使用官方 SDK，缺失时自动降级为占位图
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class ImageAgent:
    def __init__(self, doubao_client: ModelClient, cost: CostTracker) -> None:
        self.doubao = doubao_client
        self.cost = cost
        self._model = (
            os.environ.get("DOUBAO_IMAGE_MODEL")
            or os.environ.get("DOUBAO_MODEL")
            or "doubao-seedream-4-0-250828"
        )
        self._base_url = (
            os.environ.get("DOUBAO_BASE_URL")
            or os.environ.get("ARK_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/v3"
        ).rstrip("/")
        self._client = self._init_doubao_client()
        self.generate_images_tool = tool("generate_images", return_direct=True)(self._generate_images)

    def _init_doubao_client(self) -> Optional["OpenAI"]:
        if OpenAI is None:
            logger.warning("openai 库未安装，图片生成将使用占位图。请安装 openai>=1 并配置 DOUBAO_API_KEY。")
            return None
        api_key = os.environ.get("DOUBAO_API_KEY") or os.environ.get("ARK_API_KEY")
        if not api_key:
            logger.warning("DOUBAO_API_KEY/ARK_API_KEY 未设置，图片生成将回退为占位链接。")
            return None
        try:
            return OpenAI(api_key=api_key, base_url=self._base_url)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("初始化豆包图像客户端失败：%s", exc)
            return None

    def _build_cover_prompt(self, recipe: Recipe) -> str:
        steps_text = " -> ".join([f"{s.order}. {s.instruction}" for s in recipe.steps])
        dish_names = ", ".join([recipe.name] + [d.get("name", "") for d in getattr(recipe, "dishes", []) if d])
        return (
            "Illustration for a book cover, \"今天吃什么？\" handwritten journal style, by Jonna Kenty and Felicia Chiao. "
            "Scene: A vibrant and whimsical composition on a Cornell notepad's lined paper background. Various cute fruits, "
            "vegetables, and assorted foods (avocado, strawberry, lemon, a grilled chicken drumstick, a strip of bacon, "
            "a sunny-side-up egg, a bowl of ramen, and a sushi roll) are playfully doodled around the bold, handwritten title. "
            "Media & Colors: Rendered with the texture of colored pencils and neon highlighters, featuring a bright, cheerful "
            "color palette with pops of fluorescent color for highlights and accents. Layout: Energetic, asymmetrical, and "
            "spontaneous layout, ensuring the title remains the focal point.")
        

    def _build_dish_prompt(self, name: str, description: str, steps: List[str]) -> str:
        steps_text = " | ".join(steps)
        return (
            f"{name} 手绘步骤卡，单张竖版。风格：手写/涂鸦笔记，简笔画+箭头标注关键动作，色彩活泼荧光笔和彩铅质感，背景纸张纹理。"
            f" 口味/概要：{description}。步骤：{steps_text}。要求：所有步骤置于同一张卡片，排版清晰可读，保留手写感和趣味 doodle 元素。"
        )

    def _normalize_steps(self, steps_data: List) -> List[str]:
        normalized: List[str] = []
        for idx, item in enumerate(steps_data, start=1):
            if isinstance(item, str):
                normalized.append(f"{idx}. {item}")
            elif isinstance(item, dict):
                text = item.get("instruction") or item.get("step") or ""
                order = item.get("order") or idx
                normalized.append(f"{order}. {text}")
        return [s for s in normalized if s.strip()]

    def _generate_single_image(self, prompt: str) -> Optional[str]:
        if not self._client:
            return None
        self.cost.add("doubao")
        try:
            resp = self._client.images.generate(
                model=self._model,
                prompt=prompt,
                size=os.environ.get("DOUBAO_IMAGE_SIZE", "1080x1920"),
                response_format="url",
                extra_body={"watermark": False},
            )
            data = getattr(resp, "data", None)
            if data:
                first = data[0]
                url = getattr(first, "url", None) or (first.get("url") if isinstance(first, dict) else None)
                if url:
                    return str(url)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error("豆包生成图片失败：%s", exc)
        return None

    def _download_image(self, url: str, filename: str) -> Optional[str]:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            out_dir = Path(os.environ.get("DOUBAO_IMAGE_DIR", "images"))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            out_path.write_bytes(resp.content)
            return str(out_path)
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("保存图片失败 %s: %s", filename, exc)
            return None

    def _generate_images(self, recipe: dict) -> Dict[str, List[str]]:
        """生成封面 + 每道菜一张 9:16 手绘步骤卡，失败则返回占位 URL。返回 JSON: {'images': [...]}，确保封面第一张。"""
        recipe_obj = Recipe(**recipe)
        dishes_raw = recipe.get("dishes") or []
        dishes = []
        for raw in dishes_raw:
            if not raw:
                continue
            name = raw.get("name") or recipe_obj.name
            desc = raw.get("description") or recipe_obj.description
            steps_list = raw.get("steps") or recipe_obj.steps
            dishes.append((name, desc, self._normalize_steps(steps_list)))

        if not dishes:
            dishes.append((recipe_obj.name, recipe_obj.description, self._normalize_steps(recipe_obj.steps)))

        prompts = [self._build_cover_prompt(recipe_obj)] + [
            self._build_dish_prompt(name, desc, steps) for name, desc, steps in dishes
        ]

        urls: List[str] = []
        for idx, prompt in enumerate(prompts):
            url = self._generate_single_image(prompt)
            if idx == 0:
                filename = f"{recipe_obj.name}_cover_handdrawn.png"
            else:
                dish_name = dishes[idx - 1][0]
                filename = f"{dish_name}_steps_handdrawn.png"
            if not url:
                url = f"https://imgs.local/{filename}"
                local_path = None
            else:
                local_path = self._download_image(url, filename)
            urls.append(url)
        return {"images": urls}
