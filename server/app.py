# -*- coding: utf-8 -*-
"""
果蔬卡路里 · 后端识别服务（智能体架构）
提供 /api/recognize 接口：接收食物照片，返回热量与营养信息。

智能体两级调度：
1. 识别层（多源可插拔）：多模态大模型 VLM（配 GLM_API_KEY 启用，可识别任意食物并估克重）
   → 百度智能云（可选，配 BAIDU_API_KEY/BAIDU_SECRET_KEY 启用）→ 演示模式兜底；
2. 营养层（三级检索）：本地 40 条营养库 → 联网检索 OpenFoodFacts 全球开源食品库（免密钥）
   → 通用估算兜底。营养数据不再局限于本地固定字典。

启动：uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64
import hashlib
import difflib
import json
import os
import re
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "food_db.json"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app = FastAPI(title="果蔬卡路里 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 营养数据库 ----------------
with open(DB_FILE, "r", encoding="utf-8") as f:
    FOOD_DB = json.load(f)

BY_NAME = {item["name"]: item for item in FOOD_DB}
ALIAS = {}
for item in FOOD_DB:
    for a in item.get("aliases", []):
        ALIAS[a] = item["name"]


def match_food(raw_name: str):
    """把识别出的菜名映射到本地营养库（含别名与模糊匹配）。"""
    raw = raw_name.strip()
    if raw in BY_NAME:
        return BY_NAME[raw]
    if raw in ALIAS:
        return BY_NAME[ALIAS[raw]]
    for name, item in BY_NAME.items():
        if name in raw or raw in name:
            return item
    names = list(BY_NAME.keys())
    close = difflib.get_close_matches(raw, names, n=1, cutoff=0.45)
    if close:
        return BY_NAME[close[0]]
    return None


# ---------------- 结果构造 ----------------
def build_result(food: dict, confidence: float, source: str) -> dict:
    import random
    random.seed()
    weight = round(food["weight"] * random.uniform(0.85, 1.15))
    k = weight / 100.0
    fmt = lambda v: round(v * k, 1)
    return {
        "name": food["name"],
        "emoji": food["emoji"],
        "category": food["category"],
        "confidence": round(confidence, 3),
        "per100g": {
            "kcal": food["kcal"],
            "protein": food["protein"],
            "fat": food["fat"],
            "carbs": food["carbs"],
            "fiber": food["fiber"],
        },
        "weightG": weight,
        "total": {
            "kcal": round(food["kcal"] * k),
            "protein": fmt(food["protein"]),
            "fat": fmt(food["fat"]),
            "carbs": fmt(food["carbs"]),
            "fiber": fmt(food["fiber"]),
        },
        "tip": food["tip"],
        "source": source,
    }


# ---------------- 识别提供方 ----------------
_baidu_token_cache = {"token": None, "expires": 0}


def get_baidu_token() -> str:
    # 支持直接注入 access_token（如用户从平台获取，有效期30天）
    if os.environ.get("BAIDU_ACCESS_TOKEN"):
        return os.environ["BAIDU_ACCESS_TOKEN"]
    now = time.time()
    if _baidu_token_cache["token"] and now < _baidu_token_cache["expires"]:
        return _baidu_token_cache["token"]
    resp = requests.post(
        "https://aip.baidubce.com/oauth/2.0/token",
        params={
            "grant_type": "client_credentials",
            "client_id": os.environ["BAIDU_API_KEY"],
            "client_secret": os.environ["BAIDU_SECRET_KEY"],
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _baidu_token_cache["token"] = data["access_token"]
    _baidu_token_cache["expires"] = now + data.get("expires_in", 2592000) - 3600
    return _baidu_token_cache["token"]


def baidu_recognize(image_bytes: bytes):
    """
    分层识别策略，返回 (名称, 置信度, 提供方) 或 None。
    第一层（食物）：果蔬识别 v1/classify/ingredient + 菜品识别 v2/dish + 通用物体识别兜底
    第二层（非食物兜底）：植物识别 v1/plant + 动物识别 v1/animal
    """
    token = get_baidu_token()
    img_b64 = base64.b64encode(image_bytes).decode()

    def _call(endpoints, min_score=0.0):
        best_name, best_score = None, 0.0
        for url, kind in endpoints:
            try:
                resp = requests.post(
                    url,
                    params={"access_token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"image": img_b64, "top_num": 5},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error_code" in data:
                    print(f"[warn] 百度接口 {kind} 返回错误: {data.get('error_msg')}")
                    continue
                for item in data.get("result") or []:
                    # 不同接口字段不同：dish=probability/name，ingredient/general/plant/animal=score、name/keyword
                    name = item.get("name") or item.get("keyword") or ""
                    score = item.get("score") or item.get("probability") or 0.0
                    # 通用识别的植物/动物类候选不算食物（root 如"植物-蔷薇科"），转交非食物兜底层
                    root = item.get("root") or ""
                    if kind == "general" and ("植物" in root or "动物" in root):
                        continue
                    # 过滤"非菜/非果/非动物"等负面结果，只取置信度最高者
                    if name and "非" not in name and score >= min_score and score > best_score:
                        best_name, best_score = name, score
            except Exception as e:
                print(f"[warn] 百度接口 {kind} 调用失败: {e}")
        return best_name, best_score

    # 第一层：食物识别（低于 0.5 视为无效，避免低分误判，转入非食物兜底）
    name, score = _call(FOOD_ENDPOINTS, min_score=0.5)
    if name:
        return name, min(score, 0.99), "baidu_dish"

    # 第二层：非食物兜底（照片是植物/动物时给出友好识别，避免乱报热量）
    for url, kind in NONFOOD_ENDPOINTS:
        try:
            resp = requests.post(
                url,
                params={"access_token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"image": img_b64, "top_num": 3},
                timeout=15,
            )
            data = resp.json()
            if "error_code" in data:
                continue
            for item in data.get("result") or []:
                nm = item.get("name") or item.get("keyword") or ""
                sc = float(item.get("score") or 0)
                if nm and "非" not in nm and sc > 0.5:
                    return nm, min(sc, 0.99), kind
        except Exception as e:
            print(f"[warn] 百度接口 {kind} 调用失败: {e}")
    return None


# ---------------- 识别端点配置 ----------------
FOOD_ENDPOINTS = [
    # 果蔬识别（果蔬类最准，注意正确路径为 v1/classify/ingredient）
    ("https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient", "ingredient"),
    ("https://aip.baidubce.com/rest/2.0/image-classify/v2/dish", "dish"),
    # 通用物体识别兜底：覆盖成品食物/零食等，结果字段为 keyword
    ("https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general", "general"),
]
NONFOOD_ENDPOINTS = [
    ("https://aip.baidubce.com/rest/2.0/image-classify/v1/plant", "baidu_plant"),
    ("https://aip.baidubce.com/rest/2.0/image-classify/v1/animal", "baidu_animal"),
]


def demo_recognize(image_bytes: bytes):
    """演示模式：按图片哈希稳定地挑选一个食物（保证同图同果）。"""
    digest = int(hashlib.md5(image_bytes).hexdigest(), 16)
    food = FOOD_DB[digest % len(FOOD_DB)]
    return food["name"], 0.85 + (digest % 12) / 100.0


# ---------------- 接口 ----------------
class RecognizeRequest(BaseModel):
    image_base64: str = ""   # 二选一：本地图（前端拍照/相册上传的 base64）
    image_url: str = ""      # 二选一：图片 URL（服务端代下载后识别）
    user_text: str = ""      # 可选：用户命令/疑问（复检模式）


# ---------------- 智能体层 1：多模态大模型识别（可选，优先级最高） ----------------
def vlm_recognize(image_bytes: bytes, user_text: str = ""):
    """多模态大模型识别：配置 GLM_API_KEY（智谱，glm-4v-flash 免费模型）即启用。
    大模型直接看图理解，能识别任意食物/菜品并估算克重，不再受固定接口类别限制。
    user_text 非空时进入"复检模式"：把用户的命令/疑问注入 prompt，让大模型重新检验。
    返回 (名称, 置信度, 估算克重 or None, 类别 or None, 候选列表, "vlm")；未配置或失败返回 None。"""
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        return None
    model = os.environ.get("GLM_MODEL", "glm-4v-flash")
    b64 = base64.b64encode(image_bytes).decode()
    base_prompt = (
        "仔细观察照片中食物的形状、大小、颜色、果皮纹理、数量排列等特征，判断是什么食物。"
        "注意区分易混淆的食物：葡萄是成串多颗小圆果；苹果是单个大果、顶部有果柄窝；桃子表皮有绒毛；橙子/橘子表皮有油胞点。"
        "只输出一个 JSON 对象，不要其他文字："
        '{"name":"最可能的中文食物名","candidates":[{"name":"候选1","confidence":0~1},{"name":"候选2","confidence":0~1}],'
        '"category":"水果/蔬菜/肉类/海鲜/主食/菜品/零食/汤品 之一",'
        '"weight_g":照片中食物可食用部分的估算克重(按一份合理份量估算，通常30~1000克),"confidence":最可能项置信度(0~1)}。'
        '若不是食物则输出 {"name":"非食物","weight_g":0,"confidence":1,"candidates":[]}'
    )
    if user_text:
        user_text = user_text.strip()[:200]
        base_prompt = (
            f"用户对之前的识别结果提出了疑问或指令：「{user_text}」。"
            "这是复检请求，请抛开之前的结论，重新逐项检验照片：重点核对用户提到的食物名称特征是否成立，"
            "逐个排除易混淆项（数量、果形、表皮纹理、果柄），再给出最终结论。若有多个候选请按可能性排序。"
            + base_prompt
        )
    resp = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": base_prompt},
                ],
            }],
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    d = json.loads(m.group())
    name = str(d.get("name", "")).strip()
    if not name:
        return None
    weight = d.get("weight_g")
    weight = int(weight) if isinstance(weight, (int, float)) and 0 < weight <= 1200 else None
    conf = float(d.get("confidence") or 0.9)
    category = str(d.get("category") or "").strip() or None
    # top 候选列表（用于交叉验证决策）
    cands = []
    for c in (d.get("candidates") or [])[:3]:
        try:
            cands.append((str(c.get("name", "")).strip(), float(c.get("confidence") or 0)))
        except Exception:
            pass
    return name, min(conf, 0.99), weight, category, cands, "vlm"


# ---------------- 智能体层 2：联网营养检索（OpenFoodFacts 全球开源库，免密钥） ----------------
_OFF_CACHE = {}

def fetch_nutrition_online(name: str):
    """联网查询 OpenFoodFacts 全球开源食品库（数据来自全网产品标注汇集），
    返回 food dict 或 None；带缓存。彻底摆脱"仅靠本地固定营养库"的检索式局限。"""
    key = name.strip()
    if not key or key in _OFF_CACHE:
        return _OFF_CACHE.get(key)
    result = None
    try:
        r = requests.get(
            "https://world.openfoodfacts.net/cgi/search.pl",
            params={"search_terms": key, "search_simple": 1, "action": "process",
                    "json": 1, "page_size": 5},
            headers={"User-Agent": "FoodCalorieApp/1.0"},
            timeout=12,
        )
        r.raise_for_status()
        for p in (r.json().get("products") or []):
            n = p.get("nutriments") or {}
            kcal = n.get("energy-kcal_100g") or n.get("energy-kcal_100g_estimate")
            if kcal is None:
                continue
            result = {
                "name": p.get("product_name_zh") or p.get("product_name") or key,
                "emoji": "🥗",
                "category": "联网检索 · OpenFoodFacts",
                "kcal": round(float(kcal)),
                "protein": round(float(n.get("proteins_100g") or 0), 1),
                "fat": round(float(n.get("fat_100g") or 0), 1),
                "carbs": round(float(n.get("carbohydrates_100g") or 0), 1),
                "fiber": round(float(n.get("fiber_100g") or 0), 1),
                "weight": 100,
                "tip": "营养数据由智能体实时联网检索 OpenFoodFacts 全球开源食品库获得（产品包装标注数据）。",
            }
            break
    except Exception as e:
        print(f"[warn] OpenFoodFacts 检索失败: {e}")
    _OFF_CACHE[key] = result
    return result


def names_similar(a: str, b: str) -> bool:
    """两个食物名是否指向同一食物（营养库映射一致或互为子串）。"""
    fa, fb = match_food(a), match_food(b)
    if fa and fb and fa.get("name") == fb.get("name"):
        return True
    a, b = a.strip(), b.strip()
    return (a in b or b in a) and bool(a and b)


def _recognize_bytes(image_bytes: bytes, user_text: str = "") -> dict:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片超过 10MB，请压缩后上传")

    # ---- 识别层（智能体多源调度）：VLM 大模型 → 百度交叉验证 → 演示兜底 ----
    recheck_mode = bool(user_text.strip())
    raw_name, confidence, weight_override, category_override = None, 0.0, None, None
    cands, provider, cross_note = [], "demo", None
    try:
        vres = vlm_recognize(image_bytes, user_text=user_text)   # 配置 GLM_API_KEY 才会实际调用
        if vres:
            raw_name, confidence, weight_override, category_override, cands, provider = vres
    except Exception as e:
        print(f"[warn] VLM 识别失败，尝试下一层: {e}")

    # 交叉验证：VLM 置信度不足或候选接近时，请百度复核，取更可信者；
    # 复检模式（用户质疑）强制走一次百度复核
    has_baidu_keys = os.environ.get("BAIDU_API_KEY") and os.environ.get("BAIDU_SECRET_KEY")
    has_baidu_token = bool(os.environ.get("BAIDU_ACCESS_TOKEN"))
    need_cross = (
        provider == "vlm" and raw_name and "非食物" not in raw_name
        and (recheck_mode
             or confidence < 0.85
             or (len(cands) > 1 and confidence - cands[1][1] < 0.15)
             or (len(cands) == 1 and confidence < 0.9))
    )
    if need_cross and (has_baidu_keys or has_baidu_token):
        try:
            bres = baidu_recognize(image_bytes)
            if bres:
                bname, bconf, bprov = bres
                if bprov not in ("baidu_plant", "baidu_animal") and bname:
                    head = "🔁 复检" if recheck_mode else ""
                    if names_similar(bname, raw_name):
                        confidence = min(round((confidence + bconf) / 2 + 0.03, 3), 0.99)
                        cross_note = f"{head}✅ {'复检确认' if recheck_mode else '双源交叉验证一致'}：大模型与百度均识别为「{raw_name}」"
                    elif bconf > confidence:
                        cross_note = (f"{head}⚠️ {'复检修正' if recheck_mode else '交叉验证'}：大模型判断为「{raw_name}」({confidence:.2f})，"
                                      f"百度识别为「{bname}」({bconf:.2f})，已采用置信度更高的百度结果")
                        raw_name, confidence, provider = bname, bconf, bprov
                    else:
                        cross_note = (f"{head}⚠️ {'复检确认' if recheck_mode else '交叉验证'}：大模型判断为「{raw_name}」({confidence:.2f})，"
                                      f"百度识别为「{bname}」({bconf:.2f})，已采用置信度更高的大模型结果")
        except Exception as e:
            print(f"[warn] 交叉验证百度调用失败: {e}")

    if raw_name is None:
        has_baidu_keys = os.environ.get("BAIDU_API_KEY") and os.environ.get("BAIDU_SECRET_KEY")
        has_baidu_token = bool(os.environ.get("BAIDU_ACCESS_TOKEN"))
        if has_baidu_keys or has_baidu_token:
            try:
                res = baidu_recognize(image_bytes)  # (名称, 置信度, 提供方) 或 None
                if res:
                    raw_name, confidence, provider = res
            except Exception as e:
                print(f"[warn] 百度识别失败，回退演示模式: {e}")

    if raw_name is None:
        raw_name, confidence = demo_recognize(image_bytes)
        provider = "demo"

    # VLM 判定非食物：明确告知，不硬编营养数据
    if provider == "vlm" and "非食物" in raw_name:
        return {
            "name": "非食物", "emoji": "🔍", "category": "非食物",
            "confidence": round(confidence, 3),
            "per100g": {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0},
            "weightG": 0,
            "total": {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0},
            "tip": "大模型判断这张照片不是食物，热量数据不适用。想测热量请拍摄水果、菜品等食物照片～",
            "source": "vlm_nonfood",
        }

    # 非食物结果（植物/动物）：明确告知热量不适用，不硬编营养数据
    if provider in ("baidu_plant", "baidu_animal"):
        kind_label = "植物" if provider == "baidu_plant" else "动物"
        return {
            "name": raw_name,
            "emoji": "🌿" if kind_label == "植物" else "🐾",
            "category": f"非食物（{kind_label}）",
            "confidence": round(confidence, 3),
            "per100g": {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0},
            "weightG": 0,
            "total": {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0},
            "tip": f"识别到{kind_label}「{raw_name}」，这不是食物，热量数据不适用。想测热量请拍摄水果、菜品等食物照片～",
            "source": provider,
        }

    # ---- 营养层（智能体三级检索）：本地库 → 联网 OpenFoodFacts → 通用估算 ----
    food = match_food(raw_name)
    if food is None:
        food = fetch_nutrition_online(raw_name)   # 联网检索权威开源库
    if food is None:
        food = {
            "name": raw_name, "emoji": "🍽️", "category": "其他", "kcal": 200,
            "protein": 8, "fat": 8, "carbs": 22, "fiber": 1,
            "weight": 200, "tip": "本地库与联网检索均未获得该食物的营养数据，此为通用估算值。",
        }
    if weight_override or category_override:  # VLM 的克重估计与类别判断更可信
        food = dict(food)
        if weight_override:
            food["weight"] = weight_override
        if category_override:
            food["category"] = category_override
    result = build_result(food, confidence, provider)
    if cross_note:
        result["cross_validation"] = cross_note
        result["tip"] = cross_note + "\n" + result["tip"]
    return result


@app.post("/api/recognize")
def recognize_json(req: RecognizeRequest):
    """JSON 方式调用（H5 / 自定义前端）：{"image_base64": "..."} 或 {"image_url": "https://..."}"""
    if req.image_url and not req.image_base64:
        # 服务端代下载 URL 图片
        try:
            dl = requests.get(req.image_url, timeout=15)
            dl.raise_for_status()
            image_bytes = dl.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图片 URL 下载失败: {e}")
        return _recognize_bytes(image_bytes)
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 解码失败")
    return _recognize_bytes(image_bytes, user_text=req.user_text)


@app.post("/api/recheck")
def recheck(req: RecognizeRequest):
    """AI 复检：用户对识别结果存疑，输入文字命令让大模型重新检验。
    请求体 {"image_base64": "...", "user_text": "我觉得是苹果，请重新检验"}"""
    if not req.user_text.strip():
        raise HTTPException(status_code=400, detail="请输入复检命令或疑问，例如「我觉得是苹果」")
    return recheck_impl(req)


def recheck_impl(req: RecognizeRequest):
    if req.image_url and not req.image_base64:
        try:
            dl = requests.get(req.image_url, timeout=15)
            dl.raise_for_status()
            image_bytes = dl.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图片 URL 下载失败: {e}")
    else:
        try:
            image_bytes = base64.b64decode(req.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="image_base64 解码失败")
    result = _recognize_bytes(image_bytes, user_text=req.user_text)
    result["user_command"] = req.user_text.strip()
    if not result.get("cross_validation"):
        result["cross_validation"] = f"🔁 已按指令「{req.user_text.strip()}」重新检验，结论维持「{result['name']}」"
        result["tip"] = result["cross_validation"] + "\n" + result.get("tip", "")
    return result


@app.post("/api/recognize_upload")
async def recognize_upload(file: UploadFile = File(...)):
    """multipart 方式调用（小程序端 uni.uploadFile：字段名 file）"""
    image_bytes = await file.read()
    return _recognize_bytes(image_bytes)


@app.get("/api/foods")
def list_foods():
    return {"count": len(FOOD_DB), "foods": FOOD_DB}


@app.get("/api/health")
def health():
    has_vlm = bool(os.environ.get("GLM_API_KEY"))
    has_baidu = bool(
        (os.environ.get("BAIDU_API_KEY") and os.environ.get("BAIDU_SECRET_KEY"))
        or os.environ.get("BAIDU_ACCESS_TOKEN")
    )
    provider = "vlm" if has_vlm else ("baidu_dish" if has_baidu else "demo")
    return {
        "status": "ok",
        "provider": provider,
        "recognizers": {"vlm": has_vlm, "baidu": has_baidu, "openfoodfacts_online": True},
        "foods": len(FOOD_DB),
    }
