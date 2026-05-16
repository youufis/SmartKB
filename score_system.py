"""
课堂积分激励系统 - Gradio 5.x 集成模块 (v2)
改用 app.get()/app.post() 装饰器方式挂载路由
"""
import json, os
from fastapi import Request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "root", "html")
SCORES_FILE = os.path.join(DATA_DIR, "score_system", "scores.json")
STUDENT_FILES = {
    "高一": os.path.join(DATA_DIR, "高一年级学生名单.json"),
    "高二": os.path.join(DATA_DIR, "高二年级学生名单.json"),
}

os.makedirs(os.path.join(DATA_DIR, "score_system"), exist_ok=True)

def _load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_scores(scores):
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

def _save_students(grade, students):
    path = STUDENT_FILES.get(grade)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(students, f, ensure_ascii=False, indent=2)


def _load_students(grade):
    path = STUDENT_FILES.get(grade)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            students = json.load(f)
        if isinstance(students, list):
            return students
    return []


def _key(g, c, n):
    return f"{g}|{c}|{n}"

# ── API 处理器 ──

async def api_classes(request: Request):
    grade = request.query_params.get("grade", "")
    students = _load_students(grade)
    return sorted(set(s["class"] for s in students))

async def api_students(request: Request):
    grade = request.query_params.get("grade", "")
    cls = request.query_params.get("class", "")
    students = _load_students(grade)
    filtered = [s for s in students if s["class"] == cls]
    scores = _load_scores()
    for s in filtered:
        s["score"] = scores.get(_key(grade, s["class"], s["name"]), 0)
    return filtered

async def api_ranking(request: Request):
    grade = request.query_params.get("grade", "")
    cls = request.query_params.get("class", "")
    students = _load_students(grade)
    # 如果未指定班级，返回整个年级排行
    if cls:
        filtered = [s for s in students if s["class"] == cls]
    else:
        filtered = list(students)
    scores = _load_scores()
    for s in filtered:
        s["score"] = scores.get(_key(grade, s["class"], s["name"]), 0)
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered

async def api_stats(request: Request):
    grade = request.query_params.get("grade", "")
    cls = request.query_params.get("class", "")
    students = _load_students(grade)
    if cls:
        filtered = [s for s in students if s["class"] == cls]
    else:
        filtered = list(students)
    scores = _load_scores()
    total = max_s = 0
    max_name = ""
    for s in filtered:
        sc = scores.get(_key(grade, s["class"], s["name"]), 0)
        total += sc
        if sc > max_s:
            max_s, max_name = sc, s["name"]
    return {
        "total": total,
        "avg": round(total / len(filtered), 1) if filtered else 0,
        "max_score": max_s,
        "max_name": max_name,
        "count": len(filtered),
    }

async def api_score_post(request: Request):
    body = await request.json()
    scores = _load_scores()
    key = _key(body["grade"], body["class"], body["name"])
    scores[key] = scores.get(key, 0) + body["points"]
    _save_scores(scores)
    return {"success": True, "total": scores[key], "added": body["points"]}

async def api_reset_post(request: Request):
    body = await request.json()
    scores = _load_scores()
    g, c, n = body.get("grade"), body.get("class"), body.get("name")
    if n:
        scores.pop(_key(g, c, n), None)
    elif c and g:
        for k in list(scores):
            if k.startswith(f"{g}|{c}|"):
                scores.pop(k, None)
    _save_scores(scores)
    return {"success": True}

async def api_student_save(request: Request):
    body = await request.json()
    grade = body.get("grade")
    if grade not in STUDENT_FILES:
        return {"success": False, "error": "无效年级"}

    students = _load_students(grade)
    name = (body.get("name") or "").strip()
    cls = (body.get("class") or "").strip()
    original_name = (body.get("originalName") or "").strip()
    original_class = (body.get("originalClass") or "").strip()

    if not name or not cls:
        return {"success": False, "error": "姓名和班级为必填项"}

    existing = None
    if original_name and original_class:
        for s in students:
            if s.get("name") == original_name and s.get("class") == original_class:
                existing = s
                break

    if existing:
        old_name = existing.get("name")
        old_class = existing.get("class")
        existing["name"] = name
        existing["class"] = cls
        existing["gender"] = body.get("gender", existing.get("gender", ""))
        existing["language"] = body.get("language", existing.get("language", ""))
        existing["subjects"] = body.get("subjects", existing.get("subjects", ""))
        existing["major"] = body.get("major", existing.get("major", ""))

        if old_name != name or old_class != cls:
            scores = _load_scores()
            old_key = _key(grade, old_class, old_name)
            new_key = _key(grade, cls, name)
            if old_key in scores:
                scores[new_key] = scores.pop(old_key)
                _save_scores(scores)
        student = existing
    else:
        student = {
            "name": name,
            "class": cls,
            "gender": body.get("gender", ""),
            "language": body.get("language", ""),
            "subjects": body.get("subjects", ""),
            "major": body.get("major", ""),
        }
        students.append(student)

    _save_students(grade, students)
    return {"success": True, "student": student}

async def api_student_delete(request: Request):
    body = await request.json()
    grade = body.get("grade")
    name = (body.get("name") or "").strip()
    cls = (body.get("class") or "").strip()
    if grade not in STUDENT_FILES or not name or not cls:
        return {"success": False, "error": "参数错误"}

    students = _load_students(grade)
    target = None
    for idx, s in enumerate(students):
        if s.get("name") == name and s.get("class") == cls:
            target = students.pop(idx)
            break

    if not target:
        return {"success": False, "error": "未找到学生"}

    _save_students(grade, students)
    scores = _load_scores()
    scores.pop(_key(grade, cls, name), None)
    _save_scores(scores)
    return {"success": True}

# ── 挂载路由 ──

def mount_score_api(app):
    """在 demo.queue() 之后、demo.launch() 之前调用"""
    # 使用装饰器方式注册路由
    app.get("/score-api/classes")(api_classes)
    app.get("/score-api/students")(api_students)
    app.get("/score-api/ranking")(api_ranking)
    app.get("/score-api/stats")(api_stats)
    app.post("/score-api/score")(api_score_post)
    app.post("/score-api/reset")(api_reset_post)
    
    # 添加 ping 诊断端点
    async def ping(request: Request):
        routes = sorted(set(r.path for r in app.routes if hasattr(r, 'path') and 'score' in r.path))
        return {"status": "ok", "score_routes": routes}
    app.get("/score-api/ping")(ping)
    app.post("/score-api/student")(api_student_save)
    app.delete("/score-api/student")(api_student_delete)
    
    # 打印已注册的路由以验证
    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    score_routes = [r for r in routes if 'score' in r]
    #print(f"  [积分系统] 已注册 {len(score_routes)} 个路由: {score_routes}")

# ── 读取 HTML 内容 ──

def read_score_html():
    path = os.path.join(DATA_DIR, "score_system", "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    return "<p style='color:white;text-align:center;padding:40px;'>课堂积分系统加载中...</p>"




