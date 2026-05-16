"""
智能点名 · 公平版 — 后端 API
提供：年级/班级列表、学生数据、公平点名算法、历史记录（服务端持久化）
"""
import json, os, random, time
from fastapi import Request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "root", "html")
ROLLCALL_DIR = os.path.join(DATA_DIR, "rollcall_data")
os.makedirs(ROLLCALL_DIR, exist_ok=True)

STUDENT_FILES = {
    "高一": os.path.join(DATA_DIR, "高一年级学生名单.json"),
    "高二": os.path.join(DATA_DIR, "高二年级学生名单.json"),
}
SCORES_FILE = os.path.join(DATA_DIR, "score_system", "scores.json")

# ── 工具函数 ──

def _load_students(grade):
    path = STUDENT_FILES.get(grade)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []

def _load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_scores(scores):
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

def _score_key(grade, cls, name):
    return f"{grade}|{cls}|{name}"

def _history_file(grade, cls):
    safe = f"{grade}_{cls}".replace(" ", "_")
    return os.path.join(ROLLCALL_DIR, f"{safe}.json")

def _load_history(grade, cls):
    path = _history_file(grade, cls)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"weights": {}, "history": [], "updated": ""}

def _save_history(grade, cls, data):
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(_history_file(grade, cls), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── API: 年级列表 ──

async def api_grades(request: Request):
    return [g for g in STUDENT_FILES if os.path.exists(STUDENT_FILES[g])]

# ── API: 班级列表 ──

async def api_classes(request: Request):
    grade = request.query_params.get("grade", "")
    students = _load_students(grade)
    return sorted(set(s.get("class", "") for s in students if s.get("class")))

# ── API: 学生列表（含积分） ──

async def api_students(request: Request):
    grade = request.query_params.get("grade", "")
    cls = request.query_params.get("class", "")
    students = _load_students(grade)
    if cls:
        students = [s for s in students if s.get("class") == cls]
    scores = _load_scores()
    result = []
    for s in students:
        sk = _score_key(grade, s.get("class", ""), s["name"])
        result.append({
            "name": s["name"],
            "class": s.get("class", ""),
            "gender": s.get("gender", ""),
            "score": scores.get(sk, 0),
        })
    return result

# ── 公平算法：加权随机选取 ──

def _weighted_pick(weights):
    """权重越高越可能被选到，最低保底权重1"""
    names = list(weights.keys())
    if not names:
        return None
    vals = [(n, max(1, weights[n])) for n in names]
    total = sum(w for _, w in vals)
    r = random.random() * total
    for name, w in vals:
        r -= w
        if r <= 0:
            return name
    return vals[-1][0]

def _apply_decay(weights, last_time):
    """权重自然恢复：每隔几分钟权重向10恢复"""
    if not last_time:
        return time.time()
    elapsed = (time.time() - last_time) / 60  # 分钟
    if elapsed >= 2:
        for s in weights:
            weights[s] = min(10, weights[s] + elapsed * 0.3)
        return time.time()
    return last_time

# ── API: 点名选取 ──

async def api_pick(request: Request):
    body = await request.json()
    grade, cls = body.get("grade", ""), body.get("class", "")
    if not grade or not cls:
        return {"error": "缺少年级/班级"}

    state = _load_history(grade, cls)
    students = _load_students(grade)
    names = [s["name"] for s in students if s.get("class") == cls]

    # 确保每个学生有权重
    for n in names:
        state["weights"].setdefault(n, 10)

    # 权重自然恢复
    state["last_time"] = _apply_decay(state["weights"], state.get("last_time"))

    # 公平轮询：维护本轮已抽到的学生
    picked_in_round = state.setdefault("picked_in_round", [])

    # 如果本轮所有学生都已抽到，重置轮次
    if set(picked_in_round) == set(names):
        picked_in_round.clear()

    # 从未在本轮抽到的学生中，按权重选择
    available = {n: state["weights"][n] for n in names if n not in picked_in_round}
    if not available:
        # 兜底：如果所有都抽到了（不应该发生），重置
        picked_in_round.clear()
        available = {n: state["weights"][n] for n in names}

    picked = _weighted_pick(available)
    if not picked:
        return {"error": "没有学生"}

    # 被点到：权重-3，加入本轮已抽到
    state["weights"][picked] = max(1, state["weights"][picked] - 3)
    picked_in_round.append(picked)
    state["last_time"] = time.time()

    _save_history(grade, cls, state)

    covered = set(h.get("student") for h in state.get("history", []))
    return {
        "student": picked,
        "grade": grade,
        "class": cls,
        "covered": len(covered),
        "total": len(names),
        "history_count": len(state.get("history", [])),
    }

# ── API: 标记结果 ──

async def api_mark(request: Request):
    body = await request.json()
    grade, cls = body.get("grade", ""), body.get("class", "")
    student = body.get("student", "")
    result = body.get("result", "skip")
    # noScore=true: 仅记录历史不写积分（供答题页调用，避免重复计分）
    noScore = body.get("noScore", False)

    state = _load_history(grade, cls)
    points_added = 0

    if result == "correct":
        state["weights"][student] = min(10, state["weights"].get(student, 10) + 1)
        if not noScore:
            scores = _load_scores()
            sk = _score_key(grade, cls, student)
            scores[sk] = scores.get(sk, 0) + 5
            _save_scores(scores)
            points_added = 5
    elif result == "incorrect":
        if not noScore:
            scores = _load_scores()
            sk = _score_key(grade, cls, student)
            scores[sk] = scores.get(sk, 0) + 2
            _save_scores(scores)
            points_added = 2

    state.setdefault("history", []).append({
        "student": student,
        "time": time.strftime("%H:%M:%S"),
        "result": result,
        "points": points_added,
    })

    _save_history(grade, cls, state)

    scores = _load_scores()
    sk = _score_key(grade, cls, student)
    return {
        "success": True,
        "student": student,
        "result": result,
        "points_added": points_added,
        "total_score": scores.get(sk, 0),
        "history_count": len(state["history"]),
    }

# ── API: 获取历史记录 + 覆盖统计 ──

async def api_history(request: Request):
    grade = request.query_params.get("grade", "")
    cls = request.query_params.get("class", "")
    state = _load_history(grade, cls)
    students = _load_students(grade)
    names = [s["name"] for s in students if s.get("class") == cls]
    covered = set(h.get("student") for h in state.get("history", []))
    correct_count = sum(1 for h in state.get("history", []) if h.get("result") == "correct")
    return {
        "history": state.get("history", []),
        "weights": state.get("weights", {}),
        "covered": len(covered),
        "total": len(names),
        "correct_count": correct_count,
        "updated": state.get("updated", ""),
    }

# ── API: 重置 ──

async def api_reset(request: Request):
    body = await request.json()
    grade, cls = body.get("grade", ""), body.get("class", "")
    students = _load_students(grade)
    names = [s["name"] for s in students if s.get("class") == cls]
    state = {
        "weights": {n: 10 for n in names},
        "history": [],
        "picked_in_round": [],  # 重置轮次
        "last_time": time.time(),
    }
    _save_history(grade, cls, state)
    return {"success": True, "total": len(names)}

# ── 挂载 ──

def mount_rollcall_api(app):
    app.get("/rollcall-api/grades")(api_grades)
    app.get("/rollcall-api/classes")(api_classes)
    app.get("/rollcall-api/students")(api_students)
    app.post("/rollcall-api/pick")(api_pick)
    app.post("/rollcall-api/mark")(api_mark)
    app.get("/rollcall-api/history")(api_history)
    app.post("/rollcall-api/reset")(api_reset)