"""
文件下载管理 API - 动态列出 downloads 目录中的文件
挂载到 FastAPI app (Gradio 5.x 底层) 使用
"""
import os, json, shutil
from datetime import datetime
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "root", "html", "downloads")

# 需要排除的文件（只排除页面自身）
EXCLUDE = {'index.html'}

def _safe_rel_path(rel_path: str) -> str:
    """规范化相对路径，防止路径穿越，返回相对于 DOWNLOADS_DIR 的安全路径"""
    # 用 posix 风格统一
    rel_path = rel_path.replace('\\', '/').strip('/')
    norm = os.path.normpath(rel_path).replace('\\', '/')
    # 不允许跳出
    if norm.startswith('..') or norm.startswith('/'):
        return ''
    return norm

def _scan_dir(dirpath: str, base_rel: str) -> list:
    """递归扫描目录，返回 [{name, path, size, mtime}]，path 为相对于 DOWNLOADS_DIR"""
    entries = []
    for name in sorted(os.listdir(dirpath), key=str.lower):
        full = os.path.join(dirpath, name)
        rel = (base_rel + '/' + name) if base_rel else name
        if name in EXCLUDE and not base_rel:
            continue
        if os.path.isfile(full):
            stat = os.stat(full)
            entries.append({
                'name': name,
                'path': _safe_rel_path(rel),
                'size': stat.st_size,
                'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
        elif os.path.isdir(full):
            # 先加入目录自身（让前端可以展示文件夹）
            entries.append({
                'name': name,
                'path': _safe_rel_path(rel) + '/',
                'size': 0,
                'mtime': '',
                'is_dir': True
            })
            # 递归子目录文件
            entries.extend(_scan_dir(full, rel))
    return entries

async def api_list_files(request: Request):
    """动态递归扫描 downloads 目录，返回文件列表 JSON"""
    if not os.path.isdir(DOWNLOADS_DIR):
        return {"files": [], "error": "目录不存在"}
    entries = _scan_dir(DOWNLOADS_DIR, '')
    return {"files": entries}

async def api_ping(request: Request):
    """诊断端点"""
    count = 0
    if os.path.isdir(DOWNLOADS_DIR):
        for root, dirs, files in os.walk(DOWNLOADS_DIR):
            for f in files:
                if f not in EXCLUDE or root != DOWNLOADS_DIR:
                    count += 1
    return {
        "status": "ok",
        "downloads_dir": DOWNLOADS_DIR,
        "exists": os.path.isdir(DOWNLOADS_DIR),
        "file_count": count
    }

async def api_upload(request: Request):
    """上传文件到 downloads 目录（支持子目录结构）"""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    form = await request.form()
    uploaded = []
    errors = []

    # 收集所有文件及对应的路径信息
    file_map = {}  # index -> (file_item, rel_path)
    for key in form.keys():
        if key.startswith('file'):
            idx = key[4:]  # 取出数字索引
            item = form[key]
            if hasattr(item, 'filename') and item.filename:
                file_map[idx] = [item, '']  # 默认空路径
    for key in form.keys():
        if key.startswith('path'):
            idx = key[4:]
            if idx in file_map:
                file_map[idx][1] = form[key]

    for idx, (item, rel_path) in file_map.items():
        try:
            raw_filename = item.filename  # type: ignore
            if not raw_filename:
                continue
            # 使用前端传的 rel_path 构造目标路径
            rel = _safe_rel_path(rel_path)
            if not rel and rel_path:
                errors.append(f"{raw_filename}: 非法路径")
                continue
            if rel:
                full_rel = os.path.join(rel, os.path.basename(raw_filename))
            else:
                full_rel = os.path.basename(raw_filename)
            full_rel = _safe_rel_path(full_rel)
            if not full_rel:
                errors.append(f"{raw_filename}: 非法路径")
                continue

            # 检查排除
            if os.path.basename(full_rel) in EXCLUDE and not os.path.dirname(full_rel):
                errors.append(f"{full_rel}: 不允许上传此文件")
                continue

            dest = os.path.join(DOWNLOADS_DIR, full_rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            content = await item.read()  # type: ignore
            with open(dest, 'wb') as f:
                f.write(content)
            uploaded.append(full_rel)
        except Exception as e:
            errors.append(f"{raw_filename if 'raw_filename' in dir() else '?'}: {str(e)}")

    return {"success": len(uploaded) > 0, "files": uploaded, "errors": errors}

async def api_delete(request: Request):
    """删除 downloads 目录中的文件或空目录"""
    body = await request.json()
    filename = body.get("filename", "")
    if not filename:
        return {"success": False, "error": "未指定文件名"}
    rel = _safe_rel_path(filename)
    if not rel:
        return {"success": False, "error": "非法路径"}
    # 不允许删除根目录下的 index.html
    if rel == 'index.html':
        return {"success": False, "error": "不允许删除此文件"}
    filepath = os.path.join(DOWNLOADS_DIR, rel)
    if not os.path.exists(filepath):
        return {"success": False, "error": "文件或目录不存在"}

    # 删除后清理空父目录
    def _rm_and_clean(path):
        if os.path.isfile(path):
            os.remove(path)
            # 尝试删除空父目录
            parent = os.path.dirname(path)
            while parent and os.path.isdir(parent) and parent != DOWNLOADS_DIR:
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            parent = os.path.dirname(path)
            while parent and os.path.isdir(parent) and parent != DOWNLOADS_DIR:
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
        else:
            raise FileNotFoundError()

    try:
        _rm_and_clean(filepath)
        return {"success": True, "filename": rel}
    except Exception as e:
        return {"success": False, "error": str(e)}


def mount_downloads_api(app):
    """挂载下载管理 API 路由到 FastAPI app"""
    app.get("/downloads-api/list")(api_list_files)
    app.get("/downloads-api/ping")(api_ping)
    app.post("/downloads-api/upload")(api_upload)
    app.post("/downloads-api/delete")(api_delete)
    
    # 打印已注册的路由以验证
    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    dl_routes = [r for r in routes if 'downloads' in r]
    #print(f"  [下载管理] 已注册 {len(dl_routes)} 个路由: {dl_routes}")
