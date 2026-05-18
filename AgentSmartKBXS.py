import os
os.environ['MPLCONFIGDIR'] ="D:/SmartKB/matplotlib"
from dashscope import Application    
import gradio as gr
from dotenv import load_dotenv
import time
import json
import csv
import requests
import base64
import hashlib
import random
import re
import shutil
import sqlite3
import bcrypt
from score_system import mount_score_api
from smart_rollcall_api import mount_rollcall_api


# 定义初始最大允许的请求数
maxallowed_requests=50
# 添加全局变量控制是否限制最大允许的请求数
enable_request_limit = False # 默认为不启用请求限制

active_users = 0   # 在线用户数
user_sessions = {}  # 存储用户会话信息

##########################################
# 全局配置常量（将硬编码提取到这里）
##########################################
# 目录/文件相关
CHAT_HISTORY_DIR = "ChatHistory"
LOG_FILES_DIR = "LogFiles"
ROOT_DIR = "root"
RESERVED_DIR_NAME = "Reserved"
SUMMARY_DIR_NAME = "Summary"
TASK_DIR_NAME = "Task"
PROMPT_FILE_NAME = "信通课程知识要点.txt"

# 网络/UI相关
UI_PORT = 7862
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8082
ICON_PATH = "icon/logo.png"
FAVICON_PATH = "favicon.ico"

# DashScope / 模型配置
QWEN_OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
APPID = "6fcb54e8f16f4e3b94e4b9fd4eab1125"
MEMORY_ID = "77338108f9c649c4b629b8078e6c6370"
MODEL_LONG_NAME = "qwen-long"
MODEL_VL_NAME = "qwen3-vl-flash"
# MODEL_NAME="qwen3-max"
# MODEL_NAME="qwen3.5-flash"
MODEL_NAME="deepseek-v4-flash"
EMBEDDING_MODEL_NAME = "quentinz/bge-large-zh-v1.5:latest"

# 默认用户
DEFAULT_LOGGED_IN_NAME = "root"

# 任务管理相关
ACTIVE_TASKS_FILE = "active_tasks.json"
TASKS_DIR_NAME = "tasks"
TEACHERS_SUMMARY_DIR = "teachers"
ADMIN_SUMMARY_DIR = "admin"

# 文件类型与大小限制
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
DOCUMENT_EXTENSIONS = ['.txt', '.md', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv','.json','.html','.htm']
MAX_DOC_SIZE_MB = 10
MAX_IMAGE_SIZE_MB = 5


##########################################
# 数据库和认证相关函数
##########################################

def init_db():
    """初始化用户数据库"""
    conn = sqlite3.connect('users.db')  
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password BLOB, class INTEGER, name TEXT, gender INTEGER, role INTEGER DEFAULT 2)''')
    conn.commit()
    conn.close()

# 密码哈希函数
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# 角色权限检查函数
def get_user_role(username):
    """获取用户角色"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return 2  # 默认普通用户

def is_admin(username):
    """检查用户是否为管理员"""
    return get_user_role(username) == 0

def is_teacher(username):
    """检查用户是否为教师"""
    return get_user_role(username) == 1

def is_regular_user(username):
    """检查用户是否为普通用户"""
    return get_user_role(username) == 2

def can_create_task(username):
    """检查用户是否可以创建任务（管理员和教师可以）"""
    role = get_user_role(username)
    return role == 0 or role == 1  # 管理员和教师可以创建任务

def can_manage_users(username):
    """检查用户是否可以管理用户（只有管理员可以）"""
    return is_admin(username)

def can_provide_api_key(username):
    """检查用户是否可以提供API KEY（只有管理员可以）"""
    return is_admin(username)

def can_manage_html_files(username):
    """检查用户是否可以管理HTML文件（管理员和教师可以）"""
    role = get_user_role(username)
    return role == 0 or role == 1  # 管理员和教师可以管理HTML文件

##########################################
# 用户注册管理相关函数
##########################################

def register_user(username, password, class_val, name, gender, current_user, role=2):
    """注册新用户，只有管理员可以注册用户"""
    if not is_admin(current_user):
        return "权限不足：只有管理员可以注册新用户"
    
    if not username or not password:
        return "用户名和密码不能为空"
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 检查用户名是否已存在
    c.execute("SELECT username FROM users WHERE username=?", (username,))
    if c.fetchone():
        conn.close()
        return f"用户名 {username} 已存在"
    
    # 插入新用户
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password, class, name, gender, role) VALUES (?, ?, ?, ?, ?, ?)",
                 (username, hashed_password, class_val, name, gender, role))
        conn.commit()
        conn.close()
        
        role_name = "管理员" if role == 0 else "教师" if role == 1 else "普通用户"
        return f"用户 {username} 注册成功（角色：{role_name}）"
    except Exception as e:
        conn.close()
        return f"注册失败：{str(e)}"

def update_user_info(username, class_val, name, gender, current_user):
    """更新用户信息，管理员可以更新任何用户，普通用户只能更新自己的信息"""
    if current_user != "root" and current_user != username:
        return "权限不足：只能修改自己的信息"
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 检查用户是否存在
    c.execute("SELECT username FROM users WHERE username=?", (username,))
    if not c.fetchone():
        conn.close()
        return f"用户 {username} 不存在"
    
    try:
        c.execute("UPDATE users SET class=?, name=?, gender=? WHERE username=?",
                 (class_val, name, gender, username))
        conn.commit()
        conn.close()
        return f"用户 {username} 信息更新成功"
    except Exception as e:
        conn.close()
        return f"更新失败：{str(e)}"

def change_password(username, old_password, new_password, current_user):
    """修改密码，管理员可以直接修改，普通用户登录后可以直接修改自己的密码"""
    if current_user != "root" and current_user != username:
        return "权限不足：只能修改自己的密码"
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 检查用户是否存在
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    if not result:
        conn.close()
        return f"用户 {username} 不存在"
    
    # 如果是普通用户修改自己的密码，不需要验证旧密码（因为用户已经登录过）
    # 如果是管理员修改其他用户的密码，也不需要验证旧密码
    # 只有当普通用户修改自己的密码时，才不需要验证旧密码
    
    # 更新密码
    new_hashed_password = hash_password(new_password)
    try:
        c.execute("UPDATE users SET password=? WHERE username=?", (new_hashed_password, username))
        conn.commit()
        conn.close()
        return f"用户 {username} 密码修改成功"
    except Exception as e:
        conn.close()
        return f"密码修改失败：{str(e)}"

def delete_user(username, current_user):
    """删除用户，只有管理员可以删除普通用户，不能删除管理员自己"""
    if current_user != "root":
        return "权限不足：只有管理员可以删除用户"
    
    if username == "root":
        return "不能删除管理员账户"
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 检查用户是否存在
    c.execute("SELECT username FROM users WHERE username=?", (username,))
    if not c.fetchone():
        conn.close()
        return f"用户 {username} 不存在"
    
    try:
        c.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
        conn.close()
        
        # 删除用户目录（如果存在）
        user_dir = os.path.abspath(username)
        dir_deleted = ""
        if os.path.isdir(user_dir):
            try:
                shutil.rmtree(user_dir)
                dir_deleted = f"，并已删除用户目录 {username}/"
            except Exception as e:
                dir_deleted = f"，但删除用户目录失败：{str(e)}"
        
        return f"用户 {username} 删除成功{dir_deleted}"
    except Exception as e:
        conn.close()
        return f"删除失败：{str(e)}"

def get_user_info(username, current_user):
    """获取用户信息，支持模糊查询。管理员可以查看任何用户，普通用户只能查看自己的信息"""
    if current_user != "root" and current_user != username:
        return "权限不足：只能查看自己的信息"
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 先尝试精确查询
    c.execute("SELECT username, class, name, gender FROM users WHERE username=?", (username,))
    result = c.fetchone()
    
    if result:
        conn.close()
        username, class_val, name, gender = result
        
        # 格式化性别显示
        gender_str = ""
        if gender is not None:
            g = str(gender)
            if g in ('1', 'M', 'm', '男'):
                gender_str = '男'
            elif g in ('2', 'F', 'f', '女', '0'):
                gender_str = '女'
            else:
                gender_str = g
        
        return f"用户名: {username}\n班级: {class_val}\n姓名: {name}\n性别: {gender_str}"
    
    # 如果精确查询没有找到，尝试模糊查询
    fuzzy_param = '%' + username + '%'
    c.execute("SELECT username, class, name, gender FROM users WHERE username LIKE ?", (fuzzy_param,))
    fuzzy_results = c.fetchall()
    conn.close()
    
    if not fuzzy_results:
        return f"用户 {username} 不存在"
    
    if len(fuzzy_results) == 1:
        # 只有一个匹配，返回详细信息
        username, class_val, name, gender = fuzzy_results[0]
        
        # 格式化性别显示
        gender_str = ""
        if gender is not None:
            g = str(gender)
            if g in ('1', 'M', 'm', '男'):
                gender_str = '男'
            elif g in ('2', 'F', 'f', '女', '0'):
                gender_str = '女'
            else:
                gender_str = g
        
        return f"用户名: {username}\n班级: {class_val}\n姓名: {name}\n性别: {gender_str}"
    
    # 多个匹配，列出用户名
    usernames = [row[0] for row in fuzzy_results]
    return f"找到多个匹配用户：{', '.join(usernames)}\n请提供更精确的用户名"

def get_all_users(current_user):
    """获取所有用户列表，只有管理员可以查看"""
    if current_user != "root":
        return "权限不足：只有管理员可以查看用户列表"
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute("SELECT username, class, name, gender FROM users ORDER BY username")
    users = c.fetchall()
    conn.close()
    
    if not users:
        return "没有用户数据"
    
    result = "用户列表：\n"
    for user in users:
        username, class_val, name, gender = user
        
        # 格式化性别显示
        gender_str = ""
        if gender is not None:
            g = str(gender)
            if g in ('1', 'M', 'm', '男'):
                gender_str = '男'
            elif g in ('2', 'F', 'f', '女', '0'):
                gender_str = '女'
            else:
                gender_str = g
        
        result += f"用户名: {username}, 班级: {class_val}, 姓名: {name}, 性别: {gender_str}\n"
    
    return result


def standardize_gender_value(gender_value):
    if gender_value is None:
        return None
    gender_str = str(gender_value).strip()
    if gender_str in ("男", "1", "M", "m", "male", "Male"):
        return 1
    if gender_str in ("女", "0", "F", "f", "female", "Female"):
        return 0
    if gender_str == "":
        return None
    return None


def standardize_role_value(role_value):
    if role_value is None:
        return 2
    role_str = str(role_value).strip()
    if role_str in ("管理员", "0", "admin", "Admin", "administrator", "Administrator"):
        return 0
    if role_str in ("教师", "1", "teacher", "Teacher"):
        return 1
    if role_str in ("普通用户", "2", "user", "User"):
        return 2
    return 2


def bulk_delete_users(pattern, current_user):
    if current_user != "root":
        return "权限不足：只有管理员可以批量删除用户"

    if not pattern or not pattern.strip():
        return "请输入要删除的用户名包含的关键词"

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    like_pattern = f"%{pattern.strip()}%"
    c.execute("SELECT username FROM users WHERE username LIKE ? AND username != 'root'", (like_pattern,))
    users = [row[0] for row in c.fetchall()]
    if not users:
        conn.close()
        return f"未找到包含 '{pattern}' 的用户"

    c.executemany("DELETE FROM users WHERE username=?", [(u,) for u in users])
    conn.commit()
    conn.close()

    # 删除对应用户目录
    deleted_dirs = []
    failed_dirs = []
    for u in users:
        user_dir = os.path.abspath(u)
        if os.path.isdir(user_dir):
            try:
                shutil.rmtree(user_dir)
                deleted_dirs.append(u)
            except Exception as e:
                failed_dirs.append(f"{u}({e})")

    users_display = ", ".join(users[:20])
    if len(users) > 20:
        users_display += f", ... 共 {len(users)} 个用户"

    result = f"已删除 {len(users)} 个用户：{users_display}"
    if deleted_dirs:
        result += f"\n已删除 {len(deleted_dirs)} 个用户目录"
    if failed_dirs:
        result += f"\n删除目录失败：{', '.join(failed_dirs)}"
    return result


def generate_user_import_template():
    template_path = os.path.abspath("user_import_template.csv")
    header = "username,password,class,name,gender,role\n"
    sample = "s11001,Password123,1,张三,男,普通用户\n"
    try:
        with open(template_path, "w", encoding="gbk", newline="") as f:
            f.write(header)
            f.write(sample)
        return template_path
    except Exception as e:
        return None


def import_users_from_file(file_path, current_user):
    if current_user != "root":
        return "权限不足：只有管理员可以导入用户"

    if not file_path or not os.path.exists(file_path):
        return "导入文件不存在"

    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".csv":
        return "仅支持 CSV 格式的导入文件"

    success_count = 0
    failure_messages = []

    def read_csv_with_encoding(encoding):
        with open(file_path, newline="", encoding=encoding) as f:
            reader = csv.DictReader(f)
            return list(reader)

    try:
        try:
            rows = read_csv_with_encoding("utf-8-sig")
        except UnicodeDecodeError:
            rows = read_csv_with_encoding("gbk")

        row_index = 1
        for row in rows:
            row_index += 1
            if not row or not any(str(v).strip() for v in row.values()):
                continue

            username = str(row.get("username", "")).strip()
            password = str(row.get("password", "")).strip()
            class_val = str(row.get("class", "")).strip()
            name = str(row.get("name", "")).strip()
            gender = str(row.get("gender", "")).strip()
            role = str(row.get("role", "")).strip()

            if not username or not password:
                failure_messages.append(f"第{row_index}行：用户名或密码不能为空")
                continue

            if class_val == "":
                class_num = None
            else:
                try:
                    class_num = int(class_val)
                except ValueError:
                    failure_messages.append(f"第{row_index}行：班级必须为整数")
                    continue

            gender_num = standardize_gender_value(gender)
            role_num = standardize_role_value(role)

            result = register_user(username, password, class_num, name, gender_num, current_user, role_num)
            if "成功" in result:
                success_count += 1
            else:
                failure_messages.append(f"第{row_index}行：{result}")
    except Exception as e:
        return f"导入失败：{str(e)}"

    summary = [f"导入完成：成功 {success_count} 个用户，失败 {len(failure_messages)} 条"]
    if failure_messages:
        summary.append("失败明细：")
        summary.extend(failure_messages[:20])
        if len(failure_messages) > 20:
            summary.append(f"...还有 {len(failure_messages) - 20} 条失败记录")
    return "\n".join(summary)


def handle_bulk_delete_users(pattern, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    return bulk_delete_users(pattern, current_user)


def handle_import_users(import_file, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    return import_users_from_file(import_file, current_user)


def get_online_users_count():
    """获取当前在线人数"""
    global active_users
    return active_users

def update_online_users_display():
    """更新在线人数显示"""
    count = get_online_users_count()
    htmlstr = f"<p style='text-align: center;font-size: 14px;'>当前在线人数：{count}</p>"
    return gr.update(value=htmlstr)

def login(username_or_name, password, state):
    """用户登录（支持用户名和姓名双重输入）"""
    global active_users
    
    msg = ""
    
    logged_in_name = state.get("logged_in_name")
    htmlstr = f"<p style='text-align: center;font-size: 14px;'>在线人数：{active_users}</p>" #初值
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 首先尝试作为用户名查找
    c.execute("SELECT username, password FROM users WHERE username=?", (username_or_name,))
    result = c.fetchone()
    
    if not result:
        # 如果用户名查找失败，尝试作为姓名查找
        c.execute("SELECT username, password FROM users WHERE name=?", (username_or_name,))
        results = c.fetchall()
        
        if len(results) == 0:
            # 既不是用户名也不是姓名
            logged_in = False
            state["logged_in_name"] = ""
            state["class"] = ""
            state["name"] = ""
            state["gender"] = ""
            conn.close()
            return ("用户名或姓名错误", 
                    gr.update(value=htmlstr), 
                    gr.update(visible=False), 
                    gr.update(selected="login_tab"), 
                    state, 
                    gr.FileExplorer(root_dir=get_admin_chat_history_dir()), 
                    gr.update(visible=False), 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(value="<p style='text-align: center;'>请先登录以查看您的HTML资源</p>"),
                    gr.FileExplorer(root_dir=get_html_placeholder_dir()))
        elif len(results) == 1:
            # 姓名唯一，直接使用
            result = results[0]
            username = result[0]
            hashed_password = result[1]
        else:
            # 姓名重复，需要用户输入用户名
            conn.close()
            user_list = "\n".join([f"- {user[0]}" for user in results])
            return (f"姓名 '{username_or_name}' 有重复，请使用用户名登录：\n{user_list}", 
                    gr.update(value=htmlstr), 
                    gr.update(visible=False), 
                    gr.update(selected="login_tab"), 
                    state, 
                    gr.FileExplorer(root_dir=get_admin_chat_history_dir()), 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(visible=False))
    else:
        # 用户名查找成功
        username = result[0]
        hashed_password = result[1]
    
    # 密码验证
    # if username == "root" and password == "":
    #     login_success = True
    # else:
    login_success = check_password(password, hashed_password)
        
    if login_success:
        logged_in = True              
        if not logged_in_name:  # 未登录，防止重复统计
            active_users = active_users + 1  # 统计活动用户数量       
        # 在登录成功前查询用户详情（班级、姓名、性别）以便展示
        try:
            c.execute("SELECT class, name, gender FROM users WHERE username=?", (username,))
            info = c.fetchone()
            if info:
                class_val, name_val, gender_val = info
            else:
                class_val, name_val, gender_val = ("", "", "")
        except Exception:
            class_val, name_val, gender_val = ("", "", "")

        # 格式化性别显示
        gender_str = ""
        if gender_val is not None:
            g = str(gender_val)
            if g in ('1', 'M', 'm', '男'):
                gender_str = '男'
            elif g in ('2', 'F', 'f', '女', '0'):
                gender_str = '女'
            else:
                gender_str = g

        htmlstr = f"<p style='text-align: center;font-size: 14px;'>当前用户：{username}，在线人数：{active_users}</p>"
        # 登录信息显示（作为 login_msg 显示）
        if not msg:  # 如果没有连点五次的消息，显示正常登录消息
            msg = f"<p style='text-align: center;font-size: 14px;'>学号：{username}；班级：{class_val}；姓名：{name_val}；性别：{gender_str}</p>"
        else:
            msg = ""
        conn.close()
        
        # 创建以用户名命名的目录（如果不存在）
        if not os.path.exists(username):
            os.makedirs(username)
        # 确保该用户的 ChatHistory 子目录存在
        user_chat_dir = os.path.join(username, CHAT_HISTORY_DIR)
        os.makedirs(user_chat_dir, exist_ok=True)
        
        # 更新状态和用户名，并保存用户信息到 state 以便后续使用
        state["logged_in_name"] = username
        state["class"] = class_val
        state["name"] = name_val
        state["gender"] = gender_str
        logged_in_name = state.get("logged_in_name")
        # 获取用户的API KEY
        dashscope_api_key, deepseek_api_key = getapi_key(state)
        
        # 用户登录成功后显示用户帐号，班级、姓名、性别等信息

        msg=f"帐号: {username}； 班级: {class_val}； 姓名: {name_val}； 性别: {gender_str}"
            
        # 检查是否是管理员，如果是则显示完整的用户注册管理面板
        # 普通用户也可以看到用户管理面板，但只能修改自己的密码
        user_mgmt_visible = True  # 所有登录用户都可以看到用户管理面板
        
        # 获取HTML文件列表
        html_grid = get_htmlfilelst(state)
        html_content = f"""
        <div style="margin: 5px 0;">
            {html_grid}
        </div>
        """
        
        # 获取用户HTML目录
        user_html_dir = get_account_html_dir(username)
        
        # 返回时同时更新历史文件浏览器根目录为当前用户的 ChatHistory，并显示历史侧栏
        # 同时更新HTML文件浏览器根目录为当前用户的HTML目录
        # 检查用户角色，只有管理员和教师才显示教学资源页面
        is_admin_or_teacher = is_admin(username) or is_teacher(username)
        html_resources_visible = is_admin_or_teacher
        score_tab_visible = is_admin_or_teacher
        
        return (msg, gr.update(value=htmlstr), gr.update(visible=True), 
                gr.update(selected="main_tab"), state, 
                gr.FileExplorer(root_dir=user_chat_dir), 
                gr.update(visible=True), 
                gr.update(visible=user_mgmt_visible), 
                gr.update(visible=html_resources_visible), 
                gr.update(value=html_content), 
                gr.FileExplorer(root_dir=user_html_dir), 
                gr.update(visible=score_tab_visible)
                )
    else:
        logged_in = False
        state["logged_in_name"] = ""
        state["class"] = ""
        state["name"] = ""
        state["gender"] = ""
        conn.close()
        return ("密码错误", 
                gr.update(value=htmlstr), 
                gr.update(visible=False), 
                gr.update(selected="login_tab"), 
                state, 
                gr.FileExplorer(root_dir=get_admin_chat_history_dir()), 
                gr.update(visible=False), 
                gr.update(visible=False), 
                gr.update(visible=False),
                gr.update(value="<p style='text-align: center;'>请先登录以查看您的HTML资源</p>"),
                gr.FileExplorer(root_dir=get_html_placeholder_dir()),
                gr.update(visible=False))
##########################################
# 工具函数
##########################################

# 计算文件内容的MD5哈希值
def calculate_file_hash(file_path):
    """计算文件内容的MD5哈希值，用于判断文件是否相同"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 读取目录下的文件列表
def read_directory(directory_path, extflag=True):
    """读取目录下的所有文件，并将文件名作为列表中的一个元素。并返回列表"""
    # 只读取目录下的文件，不读取目录
    # 如果目录不存在，创建目录
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]      
    # 排除参考提示.txt文件,排除扩展名为json的文件
    files = [file for file in files if file != "参考提示.txt" and not file.endswith(".json") and not file.endswith(".js")]
    # 如果files为空，则返回None
    if not files:
        return [""]
    # 去掉目录路径        
    # 去掉文件后缀
    if extflag:
        files = [os.path.splitext(file)[0] for file in files]
    return files

# 获取用户HTML目录
def get_account_html_dir(logged_in_name: str | None):
    """返回指定账号的 HTML 目录路径。非管理员账号的HTML保存在各自账号目录下，管理员使用 ROOT_DIR。"""
    name = logged_in_name if logged_in_name else DEFAULT_LOGGED_IN_NAME
    # 如果是管理员root则其根目录为ROOT_DIR常量
    if name == ROOT_DIR:
        base = ROOT_DIR
    else:
        base = name
    return os.path.join(base, "html")

# 获取HTML文件列表
def get_htmlfilelst(state):
    """获取用户html目录下的文件列表，生成HTML网格布局"""
    # 简化逻辑：参考ChatHistory的显示逻辑
    logged_in_name = None
    if isinstance(state, dict):
        logged_in_name = state.get("logged_in_name")
    elif hasattr(state, 'get'):
        logged_in_name = state.get("logged_in_name")
    
    # 获取用户HTML目录
    html_dir = get_account_html_dir(logged_in_name)
    
    # 确保目录存在
    os.makedirs(html_dir, exist_ok=True)
    # 在HTML目录下创建一个名为imgs的子目录（如果不存在）则创建
    imgs_dir = os.path.join(html_dir, RESERVED_DIR_NAME)
    os.makedirs(imgs_dir, exist_ok=True)
    
    html_files = read_directory(html_dir, False)
    
    # 按文件名排序
    html_files_sorted = sorted(html_files) if html_files else []

    # 生成文件网格HTML，确保路径使用正斜杠
    html_dir_normalized = html_dir.replace('\\', '/')
    
    file_grid_html = f'''
        <style>
        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 10px;
            padding: 0;
            margin: 15px 0;
        }}

        .file-card {{
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            padding: 8px 12px;
            overflow: hidden;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}

        .file-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            border-color: #c8d1d9;
        }}

        .file-card a {{
            display: block;
            color: #24292f;
            text-decoration: none !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            font-size: 13px;
            font-weight: 400;
            line-height: 1.4;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .file-card a:hover {{
            color: #0969da;
            text-decoration: none !important;
        }}

        .file-card a::before {{
            content: "📄";
            margin-right: 6px;
            font-size: 12px;
        }}
        </style>

        <div class="file-grid">
            {"".join(
                f'<div class="file-card"><a href="/gradio_api/file={html_dir_normalized}/{f}" target="_blank">{os.path.splitext(f)[0]}</a></div>'
                for f in html_files_sorted
            )}
        </div>
        '''
    # file_grid_html=file_grid_html+f'<div class="file-grid"><div class="file-card"><a href="/gradio_api/file=root/html/puzzle/index.html" target="_blank">互动小活动</a></div>  \
    # <div class="file-card"><a href="/gradio_api/file=root/html/ai/index.html" target="_blank">人工智能通识课</a></div></div>'
    
    return file_grid_html

# 获取HTML占位目录
def get_html_placeholder_dir():
    """返回一个启动占位的HTML目录（避免在启动时绑定管理员目录）。"""
    return os.path.abspath(os.path.join('.', '.html_placeholder'))

# ---------- 用户目录辅助函数 ----------
def get_account_chat_history_dir(logged_in_name: str | None):
    """返回指定账号的 ChatHistory 目录路径。非管理员账号的历史保存在各自账号目录下，管理员使用 ROOT_DIR。"""
    name = logged_in_name if logged_in_name else DEFAULT_LOGGED_IN_NAME
    # 如果是管理员root则其根目录为ROOT_DIR常量
    if name == ROOT_DIR:
        base = ROOT_DIR
    else:
        base = name
    return os.path.join(base, CHAT_HISTORY_DIR)

def get_admin_chat_history_dir():
    """返回管理员（root）下的 ChatHistory 目录路径，用于 Summary/Task 等共享目录。"""
    return os.path.join(ROOT_DIR, CHAT_HISTORY_DIR)

def get_history_placeholder_dir():
    """返回一个启动占位的历史目录（避免在启动时绑定管理员目录）。"""
    return os.path.abspath(os.path.join('.', '.history_placeholder'))

# 获取用户的API KEY
def getapi_key(session_state=None):
    # 从 session_state 获取登录用户，否则使用默认管理员 root
    logged_in_name = session_state.get("logged_in_name") if session_state else "None"
    logged_in_name = DEFAULT_LOGGED_IN_NAME # 默认使用管理员的环境变量!!!!
    # 判断用户是否是管理员或教师
    if is_admin(logged_in_name):
        logged_in_name = DEFAULT_LOGGED_IN_NAME  # 管理员使用 ROOT_DIR 目录
        env_path=os.path.join(logged_in_name, ".env")
    else:
        logged_in_name = logged_in_name  # 使用自己的目录
        env_path=os.path.join(logged_in_name, ".env")
        # 如果目录下没有 .env 文件,则复制根目录下的 .env 文件到用户目录下
        if not os.path.exists(env_path):
            shutil.copyfile(os.path.join(".env"), env_path)
        
     # 清除全局环境变量中的缓存，防止污染
    for key in ["dashscope_api_key", "deepseek_api_key"]:
        if key in os.environ:
            del os.environ[key]
    # 清除dashscope.api_key
    #dashscope.api_key = None
            
    load_dotenv(env_path)
    dashscope_apikey  = os.getenv("dashscope_api_key")
    deepseek_apikey=os.getenv("deepseek_api_key")
    #dashscope.api_key = dashscope_apikey 
    #print(f"从 {env_path} 加载的 DashScope API KEY: {dashscope_apikey}")
    #print(f"从 {env_path} 加载的 DeepSeek API KEY: {deepseek_apikey}")
    return dashscope_apikey,deepseek_apikey



# 用户上下文相关函数
def get_user_context(session_state):
    """从session_state提取用户上下文信息"""
    if not session_state or not isinstance(session_state, dict):
        return None
    
    username = session_state.get("logged_in_name", "")
    class_info = session_state.get("class", "")
    name = session_state.get("name", "")
    gender = session_state.get("gender", "")
    
    # 如果是默认用户或未登录，返回None
    if not username or username == DEFAULT_LOGGED_IN_NAME:
        return None
    
    return {
        "username": username,
        "class": class_info,
        "name": name,
        "gender": gender
    }

def build_user_system_message(user_context):
    """构建包含用户信息的系统消息"""
    if not user_context:
        return None
    
    system_message = f"当前对话用户信息：\n"
    system_message += f"- 用户名/学号：{user_context['username']}\n"
    if user_context['class']:
        system_message += f"- 班级：{user_context['class']}\n"
    if user_context['name']:
        system_message += f"- 姓名：{user_context['name']}\n"
    if user_context['gender']:
        system_message += f"- 性别：{user_context['gender']}\n"
    
    system_message += "\n请记住这些用户信息，在适当的时候使用（如个性化称呼、提供班级相关的建议等），但不要每次回答都重复显示这些信息。"
    
    return system_message

def enhance_prompt_with_user_context(prompt, session_state):
    """增强提示词，包含用户上下文"""
    if not prompt:
        return prompt
    
    user_context = get_user_context(session_state)
    if not user_context:
        return prompt
    
    system_message = build_user_system_message(user_context)
    if not system_message:
        return prompt
    
    # 将系统消息和用户提示合并
    enhanced_prompt = f"{system_message}\n\n用户问题：{prompt}"
    return enhanced_prompt

# 获取访问host地址
def get_host(request: gr.Request):
    headers = request.headers    
    hostip=request.headers.get("host")   

    # 判断是 HTTP 还是 HTTPS
    protocol = "http"  # 默认使用 HTTP
    forwarded_proto = headers.get("x-forwarded-proto", "").lower()  # 如果使用反向代理，优先获取 x-forwarded-proto
    if forwarded_proto == "https":
        protocol = "https"
    elif str(request.url).startswith("https"):  # 将 URL 对象转换为字符串再判断
        protocol = "https"
    else:
        protocol = "http"
    link=hostip.split(":")[0]+":7862"   
    htmlstr=f"""
                    <p style='text-align: center;' id="smartkb-link">
                        更多功能请访问 <a href='{protocol}://{link}' target='_blank'>智能助手-SmartKB</a>
                        </p>
                        """
    return htmlstr

# 定义一个函数读取文本文件，按行返回一个列表
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines

##########################################
# 请求限制相关函数
##########################################

def check_ip_daily_requests(ip_address):
    """
    根据访问日志检查IP地址当天的请求次数
    :param ip_address: IP地址
    :return: (是否允许请求, 剩余次数)
    """
       
    today = time.strftime('%Y-%m-%d')
    count = 0
    
    # 查找所有当月的日志文件
    log_pattern = os.path.join(LOG_FILES_DIR, f"access_{time.strftime('%Y-%m')}.log")
    log_files = []
    if os.path.exists(log_pattern):
        log_files.append(log_pattern)
    
    # 如果今天是月初前几天，还需要检查上个月的日志文件（如果存在跨月的情况）
    # 不过对于我们的需求来说应该不需要这么复杂
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        # 检查是否为今天的日志且来自该IP
                        if line.startswith(f"[{today}") and f"IP: {ip_address}," in line:
                            count += 1
            except FileNotFoundError:
                # 日志文件还不存在
                pass
            except Exception:
                # 其他异常，但不影响主流程
                pass
            
    # 如果不启用请求限制，直接返回允许请求
    if not enable_request_limit:
        return True, float('inf')
    
    # 检查是否超过限制
    if count >= maxallowed_requests:
        return False, 0
    else:
        return True, maxallowed_requests - count

def log_access_with_limit_check(ip_address, prompt):
    """
    记录访问日志并检查是否超过限制
    :param ip_address: IP地址
    :param prompt: 用户请求
    :return: (是否允许请求, 剩余次数)
    """
    # 先检查是否超过限制
    allowed, remaining = check_ip_daily_requests(ip_address)
    
    # 记录日志
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IP: {ip_address}, Prompt: {prompt}\n"
    log_filename = os.path.join(LOG_FILES_DIR, f"access_{time.strftime('%Y-%m')}.log")
    
    # 确保日志目录存在
    os.makedirs(LOG_FILES_DIR, exist_ok=True)
    
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)
        
    return allowed, remaining

##########################################
# 文件处理相关函数
##########################################
def upload_file_and_get_id(file_path, session_state):
    """
    上传文件到DashScope并获取文件ID
    
    :param file_path: 本地文件路径
    :return: 文件ID
    """            
    api_key, _ = getapi_key(session_state) 
    with open(file_path, 'rb') as f:
        file_response = requests.post(
            f"{QWEN_OPENAI_API_BASE}/files",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            files={
                'file': f,
                'purpose': (None, 'file-extract')
            }
        )
        
    if file_response.status_code == 200:
        result = file_response.json()
        file_id = result.get('id')
        return file_id
    else:
        raise Exception(f"文件上传失败: {file_response.text}")

# 文件类型检测函数
def is_image_file(file_path):
    """判断文件是否为图像文件"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in IMAGE_EXTENSIONS

def is_document_file(file_path):
    """判断文件是否为文档文件"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in DOCUMENT_EXTENSIONS

def check_file_size(file_path, max_size_mb=10):
    """检查文件大小是否超过限制"""
    if file_path is None or not os.path.exists(file_path):
        return True  # 文件不存在则不检查
    
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024  # 转换为字节
    
    return file_size <= max_size_bytes

# 图像理解相关函数
def encode_image_to_base64(image_path):
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

##########################################
# 任务管理相关函数
##########################################

def get_user_active_task_file_path(username):
    """获取用户的活动任务文件路径 - 同时在管理员root目录和用户自己目录中存储"""
    # 主要存储在管理员root的Task目录下（用于系统统一管理）
    chat_history_dir = get_admin_chat_history_dir()
    admin_task_dir = os.path.join(chat_history_dir, TASK_DIR_NAME, username)
    os.makedirs(admin_task_dir, exist_ok=True)
    admin_task_file = os.path.join(admin_task_dir, ACTIVE_TASKS_FILE)
    
    # 同时在用户自己的目录中存储一份副本（用于用户查看自己的任务）
    user_chat_history_dir = get_account_chat_history_dir(username)
    user_task_dir = os.path.join(user_chat_history_dir, TASK_DIR_NAME)
    os.makedirs(user_task_dir, exist_ok=True)
    user_task_file = os.path.join(user_task_dir, ACTIVE_TASKS_FILE)
    
    # 返回管理员目录中的文件路径（作为主要存储）
    return admin_task_file

def load_user_active_tasks(username):
    """加载用户的活动任务列表"""
    active_tasks_file = get_user_active_task_file_path(username)
    if os.path.exists(active_tasks_file):
        try:
            with open(active_tasks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载用户 {username} 的活动任务时出错: {e}")
    return {"tasks": []}

def save_user_active_tasks(username, active_tasks):
    """保存用户的活动任务列表 - 同时在管理员目录和用户自己目录中存储"""
    # 主要存储在管理员root的Task目录下（用于系统统一管理）
    chat_history_dir = get_admin_chat_history_dir()
    admin_task_dir = os.path.join(chat_history_dir, TASK_DIR_NAME, username)
    os.makedirs(admin_task_dir, exist_ok=True)
    admin_task_file = os.path.join(admin_task_dir, ACTIVE_TASKS_FILE)
    
    # 同时在用户自己的目录中存储一份副本（用于用户查看自己的任务）
    user_chat_history_dir = get_account_chat_history_dir(username)
    user_task_dir = os.path.join(user_chat_history_dir, TASK_DIR_NAME)
    os.makedirs(user_task_dir, exist_ok=True)
    user_task_file = os.path.join(user_task_dir, ACTIVE_TASKS_FILE)
    
    try:
        # 保存到管理员目录
        with open(admin_task_file, 'w', encoding='utf-8') as f:
            json.dump(active_tasks, f, ensure_ascii=False, indent=2)
        
        # 同时保存到用户自己的目录
        with open(user_task_file, 'w', encoding='utf-8') as f:
            json.dump(active_tasks, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"保存用户 {username} 的活动任务时出错: {e}")

def create_task(creator, task_name):
    """创建新任务"""
    if not can_create_task(creator):
        return None, "权限不足：只有管理员和教师可以创建任务"
    
    # 加载用户的任务列表
    user_tasks = load_user_active_tasks(creator)
    
    # 检查是否有活动任务，如果有则将其状态改为非活动
    for task in user_tasks["tasks"]:
        if task["status"] == "active":
            task["status"] = "inactive"
    
    # 生成任务ID
    task_id = f"{creator}_{task_name}_{int(time.time())}"
    
    # 检查任务是否已存在
    for task in user_tasks["tasks"]:
        if task["creator"] == creator and task["name"] == task_name:
            # 如果任务已存在，将其状态改为活动
            task["status"] = "active"
            task["created_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            task["submissions"] = []  # 重置提交列表
            save_user_active_tasks(creator, user_tasks)
            # 更新root目录的统一任务文件
            update_unified_tasks_file()
            return task, f"任务 '{task_name}' 已重新激活"
    
    # 创建新任务
    new_task = {
        "id": task_id,
        "creator": creator,
        "name": task_name,
        "status": "active",
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "submissions": []
    }
    
    # 添加到用户的任务列表
    user_tasks["tasks"].append(new_task)
    
    # 保存用户的任务
    save_user_active_tasks(creator, user_tasks)
    
    # 更新root目录的统一任务文件
    update_unified_tasks_file()
    
    # 创建任务目录
    create_task_directories(creator, task_name)
    
    return new_task, f"任务 '{task_name}' 创建成功"

def create_task_directories(creator, task_name):
    """创建任务相关的目录结构"""
    chat_history_dir = get_admin_chat_history_dir()
    
    # 创建教师汇总目录
    teacher_summary_dir = os.path.join(chat_history_dir, SUMMARY_DIR_NAME, TEACHERS_SUMMARY_DIR, creator)
    os.makedirs(teacher_summary_dir, exist_ok=True)
    
    # 创建管理员汇总目录
    admin_summary_dir = os.path.join(chat_history_dir, SUMMARY_DIR_NAME, ADMIN_SUMMARY_DIR)
    os.makedirs(admin_summary_dir, exist_ok=True)

def update_unified_tasks_file():
    """更新root目录中的统一任务文件"""
    all_active_tasks = []
    
    # 获取管理员的活动任务
    admin_tasks = load_user_active_tasks("root")
    all_active_tasks.extend([task for task in admin_tasks["tasks"] if task["status"] == "active"])

    # 获取所有教师的活动任务
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE role = 1")
    teachers = c.fetchall()
    conn.close()

    for teacher in teachers:
        teacher_username = teacher[0]
        teacher_tasks = load_user_active_tasks(teacher_username)
        all_active_tasks.extend([task for task in teacher_tasks["tasks"] if task["status"] == "active"])
    
    # 保存到统一任务文件
    unified_tasks_path = os.path.join(get_admin_chat_history_dir(), TASK_DIR_NAME, "all_active_tasks.json")
    os.makedirs(os.path.dirname(unified_tasks_path), exist_ok=True)
    
    try:
        with open(unified_tasks_path, 'w', encoding='utf-8') as f:
            json.dump({"tasks": all_active_tasks}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"更新统一任务文件失败: {e}")

def get_all_active_tasks():
    """获取所有用户的活动任务"""
    # 首先尝试从统一的任务文件中获取
    unified_tasks_path = os.path.join(get_admin_chat_history_dir(), TASK_DIR_NAME, "all_active_tasks.json")
    
    if os.path.exists(unified_tasks_path):
        try:
            with open(unified_tasks_path, 'r', encoding='utf-8') as f:
                unified_tasks = json.load(f)
                return unified_tasks["tasks"]
        except Exception as e:
            print(f"读取统一任务文件失败: {e}")
    
    # 如果统一文件不存在或读取失败，回退到原来的逻辑
    all_active_tasks = []

    # 获取管理员的活动任务
    admin_tasks = load_user_active_tasks("root")
    all_active_tasks.extend([task for task in admin_tasks["tasks"] if task["status"] == "active"])

    # 获取所有教师的活动任务
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE role = 1")
    teachers = c.fetchall()
    conn.close()

    for teacher in teachers:
        teacher_username = teacher[0]
        teacher_tasks = load_user_active_tasks(teacher_username)
        all_active_tasks.extend([task for task in teacher_tasks["tasks"] if task["status"] == "active"])

    return all_active_tasks

def submit_to_task(student_user, task_info, conversation_content):
    """提交对话到指定任务"""
    if not task_info:
        return False, "任务信息为空"
    
    # 更新任务创建者的任务提交列表
    creator = task_info["creator"]
    user_tasks = load_user_active_tasks(creator)
    
    for task in user_tasks["tasks"]:
        if task["id"] == task_info["id"]:
            if student_user not in task["submissions"]:
                task["submissions"].append(student_user)
            break
    
    save_user_active_tasks(creator, user_tasks)
    
    # 保存到教师汇总（管理员目录）
    teacher_summary_path = os.path.join(
        get_admin_chat_history_dir(), 
        SUMMARY_DIR_NAME, 
        TEACHERS_SUMMARY_DIR, 
        task_info["creator"], 
        f"summary_{task_info['name']}.md"
    )
    save_to_summary_file(teacher_summary_path, student_user, conversation_content)
    
    # 保存到教师自己的目录（方便教师查看）
    teacher_own_summary_path = os.path.join(
        get_account_chat_history_dir(task_info["creator"]),
        SUMMARY_DIR_NAME,
        f"summary_{task_info['name']}.md"
    )
    save_to_summary_file(teacher_own_summary_path, student_user, conversation_content)
    
    # 保存到管理员汇总
    admin_summary_path = os.path.join(
        get_admin_chat_history_dir(), 
        SUMMARY_DIR_NAME, 
        ADMIN_SUMMARY_DIR, 
        f"summary_{task_info['creator']}_{task_info['name']}.md"
    )
    save_to_summary_file(admin_summary_path, student_user, conversation_content)
    
    return True, f"已提交到任务 '{task_info['name']}'"

def save_to_summary_file(summary_path, student_user, conversation_content):
    """保存提交内容到汇总文件"""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    file_exists = os.path.exists(summary_path)
    
    with open(summary_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(f"# 任务汇总\n\n")
            f.write(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
        
        f.write(f"## 学生 {student_user}\n\n")
        f.write(f"提交时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"内容:\n{conversation_content}\n\n")
        f.write("---\n\n")

def detect_task_creation(message, current_user):
    """检测对话中是否包含创建任务意图"""
    if message.startswith("提交") and message.endswith("任务"):
        # 提取任务名称：去掉"提交"和"任务"
        task_name = message[2:-2].strip()
        if task_name:
            return create_task(current_user, task_name)
    return None, None

def get_user_class(username):
    """获取用户班级信息"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT class FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_user_relevant_tasks(student_user, active_tasks):
    """获取与学生用户相关的任务（智能筛选）"""
    relevant_tasks = []
    
    # 1. 获取学生班级信息
    student_class = get_user_class(student_user)
    
    # 2. 优先选择学生所在班级教师创建的任务
    if student_class:
        for task in active_tasks:
            if is_teacher(task["creator"]):
                teacher_class = get_user_class(task["creator"])
                if teacher_class == student_class:
                    relevant_tasks.append(task)
    
    # 3. 如果没有班级相关任务，选择所有教师任务（包括没有设置班级的教师）
    if not relevant_tasks:
        for task in active_tasks:
            if is_teacher(task["creator"]):
                relevant_tasks.append(task)
    
    # 4. 同时显示管理员任务（与教师任务一起显示）
    for task in active_tasks:
        if task["creator"] == "root" and task not in relevant_tasks:
            relevant_tasks.append(task)
    
    return relevant_tasks

def detect_task_submission(message, current_user):
    """改进的任务提交检测，支持任务编号选择和智能筛选"""
    
    # 基础检测：用户输入"完成"或"结束"
    if message.strip() in ["完成", "结束"]:
        active_tasks = get_all_active_tasks()
        if not active_tasks:
            return None, "当前没有活动任务，无法提交"
        
        # 获取与学生相关的任务（智能筛选）
        relevant_tasks = get_user_relevant_tasks(current_user, active_tasks)
        
        if len(relevant_tasks) == 1:
            # 只有一个相关任务，直接提交
            return relevant_tasks[0], None
        elif len(relevant_tasks) > 1:
            # 需要用户选择
            task_list = "\n".join([f"{i+1}. {task['name']}（创建者：{task['creator']}）" for i, task in enumerate(relevant_tasks)])
            return None, f"当前有多个活动任务，请选择：\n{task_list}\n请输入任务编号（1-{len(relevant_tasks)}）："
        else:
            return None, "当前没有适合您的活动任务"
    
    # 支持用户直接指定任务编号
    if message.strip().isdigit():
        task_number = int(message.strip())
        relevant_tasks = get_user_relevant_tasks(current_user, get_all_active_tasks())
        if 1 <= task_number <= len(relevant_tasks):
            return relevant_tasks[task_number - 1], None
        else:
            return None, f"任务编号无效，请输入 1-{len(relevant_tasks)} 之间的数字"
    
    return None, None

def verify_task_submission(student_user, task_info, conversation_content):
    """验证任务提交是否成功"""
    success, message = submit_to_task(student_user, task_info, conversation_content)
    
    if success:
        # 验证文件是否实际创建
        teacher_summary_path = os.path.join(
            get_admin_chat_history_dir(), 
            SUMMARY_DIR_NAME, 
            TEACHERS_SUMMARY_DIR, 
            task_info["creator"], 
            f"summary_{task_info['name']}.md"
        )
        
        admin_summary_path = os.path.join(
            get_admin_chat_history_dir(), 
            SUMMARY_DIR_NAME, 
            ADMIN_SUMMARY_DIR, 
            f"summary_{task_info['creator']}_{task_info['name']}.md"
        )
        
        # 检查文件是否创建成功
        teacher_file_exists = os.path.exists(teacher_summary_path)
        admin_file_exists = os.path.exists(admin_summary_path)
        
        if teacher_file_exists and admin_file_exists:
            return True, f"✅ 任务提交成功！\n已保存到：{task_info['name']}（创建者：{task_info['creator']}）"
        else:
            return False, "⚠️ 任务提交失败：汇总文件未正确创建"
    
    return False, message

##########################################
# 对话历史管理
##########################################


# 添加保存对话记录的函数
def save_conversation_history(conversation_history, session_id, file_path=None, session_state=None):
    """保存对话记录到Markdown文件"""
    if not conversation_history:
        return
    # 从 session_state 获取登录用户，否则使用默认管理员 root
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
    
    # 统一使用 get_account_chat_history_dir
    chat_history_dir = get_account_chat_history_dir(logged_in_name)
    
    # 保护：非管理员用户不得写入管理员 ChatHistory
    admin_dir = os.path.abspath(get_admin_chat_history_dir())
    chat_history_dir_abs = os.path.abspath(chat_history_dir)
    if logged_in_name != ROOT_DIR and chat_history_dir_abs.startswith(admin_dir):
        # 将存储路径强制到用户自己的目录
        chat_history_dir = os.path.abspath(os.path.join(logged_in_name, CHAT_HISTORY_DIR))
    
    os.makedirs(chat_history_dir, exist_ok=True)
    
    # 创建以当前日期命名的目录 (年-月-日)
    current_date = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(chat_history_dir, current_date)
    os.makedirs(date_dir, exist_ok=True)
    
    # 处理文件列表情况，使用第一个有效文件
    actual_file_path = None
    if isinstance(file_path, list):
        # 如果是文件列表，遍历找到第一个存在的文件
        for path in file_path:
            if path is not None and isinstance(path, str) and os.path.exists(path):
                actual_file_path = path
                break
    elif file_path is not None and isinstance(file_path, str) and os.path.exists(file_path):
        # 如果是单个文件路径
        actual_file_path = file_path
    
    if session_id is not None:
        filename = f"conversation_{session_id}.md"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"
    
    file_path = os.path.join(date_dir, filename)
    
    # 检查文件是否已存在
    file_exists = os.path.exists(file_path)
    
    # 追加对话历史到Markdown文件
    with open(file_path, "a", encoding="utf-8") as f:
        # 如果是新文件，写入标题和基本信息
        if not file_exists:
            #f.write(f"# 对话记录\n\n")
            # if actual_file_path is not None:
            #     f.write(f"文件: {os.path.basename(actual_file_path)}\n\n")
            #f.write(f"会话ID: {session_id}\n\n")
            f.write(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
        
        # 只写入最新的对话条目（最后两个：用户输入和AI响应）
        # 假设conversation_history中的最后两项是刚刚添加的用户输入和AI响应
        if len(conversation_history) >= 2:
            user_item = conversation_history[-2]  # 倒数第二项应该是用户输入
            ai_item = conversation_history[-1]    # 最后一项应该是AI响应
            
            # 写入用户输入
            if user_item["role"] == "user":
                f.write(f"**用户** ({time.strftime('%Y-%m-%d %H:%M:%S')}): {user_item['content']}\n\n")
            
            # 写入AI响应
            if ai_item["role"] == "assistant":
                f.write(f"**助手** ({time.strftime('%Y-%m-%d %H:%M:%S')}): {ai_item['content']}\n\n")
        
        f.write("---\n\n")  # 添加分隔线
                
    return file_path


def create_unique_task_name(task_name):
    """创建唯一的任务名称，如果任务已存在则添加序号"""
    # 任务与汇总放在管理员目录下
    chat_history_dir = get_admin_chat_history_dir()
    os.makedirs(chat_history_dir, exist_ok=True)

    # Summary目录（管理员目录下）
    summary_dir = os.path.join(chat_history_dir, SUMMARY_DIR_NAME)
    os.makedirs(summary_dir, exist_ok=True)
    
    # 清理任务名称，避免非法字符
    cleaned_task_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    base_filename = f"summary_{cleaned_task_name}.md"
    base_filepath = os.path.join(summary_dir, base_filename)
    
    # 如果文件不存在，直接返回原任务名称
    if not os.path.exists(base_filepath):
        return task_name
    
    # 如果文件已存在，添加序号
    counter = 1
    while True:
        new_task_name = f"{task_name}_{counter}"
        new_filename = f"summary_{new_task_name}.md"
        new_filepath = os.path.join(summary_dir, new_filename)
        
        if not os.path.exists(new_filepath):
            return new_task_name
            
        counter += 1

##########################################
# AI聊天相关函数
##########################################

# 修改agent_chart函数以支持session_id 云端知识库
def agent_chatX(prompt, session_state=None):
    # 从 session_state 获取登录用户，否则使用默认
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    session_id = None
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
        session_id = session_state.get("session_id")
    dashscope_api_key, _ = getapi_key(session_state)
    yield f"请稍候...", session_id        
    
    # 增强提示词，包含用户上下文
    enhanced_prompt = enhance_prompt_with_user_context(prompt, session_state)
    
    # 准备调用参数
    call_params = {
        "api_key": dashscope_api_key,  # type: ignore
        "app_id": APPID,  # 应用ID
        "prompt": enhanced_prompt,
        "memory_id": MEMORY_ID,
        "stream": True,  # 流式输出
        "incremental_output": True,  # 增量输出
        "headers": {  # 添加头部信息支持
            "X-DashScope-OssResourceResolve": "enable"
        }
    }
    
    # 如果存在session_id，则添加到调用参数中
    if session_id:
        call_params["session_id"] = session_id
    try:    
        response = Application.call(**call_params)  
    except Exception as e:    
        yield "网络连接错误：请检查您的网络连接或稍后重试！", session_id
        return
    
    full_text = ""
    session_id_from_response = None
    
    try:
        for chunk in response:
            # 尝试从第一个chunk中获取session_id
            if session_id_from_response is None and hasattr(chunk, 'output') and hasattr(chunk.output, 'session_id'):
                session_id_from_response = chunk.output.session_id
                if session_id_from_response:
                    session_id = session_id_from_response
                    
            if chunk.output is not None and hasattr(chunk.output, 'text') and chunk.output.text:
                full_text += chunk.output.text
                yield full_text, session_id  # 流式返回每次更新的内容和session_id
            elif chunk.output is None  or not hasattr(chunk.output, 'text'):
                yield "内容安全警告：输入的文本数据可能包含不适当的内容！",session_id
    except Exception as e:    
        #yield str(e), session_id
        yield "网络连接错误：请检查您的网络连接或稍后重试！", session_id

   
def agent_chat_with_document(file_path, prompt, session_state=None):
    """使用agent_chat处理文档问答，支持 session_state 获取用户上下文"""
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    session_id = None
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
        session_id = session_state.get("session_id")
        # 如果session_state中没有session_id，则创建一个新的
        if session_id is None:
            session_id = f"{logged_in_name}_{int(time.time())}"
            session_state["session_id"] = session_id
        
    dashscope_api_key, _ = getapi_key(session_state)
    yield "请稍候...", session_id
    
    try:
        file_id = upload_file_and_get_id(file_path, session_state)
        if not file_id:
            yield "文件上传失败：无法获取文件ID", session_id
            return
            
        # 增强提示词，包含用户上下文
        enhanced_prompt = enhance_prompt_with_user_context(prompt, session_state)
        
        # 使用qwen-long模型提取文件内容（流式处理）
        full_content = ""
        try:
            # 使用全局 QWEN_OPENAI_API_BASE 和模型常量
            response = requests.post(
                f"{QWEN_OPENAI_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {dashscope_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_LONG_NAME,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "system", "content": f"fileid://{file_id}"},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    "stream": True
                },
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            data_str = decoded_line[5:]  # 移除 "data:" 前缀
                            if data_str.strip() == "[DONE]":
                                return full_content, session_id
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_content += content
                                        yield full_content, session_id
                            except json.JSONDecodeError:
                                continue
                return full_content, session_id
            else:
                print(f"提取文件内容失败: {response.status_code}, {response.text}")
                yield "无法提取文档内容", session_id
        except Exception as e:
            print(f"提取文件内容时出错: {e}")
            yield "无法提取文档内容", session_id
            
    except Exception as e:
        yield f"文件处理失败: {str(e)}", session_id

def agent_chat_with_image(file_path, prompt, session_state=None):
    """使用agent_chat处理图像问答，支持 session_state 获取用户上下文"""
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    session_id = None
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
        session_id = session_state.get("session_id")
        # 如果session_state中没有session_id，则创建一个新的
        if session_id is None:
            session_id = f"{logged_in_name}_{int(time.time())}"
            session_state["session_id"] = session_id
            
    dashscope_api_key, _ = getapi_key(session_state)
    yield "请稍候...", session_id
       
    model_name = MODEL_VL_NAME  # 图像理解模型
    
    if file_path is None:        
        yield "未提供图像文件", session_id
        return
    
    # 增强提示词，包含用户上下文
    enhanced_prompt = enhance_prompt_with_user_context(prompt, session_state)
    
    try:
        encoded_image = encode_image_to_base64(file_path)        
        response = requests.post(
            f"{QWEN_OPENAI_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {dashscope_api_key}",
                "Content-Type": "application/json",
                "Content-Length": str(len(encoded_image or "") + len(enhanced_prompt))
            },
            json={
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                        {"type": "text", "text": enhanced_prompt}
                    ]
                }],
                "stream": True
            },
            stream=True
        )
        
        full_response = ""
        for chunk in response.iter_content(chunk_size=None):
            if not chunk:
                continue
            try:           
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith("data:"):
                    data = json.loads(chunk_str[5:])
                    if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                        full_response += data["choices"][0]["delta"]["content"]
                        yield full_response, session_id         
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        yield f"图像处理失败: {str(e)}", session_id

# 缓存文件摘要，避免重复计算（key: 绝对路径 -> {hash, summary, mtime}）
FILE_SUMMARY_CACHE = {}

def get_file_summary(file_path, session_state=None):
    """根据文件类型同步获取一个简短摘要（用于作为上下文增强），带缓存：文件未变化时复用摘要"""
    global FILE_SUMMARY_CACHE
    try:
        if not file_path or not os.path.exists(file_path):
            return ""
        abs_path = os.path.abspath(file_path)
        try:
            file_hash = calculate_file_hash(abs_path)
        except Exception:
            # 退回到 mtime 作为降级判断
            file_hash = str(os.path.getmtime(abs_path))
        # 检查缓存
        cache_entry = FILE_SUMMARY_CACHE.get(abs_path)
        if cache_entry and cache_entry.get("hash") == file_hash:
            return cache_entry.get("summary", "")

        # 未命中缓存，生成摘要
        summary_text = ""
        if is_image_file(abs_path):
            prompt = "请简要描述这张图片的主要内容与要点，最多200字。"
            last = ""
            for out, sid in agent_chat_with_image(abs_path, prompt, session_state=session_state):
                last = out
            summary_text = (last or "").strip()
        elif is_document_file(abs_path):
            prompt = "请简要总结该文档的主要结论与要点，最多200字。"
            last = ""
            for out, sid in agent_chat_with_document(abs_path, prompt, session_state=session_state):
                last = out
            summary_text = (last or "").strip()
        else:
            summary_text = ""

        # 截断过长摘要以节省缓存空间
        if summary_text and len(summary_text) > 5000:
            summary_text = summary_text[:5000]

        # 存入缓存
        FILE_SUMMARY_CACHE[abs_path] = {"hash": file_hash, "summary": summary_text, "mtime": os.path.getmtime(abs_path)}
        # print(f"缓存文件摘要: {FILE_SUMMARY_CACHE[abs_path]}") # 打印缓存信息
        return summary_text
    except Exception:
        return ""

def handle_unified_query(file_path, prompt, session_state, include_file_context, request: gr.Request):
    # 从 session_state 获取 session_id
    session_id = session_state.get("session_id") if session_state else None
    
    # 记录访问日志
    if request:
        ip_address = request.client.host
        # 总是记录访问日志
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IP: {ip_address}, Prompt: {prompt}\n"
        log_filename = os.path.join(LOG_FILES_DIR, f"access_{time.strftime('%Y-%m')}.log")
        
        # 确保日志目录存在
        os.makedirs(LOG_FILES_DIR, exist_ok=True)
        
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
            
        # 检查IP请求限制（仅在启用限制时检查）
        if enable_request_limit:
            allowed, remaining = check_ip_daily_requests(ip_address)
            if not allowed:
                yield f"您今天的请求次数已达到上限（{maxallowed_requests}次），请明天再试。今日剩余次数: {remaining}", session_id
                return

    """统一处理查询，根据文件类型决定处理方式"""
    # 支持单个文件或多个文件
    # print("handle_unified_query called with file_path:", file_path)
    file_paths = []
    if file_path is not None:
        if isinstance(file_path, list):
            file_paths = [fp for fp in file_path if fp is not None and os.path.exists(fp)]
        elif os.path.exists(file_path):
            file_paths = [file_path]
    
    # 如果有文件且提示为空，则设置默认提示
    if (prompt is None or prompt.strip() == "") and file_paths:
        # 对于多个文件，提供更通用的提示
        if len(file_paths) > 1:
            if is_document_file(file_paths[0]):
                prompt = "请解读这篇文档的主要内容"
            elif is_image_file(file_paths[0]):
                prompt = "请描述这张图片的内容"
            
    # 检查所有文件的大小
    for fp in file_paths:
        if is_document_file(fp) and not check_file_size(fp):
            yield f"文档大小超过限制（10MB），请上传较小的文件。当前文件大小：{os.path.getsize(fp) / (1024*1024):.2f}MB", session_id
            return
        # 对于图像文件也可以选择检查大小限制
        elif is_image_file(fp) and not check_file_size(fp, 5):  # 图像文件限制为5MB
            yield f"图像大小超过限制（5MB），请上传较小的文件。当前文件大小：{os.path.getsize(fp) / (1024*1024):.2f}MB", session_id
            return
    
    
    # 处理所有文件
    if file_paths:
        # 新增：从文件生成简短摘要，用于上下文增强
        if include_file_context:
            summaries = []
            for fp in file_paths:
                s = get_file_summary(fp, session_state=session_state)
                if s:
                    summaries.append(f"文件 {os.path.basename(fp)} 摘要：\n{s.strip()}")
            if summaries:
                context_text = "\n\n".join(summaries)
                prompt = (context_text + "\n\n" + (prompt or "")).strip()
            
            # 增强提示词，包含用户上下文
            enhanced_prompt = enhance_prompt_with_user_context(prompt, session_state)
            # print("Enhanced Prompt with File Context:", enhanced_prompt) #调试输出

            response_gen = agent_chatX(enhanced_prompt, session_state=session_state)
            
            for response, updated_session_id in response_gen:
                    yield response, updated_session_id    
            
        else:
            # 用于累积所有文件的响应
            accumulated_response = ""
            
            # 分别处理每个文件
            for i, fp in enumerate(file_paths):
                # 如果有多个文件，在每个文件前加上标识
                if len(file_paths) > 1:
                    file_header = f"--- 文件 {i+1}/{len(file_paths)} ---\n\n"
                    # 先输出文件头
                    accumulated_response += file_header
                    yield accumulated_response, session_id
                
                # 判断文件类型并进行相应处理
                file_response = ""  # 用于存储当前文件的响应
                if is_image_file(fp):
                    # 图像文件处理
                    response_gen = agent_chat_with_image(fp, prompt, session_state=session_state)
                    for response, updated_session_id in response_gen:
                        file_response = response  # 只保存最新响应，而非累加
                        yield accumulated_response + file_response, updated_session_id
                elif is_document_file(fp):
                    # 文档文件处理
                    response_gen = agent_chat_with_document(fp, prompt, session_state=session_state)
                    for response, updated_session_id in response_gen:
                        file_response = response  # 只保存最新响应，而非累加
                        yield accumulated_response + file_response, updated_session_id
                else:
                    file_error = f"不支持的文件类型，请上传图像文件或文档文件。文件路径：{fp}\n\n"
                    file_response = file_error
                    yield accumulated_response + file_response, session_id
                # 将当前文件的完整响应添加到累积响应中
                accumulated_response += file_response
                # 在文件处理完成后添加分割线（如果不是最后一个文件）
                if i < len(file_paths) - 1:
                    accumulated_response += "\n" + "-" * 50 + "\n\n"
    else:
        # 没有文件，直接使用文本问答
        if prompt is None or prompt.strip() == "":
            prompt="介绍一下自己"
        # 增强提示词，包含用户上下文
        enhanced_prompt = enhance_prompt_with_user_context(prompt, session_state)
        # print("Enhanced Prompt:", enhanced_prompt) # 调试输出

        response_gen = agent_chatX(enhanced_prompt, session_state=session_state)
        
        for response, updated_session_id in response_gen:
                yield response, updated_session_id

# 创建一个包装函数来处理对话和历史记录
def chat_with_history(file_path, user_input, session_state, include_file_context, request: gr.Request):
    # 初始化session_state
    if session_state is None:
        session_state = {"conversation_history": [], "session_id": None}
    
    # 确保session_state是一个字典并且有正确的键
    if not isinstance(session_state, dict):
        session_state = {"conversation_history": [], "session_id": None}
    
    if "conversation_history" not in session_state:
        session_state["conversation_history"] = []
        
    if "session_id" not in session_state:
        session_state["session_id"] = None
    
    # 使用session_state中的对话历史
    conversation_history = session_state["conversation_history"]
    
    # 确保conversation_history是一个列表
    if not isinstance(conversation_history, list):
        conversation_history = []
    
    # 将用户输入添加到历史记录
    conversation_history.append({"role": "user", "content": user_input})

    # 获取当前登录用户名
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state:
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
    
    # 检查是否是提交、完成或结束操作，并提取任务名称
    is_submission = False
    task_info = None
    submission_message = None
    is_task_creation = False
    
    # 检测任务创建意图
    task_info, task_creation_msg = detect_task_creation(user_input, logged_in_name)
    if task_info:
        is_task_creation = True
        submission_message = task_creation_msg
    
    # 检测任务提交意图
    if not task_info:
        task_info, submission_msg = detect_task_submission(user_input, logged_in_name)
        if task_info:
            is_submission = True
        elif submission_msg:
            # 有提交意图但需要用户选择任务
            is_submission = True
            submission_message = submission_msg
    
    # 如果是提交操作但没有任务信息，尝试获取活动任务
    if is_submission and not task_info and not submission_message:
        active_tasks = get_all_active_tasks()
        if len(active_tasks) == 1:
            task_info = active_tasks[0]
        elif len(active_tasks) > 1:
            task_list = "\n".join([f"{i+1}. {task['name']}（{task['creator']}）" for i, task in enumerate(active_tasks)])
            submission_message = f"当前有多个活动任务，请选择：\n{task_list}\n请输入任务编号："
        else:
            submission_message = "当前没有活动任务，无法提交"
    
    # 构建包含历史记录的完整显示
    history_text = ""
    for item in conversation_history:
        if item["role"] == "user":
            history_text += f"<div class='user-message'>{item['content']}</div>\n\n"
        else:
            history_text += f"<div class='ai-message'>{item['content']}</div>\n\n"
    
    # 获取session_id
    session_id = session_state.get("session_id")
    
    # 获取AI响应
    ai_response = ""
    final_session_id = session_id
    
    # 如果是任务创建操作，直接处理而不调用AI
    if is_task_creation and task_info:
        # 直接显示任务创建成功消息，不调用AI
        ai_response = submission_message
        final_session_id = session_id
        # 显示历史记录 + 任务创建成功消息
        yield history_text + f"<div class='ai-message'>{ai_response}</div>\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=False, value="")
    
    # 如果是任务提交操作（包括任务编号选择），直接处理而不调用AI
    elif is_submission and (submission_message or task_info):
        # 直接显示任务相关消息，不调用AI
        if submission_message:
            ai_response = submission_message
        elif task_info:
            # 如果是任务编号选择，显示提交成功消息
            # 获取完整的对话历史内容
            conversation_content = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conversation_history
            ])
            success, submit_msg = verify_task_submission(logged_in_name, task_info, conversation_content)
            ai_response = submit_msg
        else:
            ai_response = "任务提交处理中..."
        
        final_session_id = session_id
        # 显示历史记录 + 任务相关消息
        yield history_text + f"<div class='ai-message'>{ai_response}</div>\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=False, value="")
    else:
        # 正常调用AI
        # 增强提示词，包含用户上下文
        enhanced_prompt = enhance_prompt_with_user_context(user_input, session_state)
        response_gen = handle_unified_query(file_path, enhanced_prompt, session_state, include_file_context, request)
        for response, updated_session_id in response_gen:
            ai_response = response  # 持续更新为最新的响应
            final_session_id = updated_session_id
            # 显示历史记录 + 当前AI响应
            yield history_text + f"<div class='ai-message'>{ai_response}</div>\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=False, value="")
    
    # 将最终AI响应添加到历史记录
    if ai_response:
        conversation_history.append({"role": "assistant", "content": ai_response})
        
    # 保存对话历史到文件
    save_conversation_history(conversation_history, final_session_id, file_path, session_state)
    
    # 构建最终的历史文本显示
    final_history_text = ""
    for item in conversation_history:
        if item["role"] == "user":
            final_history_text += f"<div class='user-message'>{item['content']}</div>\n\n"
        else:
            final_history_text += f"<div class='ai-message'>{item['content']}</div>\n\n"
    
    # 检查AI响应中是否包含HTML代码块
    html_output_value = ""
    html_visible = False
    # 使用正则表达式查找HTML代码块 (包括用```html包裹的和纯HTML代码)
    
    # 匹配带标签的代码块
    html_block_pattern = r'```(?:html|HTML)\s*(.*?)\s*```'
    # 确保ai_response不是None，如果是None则使用空字符串
    ai_response_text = str(ai_response) if ai_response is not None else ""
    #ai_response_text = ai_response if ai_response is not None else ""
    html_blocks = re.findall(html_block_pattern, ai_response_text, re.DOTALL)
    
    if html_blocks:
        # 如果找到了HTML代码块，显示第一个
        html_output_value = html_blocks[0]
        html_visible = True
    elif "<!DOCTYPE html>" in ai_response_text or "<html>" in ai_response_text:
        # 如果整个响应看起来像HTML文件
        html_output_value = ai_response_text
        html_visible = True
    
    # 特殊处理HTML内容，确保动画等功能正常工作
    if html_visible:
        # 添加allow属性以允许更多功能 sandbox=""表示不限制任何功能
        # allow-scripts allow-same-origin allow-forms allow-popups
        html_output_value = f"""
        <div style="border: 0px solid #ccc; border-radius: 5px; overflow: auto;">
            <iframe 
                srcdoc="{html_output_value.replace('"', '&quot;')}" 
                style="width: 100%; height: 768px; border: none;"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups">
            </iframe>
        </div>
        """
    
    # 如果是提交、完成或结束操作，额外保存到汇总文件
    summary_file_path = None
    if is_submission and task_info:
        # 使用新的任务提交验证功能
        success, submit_msg = verify_task_submission(logged_in_name, task_info, ai_response)
        if success:
            summary_file_path = submit_msg
        else:
            submission_message = submit_msg
    
    # 如果是提交操作，告知用户保存到了哪个汇总文件
    if is_submission:
        if submission_message:
            # 有提交消息（可能是错误或需要选择任务）
            additional_msg = f"\n\n<div class='ai-message'>{submission_message}</div>"
            yield final_history_text + additional_msg + "\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=html_visible, value=html_output_value)
        elif task_info:
            # 成功提交到任务
            task_display_name = task_info['name']
            additional_msg = f"\n\n<div class='ai-message'>✅ 您的对话已成功提交到任务：<strong>{task_display_name}</strong>（创建者：{task_info['creator']}）</div>"
            yield final_history_text + additional_msg + "\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=html_visible, value=html_output_value)
        else:
            # 提交失败
            additional_msg = f"\n\n<div class='ai-message'>⚠️ 提交失败：无法找到活动任务</div>"
            yield final_history_text + additional_msg + "\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=html_visible, value=html_output_value)
    elif is_task_creation and task_info:
        # 任务创建成功
        task_display_name = task_info['name']
        additional_msg = f"\n\n<div class='ai-message'>✅ 任务创建成功！已创建任务：<strong>{task_display_name}</strong><br>其他用户现在可以提交对话到此任务。</div>"
        yield final_history_text + additional_msg + "\n\n", {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=html_visible, value=html_output_value)
        
    # 返回更新后的状态
    yield final_history_text, {"conversation_history": conversation_history.copy(), "session_id": final_session_id, "logged_in_name": logged_in_name, "class": session_state.get("class"), "name": session_state.get("name"), "gender": session_state.get("gender")}, gr.update(visible=html_visible, value=html_output_value)

##########################################
# 示例数据生成函数
##########################################

def generate_random_examples():
    lsts=read_file(os.path.join(ROOT_DIR, "prompttype", PROMPT_FILE_NAME))
    prelst=[] # 预设问题列表
    #随机从lsts列表中选择5个元素加入prelst列表
    random.shuffle(lsts)
    #更换随机种子，保证每次运行结果不同
    random.seed(time.time())  # 使用当前时间作为种子确保每次都不一样
    sampled_items = random.sample(lsts, 5)
    prelst.extend(sampled_items)
    #把prelst列表转成二维列表，每个元素是一个列表
    prelst = [[item] for item in prelst]
    return prelst

 # root/imgs目录中的文件列表做为示例
def read_files_in_directory(directory):
    file_list = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_list.append(file_path)
    return file_list

#随机获取5个文件，保证每次结果不同
def get_random_files(directory, num_files=5):
    file_list = read_files_in_directory(directory)
    #print(file_list)
    random.seed(time.time())
    random.shuffle(file_list)
    #把列表转成二维列表，每个元素是一个列表
    file_list = [[[file_path]] for file_path in file_list] #返回列表的二维列表，用于file_input的samples，列表形式
    return file_list[:num_files] 

# 创建一个函数用于更新examples
def update_examples():
    return (gr.Dataset(components=[query_input], samples=generate_random_examples()),
            gr.Dataset(components=[file_input], samples=get_random_files(os.path.join(ROOT_DIR, "imgs"))))

##########################################
# UI界面组件定义
##########################################

# 添加图像预览组件
lblmsg=gr.HTML()
image_preview = gr.HTML(label="图像预览", visible=False)
query_input = gr.Text(label="提示词",placeholder="发消息，提问题")
file_input=gr.File(label="上传文件（可选）",
                  file_types=DOCUMENT_EXTENSIONS+IMAGE_EXTENSIONS,
                  file_count="multiple",
                   type="filepath",
                   height=200)
css="""
footer {
    display: none !important;
}
/* 添加对话历史区域的滚动样式 */
.chat-history {
    max-height: 50vh;
    overflow-y: auto;
    border: 0px solid #ddd;
    padding: 10px;
    border-radius: 5px;
}

/* 自定义滚动条样式 */
.chat-history::-webkit-scrollbar {
    width: 8px; /* 滚动条宽度 */
}

.chat-history::-webkit-scrollbar-track {
    background: #f1f1f1; /* 滚动条轨道背景色 */
    border-radius: 4px;
}

.chat-history::-webkit-scrollbar-thumb {
    background: #c1c1c1; /* 滚动条滑块颜色 */
    border-radius: 4px;
}

.chat-history::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8; /* 滚动条滑块悬停颜色 */
}

/* 隐藏Gradio默认的滚动条 */
.chat-history .wrap,
.chat-history .prose {
    overflow: unset !important;
    max-height: unset !important;
    height: unset !important;
}
/* 用户消息样式 - 右对齐 */
.user-message {
    background-color: #F0F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
    margin-left: 10%;
}

/* AI消息样式 - 左对齐 */
.ai-message {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: left;
    margin-right: 10%;
    border: 0px solid #e0e0e0;
}
"""

##########################################
# 事件处理函数
##########################################

# 添加预览HTML功能
def preview_html_code_from_output(markdown_content):
    """从输出内容中提取HTML代码并预览"""
    # 使用正则表达式提取HTML代码块
    import re
    
    # 匹配markdown中的HTML代码块
    html_block_pattern = r'```(?:html|HTML)\s*(.*?)\s*```'
    html_blocks = re.findall(html_block_pattern, markdown_content, re.DOTALL)
    
    html_output_value = ""
    
    if html_blocks:
        # 如果找到了HTML代码块
        if len(html_blocks) > 1:
            # 有多个HTML代码块，显示一个简单的列表供选择
            html_output_value = "<p>检测到多个HTML代码块：</p><ol>"
            for i, block in enumerate(html_blocks):
                html_output_value += f'<li><button onclick="document.getElementById(\'html-preview-{i}\').style.display=\'block\';">查看代码块 #{i+1}</button></li>'
            html_output_value += "</ol>"
            
            # 为每个代码块创建一个预览区域，默认隐藏
            for i, block in enumerate(html_blocks):
                display_style = "display: block;" if i == 0 else "display: none;"
                html_output_value += f"""
                <div id="html-preview-{i}" style="{display_style} border: 0px solid #ccc; border-radius: 5px; overflow: auto; margin-top: 10px;">
                    <iframe 
                        srcdoc="{block.replace('"', '&quot;')}" 
                        style="width: 100%; height: 768px; border: none;"
                        sandbox="allow-scripts allow-same-origin allow-forms allow-popups">
                    </iframe>
                </div>
                """
        else:
            # 只有一个HTML代码块
            html_code = html_blocks[0]
            html_output_value = f"""
            <div style="border: 0px solid #ccc; border-radius: 5px; overflow: auto;">
                <iframe 
                    srcdoc="{html_code.replace('"', '&quot;')}" 
                    style="width: 100%; height: 768px; border: none;"
                    sandbox="allow-scripts allow-same-origin allow-forms allow-popups">
                </iframe>
            </div>
            """
    elif "<!DOCTYPE html>" in markdown_content or "<html>" in markdown_content:
        # 如果整个内容看起来像HTML文件
        html_code = markdown_content
        html_output_value = f"""
        <div style="border: 0px solid #ccc; border-radius: 5px; overflow: auto;">
            <iframe 
                srcdoc="{html_code.replace('"', '&quot;')}" 
                style="width: 100%; height: 768px; border: none;"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups">
            </iframe>
        </div>
        """
    else:
        # 尝试从markdown中提取HTML片段
        # 匹配任何HTML标签
        html_pattern = r'<[^>]+>.*?</[^>]+>|<[^>]+/>'
        html_matches = re.findall(html_pattern, markdown_content, re.DOTALL)
        if html_matches:
            html_code = ''.join(html_matches)
            html_output_value = f"""
            <div style="border: 0px solid #ccc; border-radius: 5px; overflow: auto;">
                <iframe 
                    srcdoc="{html_code.replace('"', '&quot;')}" 
                    style="width: 100%; height: 768px; border: none;"
                    sandbox="allow-scripts allow-same-origin allow-forms allow-popups">
                </iframe>
            </div>
            """
        else:
            return gr.update(visible=True, value="<p>在输出内容中未找到有效的HTML代码</p>")
    
    return gr.update(visible=True, value=html_output_value)

def load_chat_history_with_path_from_explorer(file_path, session_state=None):
    """从文件浏览器加载并显示选定的历史对话，确保用户只能访问自己的目录"""
    if not file_path:
        return "", None, gr.update(visible=False), gr.update(value=""), gr.FileExplorer(root_dir=get_account_chat_history_dir(session_state.get("logged_in_name") if session_state else None))

    # 获取当前登录用户，确定允许访问的目录
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln

    allowed_dir = os.path.abspath(get_account_chat_history_dir(logged_in_name))

    try:
        target_path = os.path.abspath(file_path)
    except Exception:
        return "", None, gr.update(visible=False), gr.update(value=""), gr.FileExplorer(root_dir=get_account_chat_history_dir(logged_in_name))

    # 非管理员用户只能访问自己的ChatHistory目录
    if not target_path.startswith(allowed_dir):
        return "", None, gr.update(visible=False), gr.update(value="您无权访问该文件"), gr.FileExplorer(root_dir=get_account_chat_history_dir(logged_in_name))

    if os.path.isfile(target_path) and os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 检查内容是否包含HTML代码块
        html_block_pattern = r'```(?:html|HTML)\s*(.*?)\s*```'
        html_blocks = re.findall(html_block_pattern, content, re.DOTALL)

        # 根据是否有HTML代码块决定是否显示预览按钮
        button_visibility = gr.update(visible=len(html_blocks) > 0)

        return content, [target_path], button_visibility, gr.update(value=""), gr.FileExplorer(root_dir=get_account_chat_history_dir(logged_in_name))
    else:
        return "", None, gr.update(visible=False), gr.update(value=""), gr.FileExplorer(root_dir=get_account_chat_history_dir(logged_in_name))

# 从文件浏览器删除历史对话文件或目录
def delete_chat_history_file_from_explorer(file_path, session_state=None):
    """从文件浏览器删除选中的历史对话文件，确保用户仅能删除自己目录下的文件"""
    msg = ""
    if not file_path:
        msg = "未选择文件"
        # 获取当前登录用户目录
        logged_in_name = DEFAULT_LOGGED_IN_NAME
        if session_state and isinstance(session_state, dict):
            ln = session_state.get("logged_in_name")
            if ln:
                logged_in_name = ln
        target_dir = get_account_chat_history_dir(logged_in_name)
        return gr.FileExplorer(root_dir=target_dir), msg, None

    # 获取当前登录用户目录
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln

    chat_history_dir = os.path.abspath(get_account_chat_history_dir(logged_in_name))
    try:
        target_path = os.path.abspath(file_path)
    except Exception:
        return gr.FileExplorer(root_dir=chat_history_dir), "路径错误", None

    # 非管理员用户不能删除管理员目录或其他用户目录下的文件
    if not target_path.startswith(chat_history_dir):
        return gr.FileExplorer(root_dir=chat_history_dir), "您无权删除该文件", None

    if os.path.exists(target_path):
        if os.path.isfile(target_path):
            try:
                os.remove(target_path)
                msg = f"文件 {os.path.basename(target_path)} 已删除"
            except Exception as e:
                msg = f"删除文件时出错: {e}"
        if os.path.isdir(target_path):                
            try:
                shutil.rmtree(target_path)
                msg = f"目录 {os.path.basename(target_path)} 已删除"
            except Exception as e:
                msg = f"删除目录时出错: {e}"

    Reserved_dir = os.path.join(chat_history_dir, RESERVED_DIR_NAME)
    if not os.path.exists(Reserved_dir):
        os.makedirs(Reserved_dir)

    # 返回删除后对应用户的 FileExplorer 根目录
    # 获取当前选择的目录的父目录，这样删除后可以停留在当前目录
    if os.path.isfile(target_path):
        # 如果是文件，返回文件所在目录
        target_dir = os.path.dirname(target_path)
    else:
        # 如果是目录，返回父目录
        target_dir = os.path.dirname(target_path) if target_path != chat_history_dir else chat_history_dir
    
    # 检查目标目录是否存在，如果不存在或者等于根目录，则返回Reserved目录
    if (not os.path.exists(target_dir)) or target_dir == chat_history_dir:
        target_dir = Reserved_dir
    return gr.FileExplorer(root_dir=target_dir), msg, None

# 对话完成后刷新文件浏览器到当前用户目录
def refresh_file_explorer_after_chat(session_state, current_file_explorer=None):
    """对话完成后刷新文件浏览器，保持当前选中的目录"""
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
    
    # 获取用户根目录
    user_root_dir = get_account_chat_history_dir(logged_in_name)
    
    # 如果提供了当前文件浏览器路径，则尝试保持当前目录
    if current_file_explorer and os.path.exists(current_file_explorer):
        # 如果是文件，取其所在目录
        if os.path.isfile(current_file_explorer):
            target_dir = os.path.dirname(current_file_explorer)
        else:
            # 如果是目录，直接使用
            target_dir = current_file_explorer
        
        # 确保目标目录在用户目录范围内
        if not target_dir.startswith(os.path.abspath(user_root_dir)):
            target_dir = user_root_dir
    else:
        # 否则返回到用户根目录
        target_dir = user_root_dir
        
    # 确保返回的是绝对路径，与 delete_chat_history_file_from_explorer 保持一致
    target_dir = os.path.abspath(target_dir)
    
    Reserved_dir = os.path.join(user_root_dir, RESERVED_DIR_NAME)
    if not os.path.exists(Reserved_dir):
        os.makedirs(Reserved_dir)
    # 检查目标目录是否存在，如果不存在或者等于根目录，则返回Reserved目录
    if (not os.path.exists(target_dir)) or os.path.isdir(target_dir):
        target_dir = Reserved_dir
    return gr.FileExplorer(root_dir=target_dir)  
def refresh_chat_history_explorer_after_delete(file_path, session_state=None):
    """删除文件后刷新文件浏览器为当前登录用户的目录"""
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
    return gr.FileExplorer(root_dir=get_account_chat_history_dir(logged_in_name))

def refresh_html_file_explorer_after_upload_delete(file_path, session_state=None):
    """上传或删除HTML文件后刷新HTML文件浏览器为当前登录用户的HTML目录"""
    logged_in_name = DEFAULT_LOGGED_IN_NAME
    if session_state and isinstance(session_state, dict):
        ln = session_state.get("logged_in_name")
        if ln:
            logged_in_name = ln
    # 获取用户HTML目录
    html_dir = get_account_html_dir(logged_in_name)
    return gr.FileExplorer(root_dir=html_dir)

# 添加文件上传事件处理，用于图像预览
def update_image_preview(file_path):
    # 处理多文件上传情况
    file_paths = []
    if isinstance(file_path, list):
        # 如果是文件列表
        file_paths = file_path
    elif file_path:
        # 如果是单个文件
        file_paths = [file_path]
    
    # 收集所有图像文件
    image_files = []
    for path in file_paths:
        if path is not None and isinstance(path, str) and os.path.exists(path):
            if is_image_file(path):
                image_files.append(path)
    
    # 根据图像数量确定图像高度
    if len(image_files) == 1:
        img_height = 200
    elif len(image_files) == 2:
        img_height = 100
    elif len(image_files) == 3:
        img_height = 80
    else:
        img_height = 50
    
    # 如果有图像文件，创建一个包含所有图像的HTML显示
    if image_files:
        # 统一使用HTML组件显示图像，使用flex布局实现自动换行，先按列显示
        images_html = "<div style='display: flex; flex-direction: row; flex-wrap: wrap; gap: 5px; justify-content: flex-start; max-height: 200px; overflow-y: auto;'>"
        for img_path in image_files:
            # 图像先按列显示，不够宽度时自动换行
            images_html += f"<img src='/gradio_api/file={img_path}' style='max-height: {img_height}px; object-fit: contain;'>"
        images_html += "</div>"
        
        return gr.HTML(value=images_html, visible=True)
    else:
        # 如果没有找到图像文件，返回隐藏的HTML组件
        return gr.HTML(visible=False)

# 动态更新用户管理面板可见性
def update_user_mgmt_visibility(session_state):
    """根据用户权限动态更新用户管理面板的可见性"""
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return [gr.update(visible=False) for _ in range(17)]  # 隐藏所有组件
    
    is_admin = current_user == "root"
    
    # 生成模板文件路径（仅管理员）
    template_path = generate_user_import_template() if is_admin else None
    
    # 返回各个组件的可见性状态
    return [
        gr.update(visible=is_admin),  # 注册用户按钮
        gr.update(visible=is_admin),  # 更新信息按钮
        gr.update(visible=True),      # 修改密码按钮（所有用户可见）
        gr.update(visible=is_admin),  # 删除用户按钮
        gr.update(visible=is_admin),  # 查询用户按钮
        gr.update(visible=is_admin),  # 查看所有用户按钮
        gr.update(visible=is_admin),  # 批量删除用户名包含输入框
        gr.update(visible=is_admin),  # 批量删除按钮
        gr.update(visible=is_admin),  # 导入用户文件输入
        gr.update(visible=is_admin),  # 导入用户按钮
        gr.update(visible=is_admin, value=template_path),  # 导入模板文件输出
        gr.update(visible=True),      # 用户名输入框（所有用户可见，用于输入要修改密码的用户名）
        gr.update(visible=is_admin),  # 班级输入框
        gr.update(visible=is_admin),  # 姓名输入框
        gr.update(visible=is_admin),  # 性别选择框
        gr.update(visible=is_admin),  # 角色选择框（只有管理员可见）
        gr.update(visible=True)       # 密码输入框（所有用户可见，用于输入新密码）
    ]
# 用户注册管理功能
def handle_register_user(username, password, class_val, name, gender, role, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    
    # 转换性别为数字
    gender_num = 1 if gender == "男" else 0
    
    # 转换角色为数字
    role_num = 2  # 默认普通用户
    if role == "教师":
        role_num = 1
    elif role == "管理员":
        role_num = 0
    
    return register_user(username, password, class_val, name, gender_num, current_user, role_num)

def handle_update_user_info(username, class_val, name, gender, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    
    # 转换性别为数字
    gender_num = 1 if gender == "男" else 0
    return update_user_info(username, class_val, name, gender_num, current_user)

def handle_change_password(username, password, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    
    # 验证用户名不能为空
    if not username or username.strip() == "":
        return "用户名不能为空"
    
    # 验证密码不能为空
    if not password or password.strip() == "":
        return "密码不能为空"
    
    # 普通用户只能修改自己的密码，管理员可以修改任何用户的密码
    if current_user != "root":
        # 普通用户必须输入自己的用户名
        if username != current_user:
            return "权限不足：只能修改自己的密码"
    
    # 对于普通用户，直接修改密码，不需要验证旧密码
    # 对于管理员，直接修改密码，不需要旧密码验证
    old_password = ""
    return change_password(username, old_password, password, current_user)

def handle_delete_user(username, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    
    return delete_user(username, current_user)

def handle_get_user_info(username, session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    
    return get_user_info(username, current_user)

def handle_get_all_users(session_state):
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录"
    
    return get_all_users(current_user)

# 读取关于与帮助文档
def load_about_help_content():
    """读取关于与帮助的Markdown文档内容"""
    about_file_path = "about_help.md"
    try:
        if os.path.exists(about_file_path):
            with open(about_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        else:
            return "# 关于与帮助\n\n系统帮助文档正在建设中..."
    except Exception as e:
        return f"# 关于与帮助\n\n读取帮助文档时出错：{str(e)}"

# HTML资源页面相关函数
def _rewrite_html_links(html_content: str, base_url: str) -> str:
    """将 HTML 中的相对链接重写为 Gradio file 服务链接"""
    import re

    def _rewrite_attr(match):
        prefix = match.group(1)  # href=" 或 data-href="
        url = match.group(2)     # 链接值
        # 跳过绝对链接、锚点、javascript、data: 等
        if url.startswith(("http://", "https://", "#", "javascript:", "data:", "mailto:")):
            return match.group(0)
        # 跳过已经是 gradio_api/file= 开头的
        if url.startswith("/gradio_api/"):
            return match.group(0)
        # 重写相对路径
        new_url = base_url + url
        return f'{prefix}{new_url}"'

    # 重写 href="..." 和 data-href="..."
    content = re.sub(r'(href=")([^"]*)(")', _rewrite_attr, html_content)
    content = re.sub(r"(data-href=')([^']*)(')", _rewrite_attr, content)
    # 也处理双引号的 data-href
    content = re.sub(r'(data-href=")([^"]*)(")', _rewrite_attr, content)
    return content


def load_index_html():
    """加载 root/html/index.html 导航页内容，重写链接为 Gradio file 服务路径"""
    import urllib.parse
    index_path = os.path.join(ROOT_DIR, "html", "index.html")
    try:
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 重写所有相对链接，加上 gradio file 前缀
            base_url = "/gradio_api/file=" + urllib.parse.quote("root/html/")
            content = _rewrite_html_links(content, base_url)
            # 用 srcdoc 嵌入（此时链接已全部为绝对路径，不会404）
            return gr.update(value=f"""<div style="position:relative;width:100%;">
                <iframe srcdoc="{content.replace('"', '&quot;')}"
                    style="width:100%;height:100vh;border:none;"
                    onload="
                        var self = this;
                        function resize() {{
                            try {{
                                var h = self.contentWindow.document.documentElement.scrollHeight;
                                if (h > 100) {{ self.style.height = h + 'px'; }}
                            }} catch(e) {{}}
                        }}
                        resize();
                        setTimeout(resize, 500);
                        setTimeout(resize, 1500);
                        setTimeout(resize, 3000);
                    ">
                </iframe>
            </div>""")
        else:
            return gr.update(value="<p style='text-align:center;color:var(--text-secondary);'>导航页面未找到</p>")
    except Exception as e:
        return gr.update(value=f"<p style='text-align:center;color:red;'>加载导航页失败：{str(e)}</p>")

def update_html_resources(session_state):
    """更新HTML资源页面，显示用户的HTML文件列表"""


    # 获取HTML文件列表
    html_grid= get_htmlfilelst(session_state)

    html_content = f"""
    <div style="margin: 5px 0;">
        {html_grid}
    </div>
    """
    return gr.update(value=html_content)

# HTML文件上传和管理相关函数
def handle_html_file_upload(files, session_state):
    """处理HTML文件上传，只允许管理员和教师上传，限制5MB大小，多文件上传"""
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录", gr.update(), gr.update()
    
    # 检查权限
    if not can_manage_html_files(current_user):
        return "权限不足：只有管理员和教师可以上传HTML文件", gr.update(), gr.update()
    
    if not files:
        return "未选择文件", gr.update(), gr.update()
    
    # 获取用户html目录
    html_dir = os.path.join(current_user, "html")
    os.makedirs(html_dir, exist_ok=True)
    
    uploaded_files = []
    error_messages = []
    
    def process_single_file(src_path, dst_path, display_name):
        """处理单个文件"""
        nonlocal uploaded_files, error_messages
        
        # 检查文件大小（5MB限制）
        file_size = os.path.getsize(src_path)
        max_size_bytes = 5 * 1024 * 1024  # 5MB
        
        if file_size > max_size_bytes:
            error_msg = f"文件 '{display_name}' 大小超过5MB限制（{file_size/(1024*1024):.2f}MB）"
            error_messages.append(error_msg)
            return False
        
        # 检查文件扩展名
        _, ext = os.path.splitext(os.path.basename(src_path).lower())
        allowed_extensions = ['.html', '.htm', '.css', '.js', '.txt', '.md', '.json', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        
        if ext not in allowed_extensions:
            error_msg = f"文件 '{display_name}' 类型不支持（{ext}）"
            error_messages.append(error_msg)
            return False
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(os.path.basename(dst_path))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            new_file_name = f"{name}_{timestamp}{ext}"
            dst_path = os.path.join(os.path.dirname(dst_path), new_file_name)
            display_name = new_file_name
        
        try:
            # 复制文件
            shutil.copy2(src_path, dst_path)
            uploaded_files.append(display_name)
            return True
        except Exception as e:
            error_msg = f"文件 '{display_name}' 上传失败：{str(e)}"
            error_messages.append(error_msg)
            return False
    
    # 处理所有文件
    for file_item in files:
        # 获取文件路径
        if isinstance(file_item, str):
            file_path = file_item
        elif hasattr(file_item, 'name'):
            file_path = file_item.name
        elif isinstance(file_item, dict) and 'name' in file_item:
            file_path = file_item['name']
        else:
            file_path = str(file_item)
        
        if not os.path.exists(file_path):
            error_messages.append(f"文件不存在: {file_path}")
            continue
        
        # 只处理文件，不处理目录
        if os.path.isdir(file_path):
            error_messages.append(f"跳过目录: {os.path.basename(file_path)} (仅支持文件上传)")
            continue
        
        file_name = os.path.basename(file_path)
        target_path = os.path.join(html_dir, file_name)
        
        process_single_file(file_path, target_path, file_name)
    
    # 构建结果消息
    if uploaded_files:
        success_msg = f"成功上传 {len(uploaded_files)} 个文件"
        if len(uploaded_files) <= 10:
            success_msg += f"：{', '.join(uploaded_files)}"
        else:
            success_msg += f"，前10个文件：{', '.join(uploaded_files[:10])}..."
        
        if error_messages:
            success_msg += f"\n\n上传失败：\n" + "\n".join(error_messages[:10])  # 只显示前10个错误
            if len(error_messages) > 10:
                success_msg += f"\n...还有 {len(error_messages) - 10} 个错误"
        
        # 更新文件浏览器和HTML文件列表
        return success_msg, gr.FileExplorer(root_dir=html_dir), gr.update(value=get_htmlfilelst(session_state))
    else:
        error_msg = "文件上传失败：\n" + "\n".join(error_messages[:10]) if error_messages else "文件上传失败"
        if error_messages and len(error_messages) > 10:
            error_msg += f"\n...还有 {len(error_messages) - 10} 个错误"
        return error_msg, gr.update(), gr.update()

def handle_html_file_delete(file_path, session_state):
    """删除HTML文件，只允许管理员和教师删除"""
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return "请先登录", gr.update(), gr.update()
    
    # 检查权限
    if not can_manage_html_files(current_user):
        return "权限不足：只有管理员和教师可以删除HTML文件", gr.update(), gr.update()
    
    if not file_path:
        return "未选择文件", gr.update(), gr.update()
    
    # 获取用户html目录
    html_dir = os.path.join(current_user, "html")
    
    # 确保目标路径在用户html目录内（安全限制）
    try:
        # 处理可能的文件路径格式问题
        if isinstance(file_path, list):
            file_path = file_path[0] if file_path else ""
        
        target_path = os.path.abspath(str(file_path))
        html_dir_abs = os.path.abspath(html_dir)
        
        if not target_path.startswith(html_dir_abs):
            return "权限不足：只能删除自己HTML目录下的文件", gr.update(), gr.update()
    except Exception as e:
        return f"文件路径错误: {str(e)}", gr.update(), gr.update()
    
    if os.path.exists(target_path):
        try:
            if os.path.isfile(target_path):
                os.remove(target_path)
                msg = f"文件 {os.path.basename(target_path)} 已删除"
            elif os.path.isdir(target_path):
                shutil.rmtree(target_path)
                msg = f"目录 {os.path.basename(target_path)} 已删除"
            else:
                return "路径不是文件或目录", gr.update(), gr.update()
            
            # 更新文件浏览器和HTML文件列表
            return msg, gr.FileExplorer(root_dir=html_dir), gr.update(value=get_htmlfilelst(session_state))
        except Exception as e:
            return f"删除失败：{str(e)}", gr.update(), gr.update()
    else:
        return "文件不存在", gr.update(), gr.update()

def update_html_management_visibility(session_state):
    """根据用户权限更新HTML文件管理组件的可见性"""
    current_user = session_state.get("logged_in_name", "")
    if not current_user:
        return [gr.update(visible=False), gr.update(visible=False)]
    
    can_manage = can_manage_html_files(current_user)
    
    return [
        gr.update(visible=can_manage),  # 文件上传组件
        gr.update(visible=can_manage)   # 文件浏览器组件
    ]


##########################################
# Gradio应用主逻辑
##########################################

init_db()  # 初始化数据库

with gr.Blocks(title="教育智能体-高中信通版",theme="soft",css=css) as demo:  
    # 添加session state
    session_state = gr.State(value={"conversation_history": [], "session_id": None}) 
    # 添加计数器状态
    login_counter_state = gr.State(value={"count": 0, "timestamps": []})
    with gr.Row():
        htmlstr=f"""
            <p style='text-align: center;font-size: 18px;font-weight: bold;'>
            <img src='/gradio_api/file={ICON_PATH}' alt='UNET' style='width: 24px; height: 24px; display: inline; vertical-align: middle;'>
            教育智能体-高中信通版 </p>
            """
        gr.HTML(htmlstr)
    with gr.Tabs() as tabs:
        with gr.Tab("🔐 用户登录", id="login_tab") as login_tab:
            with gr.Row():
                login_username = gr.Textbox(label="用户名",placeholder="输入用户名或姓名")
                login_password = gr.Textbox(label="密码", type="password", placeholder="输入密码")
            with gr.Row():
                login_button = gr.Button("登录", variant="primary",icon="icon/login.png")
            with gr.Accordion("用户管理",open=False,visible=False) as user_mgmt:
                with gr.Row():
                    mgmt_username = gr.Textbox(label="用户名", placeholder="用户名")
                    mgmt_password = gr.Textbox(label="密码", type="password", placeholder="密码")
                    mgmt_class = gr.Number(label="班级", placeholder="班级")
                with gr.Row():
                    mgmt_name = gr.Textbox(label="姓名", placeholder="姓名")
                    mgmt_gender = gr.Radio(choices=["男", "女"], label="性别", value="男")
                    mgmt_role = gr.Radio(choices=["普通用户", "教师", "管理员"], label="角色", value="普通用户")
                with gr.Row():
                    register_button = gr.Button("注册用户", variant="primary",icon="icon/reg.png")
                    update_info_button = gr.Button("更新信息", variant="secondary",icon="icon/refresh.png")
                    change_pwd_button = gr.Button("修改密码", variant="secondary",icon="icon/edit.png")
                    delete_user_button = gr.Button("删除用户", variant="stop",icon="icon/delete.png")
                    search_button = gr.Button("查询用户", variant="secondary",icon="icon/search.png")
                    list_users_button = gr.Button("查看所有用户", variant="secondary",icon="icon/view.png")
                with gr.Row():
                    template_file = gr.File(label="导入模板文件", interactive=False)
                    bulk_delete_pattern = gr.Textbox(label="批量删除用户名包含", placeholder="例如：s11")
                with gr.Row():
                    import_user_file = gr.File(label="导入用户文件", file_types=[".csv"], file_count="single", type="filepath")
                    import_users_button = gr.Button("导入用户", variant="primary", icon="icon/upload.png")
                    bulk_delete_button = gr.Button("批量删除用户", variant="stop", icon="icon/delete.png")
                    
            with gr.Row():
                login_msg = gr.Markdown()
            with gr.Row():
                lblmsg.render()
        with gr.Tab("🤖 教育智能体", id="main_tab", visible=False) as main_tab:
            with gr.Row():
                with gr.Column(scale=2):
                    query_input.render()
                    examples1 = gr.Examples(
                    examples=generate_random_examples(), inputs=[query_input],label="示例")
                    examples2=gr.Examples(
                        examples=get_random_files(os.path.join(ROOT_DIR, "imgs")), inputs=[file_input],label="示例"
                    )            
                    
                    
                with gr.Column(scale=1):
                    image_preview.render()
                    file_input.render()
                    include_file_context = gr.Checkbox(label="上下文增强", value=False)
                    
            with gr.Row():
                query_button = gr.Button("发送消息",icon="icon/submit.png",variant="primary",scale=2)   
                new_topic_button = gr.Button("新话题", icon="icon/newtopic.png", variant="stop",scale=1) 
                stop_button = gr.Button("停止", icon="icon/stop.png", variant="stop",scale=1)
                refresh_button = gr.Button("换一换", icon="icon/refresh.png", variant="stop",scale=1)
                preview_html_button = gr.Button("预览", icon="icon/preview.png", variant="secondary", scale=1,visible=False)
                
            with gr.Row():                 
                query_output = gr.Markdown(label="对话历史",elem_classes=["chat-history"])
                #query_output = gr.Markdown(label="对话历史")
            with gr.Row():
                copy_button = gr.Button("复制内容", icon="icon/copy.png",visible=False, elem_classes=["copy-button"])
            with gr.Row():
                html_output=gr.HTML(label="HTML预览",visible=False)
                
            # 添加历史记录部分（默认隐藏，登录后显示对应用户目录）
            with gr.Sidebar("历史对话记录管理",open=False,visible=True) as history_sidebar:
                with gr.Row():
                        # 使用占位目录，避免在启动时绑定管理员目录
                        placeholder_dir = get_history_placeholder_dir()
                        os.makedirs(placeholder_dir, exist_ok=True)
                        history_file_explorer = gr.FileExplorer(
                            label="历史对话", 
                            root_dir=placeholder_dir,
                            file_count="single",
                            #glob="**/*.md",
                            #glob="**/*.*",
                            interactive=True,
                            #height=300
                        )
                with gr.Row():
                    delete_file_button = gr.Button("删除选择", icon="icon/delete.png", variant="stop")
        with gr.Tab("📚 教学资源", id="html_resources_tab", visible=False) as html_resources_tab:
            with gr.Row():
                Html_Nav=gr.HTML(value="")
            with gr.Accordion("HTML资源",open=False,) as html_tab:
                with gr.Row():
                    html_files_grid = gr.HTML(label="HTML文件列表", value="<p style='text-align: center;'>请先登录以查看您的HTML资源</p>")
            with gr.Accordion("资源管理",open=False,) as html_resources_mgmt:
                with gr.Row():
                # 管理员和教师可以上传HTML资源文件
                    file_upload = gr.File(label="上传资源文件", file_count="multiple", interactive=True)
                    # 使用占位目录，避免在启动时绑定管理员目录
                    placeholder_dir = get_history_placeholder_dir()
                    os.makedirs(placeholder_dir, exist_ok=True)
                    file_explorer = gr.FileExplorer(
                        label="HTML文件管理", 
                        root_dir=placeholder_dir,
                        file_count="single",
                        #glob="**/*.md",
                        #glob="**/*.*",
                        interactive=True,
                        #height=300
                    )
                with gr.Row():
                    upload_button = gr.Button("上传资源", variant="primary", icon="icon/upload.png")
                    delete_button = gr.Button("删除选中", variant="stop", icon="icon/delete.png")
                with gr.Row():
                    html_upload_msg = gr.Markdown()
        with gr.Tab("🏆 课堂积分", id="score_tab", visible=False) as score_tab:
            score_html = gr.HTML()
        with gr.Tab("ℹ️ 关于与帮助", id="about_help_tab") as about_help_tab:
            about_md=gr.Markdown(value=load_about_help_content())
        
    # 登录和注册功能
    login_button.click(
        fn=login,
        inputs=[login_username, login_password, session_state],
        outputs=[login_msg, lblmsg, main_tab, tabs, session_state, history_file_explorer, history_sidebar, user_mgmt, html_resources_tab, html_files_grid, file_explorer, score_tab]
    ).then(
        fn=update_user_mgmt_visibility,
        inputs=[session_state],
        outputs=[register_button, update_info_button, change_pwd_button, delete_user_button, search_button, list_users_button, bulk_delete_pattern, bulk_delete_button, import_user_file, import_users_button, template_file, mgmt_username, mgmt_class, mgmt_name, mgmt_gender, mgmt_role, mgmt_password]
    ).then(
        fn=lambda: ("", ""),  # 清空用户名和密码输入框
        inputs=None,
        outputs=[login_username, login_password]
    )
    
    login_password.submit(
        fn=login,
        inputs=[login_username, login_password, session_state],
        outputs=[login_msg, lblmsg, main_tab, tabs, session_state, history_file_explorer, history_sidebar, user_mgmt, html_resources_tab, html_files_grid, file_explorer, score_tab]
    ).then(
        fn=update_user_mgmt_visibility,
        inputs=[session_state],
        outputs=[register_button, update_info_button, change_pwd_button, delete_user_button, search_button, list_users_button, bulk_delete_pattern, bulk_delete_button, import_user_file, import_users_button, template_file, mgmt_username, mgmt_class, mgmt_name, mgmt_gender, mgmt_role, mgmt_password]
    ).then(
        fn=lambda: ("", ""),  # 清空用户名和密码输入框
        inputs=None,
        outputs=[login_username, login_password]
    )
    
    
    
    # 注册用户按钮事件
    register_button.click(
        fn=handle_register_user,
        inputs=[mgmt_username, mgmt_password, mgmt_class, mgmt_name, mgmt_gender, mgmt_role, session_state],
        outputs=[login_msg]
    )
    
    # 更新用户信息按钮事件
    update_info_button.click(
        fn=handle_update_user_info,
        inputs=[mgmt_username, mgmt_class, mgmt_name, mgmt_gender, session_state],
        outputs=[login_msg]
    )
    
    # 修改密码按钮事件
    change_pwd_button.click(
        fn=handle_change_password,
        inputs=[mgmt_username, mgmt_password, session_state],  # 使用 mgmt_password 而不是 new_password
        outputs=[login_msg]
    )
    
    # 删除用户按钮事件
    delete_user_button.click(
        fn=handle_delete_user,
        inputs=[mgmt_username, session_state],
        outputs=[login_msg]
    )
    
    # 查询用户按钮事件
    search_button.click(
        fn=handle_get_user_info,
        inputs=[mgmt_username, session_state],
        outputs=[login_msg]
    )
    
    # 查看所有用户按钮事件
    list_users_button.click(
        fn=handle_get_all_users,
        inputs=[session_state],
        outputs=[login_msg]
    )

    # 批量删除用户按钮事件
    bulk_delete_button.click(
        fn=handle_bulk_delete_users,
        inputs=[bulk_delete_pattern, session_state],
        outputs=[login_msg]
    )

    # 导入用户按钮事件
    import_users_button.click(
        fn=handle_import_users,
        inputs=[import_user_file, session_state],
        outputs=[login_msg]
    )
        
    query_event =query_button.click(
        fn=chat_with_history,
        inputs=[file_input, query_input, session_state, include_file_context], 
        outputs=[query_output, session_state, html_output] 
    ).then(
        fn=refresh_file_explorer_after_chat,
        inputs=[session_state, history_file_explorer],
        outputs=[history_file_explorer]
     ).then(
        fn=refresh_chat_history_explorer_after_delete,
        inputs=[history_file_explorer, session_state],
        outputs=[history_file_explorer]
    ).then(
        fn=lambda output: gr.update(visible=bool(output.strip())) if output else gr.update(visible=False),
        inputs=[query_output],
        outputs=[copy_button]
    )

    submit_event=query_input.submit(
        fn=chat_with_history,
        inputs=[file_input, query_input, session_state, include_file_context], 
        outputs=[query_output, session_state, html_output]
    ).then(
        fn=refresh_file_explorer_after_chat,
        inputs=[session_state, history_file_explorer],
        outputs=[history_file_explorer]
     ).then(
        fn=refresh_chat_history_explorer_after_delete,
        inputs=[history_file_explorer, session_state],
        outputs=[history_file_explorer]
    ).then(
        fn=lambda output: gr.update(visible=bool(output.strip())) if output else gr.update(visible=False),
        inputs=[query_output],
        outputs=[copy_button]
    )
    
    preview_html_button.click(
        fn=preview_html_code_from_output,
        inputs=[query_output],
        outputs=[html_output]
    )
    
    refresh_button.click(fn=update_examples, inputs=None, outputs=[examples1.dataset,examples2.dataset])
    
    stop_button.click(
        fn=None,
        inputs=None, 
        outputs=None,
        cancels=[query_event, submit_event]
    )
    
    new_topic_button.click(
        fn=lambda current_state: ("", "","" ,None,gr.update(visible=False),gr.update(visible=False),{"conversation_history": [], "session_id": None, "logged_in_name": current_state.get("logged_in_name"), "class": current_state.get("class"), "name": current_state.get("name"), "gender": current_state.get("gender")}), 
        inputs=[session_state], 
        outputs=[query_input, query_output, html_output,file_input,preview_html_button,copy_button,session_state]
    )

    
    history_file_explorer.change(
        fn=load_chat_history_with_path_from_explorer,
        inputs=[history_file_explorer, session_state],
        outputs=[query_output, file_input, preview_html_button, html_output, history_file_explorer]
    ).then(
        fn=lambda output: gr.update(visible=bool(output.strip())) if output else gr.update(visible=False),
        inputs=[query_output],
        outputs=[copy_button]
    )
    
    # 删除按钮事件处理
    delete_file_button.click(
        fn=delete_chat_history_file_from_explorer,
        inputs=[history_file_explorer, session_state],
        outputs=[history_file_explorer, query_output, file_input]
     ).then(
        fn=refresh_chat_history_explorer_after_delete,
        inputs=[history_file_explorer, session_state],
        outputs=[history_file_explorer]
    )
    # 添加复制按钮的点击事件处理
    copy_button.click(
        fn=None,
        inputs=[query_output],
        outputs=[],
        js="""
        (output) => {
            try {
                const prose = document.querySelector('.chat-history .prose') || document.querySelector('.chat-history');
                const text = (prose && (prose.innerText || prose.textContent)) || output || '';
                if (prose) {
                    const range = document.createRange();
                    range.selectNodeContents(prose);
                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(range);
                    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
                    sel.removeAllRanges();
                    return [];
                }
                // 回退到纯文本 textarea
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-99999px';
                ta.style.top = '0';
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand('copy'); } catch (e) { /* ignore */ }
                document.body.removeChild(ta);
            } catch (e) {
                console.error('copy-button error', e);
            }
            return [];
        }
        """
    )
    
    file_input.change(update_image_preview, inputs=[file_input], outputs=[image_preview])
    # 添加用户离开时减少在线人数的函数
    def decrease_active_users():
        """当用户离开页面时减少在线人数"""
        global active_users
        if active_users > 0:
            active_users -= 1

    # 添加随机示例刷新功能
    demo.load(fn=update_examples, inputs=None, outputs=[examples1.dataset,examples2.dataset])
    
    # 加载关于与帮助内容
    # demo.load(fn=load_about_help_content, inputs=None, outputs=[about_md])
    

    login_tab.select(
        fn=update_online_users_display,
        inputs=None,
        outputs=[lblmsg]
    )
    
    # 页面加载时显示初始在线人数
    demo.load(
        fn=update_online_users_display,
        inputs=None,
        outputs=[lblmsg]
    )
    
    # 添加页面卸载事件处理
    demo.unload(fn=decrease_active_users)
    
   
    # 当用户切换到HTML资源标签页时，自动刷新内容
    html_resources_tab.select(
        fn=load_index_html,
        inputs=None,
        outputs=[Html_Nav]
    ).then(
        fn=update_html_resources,
        inputs=[session_state],
        outputs=[html_files_grid]
    ).then(
        fn=update_html_management_visibility,
        inputs=[session_state],
        outputs=[file_upload, file_explorer]
    )
    
    # 课堂积分标签页 - 加载HTML内容
    def load_score_html():
        import urllib.parse
        html_path = "root/html/score_system/index.html"
        src = f"/gradio_api/file={urllib.parse.quote(html_path)}"
        return gr.update(value=f"""<iframe src="{src}"
            style="width:100%;height:100vh;border:none;min-height:700px;">
        </iframe>""")
    
    score_tab.select(
        fn=load_score_html,
        inputs=None,
        outputs=[score_html]
    )
    
    # 页面加载时也预加载积分页面
    demo.load(
        fn=load_score_html,
        inputs=None,
        outputs=[score_html]
    )
    
    # 页面加载时预加载教学资源导航
    demo.load(
        fn=load_index_html,
        inputs=None,
        outputs=[Html_Nav]
    )
    
    # HTML文件上传按钮事件文件管理器刷新BUG，先刷一个临时目录，再刷正式目录
    upload_button.click(
        fn=handle_html_file_upload,
        inputs=[file_upload, session_state],
        outputs=[html_upload_msg, file_explorer, html_files_grid]
    ).then(
        fn=lambda state: gr.FileExplorer(root_dir=os.path.join(state.get("logged_in_name", DEFAULT_LOGGED_IN_NAME), os.path.join("html", RESERVED_DIR_NAME))),
        inputs=[session_state],
        outputs=[file_explorer]
     ).then(
        fn=lambda state: gr.FileExplorer(root_dir=os.path.join(state.get("logged_in_name", DEFAULT_LOGGED_IN_NAME), "html")),
        inputs=[session_state],
        outputs=[file_explorer]
    )
    
    # HTML文件删除按钮事件
    delete_button.click(
        fn=handle_html_file_delete,
        inputs=[file_explorer, session_state],
        outputs=[html_upload_msg, file_explorer, html_files_grid]
     ).then(
        fn=lambda state: gr.FileExplorer(root_dir=os.path.join(state.get("logged_in_name", DEFAULT_LOGGED_IN_NAME), os.path.join("html", RESERVED_DIR_NAME))),
        inputs=[session_state],
        outputs=[file_explorer]
    ).then(
        fn=lambda state: gr.FileExplorer(root_dir=os.path.join(state.get("logged_in_name", DEFAULT_LOGGED_IN_NAME), "html")),
        inputs=[session_state],
        outputs=[file_explorer]
    )
    
    
    with gr.Row():
            linkurl=gr.Markdown()
    with gr.Row():
            gr.Markdown("""
                        <p style='text-align: center;'>
                        Copyright © 2025 By [UNET] All rights reserved.
                        </p>""")
    # demo.load(fn=get_host,inputs=None,outputs=linkurl)
    demo.queue(default_concurrency_limit=8,max_size=20)
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
        show_api=False,
        share=False,
        prevent_thread_lock=True,
        allowed_paths=['./'],
        favicon_path=FAVICON_PATH,
    )
    # 必须先 launch 才能拿到真正的 app 对象（queue 和 launch 创建的不是同一个）
    mount_score_api(demo.app)
    mount_rollcall_api(demo.app)
    # 阻塞主线程保持服务运行
    demo.block_thread()
