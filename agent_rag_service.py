from shared_utils import (
    cv2,FunctionAgent,asyncio,OpenAI,time,os,
    QWEN_OPENAI_API_BASE,
    base64,requests,json,re,sys,
    AgentWorkflow,Context,AgentStream,
    io,Settings,OllamaEmbedding,chromadb,ChromaVectorStore,
    StorageContext,VectorStoreIndex,
    JsonSerializer,read_file,dashscope,
    HTTPStatus, ImageSynthesis,VideoSynthesis,np,wave,getapi_key,getnvr_url,
    default_voicesid,default_voices,ChatMessage,
    VectorIndexRetriever,VectorStoreQueryMode,ContextChatEngine,ChatMemoryBuffer,BaseRetriever
)
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback
from typing import Dict, Any, AsyncGenerator, Optional
import threading
import asyncio
from queue import Queue, Empty


# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class AgentRagService:
    def __init__(self, model_name: str, embedding_model_name: str, logged_in_name: str, nvr1_url: str = "", nvr2_url: str = "", size: str = "1024*768", isplus: str = "False", voice: str = "严肃男"):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.logged_in_name = logged_in_name
        self.nvr1_url = nvr1_url
        self.nvr2_url = nvr2_url
        self.size = size
        self.isplus = isplus
        self.voice = voice
        self.memory = None

        # 获取用户的API KEY
        self.dashscope_api_key, self.deepseek_api_key = getapi_key(logged_in_name)
        dashscope.api_key = self.dashscope_api_key

        # 设置语音ID
        voiceidx = default_voices.index(voice) if voice in default_voices else 0
        self.voiceid = default_voicesid[voiceidx]

        # 设置LLM
        self.llm = OpenAI(
            model=model_name,
            api_key=self.dashscope_api_key,
            api_base=QWEN_OPENAI_API_BASE,
            extra_body={"enable_search": True}
        )

        if self.memory is None:
            #self.memory = Memory(token_limit=8192)
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=8192) #旧版本兼容
        # 初始化工作流
        self.iva_workflow = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[self.query_knowledge_base, self.get_camera_image, self.vision_query_image,
                                self.get_camera_video, self.vision_query_video,
                                self.get_current_datetime, self.generate_image_show,
                                self.generate_audio_show, self.generate_video_show, self.set_name,
                                self.generate_lecture_video_by_topic, self.generate_lecture_script,self.web_search
                                ],    
            llm=self.llm,
            initial_state={"name":"IVAgent"},
            system_prompt="""你是一个由伦教中学刘玉军老师设计开发的教育智能体，
            专为高中信息技术与通用技术教学服务, 具备查询本地知识库、生成教学资源、进行学习诊断与评估、创建教学内容等多种功能：

            基本原则：
            - 在执行任何教学相关任务前，都必须优先使用 query_knowledge_base() 函数查询本地知识库获取准确信息
            - 如果本地知识库中没有找到相关信息或查询结果为空，必须使用 web_search() 函数进行联网搜索获取最新、最准确的信息
            - 对于时间敏感、事实查询、实时数据等需要最新信息的问题，必须主动调用 web_search() 函数

            联网搜索自动触发条件：
            - 本地知识库查询结果为空或相关度较低时
            - 用户询问时间敏感问题：如"最新"、"现在"、"今天"、"当前"、"实时"、"近期"、"最近"等
            - 需要最新信息的新闻事件：如"新闻"、"事件"、"报道"、"消息"、"动态"、"疫情"、"股市"、"天气"、"黄金价格"、"汇率"等
            - 事实查询：如"是什么"、"什么是"、"定义"、"解释"、"介绍"、"概念"、"who"、"what"、"when"等
            - 数据查询：如"数据"、"统计"、"排名"、"价格"、"汇率"、"股价"、"数字"、"百分比"等
            - 人物信息：如"人物"、"个人资料"、"简历"、"传记"、"简介"、"profile"等
            - 地点信息：如"位置"、"在哪里"、"地址"、"景点"、"旅游"、"城市"、"国家"等
            - 专业领域：如"科技"、"科学"、"研究"、"发现"、"论文"、"学术"、"专家"、"学者"等
            - 开放性问题：如"如何"、"怎样"、"为什么"、"为何"、"how"、"why"等，特别是涉及具体实体时

            具体功能：
            1. 使用 query_knowledge_base() 函数查询本地知识库，获取与用户请求相关的知识内容
            2. 使用 get_camera_image() 函数获取摄像头图像，并返回图像的 image_file_path
            3. 使用 vision_query_image() 函数描述图像，接收 get_camera_image() 返回的image_file_path 作为参数
            4. 使用 get_camera_video() 函数获取摄像头视频，并返回视频的 video_file_path
            5. 使用 vision_query_video() 函数描述视频的具体过程，接收 get_camera_video() 函数返回的video_file_path作为参数
            6. 使用 get_current_datetime() 函数获取当前日期和时间，并返回一个包含日期和时间的字符串
            7. 使用 set_name() 函数设置智能体名称，用于自我介绍
            8. 使用 web_search() 函数进行联网搜索，获取最新、最准确的外部信息。这是必须掌握的关键技能，当本地知识库无法提供实时数据时，必须使用此函数。

            教学内容生成类功能（执行前必须先查询本地知识库）：
            9. 使用 generate_image_show() 函数文生图片，如果用户要求生成图片，则根据用户要求的文字描述，结合本地知识库信息生成图片
            10. 使用 generate_audio_show() 函数语音合成生成音频，如果用户要求生成音频，则根据用户要求的文字内容，结合本地知识库信息生成音频
            11. 使用 generate_video_show() 函数文生视频，如果用户要求生成视频，则根据用户要求的文字描述，结合本地知识库信息生成视频
            12. 使用 generate_lecture_video_by_topic() 函数生成讲解视频，当用户明确要求生成某个主题的讲解视频时使用此函数，必须先查询本地知识库获取相关内容
            13. 使用 generate_lecture_script() 函数生成讲解稿文本，当用户明确要求生成某个主题的讲解稿时使用此函数，必须先查询本地知识库获取相关内容
            14. 教学动画生成（纯前端HTML5+JS）
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取相关的教学内容
                - 不调用任何外部API，生成可直接运行的HTML文件
                - 生成完整、自包含的 HTML + SVG + CSS + JavaScript 动画代码
                - 特点：
                    - 支持参数调节（如速度、颜色、节点数）
                    - 界面简洁、重点突出、响应流畅
                    - 适配移动端与桌面端
                    - 提供预览说明：  
                        > *此动画支持在支持 HTML 渲染的环境中直接交互预览。如果未显示动画，请将下方完整代码保存为 .html 文件后用浏览器打开*
            15. 教学互动游戏生成（纯前端HTML5+JS）
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取相关的教学内容
                - 不调用任何外部API，生成可直接运行的HTML文件
                - 生成完整、自包含的 HTML + SVG + CSS + JavaScript 互动游戏代码
                - 丰富多样的游戏类型：随机生成不同形式的互动游戏
                - 特点：
                    - 知识点深度融合：游戏内容完全围绕指定知识点设计，题目智能生成
                    - 多样化互动形式：
                        - 连连看：匹配相关概念、公式、图片、术语等
                        - 消消乐：消除相同知识点、正确答案组合或配对项
                        - 知识闯关：分层级递进式答题挑战，逐步解锁
                        - 拖拽匹配：概念与解释、问题与答案、图片与名称拖拽配对
                        - 选择题：单选、多选、判断题，支持图片选择题
                        - 拼图游戏：将知识点碎片拼成完整概念或图表
                        - 记忆翻牌：翻开卡片配对知识点，锻炼记忆能力
                        - 知识接龙：按逻辑顺序排列知识点或事件
                        - 分类游戏：将知识点拖拽到正确的分类框中
                        - 填空补全：拖拽正确答案填入空白处
                        - 时间轴：按时间顺序排列历史事件或发展过程
                        - 地图标注：在地图上标注地理事物、历史地点等
                        - 公式推导：拖拽步骤完成公式推导过程
                        - 概念树：构建知识结构图，理解概念层级关系
                        - 答题转盘：转盘选择答案的趣味答题
                        - 知识迷宫：通过回答问题找到正确路径走出迷宫
                        - 抢答模式：限时抢答，增加紧张感和趣味性
                        - 角色扮演：模拟真实场景应用知识点
                        - 解密游戏：通过知识点解答逐步解开谜题
                        - 知识竞赛：多人对战模式，PK答题
                    - 智能反馈系统：答错时提供详细知识点解析和正确答案说明
                    - 成绩统计：实时显示得分、正确率、用时、等级等数据
                    - 参数调节：支持难度等级、题目数量、游戏速度、时间限制等参数设置
                    - 界面炫酷：现代化UI设计，丰富的动画效果和音效反馈
                    - 响应式适配：完美支持移动端与桌面端
                    - 操作简单：直观的用户界面，易于上手操作，必须有重新开始按钮
                    - 随机生成：每次可随机选择不同游戏形式，保持新鲜感。重新开始时，则清除原来答题记录，并重新随机更改题目和题干顺序
                    - 提供预览说明：
                        > *此互动游戏支持在支持 HTML 渲染的环境中直接交互体验。如果未显示游戏，请将下方完整代码保存为 .html 文件后用浏览器打开*

            通用任务类功能：
            16. 你能根据用户的指令要求，选择性地使用这些函数完成任务

            教学评估类功能：
            17. 学习诊断与反馈：当用户请求对某个知识点进行学习诊断时，你需要完成以下流程：
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取关于该知识点的详细内容
                - 基于知识库返回的内容设计诊断题目
                - 如果知道用户信息则显示用户基本信息，否则询问：学号、班级、姓名（用于个性化跟踪）
                - 依次提出三题（布鲁姆认知层级）：
                    * 识记（记忆定义/术语）
                    * 理解（解释/转述）
                    * 应用（新情境中解决问题）
                - 用户每答一题，再出下一题
                - 基于回答提供个性化反馈
                - 提供综合反馈结构：
                    * 【掌握水平】
                    ✅ 掌握（三题基本正确）
                    ⚠️ 需加强（部分正确，存在偏差）
                    ❌ 未掌握（关键概念混淆或无法作答）
                    * 【关键问题】
                    1句话精准定位认知障碍（例："混淆了'速度'与'加速度'的物理含义"）
                    * 【建议行动】
                    1–2条可操作建议（优先）：
                    - 概念澄清（"重读教材第X节"）
                    - 即时练习（"完成3道基础应用题"）
                    - 现实联结（"观察家中电器，用欧姆定律解释"）
                    - 推荐资源（"观看5分钟动画《XX原理可视化》"）
                触发词示例："请对'______'进行学习诊断。"、"学生刚学完'______'，请出3题并反馈。"、"评估学生对'______'的掌握情况。"

            18. 在线练习考试：当用户请求为某个知识点出练习题时，你需要完成以下流程：
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取关于该知识点的详细内容
                - 基于知识库返回的内容自动生成题目
                - 如果知道用户信息则显示用户基本信息，否则询问：学号、班级、姓名（用于个性化跟踪）
                - 自动生成10道单选题（4基础 + 4中等 + 2提高），每题10分，总分100
                - 一次性展示全部题目
                - 用户连续输入答案（如：A B C D A C B D A B）
                - 收集用户全部答案后，自动进行评分
                - 提供反馈结构：
                    * 【考试成绩】
                    🌟 优秀（90–100）｜🎯 良好（70–89）｜📚 需努力（60–69）｜🔧 未通过（<60）
                    * 【详细分析】
                    - 错题编号 + 正确答案
                    - 错误解析（推理过程）
                    - 知识要点（核心概念/公式）
                    - 避坑指南（常见思维误区）
                    * 【改进建议】
                    2–3条具体建议（如：复习第X节、做3–5道相似题、制作思维导图）
                触发词示例："请为'______'出10道练习题。"、"我想练习'______'，给我10道选择题。"、"关于'______'的在线测试。"、"出10道'______'的单选题让我练习。"

            19. 深度图文讲解：当用户请求对某个知识点进行深度讲解时，你需要按以下结构输出：
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取关于该知识点的详细内容
                - 按结构输出：**导入 → 概念解析 → 实例分析 → 总结归纳**
                - 语言通俗，符合中学生认知，避免学术堆砌
                - 支持：公式、代码块、表格、流程图、结构化列表
                - 优先使用文本/结构化形式：
                    * Markdown 层级列表（思维导图）
                    * Mermaid 语法（流程图、概念图、时序图）
                    * 格式化表格、代码块、箭头符号
                - 仅在必要时生成图像（如实验装置示意图、技术产品设计草图、复杂函数图像等）
                - 生成的图像要求：中文标注，无水印，无版权风险，教材风格：简洁、专业、去装饰化

            20. 教案自动生成：当用户请求生成教案时，你需要完成以下流程：
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取相关的教学内容
                - 自动生成完整教案，包含：
                    * 教学目标（核心素养导向）
                    * 教学重难点
                    * 教学方法（讲授/探究/PBL等）
                    * 教学流程（导入、新授、活动、巩固、小结、作业）
                    * 学生活动设计
                    * 板书设计
                    * 教学评价与反思建议
                - 支持：1课时 / 2课时 / 单元整体设计

            21. 试题命制与评估：当用户请求生成试题时，你需要完成以下流程：
                - 首先使用 query_knowledge_base() 函数查询本地知识库获取相关的知识点内容
                - 基于知识库内容生成指定题型：单选、多选、填空、判断、简答、综合应用题
                - 支持难度分级：基础 / 提升 / 拓展
                - 提供参考答案与详细解析
                - 适用于随堂测验、单元检测、复习练习

            重要约束：
            - 你只能使用 query_knowledge_base()、get_camera_image() 、vision_query_image()、get_camera_video()、vision_query_video()、generate_image_show() 、generate_audio_show()、generate_video_show()、generate_lecture_video_by_topic() 、generate_lecture_script() get_current_datetime()、set_name()、web_search() 函数，不要使用其他函数
            - 在对描述后的图像内容进行小结建议时，不要重复输出图像的描述内容
            - 执行任何教学相关的任务前，都必须先查询本地知识库以获取准确信息
            - 当本地知识库查询结果不足或缺失时，必须使用web_search()函数获取最新信息
            - 对于实时数据查询（如价格、汇率、天气等），必须使用web_search()函数
            - 如果输出的是HTML代码码，请使用HMTL围栏标记进行输出源码。 在HTML代码输出之前：print("```html\n", end="", flush=True)   在HTML代码结束时：print("\n```", end="", flush=True)
            """
        )

    #获取摄像头的图像，并保存到cap目录中，并返回图像文件路径image_file_path。
    def get_camera_image(self, prompt: str):
        """
        功能：获取摄像头的图像，并保存到cap目录中，并返回图像文件路径image_file_path。
        参数：prompt：提示文本内容。
        返回：图像文件路径image_file_path
        说明：摄像头的图像保存为jpg格式
        """
        #判断是否有cap目录，如果没有，则创建
        if not os.path.exists(os.path.join(self.logged_in_name,"cap")):
            os.makedirs(os.path.join(self.logged_in_name,"cap"))
        camidstr=read_file(os.path.join("nvr","nvr.txt"))
        #分析prompt文本内容，是否包括有camidstr列表中的元素，如果包含，则返回元素索引camid
        camid = None
        for idx, cam in enumerate(camidstr): # type: ignore
            if re.search(rf'\b{re.escape(cam)}\b', prompt):
                camid = idx + 1
                break
        
        if camid:
            if camid<12:
            #camid = random.randint(1, 13)
                camid = str(camid).zfill(2)
                url = f'rtsp://{self.nvr1_url}/Streaming/Channels/{camid}01?transportmode=multicas'
            else:
                camid=camid-12
                camid = str(camid).zfill(2)
                #课室内22和外23
                url = f'rtsp://{self.nvr2_url}/Streaming/Channels/{camid}01?transportmode=multicas'
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            if not ret:
                print("RTSP 摄像头无法访问，使用本地摄像头...")
                cap.release()
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                print("本地摄像头无法打开...")
                cap.release()
                return None
            
        image_file_path = ""
        i=0
        while cap.isOpened():
            ret, frame = cap.read()
            #当i==10，则跳出循环,并保存图像
            i=i+1
            if i==15:  
                if ret:   
                    [h,w,c]= frame.shape #获取图片大小
                    if w>1920:
                        frame=cv2.resize(frame, (w//2, h//2))#缩小图像
                    current_time = time.strftime('%Y%m%d%H%M%S')
                    file_name = f'{current_time}.jpg'
                    image_file_path = os.path.join(self.logged_in_name,'cap', file_name)
                    cv2.imwrite(image_file_path, frame)
                    #图像居中显示
                    htmlstr=f"<p style='text-align: center;'> <img src='/gradio_api/file={self.logged_in_name}/cap/{file_name}'  style='display: inline; vertical-align: middle;'></p>"
                    print(htmlstr)
                    #print(file_path)  
                    break            
                else:
                    print("无法读取摄像头图像。")
                    return None
        cap.release()    
        return image_file_path
        

    #获取摄像头的视频，并保存到cap目录中，并返回视频文件路径video_file_path。
    def get_camera_video(self, prompt: str):
        """
        功能：获取摄像头的视频，并保存到cap目录中，并返回视频文件路径video_file_path。
        参数：prompt：提示文本内容。
        返回：视频文件路径video_file_path
        说明：视频文件保存为mp4格式，视频时长为10秒。
        """
        #判断是否有cap目录，如果没有，则创建
        if not os.path.exists(os.path.join(self.logged_in_name,"cap")):
            os.makedirs(os.path.join(self.logged_in_name,"cap"))
            
        camidstr=read_file(os.path.join("nvr","nvr.txt"))
        #分析prompt文本内容，是否包括有camidstr列表中的元素，如果包含，则返回元素索引camid
        camid = None
        for idx, cam in enumerate(camidstr): # type: ignore
            if re.search(rf'\b{re.escape(cam)}\b', prompt):
                camid = idx + 1
                break
        if camid:
            if camid < 12:
                camid = str(camid).zfill(2)
                url = f'rtsp://{self.nvr1_url}/Streaming/Channels/{camid}01?transportmode=multicas'
            else:
                camid = camid - 12
                camid = str(camid).zfill(2)
                url = f'rtsp://{self.nvr2_url}/Streaming/Channels/{camid}01?transportmode=multicas'
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            if not ret:
                print("RTSP 摄像头无法访问，使用本地摄像头...")
                cap.release()
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                print("本地摄像头无法打开...")
                cap.release()
                return None
        
        #print("get_camera_video:",prompt)
        
        start_time = time.time()
        frame_count = 0

        # 获取视频的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 缩小视频比例一半
        new_width = width // 2
        new_height = height // 2

        # 创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # type: ignore #H264编码，
        file_name = f'video_{time.strftime("%Y%m%d%H%M%S")}.mp4'
        video_file_path = os.path.join(self.logged_in_name,'cap', file_name)
        out = cv2.VideoWriter(video_file_path, fourcc, fps, (new_width, new_height))

        while cap.isOpened() : 
            ret, frame = cap.read() 
            if not ret:
                print("无法读取摄像头图像。")
                break
            
            # 缩小帧的尺寸
            new_frame = cv2.resize(frame, (new_width, new_height))
            # 写入视频帧,跳过前10帧
            frame_count =frame_count+1
            if frame_count>10 :
                out.write(new_frame)
            
            end_time = time.time()
            if end_time - start_time >= 10:  # 如果已经超过10秒，则跳出循环
                break

        cap.release() # type: ignore  
        out.release()  # 释放VideoWriter对象
        #视频居中显示
        htmlstr= f""" <div style='display: flex; justify-content: center; align-items: center;'>
                    <video width='640' height='480' controls>
                    <source src='/gradio_api/file={self.logged_in_name}/cap/{file_name}' type='video/mp4'>
                    您的浏览器不支持HTML5视频标签。</video>
                    </div>
                    """
        print(htmlstr)
        sys.stdout.flush()
        return video_file_path


    def vision_query_image(self, image_file_path: str):  
        """
        功能：根据图像的image_file_path，描述图像的内容，并返回描述。
        参数：image_file_path：图像的文件路径。
        返回：描述文本。
        说明：根据图像的image_file_path，描述图像的内容，并返回描述。
        """
        if image_file_path==None:
            return "打开摄像头失败"
        prompt="请用中文描述这个图像的内容。"
        #print("vision_query_image:",image_file_path)
        with open(image_file_path, "rb") as image_file:
            base64str=base64.b64encode(image_file.read()).decode('utf-8')                     
        response = requests.post(
            f"{QWEN_OPENAI_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.dashscope_api_key}",
                "Content-Type": "application/json",
                "Content-Length": str(len(base64str or "") + len(prompt))
            },
            json={
                "model": "qwen3-vl-plus",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64str}"}},
                        {"type": "text", "text": prompt}
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
                        text=data["choices"][0]["delta"]["content"]
                        full_response =full_response+text                                      
                        print(text,end="",flush=True)
                        sys.stdout.flush()
                      
            except json.JSONDecodeError:
                continue   
        return full_response
     
    #根据视频的video_file_path，描述视频的具体过程，并返回视频的描述。
    def vision_query_video(self, video_file_path: str):
        '''
        功能：根据视频的video_file_path，描述视频的具体过程，并返回视频的描述。
        参数：video_file_path：视频的文件路径。
        返回：视频的描述文本。
        说明：根据视频的video_file_path，描述视频的具体过程，并返回视频的描述。
        '''
        prompt="描述这个视频的具体过程"
        if not video_file_path:
            return "打开摄像头失败"
        #视频的base64编码
        #print("vision_query_video:",file_path)
        with open(video_file_path, "rb") as video_file:
            videobase64str = base64.b64encode(video_file.read()).decode('utf-8')
           

        response = requests.post(
            f"{QWEN_OPENAI_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3-vl-plus",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{videobase64str}"},
                        },
                        {"type": "text", "text": prompt}
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
                        text = data["choices"][0]["delta"]["content"]
                        full_response += text
                        print(text, end="", flush=True)
                        sys.stdout.flush()
            except json.JSONDecodeError:
                continue
        return full_response

    #获取当前日期和时间，并返回一个包含日期和时间的字符串。
    def get_current_datetime(self):
        """
        功能：获取当前日期和时间，并返回一个包含日期和时间的字符串。
        参数：无
        返回值：包含日期和时间的字符串。
        """
        # 获取当前日期和时间，并格式化为字符串
        # 格式为："年-月-日 时:分:秒"
        # 例如："2023-04-15 12:30:45"
        # 使用time模块的strftime函数实现
        current_datetime = time.strftime("%Y-%m-%d %H:%M")
        return current_datetime

    #函数，用于设置智能体名称，用于自我介绍。
    async def set_name(self, ctx:Context, name:str) -> str:
        """
        功能：设置智能体名称，用于自我介绍。
        参数：ctx：上下文对象。
        name：智能体名称。
        返回值：智能体名称。
        说明：设置智能体名称，用于自我介绍。
        """
        state=await ctx.get("state") # type: ignore
        state["name"]=name # type: ignore
        await ctx.set("state",state) # type: ignore
        return f"{name}"

    #文生图片并显示
    def generate_image_show(self, prompt: str):
        """
        功能：根据提示文本内容，生成图片，并返回图片的文件路径。
        参数：prompt：提示文本内容。
        返回：图片的文件路径。
        说明：根据提示文本内容，生成图片，并返回图片的文件路径。
        """
         # 设置模型名称（是否启用增强版）
        if self.isplus=="True":
            #print("使用增强版模型")
            modelname = "wanx2.1-t2i-plus"  
        else:
            modelname="qwen-image"#"wanx2.1-t2i-turbo"

        # 创建异步任务
        #print("----create task----")
        try:
            rsp = ImageSynthesis.async_call(
                api_key=self.dashscope_api_key, # type: ignore
                model=modelname,
                prompt=prompt,
                n=1,
                #size=size
            )
        except Exception as e:
            #print(f"调用图像生成服务失败: {e}")
            return None

        if rsp.status_code != HTTPStatus.OK:
            #print(f"Failed to create async task: {rsp.message}")
            return None

        # 等待任务完成并获取图像 URL
        #print("----wait task done then get image url----")
        image_url = None
        try:
            for _ in range(30):  # 最多等待30次，每次2秒
                time.sleep(2)
                result_rsp = ImageSynthesis.wait(rsp)
                if result_rsp.status_code == HTTPStatus.OK:
                    for result in result_rsp.output.results:
                        image_url = result.url
                    break
                elif result_rsp.code == 'TaskIdInvalid':
                    #print("无效的任务ID，请确认任务是否已创建成功。")
                    return None
            else:
                #print("等待超时，图像生成未完成。")
                return None
        except Exception as e:
            #print(f"获取图像结果失败: {e}")
            return None

        if not image_url:
            #print("未能获取到图像 URL。")
            return None

        # 构建本地保存路径
        output_dir = os.path.join(self.logged_in_name, "imgoutput")
        os.makedirs(output_dir, exist_ok=True)

        current_time = time.strftime('%Y%m%d%H%M%S')
        file_name = f"{current_time}.png"
        file_path = os.path.join(output_dir, file_name)

        # 将路径中的反斜杠替换为正斜杠，确保在Web环境中能正确解析
        file_path = file_path.replace("\\", "/")
        
        # 下载并保存图像
        try:
            response = requests.get(image_url)
            if response.status_code == HTTPStatus.OK:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                #print(f"Image saved to {file_path}")
                #图像居中显示
                htmlstr=f"<p style='text-align: center;'> <img src='/gradio_api/file={file_path}'  style='display: inline; vertical-align: middle;'></p>"
                print(htmlstr)
                sys.stdout.flush()
                return file_path
            else:
                #print(f"下载图像失败，HTTP状态码: {response.status_code}")
                return None
        except Exception as e:
            #print(f"保存图像文件失败: {e}")
            return None
             

    #语音合成并插入网络音频文件
    def generate_audio_show(self, prompt: str):
        """
        功能：根据提示文本内容，生成音频，并返回音频的文件路径。
        参数：prompt：提示文本内容。
        返回：音频的文件路径。
        说明：根据提示文本内容，生成音频，并返回音频的文件路径。
        """
        
        class Callback(ResultCallback):
            def __init__(self):
                self.audio_data = []

            def on_data(self, data: bytes) -> None:
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_data.append(audio_chunk)
            def get_audio_array(self):
                return np.concatenate(self.audio_data) 
        

        
        callback = Callback()
        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v2",
            voice=self.voiceid,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=callback,
        )

        sample_rate = 22050

        for text in prompt:
            if text.strip():
                synthesizer.streaming_call(text)
                time.sleep(0.1)

        synthesizer.streaming_complete()

        audio_array = callback.get_audio_array()

        current_time = time.strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(self.logged_in_name, "audiooutput")
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{current_time}.wav"
        file_path = os.path.join(output_dir, file_name)

        # 将路径中的反斜杠替换为正斜杠，确保在Web环境中能正确解析
        file_path = file_path.replace("\\", "/")
        
        wav_data = audio_array.tobytes()
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setframerate(sample_rate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(wav_data)

        #音频居中显示
        htmlstr=f"<p style='text-align: center;'> <audio controls><source src='/gradio_api/file={file_path}' type='audio/mpeg'></audio></p>"
        print(htmlstr)
        sys.stdout.flush()
        
        return file_path
        
    #文生视频并显示
    def generate_video_show(self, prompt: str):
        """
        功能：根据提示文本内容，生成视频，并返回视频的文件路径。
        参数：prompt：提示文本内容。
        返回：视频的文件路径。
        说明：根据提示文本内容，生成视频，并返回视频的文件路径。
        """
        
        output_dir = os.path.join(self.logged_in_name, "videooutput")
        os.makedirs(output_dir, exist_ok=True)

        # 设置模型名称
        if self.isplus=="True":
            #print("使用增强版模型")
            modelname = "wanx2.1-t2v-plus"
        else:
            modelname="wanx2.1-t2v-turbo"

        # 创建异步任务
        try:
            rsp = VideoSynthesis.async_call(
                api_key=self.dashscope_api_key, # type: ignore
                model=modelname,
                prompt=prompt,
                size=self.size,
            )
        except Exception as e:
            #print(f"调用视频生成服务失败: {e}")
            return None,
        
        if rsp.status_code != HTTPStatus.OK:
            #print(f"Failed to create async task: {rsp.message}")
            return None

        # 等待任务完成并获取视频 URL
        video_url = None
        try:
            for _ in range(30):  # 最多等待30次，每次2秒
                time.sleep(2)
                result_rsp = VideoSynthesis.wait(rsp)
                if result_rsp.status_code == HTTPStatus.OK:
                    video_url = result_rsp.output.video_url
                    break
                elif result_rsp.code == 'TaskIdInvalid':
                    #print("无效的任务ID，请确认任务是否已创建成功。")
                    return None
            else:
                #print("等待超时，视频生成未完成。")
                return None
        except Exception as e:
            #print(f"获取视频结果失败: {e}")
            return None

        if not video_url:
            #print("未能获取到视频 URL。")
            return None

        # 下载并保存视频
        current_time = time.strftime('%Y%m%d%H%M%S')
        file_name = f"{current_time}.mp4"
        file_path = os.path.join(output_dir, file_name)
        
        # 将路径中的反斜杠替换为正斜杠，确保在Web环境中能正确解析
        file_path = file_path.replace("\\", "/")

        try:
            response = requests.get(video_url)
            if response.status_code == HTTPStatus.OK:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                #视频居中显示
                htmlstr=f"<p style='text-align: center;'> <video controls><source src='/gradio_api/file={file_path}' type='video/mp4'></video></p>"
                print(htmlstr)
                sys.stdout.flush()
                return file_path
            else:
                #print(f"下载视频失败，HTTP状态码: {response.status_code}")
                return None
        except Exception as e:
            #print(f"保存视频文件失败: {e}")
            return None
        
    ####################讲解视频生成################################################    
    def generate_teacher_image(self, topic: str) -> tuple[str, str]:
        """
        生成教师形象图片
        
        Args:
            topic: 主题内容
            
        Returns:
            生成的图片文件路径和性别信息
        """
        try:
            # 随机选择性别
            import random
            gender = random.choice(["男", "女"])
            
            # 构造教师形象提示词，基于主题生成合适的教师形象
            if gender == "男":
                prompt = f"一位专业的男性教师，正在讲解{topic}相关内容，穿着得体，背景适合教学环境，正面视角，写实摄影风格，高清8K"
            else:
                prompt = f"一位专业的女性教师，正在讲解{topic}相关内容，穿着得体，背景适合教学环境，正面视角，写实摄影风格，高清8K"
            
            # 调用图像生成API
            rsp = ImageSynthesis.async_call(
                api_key=self.dashscope_api_key, # type: ignore
                model="wanx2.1-t2i-turbo",
                prompt=prompt,
                n=1
            )
            
            if rsp.status_code != HTTPStatus.OK:
                raise Exception(f"图像生成失败: {rsp.message}")
            
            # 等待任务完成
            for _ in range(30):  # 最多等待60秒
                time.sleep(2)
                result_rsp = ImageSynthesis.wait(rsp)
                if result_rsp.status_code == HTTPStatus.OK:
                    image_url = result_rsp.output.results[0].url
                    break
                elif result_rsp.code == 'TaskIdInvalid':
                    raise Exception("无效的任务ID")
            else:
                raise Exception("图像生成超时")
            
            # 保存图片
            output_dir = os.path.join(self.logged_in_name, "imageoutput")
            os.makedirs(output_dir, exist_ok=True)
            
            current_time = time.strftime('%Y%m%d%H%M%S')
            file_name = f"teacher_{current_time}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # 将路径中的反斜杠替换为正斜杠，确保在Web环境中能正确解析
            file_path = file_path.replace("\\", "/")
            
            # 下载并保存图片
            response = requests.get(image_url)
            if response.status_code == HTTPStatus.OK:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                # 图像居中显示，模仿generate_image_show的输出方式
                htmlstr=f"<p style='text-align: center;'> <img src='/gradio_api/file={file_path}'  style='display: inline; vertical-align: middle;'></p>"
                print(htmlstr)
                sys.stdout.flush()  # 强制刷新输出缓冲区
                return file_path, gender
            else:
                raise Exception(f"图片下载失败: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"生成教师形象时出错: {str(e)}")


    def query_knowledge_base(self, topic: str) -> str:
        """
        功能：根据提示文本内容，查询本地知识库，并返回查询结果。
        参数：topic：提示文本内容。
        返回：查询结果。
        """
        Settings.llm = self.llm
        # 设置嵌入模型
        Settings.embed_model = OllamaEmbedding(
            model_name=self.embedding_model_name,
            embedding_dim=1024
        )
        kbname="root"
        # 初始化ChromaDB
        #db = chromadb.PersistentClient(path=os.path.join(self.logged_in_name,"chroma_db"))
        db = chromadb.PersistentClient(path=os.path.join(kbname,"chroma_db"))
        chroma_collection = db.get_or_create_collection(
            #name=logged_in_name,
            name=kbname,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 32
            },
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 判断是否有知识库，如果没有，返回提示
        if chroma_collection.count() == 0:
            return "知识库为空，请先添加知识库文档。\n\n"
        else:
            pass
            # print("知识库查询:", topic + "\n\n")
        
        try:
            # 从向量存储创建索引
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            
            # 初始检索，获取更多候选结果用于重排序

            # 初始检索，获取更多候选结果用于重排序
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=10,  # 增加检索结果数量供重排序使用
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                alpha=0.3
            )

            # 执行初始检索
            retrieved_nodes = retriever.retrieve(topic)
            
            # 如果有检索到节点，则进行重排序
            if retrieved_nodes:
                # 提取文档内容用于重排序
                documents = [node.get_content() for node in retrieved_nodes]
                
                # 使用Qwen3-Rerank进行重排序
                reranked_nodes = self._rerank_documents(topic, retrieved_nodes, documents)
            else:
                reranked_nodes = retrieved_nodes

            # 创建新的检索器使用重排序后的结果
            class RerankedRetriever(BaseRetriever):
                def __init__(self, nodes_with_scores, similarity_top_k=5):
                    self.nodes_with_scores = nodes_with_scores[:similarity_top_k]
                    super().__init__()
                    
                def _retrieve(self, query_str, **kwargs):  # type: ignore
                    return self.nodes_with_scores

            # 使用重排序后的前5个结果
            final_retriever = RerankedRetriever(reranked_nodes, similarity_top_k=5)

            # 初始化对话记忆
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=8000,
            )

            # 创建聊天引擎
            chat_engine = ContextChatEngine(
                retriever=final_retriever,
                memory=memory,
                llm=Settings.llm,
                prefix_messages=[]
            )

            # 流式输出
            full_response = ""
            response_stream = chat_engine.stream_chat(topic)
            
            for chunk in response_stream.response_gen:
                full_response += chunk
                print(chunk, end="", flush=True)
            
            print("\n\n")
            return full_response

        except Exception as e:
            print(f"Error in query_knowledge_base: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的堆栈跟踪信息
            raise

    def _rerank_documents(self, query, nodes, documents):
        """使用dashscope的TextReRank对文档进行重排序"""
        try:
            # 调用dashscope的TextReRank API
            resp = dashscope.TextReRank.call(
                model="qwen3-rerank",
                query=query,
                documents=documents,
                top_n=len(documents),  # 返回所有文档的排序结果
                return_documents=True
            )
            
            if resp.status_code == HTTPStatus.OK:
                # 根据重排序结果重新组织nodes
                reranked_nodes = []
                for item in resp.output.results:
                    original_index = item.index
                    # 保持原有的NodeWithScore结构，但更新分数为重排序的分数
                    node_with_score = nodes[original_index]
                    node_with_score.score = item.relevance_score
                    reranked_nodes.append(node_with_score)
                return reranked_nodes
            else:
                # 如果重排序失败，返回原始节点
                return nodes
                
        except Exception as e:
            # 发生异常时返回原始节点
            return nodes

    def web_search(self, query: str) -> str:
        """
        功能：执行联网搜索，获取最新、最准确的外部信息
        参数：query：搜索查询内容
        返回：搜索结果
        """
        # 构建消息
        messages = [ChatMessage(role="user", content=query)]
        msglst = [{
            "role": "user",
            "content": query
        }]
        
        url = f'{QWEN_OPENAI_API_BASE}/chat/completions'
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.dashscope_api_key}"
        }
        data = {
            "model": self.model_name,
            "messages": msglst,
            "enable_search": True,
            "stream": True,  # 流式返回结果
            "stream_options": {"include_usage": True}
        }

        full_response = ""
        try:
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_lines():
                        if not chunk:
                            continue
                        try:
                            chunk_str = chunk.decode('utf-8')
                            if chunk_str.startswith("data:"):
                                chunk_str = chunk_str[5:].strip()
                                if chunk_str == "[DONE]":
                                    continue
                                data_chunk = json.loads(chunk_str)
                                if data_chunk.get("choices") and data_chunk["choices"][0].get("delta", {}).get("content"):
                                    res = data_chunk["choices"][0]["delta"]["content"]
                                    full_response += res
                                    # 流式输出到控制台，以便调用方可以实时获取结果
                                    print(res, end="", flush=True)
                        except json.JSONDecodeError as e:
                            print(f"解析数据失败：{str(e)}", flush=True)
                else:
                    # 如果API调用失败，尝试使用LLM的普通回答
                    response = self.llm.chat(messages)
                    result = response.message.content if hasattr(response.message, 'content') else str(response)
                    print(result, end="", flush=True)
                    return str(result)
        except Exception as e:
            print(f"Error in web_search: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 如果出错，使用LLM的普通回答
            try:
                messages = [ChatMessage(role="user", content=query)]
                response = self.llm.chat(messages)
                result = response.message.content if hasattr(response.message, 'content') else str(response)
                print(result, end="", flush=True)
                return str(result)
            except Exception as fallback_e:
                error_msg = f"联网搜索失败: {str(fallback_e)}"
                print(error_msg, flush=True)
                return error_msg

        print("\n\n", flush=True)  # 添加换行
        return full_response

    def generate_lecture_script(self, topic: str) -> str:
        """
        生成讲解稿
        
        Args:
            topic: 讲解主题
            
        Returns:
            生成的讲解稿文本
        """
        try:
            # 首先尝试从本地知识库查询相关内容
            #print(f"正在生成讲解稿，主题: {topic}", "\n\n")
            
            knowledge_content = self.query_knowledge_base(topic)
            
            #print("查询到的知识库内容:", knowledge_content, "\n\n")
            
            # 构造讲解稿生成提示
            if knowledge_content and len(knowledge_content.strip()) > 0:
                prompt = f"""请根据以下知识库内容，生成一段关于"{topic}"的讲解稿，要求如下：
    知识库内容：
    {knowledge_content}

    生成要求：
    1. 总时长约18秒（约60-70字）
    2. 结构分为三部分：导入（3秒）+ 核心内容（12秒）+ 总结（3秒）
    3. 语言口语化，避免术语堆砌，适当使用比喻
    4. 内容准确，表达清晰流畅
    5. 必须基于提供的知识库内容进行创作

    直接输出讲解稿内容，无需额外说明。
    """
            else:
                prompt = f"""请生成一段关于"{topic}"的讲解稿，要求如下：
    1. 总时长约18秒（约60-70字）
    2. 结构分为三部分：导入（3秒）+ 核心内容（12秒）+ 总结（3秒）
    3. 语言口语化，避免术语堆砌，适当使用比喻
    4. 内容准确，表达清晰流畅

    直接输出讲解稿内容，无需额外说明。
    """
            
            # 调用LLM生成讲解稿
            # 使用 ChatMessage 对象
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content=prompt)
            ]
            response = self.llm.stream_chat(messages)
            script =""
            for chunk in response:
                if chunk.delta:
                    script += chunk.delta
                    print(chunk.delta, end="", flush=True)
                    sys.stdout.flush()  # 强制刷新输出缓冲区
            
                
            word_count = len(script)
            # 估算时长（平均每秒5个字）
            estimated_duration = word_count / 5
            
            # 输出讲解稿内容，使用居中的div展示
            htmlstr = f"<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9;'><p><strong>讲解稿:</strong></p><p>{script}</p><p><small>({word_count}字，约{estimated_duration:.1f}秒)</small></p></div>"
            print(htmlstr)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            return script
            
        except Exception as e:
            raise Exception(f"生成讲解稿时出错: {str(e)}")



    def generate_lecture_audio(self, script: str, gender: str = "female") -> str:
        """
        生成讲解音频
        
        Args:
            script: 讲解稿文本
            gender: 音色性别 ("female" 或 "male")
            
        Returns:
            生成的音频文件路径
        """
        try:
            # 根据性别选择音色
            if gender.lower() == "male":
                voiceid = "longjielidou_v2"  # 阳光男
            else:
                voiceid = "longling_v2"  # 甜美女
            
            # 使用与generate_audio_show相同的实现方式
            class Callback(ResultCallback):
                def __init__(self):
                    self.audio_data = []

                def on_data(self, data: bytes) -> None:
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    self.audio_data.append(audio_chunk)
                    
                def get_audio_array(self):
                    return np.concatenate(self.audio_data) 
            
            # 创建语音合成器，使用cosyvoice-v2模型
            callback = Callback()
            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v2",
                voice=voiceid,
                format=AudioFormat.PCM_22050HZ_MONO_16BIT,
                callback=callback,
            )

            sample_rate = 22050

            # 控制讲解稿长度以确保音频不超过18秒
            # 中文朗读速度约为每秒5个汉字，18秒约90个汉字
            max_chars = 85  # 留一些余量
            if len(script) > max_chars:
                # 截断文本到合适长度
                truncated_script = script[:max_chars]
                # 确保在句子边界截断，避免在单词中间切断
                last_punct = max(truncated_script.rfind('。'), truncated_script.rfind('！'), truncated_script.rfind('？'), truncated_script.rfind('，'))
                if last_punct > 70:  # 如果标点符号在合理位置
                    script = truncated_script[:last_punct+1]
                else:
                    script = truncated_script

            # 流式合成语音
            for text in script:
                if text.strip():
                    synthesizer.streaming_call(text)
                    time.sleep(0.1)

            synthesizer.streaming_complete()

            audio_array = callback.get_audio_array()

            # 保存音频文件
            current_time = time.strftime('%Y%m%d%H%M%S')
            output_dir = os.path.join(self.logged_in_name, "audiooutput")
            os.makedirs(output_dir, exist_ok=True)
            file_name = f"lecture_{current_time}.wav"
            file_path = os.path.join(output_dir, file_name)
            
            # 将路径中的反斜杠替换为正斜杠，确保在Web环境中能正确解析
            file_path = file_path.replace("\\", "/")

            wav_data = audio_array.tobytes()
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setframerate(sample_rate)
                wav_file.setsampwidth(2)
                wav_file.setnchannels(1)
                wav_file.writeframes(wav_data)
                
            # 音频居中显示，模仿generate_audio_show的输出方式
            htmlstr=f"<p style='text-align: center;'> <audio controls><source src='/gradio_api/file={file_path}' type='audio/mpeg'></audio></p>"
            print(htmlstr)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            return file_path
            
        except Exception as e:
            raise Exception(f"生成语音时出错: {str(e)}")


    def generate_lecture_video(self, image_path: str, audio_path: str) -> str:
        """
        生成讲解视频
        
        Args:
            image_path: 教师形象图片路径
            audio_path: 讲解音频路径
            
        Returns:
            生成的视频文件路径
        """
        try:
            # 上传文件获取URL
            def upload_file_to_oss(file_path: str, model_name: str) -> str:
                """上传文件到OSS并获取临时公网URL"""
                # 1. 获取上传凭证
                url = "https://dashscope.aliyuncs.com/api/v1/uploads"
                headers = {
                    "Authorization": f"Bearer {self.dashscope_api_key}",
                    "Content-Type": "application/json"
                }
                params = {
                    "action": "getPolicy",
                    "model": model_name
                }

                response = requests.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    raise Exception(f"Failed to get upload policy: {response.text}")
                
                policy_data = response.json()['data']

                # 2. 上传文件到OSS
                file_name = os.path.basename(file_path)
                key = f"{policy_data['upload_dir']}/{file_name}"
                with open(file_path, 'rb') as file:
                    # 根据文件扩展名确定Content-Type
                    content_type = "application/octet-stream"  # 默认类型
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if file_name.lower().endswith('.png'):
                            content_type = "image/png"
                        elif file_name.lower().endswith(('.jpg', '.jpeg')):
                            content_type = "image/jpeg"
                    elif file_name.lower().endswith('.gif'):
                        content_type = "image/gif"
                    elif file_name.lower().endswith(('.mp3', '.wav')):
                        content_type = "audio/mpeg"
                    
                    files = {
                        'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
                        'Signature': (None, policy_data['signature']),
                        'policy': (None, policy_data['policy']),
                        'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
                        'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
                        'key': (None, key),
                        'success_action_status': (None, '200'),
                        'file': (file_name, file, content_type)
                    }

                    response = requests.post(policy_data['upload_host'], files=files)
                    if response.status_code != 200:
                        raise Exception(f"Failed to upload file: {response.text}")

                return f"oss://{key}"
                
            image_url = upload_file_to_oss(image_path, "wan2.2-s2v")
            audio_url = upload_file_to_oss(audio_path, "wan2.2-s2v")
            
            # 提交视频生成任务
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
            headers = {
                "Authorization": f"Bearer {self.dashscope_api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable"
            }
            
            data = {
                "model": "wan2.2-s2v",
                "input": {
                    "image_url": image_url,
                    "audio_url": audio_url
                },
                "parameters": {
                    "resolution": "480P"
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code != HTTPStatus.OK:
                raise Exception(f"视频生成任务提交失败: {response.text}")
            
            result = response.json()
            if "output" not in result or "task_id" not in result["output"]:
                raise Exception("API响应格式不正确")
            
            task_id = result["output"]["task_id"]
            progress_html = f"<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fffbe6;'><p>视频生成任务已提交，任务ID: {task_id}，正在等待生成完成...</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            
            # 轮询任务状态
            poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            poll_headers = {"Authorization": f"Bearer {self.dashscope_api_key}"}
            
            for i in range(600):  # 最多等待10分钟(600秒)
                time.sleep(5)
                poll_response = requests.get(poll_url, headers=poll_headers)
                poll_result = poll_response.json()
                
                # 每隔一定时间输出进度信息，让用户知道仍在工作中
                if i % 12 == 0:  # 每分钟输出一次进度（5秒*12=60秒）
                    elapsed_minutes = (i * 5) // 60
                    progress_html = f"<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fffbe6;'><p>视频生成中，已用时约 {elapsed_minutes} 分钟，请耐心等待...</p></div>"
                    print(progress_html)
                    sys.stdout.flush()  # 强制刷新输出缓冲区
                
                # 检查任务状态
                if "output" not in poll_result or "task_status" not in poll_result["output"]:
                    raise Exception(f"轮询响应格式不正确: {poll_result}")
                    
                task_status = poll_result["output"]["task_status"]
                
                if task_status == "SUCCEEDED":
                    # 注意：这里的键名是"results"而不是"result"
                    if "results" not in poll_result["output"] or "video_url" not in poll_result["output"]["results"]:
                        raise Exception(f"任务成功但未返回video_url: {poll_result}")
                        
                    video_url = poll_result["output"]["results"]["video_url"]
                    progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f6ffed;'><p>✅ 视频生成完成，正在下载...</p></div>"
                    print(progress_html)
                    sys.stdout.flush()  # 强制刷新输出缓冲区
                    break
                elif task_status in ["FAILED", "CANCELLED"]:
                    # 获取错误信息
                    error_message = poll_result.get("output", {}).get("message", "未知错误")
                    raise Exception(f"视频生成失败: {error_message}")
            else:
                raise Exception("视频生成超时")
            
            # 下载并保存视频
            output_dir = os.path.join(self.logged_in_name, "videooutput")
            os.makedirs(output_dir, exist_ok=True)
            
            current_time = time.strftime('%Y%m%d%H%M%S')
            file_name = f"lecture_{current_time}.mp4"
            file_path = os.path.join(output_dir, file_name)
            
            # 将路径中的反斜杠替换为正斜杠，确保在Web环境中能正确解析
            file_path = file_path.replace("\\", "/")
            
            video_response = requests.get(video_url)
            if video_response.status_code == HTTPStatus.OK:
                with open(file_path, 'wb') as f:
                    f.write(video_response.content)
                # 视频居中显示，模仿generate_video_show的输出方式
                htmlstr=f"<p style='text-align: center;'> <video controls><source src='/gradio_api/file={file_path}' type='video/mp4'></video></p>"
                print(htmlstr)
                sys.stdout.flush()  # 强制刷新输出缓冲区
                return audio_url
            else:
                raise Exception(f"视频下载失败: {video_response.status_code}")
                
        except Exception as e:
            raise Exception(f"生成视频时出错: {str(e)}")


    def generate_lecture_video_by_topic(self, topic: str) -> dict[str, Any]: # type: ignore
        """
        根据主题生成完整的讲解视频内容
        
        Args:
            topic: 讲解主题
            gender: 讲解员性别
            
        Returns:
            包含所有生成内容路径的字典
        """
        try:
            progress_html = f"<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #e6f7ff;'><p><strong>开始生成'{topic}'的讲解视频...</strong></p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            
            # 1. 生成教师形象
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fffbe6;'><p>第1步：正在生成教师形象...</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            image_path, teacher_gender = self.generate_teacher_image(topic)
            progress_html = f"<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f6ffed;'><p>✅ 第1步完成：{teacher_gender}教师形象已生成</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            
            # 2. 生成讲解稿
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fffbe6;'><p>第2步：正在生成讲解稿...</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            script = self.generate_lecture_script(topic)
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f6ffed;'><p>✅ 第2步完成：讲解稿已生成</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            
            # 3. 生成讲解音频（使用与教师形象匹配的性别）
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fffbe6;'><p>第3步：正在生成讲解音频...</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            # 根据教师形象性别确定音色性别
            audio_gender = "male" if teacher_gender == "男" else "female"
            audio_path = self.generate_lecture_audio(script, audio_gender)
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f6ffed;'><p>✅ 第3步完成：讲解音频已生成</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            
            # 4. 生成讲解视频
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fffbe6;'><p>第4步：正在生成讲解视频...</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            video_path = self.generate_lecture_video(image_path, audio_path)
            progress_html = "<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f6ffed;'><p>✅ 第4步完成：讲解视频已生成</p></div>"
            print(progress_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            
            return {
                "image_path": image_path,
                "teacher_gender": teacher_gender,
                "script": script,
                "audio_path": audio_path,
                "video_path": video_path
            }
            
        except Exception as e:
            error_html = f"<div style='text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fff2f0; color: #ff4d4f;'><p><strong>❌ 生成讲解视频过程中出错: {str(e)}</strong></p></div>"
            print(error_html)
            sys.stdout.flush()  # 强制刷新输出缓冲区
            raise

    #保存日志
    def save_log(self, prompt, response):
        """
        功能：保存日志到文件中。
        参数：prompt：提示文本内容。
        response：响应文本内容。
        返回值：无
        说明：保存日志到文件中。
        """
         # 获取当前年月，用于日志文件命名
        current_time = time.strftime("%Y%m")
        log_filename = f"{current_time}.log"
        log_filepath = os.path.join(self.logged_in_name,"cap", log_filename)

        # 确保目录存在
        os.makedirs(os.path.join(self.logged_in_name,"cap"), exist_ok=True)
        # 打开日志文件，以追加模式写入
        with open(log_filepath, "a", encoding="utf-8") as log_file:
            log_file.write(f"Prompt: {prompt}\n")
            log_file.write(f"Response: {response}\n")
            log_file.write("-" * 50 + "\n")  # 分隔线
            
    #定义一个函数，用于执行workflow工作流程。
    async def runworkflow_image(self, prompt):  
        """
        功能：执行workflow工作流程。
        参数：prompt：提示文本内容。
        ctx_dict：上下文字典。
        返回值：ctx_dict：上下文字典。
        说明：执行workflow工作流程。
        """
        
        # 流式输出响应    #和上下文处理有bug
        response=self.iva_workflow.run(prompt,memory=self.memory)
        full_response = ""
        

        
        async for event in response.stream_events():
            if isinstance(event, AgentStream):
                full_response += event.delta
                # 输出内容
                print(event.delta, end="", flush=True)     
                
        # self.save_log(prompt, full_response)
        

    #创建智能体，获取摄像头的视频，并返回视频的video_file_path。
    def create_video_agents(self):
        get_camera_video_agent=FunctionAgent(
            name="get_camera_video_agent",
            description="获取摄像头的视频，并返回视频的video_file_path。",
            system_prompt=("1、你可以使用get_camera_video()函数获取摄像头的视频,并返回视频的video_file_path。"),
            llm=self.llm,
            tools=[self.get_camera_video],
            can_handoff_to=["vision_query_video_agent"],
        )

        #创建智能体，描述视频的具体过程，接收 get_camera_video() 函数返回的video_file_path作为参数。
        vision_query_video_agent=FunctionAgent(
            name="vision_query_video_agent",
            description="vision_query_video()函数描述视频的具体过程，接收 get_camera_video() 函数返回的video_file_path作为参数。",
            system_prompt=("接收 get_camera_video() 函数返回的video_file_path作为参数，描述视频的具体过程。"),
            llm=self.llm,
            tools=[self.vision_query_video],
            can_handoff_to=["write_agent"],
        )

        #创建智能体，对视频的描述进行总结建议。
        write_agent=FunctionAgent(
            name="write_agent",
            description="对视频的描述进行总结建议。",
            system_prompt=("根据视频的描述，进行总结建议。"),
            llm=self.llm,
            tools=None,
            can_handoff_to=None,
        )
        
        return get_camera_video_agent, vision_query_video_agent, write_agent

    #创建AgentWorkflow对象，使用智能体完成视频智能体的工作流程。
    def create_video_workflow(self):
        get_camera_video_agent, vision_query_video_agent, write_agent = self.create_video_agents()
        agent_workflow = AgentWorkflow(
            agents=[get_camera_video_agent, vision_query_video_agent, write_agent],    
            root_agent='get_camera_video_agent',
            initial_state=None,
        )
        return agent_workflow

    #定义一个函数，用于执行agent_workflow工作流程。
    async def runworkflow_video(self, prompt):  
        """
        功能：执行agent_workflow工作流程。
        参数：prompt：提示文本内容。
        ctx_dict：上下文字典。
        返回值：ctx_dict：上下文字典。
        说明：执行agent_workflow工作流程。
        """
        

        # 流式输出响应        
        response= self.create_video_workflow().run(
            user_msg=prompt,
            memory=self.memory,
            )
        full_response = ""
        async for event in response.stream_events():
            if isinstance(event, AgentStream):
                full_response += event.delta
                print(event.delta, end="", flush=True)    
        # self.save_log(prompt,full_response)


    # 主执行函数
    async def run_agent_workflow(self, prompt):
        """
        根据提示词内容执行相应的工作流
        参数：prompt：提示文本内容
        ctx_dict：上下文字典
        返回：执行结果
        """
        if "远程视频" in prompt:
            return await self.runworkflow_video(prompt)
        else:
            return await self.runworkflow_image(prompt)


# 实例缓存（模块内全局）
service_cache = {}

def get_agent_rag_service(model_name, embedding_model_name, logged_in_name, nvr1_url="", nvr2_url="", size="1024*768", isplus="False", voice="严肃男"):
    """获取或创建一个AgentRagService实例"""
    key = (model_name, embedding_model_name, logged_in_name, nvr1_url, nvr2_url, size, isplus, voice)
    if key not in service_cache:
        service_cache[key] = AgentRagService(model_name, embedding_model_name, logged_in_name, nvr1_url, nvr2_url, size, isplus, voice)
    return service_cache[key]


async def run_agent_workflow_stream(prompt, session_state, model_name, embedding_model_name, size="1024*768", isplus="False", voice="严肃男"):
    """
    流式运行agent工作流的函数，用于agent_chativ函数调用
    """


    # 从 session_state 获取登录用户
    logged_in_name = session_state.get("logged_in_name", "root") if session_state and isinstance(session_state, dict) else "root"
    
    # 获取NVR URLs
    nvr1_url, nvr2_url = getnvr_url(logged_in_name)
    
    service = get_agent_rag_service(model_name, embedding_model_name, logged_in_name, nvr1_url, nvr2_url, size, isplus, voice) # type: ignore
    
    # 创建队列用于线程间通信
    output_queue = Queue()
    
    def run_workflow_in_thread():
        original_stdout = sys.stdout
        try:
            class QueueWriter:
                def write(self, s):
                    if s and s.strip():  # 避免发送空白字符
                        output_queue.put(s)
                def flush(self):
                    pass
            
            # 重定向标准输出
            sys.stdout = QueueWriter()
            
            # 运行工作流
            if "远程视频" in prompt:
                asyncio.run(service.runworkflow_video(prompt))
            else:
                asyncio.run(service.runworkflow_image(prompt))
        except Exception as e:
            output_queue.put(f"\n错误: {str(e)}")
        finally:
            # 恢复原始stdout
            sys.stdout = original_stdout
            # 发送结束标记
            output_queue.put(None)  # None作为结束标记

    # 启动工作流线程
    thread = threading.Thread(target=run_workflow_in_thread)
    thread.start()
    
    # 累积输出内容
    full_output = ""
    
    # 持续从队列读取并输出
    while True:
        try:
            # 等待最多2秒获取输出
            item = output_queue.get(timeout=2)
            if item is None:  # 结束标记
                break
            full_output += item
            yield full_output  # 流式返回累积内容
        except Empty:
            # 检查线程是否仍在运行
            if not thread.is_alive():
                break
            continue

    # 等待线程结束
    thread.join()