from dashscope import Application
import gradio as gr

# 添加全局变量用于存储session_id
session_id = None

def agent_fn(prompt):
    """
    调用DashScope应用程序接口并以流式方式返回结果    
    Args:
        prompt (str): 用户输入的提示词        
    Yields:
        str: 流式返回的响应文本，每次返回累积的完整文本
    """
    global session_id
    appid="YOUR_APP_ID"  # 替换为你的应用ID
    dashscope_api_key ="your_api_key"  # 替换为你的API Key
    yield f"请稍候..."        
    
    # 准备调用参数
    call_params = {
        "api_key": dashscope_api_key,  # type: ignore
        "app_id": appid,  # 应用ID替换YOUR_APP_ID
        "prompt": prompt,
        "stream": True,  # 流式输出
        "incremental_output": True,  # 增量输出
    }
    
    # 如果存在session_id，则添加到调用参数中
    if session_id:
        call_params["session_id"] = session_id
    
    #print("调用参数:", call_params)  # 调试输出调用参数
    
    response = Application.call(**call_params)  
    #print('%s\n' % (response.output.text))  # 处理只输出文本text
    full_text = ""
    session_id_from_response = None
    try:
        for chunk in response:
            # 尝试从第一个chunk中获取session_id
            if session_id_from_response is None and hasattr(chunk, 'output') and hasattr(chunk.output, 'session_id'):
                session_id_from_response = chunk.output.session_id
                if session_id_from_response:
                    session_id = session_id_from_response
                    
            if chunk.output.text:
                full_text += chunk.output.text
                yield full_text  # 流式返回每次更新的内容
    except Exception as e:    
        yield str(e)



with gr.Blocks(title="智能体",theme="soft") as demo:   
    with gr.Row():
        htmlstr="""
        <p style='text-align: center;font-size: 18px;font-weight: bold;'>
        智能体 </p>
        """
        gr.HTML(htmlstr)
    with gr.Row():
        query_input = gr.Textbox(label="提示词",placeholder="发消息，提问题")
            
    with gr.Row():
        query_button = gr.Button("发送消息")   
       
    with gr.Row():
        query_output = gr.Markdown(label="回复结果")
    
    # 点击按钮时触发
    query_button.click(agent_fn, inputs=[query_input], outputs=[query_output])
    

    with gr.Row():
            gr.Markdown("""
                        <p style='text-align: center;'>
                        Copyright © 2025 By [DEMO] All rights reserved.
                        </p>""")
demo.launch(
    server_name="0.0.0.0",
    server_port=8088,
    inbrowser=True,
    show_api=False,
)