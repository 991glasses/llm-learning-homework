import sys
import os
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, LOG
from translator import PDFTranslator, TranslationConfig


def translation(input_file, source_language, target_language, style, model):
    LOG.debug(
        f"[翻译任务]\n源文件: {input_file.name}\n源语言: {source_language}\n目标语言: {target_language}\n风格：{style}\nmodel:{model}"
    )

    output_file_path = Translator.translate_pdf(
        input_file.name,
        source_language=source_language,
        target_language=target_language,
        style=style,
        model=model,
    )

    return output_file_path


def launch_gradio():
    iface = gr.Interface(
        fn=translation,
        title="OpenAI-Translator v2.0(PDF 电子书翻译工具)",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese"),
            gr.Dropdown(label="风格", choices=["小说", "新闻稿", "文艺作家"]),
            gr.Dropdown(label="AI大模型", choices=["OpenAI", "GLM"]),
        ],
        outputs=[gr.File(label="下载翻译文件")],
        allow_flagging="never",
    )

    # iface.launch(share=True, server_name="0.0.0.0")
    iface.launch(share=True)


def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config = TranslationConfig()
    config.initialize(args)
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator
    Translator = PDFTranslator(config.openai_model_name, config.glm_endpoint)


if __name__ == "__main__":
    # 初始化 translator
    initialize_translator()
    # 启动 Gradio 服务
    launch_gradio()
