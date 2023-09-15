### AI 大模型应用开发实战营作业

该项目fork自 [彭老师的项目](https://github.com/DjangoPeng/openai-quickstart)，用于提交9.3日和9.10日的作业：

1. 在 openai-translator gradio 图形化界面基础上，支持风格化翻译，如：小说、新闻稿、作家风格等。
2. 基于 ChatGLM2-6B 实现图形化界面 的 openai-translator

gradio 界面的改动请查看 langchain/openai-translator/ai_translator/gradio_server.py 中的代码。

调用大模型的核心代码请查看 langchain/openai-translator/ai_translator/translator/translation_chain.py ，里面有注释说明。

ChatGLM2-6B 的模型我在台式机上运行起来了。集成显卡跑不动，还好内存是 64G，不用 cuda()，而是使用 float()，可以跑起来，就是速度比较慢。
