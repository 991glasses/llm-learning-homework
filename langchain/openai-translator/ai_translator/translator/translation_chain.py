from langchain.chat_models import ChatOpenAI
from langchain.llms import ChatGLM
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils import LOG


class TranslationChain:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        glm_endpoint: str = "http://127.0.0.1:8000",
        verbose: bool = True,
    ):
        # 初始化 OpenAI 模型的 chain
        self.init_openai_chain(model_name, verbose)
        # 初始化 GLM 模型的 chain
        self.init_glm_chain(glm_endpoint, verbose)

    def init_openai_chain(self, model_name, verbose):
        # 创建小说风格的翻译 chain，将 temperature 设置为 0.3，保持一定的变化
        system_prompt = """You are a translation expert, proficient in various languages. \n
            Translate the following content from {source_language} to {target_language} using novel style."""
        self.openai_novel_style_chain = self.generate_openai_chain(
            system_prompt, 0.3, model_name, verbose
        )

        # 创建新闻稿风格的翻译 chain，将 temperature 设置为 0，以保持翻译结果的稳定性
        system_prompt = """You are a translation expert, proficient in various languages. \n
            Translate the following content from {source_language} to {target_language} using the style of a press release."""
        self.openai_press_style_chain = self.generate_openai_chain(
            system_prompt, 0, model_name, verbose
        )

        # 创建文艺作家风格的翻译 chain，将 temperature 设置为 0.7，让翻译内容多变
        system_prompt = """You are a translation expert, proficient in various languages. \n
            Translate the following content from {source_language} to {target_language} using the style of a literary writer."""
        self.openai_literary_style_chain = self.generate_openai_chain(
            system_prompt, 0.7, model_name, verbose
        )

        # 创建无风格的翻译 chain，将 temperature 设置为 0，以保持翻译结果的稳定性
        system_prompt = """You are a translation expert, proficient in various languages. \n
            Translates {source_language} to {target_language}."""
        self.openai_none_style_chain = self.generate_openai_chain(
            system_prompt, 0, model_name, verbose
        )

    def generate_openai_chain(
        self,
        system_promt,
        temperature,
        model_name,
        verbose,
    ) -> LLMChain:
        # 翻译任务指令始终由 System 角色承担
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_promt)

        # 待翻译文本由 Human 角色输入
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        chat = ChatOpenAI(
            model_name=model_name, temperature=temperature, verbose=verbose
        )
        return LLMChain(llm=chat, prompt=chat_prompt_template, verbose=verbose)

    def init_glm_chain(self, endpoint, verbose):
        # 创建小说风格的翻译 chain，将 temperature 设置为 3，保持一定的变化（GLM 的 temperature 范围是 0～10）
        prompt_str = """假设你是一位擅长各国语言的翻译专家，将下面的内容从{source_language}翻译成{target_language}，尽量使用小说文体的措辞风格：\n{text}"""
        self.glm_novel_style_chain = self.generate_glm_chain(
            prompt_str, 3, endpoint=endpoint, verbose=verbose
        )

        # 创建新闻稿风格的翻译 chain，将 temperature 设置为 0，以保持翻译结果的稳定性
        prompt_str = """假设你是一位擅长各国语言的翻译专家，将下面的内容从{source_language}翻译成{target_language}，尽量使用新闻稿的措辞风格：\n{text}"""
        self.glm_press_style_chain = self.generate_glm_chain(
            prompt_str, 0, endpoint=endpoint, verbose=verbose
        )

        # 创建文艺作家风格的翻译 chain，将 temperature 设置为 7，让翻译内容多变
        prompt_str = """假设你是一位擅长各国语言的翻译专家，将下面的内容从{source_language}翻译成{target_language}，尽量使用文艺作家的措辞风格：\n{text}"""
        self.glm_literary_style_chain = self.generate_glm_chain(
            prompt_str, 7, endpoint=endpoint, verbose=verbose
        )

        # 创建无风格的翻译 chain，将 temperature 设置为 0，以保持翻译结果的稳定性
        prompt_str = """假设你是一位擅长各国语言的翻译专家，将下面的内容从{source_language}翻译成{target_language}：\n{text}"""
        self.glm_none_style_chain = self.generate_glm_chain(
            prompt_str, 0, endpoint=endpoint, verbose=verbose
        )

    def generate_glm_chain(
        self,
        prompt_str,
        temperature,
        endpoint,
        verbose,
    ) -> LLMChain:
        # GLM 没有 system 角色，使用简单的 PromptTemplate 封装即可
        prompt = PromptTemplate.from_template(prompt_str)
        chat = ChatGLM(endpoint_url=endpoint, temperature=temperature, verbose=verbose)
        return LLMChain(llm=chat, prompt=prompt, verbose=verbose)

    def run(
        self,
        text,
        source_language,
        target_language,
        style,
        model: str = "OpenAI",
    ) -> (str, bool):
        result = ""
        try:
            if model == "OpenAI":
                # 使用 OpenAI 模型翻译
                result = self.run_openai(text, source_language, target_language, style)
            else:
                # 使用 GLM 模型翻译
                result = self.run_glm(text, source_language, target_language, style)

            LOG.debug(f"translation result: {result}")
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result, False

        return result, True

    def run_openai(
        self,
        text,
        source_language,
        target_language,
        style,
    ):
        if style == "小说":
            LOG.debug(f"openai novel style translation")
            return self.openai_novel_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        elif style == "新闻稿":
            LOG.debug(f"openai press style translation")
            return self.openai_press_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        elif style == "文艺作家":
            LOG.debug(f"openai literary style translation")
            return self.openai_literary_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        else:
            LOG.debug(f"openai none style translation")
            return self.openai_none_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )

    def run_glm(
        self,
        text,
        source_language,
        target_language,
        style,
    ):
        if style == "小说":
            LOG.debug(f"glm novel style translation")
            return self.glm_novel_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        elif style == "新闻稿":
            LOG.debug(f"glm press style translation")
            return self.glm_press_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        elif style == "文艺作家":
            LOG.debug(f"glm literary style translation")
            return self.glm_literary_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        else:
            LOG.debug(f"glm none style translation")
            return self.glm_none_style_chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
