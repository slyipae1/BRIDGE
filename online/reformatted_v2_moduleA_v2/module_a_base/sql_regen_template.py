from __future__ import annotations

from dataclasses import dataclass

from ..vendor_sphinteract.prompts import sql_generation_v2


@dataclass
class ZeroShotSqlGenerationTemplate:
    suffix: str = sql_generation_v2

    def format(self, **kwargs) -> str:
        return self.suffix.format(**kwargs)


def build_zero_shot_sql_gen_template() -> ZeroShotSqlGenerationTemplate:
    return ZeroShotSqlGenerationTemplate()


def build_vectorstore_sql_gen_template(
    *,
    k_shot: int,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    persist_dir: str,
):
    from langchain.prompts.few_shot import FewShotPromptTemplate
    from langchain.prompts.prompt import PromptTemplate
    from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        model=embedding_model,
    )
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    feedback_example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=k_shot,
    )
    feedback_example_prompt = PromptTemplate(
        input_variables=["nl", "gold", "feedback"],
        template="\nExample Question: {nl}\nExample Feedback:{feedback}\nExample Answer: {gold}",
    )
    return FewShotPromptTemplate(
        example_selector=feedback_example_selector,
        example_prompt=feedback_example_prompt,
        suffix=sql_generation_v2,
        input_variables=["question", "schema", "sqls", "cqas", "metadata"],
    )


def load_sql_generation_template(
    *,
    k_shot: int,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    persist_dir: str,
    allow_empty_few_shot_fallback: bool,
    force_zero_shot: bool = False,
):
    if force_zero_shot:
        return build_zero_shot_sql_gen_template()
    try:
        return build_vectorstore_sql_gen_template(
            k_shot=k_shot,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            persist_dir=persist_dir,
        )
    except Exception:
        if allow_empty_few_shot_fallback:
            return build_zero_shot_sql_gen_template()
        raise
