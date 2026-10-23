
from app.agent.rewrite import parse_rewrite_query, rewrite_query
from app.llm.fake import FakeChatModel


def test_rewrite():
    # 正常改写
    model = FakeChatModel('{"query":"var 变量提升 作用域"}')
    assert rewrite_query("var 有什么特点", "var", [], model) == "var 变量提升 作用域"

    # 模型输出无效，当前 query 不同于原问题
    model = FakeChatModel("bad output")
    assert rewrite_query("var 有什么特点", "var", [], model) == "var 有什么特点"

    # 模型输出无效，当前 query 已经等于原问题
    assert rewrite_query("var 有什么特点", "var 有什么特点", [], model) == "var 有什么特点 相关概念"

    # 模型返回相同查询，不能形成死循环
    model = FakeChatModel('{"query":"var"}')
    assert rewrite_query("var 有什么特点", "var", [], model) == "var 有什么特点"

    assert parse_rewrite_query("```text\nconst let 区别\n```") == "const let 区别"
