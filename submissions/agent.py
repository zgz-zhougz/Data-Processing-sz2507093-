import os
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from tools import DataProcessingTools
from typing import TypedDict, Optional, Dict, List
# 加载环境变量
load_dotenv()
class AnalysisState(TypedDict):
    df: pd.DataFrame
    cleaned_df: Optional[pd.DataFrame]  # 可选字段，初始为None
    clean_log: Dict
    eda_results: Dict
    visuals: List
    stats_results: Dict
    report: str

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


# ========== 2. 修正节点函数（返回字典，更新状态） ==========
def clean_data_node(state: AnalysisState) -> dict:
    """数据清洗节点：返回更新的状态字段"""
    tools = DataProcessingTools(state["df"])
    clean_res = tools.clean_data()
    return {
        "cleaned_df": clean_res["清洗后数据"],
        "clean_log": clean_res["清洗日志"]
    }

def eda_node(state: AnalysisState) -> dict:
    """EDA节点：返回更新的状态字段"""
    tools = DataProcessingTools(state["cleaned_df"])
    eda_res = tools.eda_analysis()
    return {"eda_results": eda_res}

def visualization_node(state: AnalysisState) -> dict:
    """可视化节点：返回更新的状态字段"""
    tools = DataProcessingTools(state["cleaned_df"])
    visuals = tools.generate_visuals()
    return {"visuals": visuals}

def stats_node(state: AnalysisState) -> dict:
    """统计检验节点：返回更新的状态字段"""
    target_col = None
    if state["cleaned_df"] is not None:
        for col in state["cleaned_df"].columns:
            if col.lower() in ["survived", "target", "label"]:
                target_col = col
                break
    tools = DataProcessingTools(state["cleaned_df"])
    stats_res = tools.statistical_tests(target_col)
    return {"stats_results": stats_res}

def report_node(state: AnalysisState) -> dict:
    """报告生成节点：返回更新的状态字段"""
    prompt = PromptTemplate(
        input_variables=["clean_log", "eda", "stats", "visuals"],
        template="""
        请基于以下数据分析结果生成一份专业的Markdown格式分析报告，结构如下：
        1. 数据概述（基本信息+清洗过程）
        2. 探索性分析（描述统计+变量分布）
        3. 统计检验结果（相关性+显著性检验）
        4. 核心洞见（基于统计结果的深度总结）
        5. 建议（基于洞见的实用建议）

        数据清洗日志：{clean_log}
        EDA结果：{eda}
        统计检验：{stats}
        可视化列表：{visuals}

        要求：
        - 洞见需基于统计显著性结果，避免表面描述
        - 语言专业但易懂，重点突出
        - 格式规范，使用Markdown标题/列表/表格
        """
    )

    prompt_str = prompt.format(
        clean_log=state["clean_log"],
        eda=state["eda_results"],
        stats=state["stats_results"],
        visuals=state["visuals"]
    )
    response = llm.invoke(prompt_str)
    return {"report": response.content}

# ========== 4. 构建LangGraph工作流（传入TypedDict作为state_schema） ==========
def build_analysis_graph():
    graph = StateGraph(state_schema=AnalysisState)
    
    # 添加节点：将"report"改为"generate_report"
    graph.add_node("clean", clean_data_node)
    graph.add_node("eda", eda_node)
    graph.add_node("visual", visualization_node)
    graph.add_node("stats", stats_node)
    graph.add_node("generate_report", report_node)  # 关键修改：节点名避免与状态键重复
    
    # 定义边：同步修改指向新节点名
    graph.add_edge("clean", "eda")
    graph.add_edge("eda", "visual")
    graph.add_edge("visual", "stats")
    graph.add_edge("stats", "generate_report")  # 关键修改：stats → generate_report
    graph.add_edge("generate_report", END)      # 关键修改：generate_report → END
    
    graph.set_entry_point("clean")
    return graph.compile()
# ========== 5. 执行分析主函数 ==========
def run_analysis(df: pd.DataFrame) -> str:
    # 初始化TypedDict状态（字典格式，符合TypedDict定义）
    initial_state: AnalysisState = {
        "df": df,
        "cleaned_df": None,
        "clean_log": {},
        "eda_results": {},
        "visuals": [],
        "stats_results": {},
        "report": ""
    }
    # 执行工作流
    graph = build_analysis_graph()
    final_state = graph.invoke(initial_state)
    # 返回生成的分析报告
    return final_state["report"]