import streamlit as st
import pandas as pd
import os
from agent import run_analysis
from tools import DataProcessingTools

# 页面配置
st.set_page_config(
    page_title="AI数据分析代理",
    page_icon="📊",
    layout="wide"
)

# 页面标题
st.title("📊 AI驱动的大规模数据分析代理")
st.divider()

# 侧边栏上传
with st.sidebar:
    st.header("数据上传")
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    use_sample = st.button("使用泰坦尼克示例数据")

# 主逻辑
if uploaded_file or use_sample:
    # 加载数据
    if use_sample:
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        st.success("✅ 加载泰坦尼克示例数据成功！")
    else:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ 加载数据成功：{df.shape[0]} 行 × {df.shape[1]} 列")

    # 显示原始数据预览
    st.subheader("原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    # 执行分析
    with st.spinner("🔍 正在执行数据分析（清洗→EDA→统计检验→报告生成）..."):
        report = run_analysis(df)

    # 显示报告
    st.subheader("📋 自动化分析报告")
    st.markdown(report, unsafe_allow_html=True)

    # 下载报告
    st.download_button(
        label="📥 下载Markdown报告",
        data=report,
        file_name="data_analysis_report.md",
        mime="text/markdown"
    )

    # 显示可视化
    st.subheader("📈 可视化结果")
    tools = DataProcessingTools(df)
    tools.clean_data()
    visuals = tools.generate_visuals()
    for viz_name, viz_path in visuals:
        st.subheader(viz_name)
        st.image(viz_path, use_container_width=True)

else:
    st.info("请上传CSV文件或点击使用示例数据开始分析")
    with st.expander("📖 示例数据说明"):
        st.markdown("""
        泰坦尼克数据集包含字段：
        - Survived: 是否幸存（0=否，1=是）
        - Pclass: 舱位等级（1/2/3等舱）
        - Sex: 性别
        - Age: 年龄
        - Fare: 票价
        - Embarked: 登船港口（S/C/Q）
        """)

# 页脚
st.divider()
st.caption("© 2025 AI数据分析代理 | 基于LangGraph + OpenAI构建")