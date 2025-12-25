import streamlit as st
import pandas as pd
import os
from agent import run_analysis
from tools import DataProcessingTools

# 页面配置
st.set_page_config(
    page_title="大模型驱动的QAR分析代理",
    page_icon="📊",
    layout="wide"
)

# 页面标题
st.title("📊 大模型驱动的QAR分析代理")
st.divider()

# 侧边栏上传
with st.sidebar:
    st.header("数据上传")
    uploaded_file = st.file_uploader("上传QAR数据（CSV）", type=["csv", "bin"])
    # 新增文件类型选择
    file_type = st.radio("文件类型", ["csv", "bin"], index=0)
    use_sample = st.button("使用示例QAR数据")
    
    # 字段映射配置
    st.header("字段映射配置")
    st.info("如果程序无法识别你的QAR字段名，请在此配置")
    
    # 展开的配置选项
    with st.expander("自定义字段映射"):
        timestamp_field = st.text_input("时间戳字段名", value="Time", help="例如: Time, Timestamp")
        phase_field = st.text_input("飞行阶段字段名", value="FLT_PHASE", help="例如: FLT_PHASE, Phase")
        
        # 发动机参数配置
        st.subheader("发动机参数")
        n1_field = st.text_input("N1转速字段", value="LOCAL_N1_L,LOCAL_N1_R", help="左右发用逗号分隔")
        n2_field = st.text_input("N2转速字段", value="LOCAL_N2_SENSOR_L,LOCAL_N2_SENSOR_R", help="左右发用逗号分隔")
        egt_field = st.text_input("EGT字段", value="SEL_EGT_L,SEL_EGT_R", help="左右发用逗号分隔")
        
        # 飞行参数配置
        st.subheader("飞行参数")
        mach_field = st.text_input("马赫数字段", value="CALC_MACH_NUM_L,CALC_MACH_NUM_R", help="左右发用逗号分隔")
        alt_field = st.text_input("高度字段", value="CALCULATED_ALT_L,CALCULATED_ALT_R", help="左右发用逗号分隔")
        temp_field = st.text_input("温度字段", value="AMBIENT_TMP_L,AMBIENT_TMP_R", help="左右发用逗号分隔")
    
    # 飞行阶段映射配置
    with st.expander("飞行阶段映射"):
        st.info("QAR飞行阶段数字映射（默认）")
        st.text("0: 起飞前\n1: 起飞Takeoff\n2: 爬升CLB\n3: 巡航CRZ\n4: 下降DES\n5: 进近APP\n6: 盘旋/复飞GoAround\n7: 结束Done")
        
        # 如果需要自定义映射
        custom_phase = st.checkbox("自定义飞行阶段映射")
        if custom_phase:
            phase_mapping_input = st.text_area(
                "飞行阶段映射（格式: 数字:名称,每行一个）",
                value="0:起飞前\n1:起飞Takeoff\n2:爬升CLB\n3:巡航CRZ\n4:下降DES\n5:进近APP\n6:盘旋/复飞GoAround\n7:结束Done"
            )
    
    # 大文件处理配置
    with st.expander("大文件处理"):
        chunk_size = st.slider("分块大小（行数）", 10000, 100000, 50000, 10000, 
                              help="处理大文件时每块的行数，内存充足可调大")
        max_file_mb = st.slider("最大文件大小（MB）", 100, 1000, 300, 50, 
                               help="超过此大小自动启用分块处理")
    
    # 应用配置按钮
    if st.button("应用配置"):
        st.success("✅ 配置已更新")
    
    # 参数筛选配置
    st.header("📊 参数筛选")
    st.info("选择需要保留的参数，减少数据量，提高分析效率")
    
    # 筛选模式选择
    filter_mode = st.radio(
        "筛选模式",
        ["自动推荐", "手动选择", "保留全部"],
        help="自动推荐会基于数据类型推荐关键参数"
    )
    
    # 参数选择容器（初始隐藏）
    param_selection = None
    if filter_mode == "手动选择":
        with st.expander("手动选择参数"):
            st.info("数据加载后，可在此处选择需要保留的参数")
            param_selection = st.empty()  # 占位符，后续动态填充
    
    # 预设模板
    with st.expander("预设参数模板"):
        template = st.selectbox(
            "选择预设模板",
            ["自定义", "发动机监控", "飞行性能", "系统状态"]
        )
        if template != "自定义":
            st.write(f"**{template}模板包含:**")
            if template == "发动机监控":
                st.text("N1转速, N2转速, EGT, 燃油流量, 推力")
            elif template == "飞行性能":
                st.text("高度, 速度, 马赫数, 姿态角, 垂直速度")
            elif template == "系统状态":
                st.text("液压压力, 电气参数, 系统开关, 压力值")

# 主逻辑
if uploaded_file or use_sample:
    # 创建工具实例并应用配置
    tools = DataProcessingTools(pd.DataFrame())
    
    # 应用用户配置的字段映射
    if timestamp_field:
        tools.field_mapping["timestamp"] = timestamp_field
    if phase_field:
        tools.field_mapping["flight_phase"] = phase_field
    
    # 应用发动机参数映射
    if n1_field:
        tools.field_mapping["n1_speed"] = [x.strip() for x in n1_field.split(",")]
    if n2_field:
        tools.field_mapping["n2_speed"] = [x.strip() for x in n2_field.split(",")]
    if egt_field:
        tools.field_mapping["egt"] = [x.strip() for x in egt_field.split(",")]
    
    # 应用飞行参数映射
    if mach_field:
        tools.field_mapping["mach"] = [x.strip() for x in mach_field.split(",")]
    if alt_field:
        tools.field_mapping["altitude"] = [x.strip() for x in alt_field.split(",")]
    if temp_field:
        tools.field_mapping["temperature"] = [x.strip() for x in temp_field.split(",")]
    
    # 应用大文件处理配置
    tools.chunk_size = chunk_size
    tools.max_file_size = max_file_mb * 1024 * 1024
    
    # 应用自定义飞行阶段映射
    if custom_phase and phase_mapping_input:
        try:
            new_mapping = {}
            for line in phase_mapping_input.strip().split("\n"):
                if ":" in line:
                    num, name = line.split(":", 1)
                    new_mapping[int(num.strip())] = name.strip()
            tools.phase_mapping = new_mapping
        except Exception as e:
            st.warning(f"飞行阶段映射格式错误，使用默认配置: {e}")
    
    if use_sample:
        # 加载示例QAR数据（CSV格式）
        df = pd.read_csv("sample_qar_data.csv")
        st.success("✅ 加载示例QAR数据成功！")
    else:
        # 保存上传文件并解析
        with open(f"temp.{file_type}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 显示处理模式
        file_size = len(uploaded_file.getbuffer())
        if file_size > tools.max_file_size:
            st.info(f"🔄 大文件检测：{file_size/1024/1024:.1f}MB，将使用分块处理模式")
        
        df = tools.parse_qar_data(f"temp.{file_type}", file_type)
        st.success(f"✅ 加载QAR数据成功：{df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 显示QAR核心字段识别结果
    st.info(f"**字段识别结果**  \n时间戳: `{tools.qar_core_fields['timestamp']}`  \n飞行阶段: `{tools.qar_core_fields['flight_phase']}`  \n发动机参数: `{tools.qar_core_fields['engine_params']}`  \n飞行参数: `{tools.qar_core_fields['flight_params']}`")

    # 参数筛选功能
    selected_params = None
    
    # 获取参数分类信息
    param_categories = tools.get_parameter_categories(df)
    
    if filter_mode == "自动推荐":
        # 自动推荐关键参数
        recommended_params = []
        if "核心参数" in param_categories:
            recommended_params.extend(param_categories["核心参数"])
        if "发动机参数" in param_categories:
            recommended_params.extend(param_categories["发动机参数"][:6])  # 最多6个发动机参数
        if "飞行参数" in param_categories:
            recommended_params.extend(param_categories["飞行参数"][:4])  # 最多4个飞行参数
        
        st.info(f"🔄 自动推荐保留 {len(recommended_params)} 个关键参数")
        with st.expander("查看推荐参数列表"):
            st.write(recommended_params)
        
        # 应用筛选
        df = tools.filter_parameters(df, recommended_params)
        selected_params = recommended_params
        
    elif filter_mode == "手动选择":
        # 动态显示参数选择界面
        st.subheader("🔧 手动选择参数")
        
        # 显示参数分类
        for category, params in param_categories.items():
            with st.expander(f"{category} ({len(params)}个参数)"):
                if params:
                    # 使用多选框让用户选择
                    selected_from_category = st.multiselect(
                        f"选择{category}中的参数",
                        params,
                        default=params[:min(3, len(params))],  # 默认选前3个
                        key=f"select_{category}"
                    )
                    if selected_from_category:
                        if selected_params is None:
                            selected_params = []
                        selected_params.extend(selected_from_category)
        
        if selected_params:
            st.success(f"✅ 已选择 {len(selected_params)} 个参数")
            with st.expander("查看已选参数"):
                st.write(selected_params)
            
            # 应用筛选
            df = tools.filter_parameters(df, selected_params)
        else:
            st.warning("⚠️ 未选择任何参数，将保留全部数据")
    
    elif filter_mode == "保留全部":
        st.info("🔄 保留全部参数，不进行筛选")
        selected_params = list(df.columns)
    
    # 显示筛选后的数据信息
    if selected_params:
        st.info(f"📊 筛选后数据规模: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 飞行阶段筛选功能
    if "FLT_PHASE" in df.columns:
        st.subheader("✈️ 飞行阶段筛选")
        
        # 获取飞行阶段分布
        phase_summary = tools.get_flight_phases_summary(df)
        
        if phase_summary:
            # 显示各阶段数据量
            with st.expander("查看飞行阶段分布"):
                for phase, info in phase_summary.items():
                    st.write(f"**{phase}**: {info['rows']} 行 ({info['percentage']}%)")
            
            # 让用户选择飞行阶段
            available_phases = list(phase_summary.keys())
            selected_phases = st.multiselect(
                "选择要保留的飞行阶段",
                available_phases,
                default=available_phases,  # 默认全选
                help="选择需要保留的飞行阶段，未选择的阶段将被过滤掉"
            )
            
            if selected_phases and len(selected_phases) < len(available_phases):
                # 应用飞行阶段筛选
                df = tools.filter_flight_phases(df, selected_phases)
                st.success(f"✅ 已筛选保留 {len(selected_phases)} 个飞行阶段")
            else:
                st.info("🔄 保留所有飞行阶段")
        else:
            st.warning("⚠️ 无法获取飞行阶段信息")
    
    # Token验证和估算
    st.subheader("📝 Token数量验证")
    
    # 先执行EDA和统计检验，用于token估算
    with st.spinner("正在分析数据并估算token消耗..."):
        # 确保工具实例使用最新的筛选后数据
        tools.df = df.copy()
        tools.cleaned_df = None  # 重置，让clean_data重新处理
        eda_results = tools.eda_analysis()
        stats_results = tools.statistical_tests()
        
        # 估算token数量
        estimated_tokens = tools.estimate_report_tokens(df, eda_results, stats_results)
        
        # 验证是否超限
        is_safe, margin, safe_limit = tools.validate_token_limit(estimated_tokens)
        
        # 显示验证结果
        if is_safe:
            st.success(f"✅ Token数量安全: {estimated_tokens} / {safe_limit} (剩余: {margin})")
            st.info("💡 可以正常生成报告")
        else:
            st.error(f"❌ Token数量超限: {estimated_tokens} / {safe_limit}")
            st.warning(f"⚠️ 超出 {estimated_tokens - safe_limit} 个token")
            st.info("💡 请减少数据量或选择更少的飞行阶段")
            
            # 提供解决方案
            with st.expander("查看建议"):
                st.write("- 减少飞行阶段数量")
                st.write("- 减少保留的参数数量")
                st.write("- 使用分块处理大文件")
    
    # 显示原始数据预览
    st.subheader("原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    # 执行分析（只有token安全时才执行）
    if is_safe:
        with st.spinner("🔍 正在执行数据分析（清洗→EDA→统计检验→报告生成）..."):
            report = run_analysis(df)
    else:
        st.error("❌ Token数量超限，无法生成报告。请调整筛选条件。")
        st.stop()

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
