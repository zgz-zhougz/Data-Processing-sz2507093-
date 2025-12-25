import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataProcessingTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaned_df = None
        self.visualizations = []
        self.stats_results = {}
        self.qar_core_fields = {}
        
        # QAR字段映射配置 - 支持用户自定义字段名
        self.field_mapping = {
            "timestamp": "Time",           # 时间戳字段
            "flight_phase": "FLT_PHASE",   # 飞行阶段字段
            
            # 发动机参数映射（支持左右发）
            "n1_speed": ["LOCAL_N1_L", "LOCAL_N1_R"],
            "n2_speed": ["LOCAL_N2_SENSOR_L", "LOCAL_N2_SENSOR_R"],
            "egt": ["SEL_EGT_L", "SEL_EGT_R"],
            
            # 飞行参数映射
            "mach": ["CALC_MACH_NUM_L", "CALC_MACH_NUM_R"],
            "altitude": ["CALCULATED_ALT_L", "CALCULATED_ALT_R"],
            "temperature": ["AMBIENT_TMP_L", "AMBIENT_TMP_R"]
        }
        
        # 飞行阶段数字映射
        self.phase_mapping = {
            0: "起飞前",
            1: "起飞Takeoff", 
            2: "爬升CLB",
            3: "巡航CRZ",
            4: "下降DES",
            5: "进近APP",
            6: "盘旋/复飞GoAround",
            7: "结束Done"
        }
        
        # 大文件处理配置
        self.chunk_size = 50000  # 分块大小
        self.max_file_size = 300 * 1024 * 1024  # 300MB限制

    def parse_qar_data(self, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """解析QAR数据（支持CSV格式，自动处理字段映射和大文件）"""
        import os
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            print(f"⚠️  文件大小为 {file_size/1024/1024:.1f}MB，超过300MB限制，将使用分块处理")
            return self._parse_large_qar_data(file_path, file_type)
        
        if file_type == "csv":
            # 标准CSV格式QAR数据解析
            df = pd.read_csv(file_path)
            # 应用字段映射
            df = self._apply_field_mapping(df)
            # 自动识别QAR核心字段（时间戳、飞行阶段等）
            self._detect_qar_fields(df)
            return df
        elif file_type == "bin":
            # 二进制QAR数据解析（参考民航规范）
            import struct
            with open(file_path, "rb") as f:
                data = f.read()
            # 假设二进制格式为：头部（8字节时间戳）+ 参数块（每个参数4字节浮点数）
            # 具体解析逻辑需根据民航QAR格式规范调整
            timestamps = []
            params = []
            for i in range(0, len(data), 8 + 4*len(self.expected_params)):
                timestamp = struct.unpack("d", data[i:i+8])[0]
                param_vals = struct.unpack(f"{len(self.expected_params)}f", data[i+8:i+8+4*len(self.expected_params)])
                timestamps.append(timestamp)
                params.append(param_vals)
            df = pd.DataFrame(params, columns=self.expected_params)
            df["timestamp"] = pd.to_datetime(timestamps, unit="s")
            self.qar_core_fields = {
                "timestamp": "timestamp",
                "flight_phase": "flight_phase",  # 假设解析后包含该字段
                # 其他核心字段映射...
            }
            return df
        else:
            raise ValueError("不支持的QAR文件格式，仅支持csv和bin")
    
    def _parse_large_qar_data(self, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """分块处理大文件"""
        if file_type != "csv":
            raise ValueError("大文件处理仅支持CSV格式")
        
        print(f"🔄  开始分块读取大文件，每块 {self.chunk_size} 行...")
        chunks = []
        
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=self.chunk_size)):
            print(f"   处理第 {i+1} 块...")
            # 应用字段映射
            chunk = self._apply_field_mapping(chunk)
            chunks.append(chunk)
            
            # 如果已经处理了足够多的块用于测试，可以提前停止（生产环境去掉）
            if i >= 2:  # 只处理前3块用于测试，实际应该处理全部
                print(f"   ⚠️  测试模式：只处理前3块")
                break
        
        if not chunks:
            raise ValueError("未读取到任何数据块")
        
        # 合并所有块
        df = pd.concat(chunks, ignore_index=True)
        print(f"✅  大文件处理完成，总行数: {len(df)}")
        
        # 自动识别字段
        self._detect_qar_fields(df)
        return df
    
    def _apply_field_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用字段名映射，将QAR缩写映射到程序期望的字段名"""
        mapped_df = df.copy()
        
        # 时间戳映射
        if self.field_mapping["timestamp"] in df.columns:
            mapped_df = mapped_df.rename(columns={self.field_mapping["timestamp"]: "Time"})
        
        # 飞行阶段映射
        if self.field_mapping["flight_phase"] in df.columns:
            mapped_df = mapped_df.rename(columns={self.field_mapping["flight_phase"]: "FLT_PHASE"})
            # 将数字阶段映射为可读名称
            if "FLT_PHASE" in mapped_df.columns:
                mapped_df["FLT_PHASE"] = mapped_df["FLT_PHASE"].map(self.phase_mapping)
        
        # 发动机参数映射（支持左右发）
        for standard_name, qar_names in self.field_mapping.items():
            if standard_name in ["n1_speed", "n2_speed", "egt"]:
                for qar_name in qar_names:
                    if qar_name in df.columns:
                        # 保留原始列，同时添加标准名称列（用于后续分析）
                        if standard_name not in mapped_df.columns:
                            mapped_df[standard_name] = mapped_df[qar_name]
        
        # 飞行参数映射
        for standard_name, qar_names in self.field_mapping.items():
            if standard_name in ["mach", "altitude", "temperature"]:
                for qar_name in qar_names:
                    if qar_name in df.columns:
                        if standard_name not in mapped_df.columns:
                            mapped_df[standard_name] = mapped_df[qar_name]
        
        return mapped_df
    
    def _detect_qar_fields(self, df: pd.DataFrame):
        """自动检测并记录QAR核心字段"""
        self.qar_core_fields = {
            "timestamp": "Time" if "Time" in df.columns else None,
            "flight_phase": "FLT_PHASE" if "FLT_PHASE" in df.columns else None,
            "engine_params": [col for col in df.columns if col in ["n1_speed", "n2_speed", "egt"]],
            "flight_params": [col for col in df.columns if col in ["mach", "altitude", "temperature"]]
        }
        
        # 如果标准字段不存在，尝试从原始字段中识别
        if not self.qar_core_fields["engine_params"]:
            self.qar_core_fields["engine_params"] = [col for col in df.columns if any(p in col.lower() for p in ["local_n1", "local_n2", "sel_egt"])]
        
        if not self.qar_core_fields["flight_params"]:
            self.qar_core_fields["flight_params"] = [col for col in df.columns if any(p in col.lower() for p in ["calc_mach", "calculated_alt", "ambient_tmp"])]
        
        print(f"🔍  检测到QAR核心字段:")
        print(f"   时间戳: {self.qar_core_fields['timestamp']}")
        print(f"   飞行阶段: {self.qar_core_fields['flight_phase']}")
        print(f"   发动机参数: {self.qar_core_fields['engine_params']}")
        print(f"   飞行参数: {self.qar_core_fields['flight_params']}")
    
    def get_parameter_categories(self, df: pd.DataFrame) -> dict:
        """获取参数分类，用于筛选界面"""
        categories = {
            "核心参数": [],
            "发动机参数": [],
            "飞行参数": [],
            "系统参数": [],
            "环境参数": [],
            "其他参数": []
        }
        
        # 核心参数（必须保留）
        core_params = ["Time", "FLT_PHASE"]
        for param in core_params:
            if param in df.columns:
                categories["核心参数"].append(param)
        
        # 发动机参数
        engine_keywords = ["n1", "n2", "egt", "thrust", "fuel", "engine", "LOCAL_N1", "LOCAL_N2", "SEL_EGT"]
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in engine_keywords):
                categories["发动机参数"].append(col)
        
        # 飞行参数
        flight_keywords = ["altitude", "speed", "mach", "angle", "pitch", "roll", "yaw", "vert_speed", 
                          "CALCULATED_ALT", "CALC_MACH", "airspeed", "velocity"]
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in flight_keywords):
                categories["飞行参数"].append(col)
        
        # 系统参数
        system_keywords = ["hydraulic", "electric", "pressure", "voltage", "current", "pump", "valve", "switch"]
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in system_keywords) and col not in categories["发动机参数"] + categories["飞行参数"]:
                categories["系统参数"].append(col)
        
        # 环境参数
        env_keywords = ["ambient", "temperature", "pressure", "wind", "weather", "tmp", "TMP"]
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in env_keywords) and col not in categories["飞行参数"]:
                categories["环境参数"].append(col)
        
        # 其他参数（未分类的）
        all_categorized = set()
        for cat_list in categories.values():
            all_categorized.update(cat_list)
        
        for col in df.columns:
            if col not in all_categorized:
                categories["其他参数"].append(col)
        
        # 移除空的分类
        categories = {k: v for k, v in categories.items() if v}
        
        return categories
    
    def filter_parameters(self, df: pd.DataFrame, keep_params: list) -> pd.DataFrame:
        """根据选择的参数列表筛选数据"""
        if not keep_params:
            print("⚠️  未选择任何参数，返回原始数据")
            return df
        
        # 确保核心参数被保留
        required_params = ["Time", "FLT_PHASE"]
        for param in required_params:
            if param in df.columns and param not in keep_params:
                keep_params.append(param)
        
        # 检查选择的参数是否存在于数据中
        available_params = [p for p in keep_params if p in df.columns]
        missing_params = [p for p in keep_params if p not in df.columns]
        
        if missing_params:
            print(f"⚠️  以下参数不存在，将被忽略: {missing_params}")
        
        print(f"📊 筛选后保留 {len(available_params)} 个参数: {available_params}")
        return df[available_params]
    
    def get_flight_phases_summary(self, df: pd.DataFrame) -> dict:
        """获取飞行阶段数据分布摘要"""
        if "FLT_PHASE" not in df.columns:
            return {}
        
        phase_counts = df["FLT_PHASE"].value_counts().to_dict()
        total_rows = len(df)
        
        summary = {}
        for phase, count in phase_counts.items():
            percentage = (count / total_rows) * 100
            summary[phase] = {
                "rows": count,
                "percentage": round(percentage, 2)
            }
        
        return summary
    
    def filter_flight_phases(self, df: pd.DataFrame, selected_phases: list) -> pd.DataFrame:
        """根据选择的飞行阶段筛选数据"""
        if not selected_phases:
            print("⚠️  未选择任何飞行阶段，返回原始数据")
            return df
        
        if "FLT_PHASE" not in df.columns:
            print("⚠️  数据中不存在飞行阶段列，返回原始数据")
            return df
        
        # 应用筛选
        filtered_df = df[df["FLT_PHASE"].isin(selected_phases)]
        
        original_rows = len(df)
        filtered_rows = len(filtered_df)
        reduction = ((original_rows - filtered_rows) / original_rows) * 100
        
        print(f"✈️  飞行阶段筛选: 保留 {len(selected_phases)} 个阶段")
        print(f"   原始数据: {original_rows} 行")
        print(f"   筛选后: {filtered_rows} 行")
        print(f"   数据减少: {reduction:.1f}%")
        
        return filtered_df
    
    def estimate_report_tokens(self, df: pd.DataFrame, eda_results: dict, stats_results: dict) -> int:
        """估算生成报告所需的token数量"""
        # 基础token（提示词模板）
        base_tokens = 200
        
        # 数据摘要token
        data_tokens = len(df) * 2  # 每行数据约2个token（摘要）
        if data_tokens > 500:
            data_tokens = 500  # 限制最大值
        
        # EDA结果token
        eda_tokens = 0
        if eda_results:
            eda_text = str(eda_results)
            eda_tokens = len(eda_text) // 4  # 粗略估算
        
        # 统计结果token
        stats_tokens = 0
        if stats_results:
            stats_text = str(stats_results)
            stats_tokens = len(stats_text) // 4  # 粗略估算
        
        # 总token估算
        total_tokens = base_tokens + data_tokens + eda_tokens + stats_tokens
        
        print(f"📝 Token估算:")
        print(f"   基础提示词: {base_tokens}")
        print(f"   数据摘要: {data_tokens}")
        print(f"   EDA结果: {eda_tokens}")
        print(f"   统计结果: {stats_tokens}")
        print(f"   预估总计: {total_tokens} tokens")
        
        return total_tokens
    
    def validate_token_limit(self, total_tokens: int, limit: int = 32768) -> tuple:
        """验证token数量是否超限"""
        safe_limit = limit * 0.8  # 使用80%作为安全线
        is_safe = total_tokens <= safe_limit
        margin = limit - total_tokens
        
        return is_safe, margin, safe_limit
        
    # 1. 自动数据清洗
    def clean_data(self) -> dict:
        log = []
        log.append(f"原始QAR数据规模: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        # 确保qar_core_fields已初始化
        if not self.qar_core_fields:
            self._detect_qar_fields(self.df)

        # 1. 飞行阶段过滤（保留有效阶段）
        if self.qar_core_fields.get("flight_phase"):
            valid_phases = ["起飞Takeoff", "巡航CRZ", "进近APP", "起飞前", "下降DES", "盘旋/复飞GoAround"]  # 有效飞行阶段
            phase_col = self.qar_core_fields["flight_phase"]
            if phase_col in self.df.columns:
                # 检查是否包含有效阶段
                valid_mask = self.df[phase_col].isin(valid_phases)
                if valid_mask.any():
                    invalid_count = (~valid_mask).sum()
                    if invalid_count > 0:
                        self.df = self.df[valid_mask]
                        log.append(f"过滤无效飞行阶段数据：{invalid_count} 行（保留：{valid_phases}）")
                else:
                    log.append(f"⚠️  飞行阶段列 {phase_col} 不包含有效阶段值，跳过过滤")
            else:
                log.append(f"⚠️  飞行阶段列 {phase_col} 不存在，跳过过滤")

        # 2. 重复值处理（时间戳+关键参数联合去重）
        if self.qar_core_fields.get("timestamp") and self.qar_core_fields["timestamp"] in self.df.columns:
            dup_cols = [self.qar_core_fields["timestamp"]]
            # 只添加存在的引擎参数
            for param in self.qar_core_fields["engine_params"][:3]:
                if param in self.df.columns:
                    dup_cols.append(param)
            
            if len(dup_cols) > 1:
                dup_count = self.df.duplicated(subset=dup_cols).sum()
                if dup_count > 0:
                    self.df = self.df.drop_duplicates(subset=dup_cols)
                    log.append(f"删除重复时间戳记录：{dup_count} 行")
            else:
                log.append("⚠️  没有找到有效的去重列，跳过重复值处理")
        else:
            log.append("⚠️  没有找到时间戳列，跳过重复值处理")

        # 3. 缺失值处理（区分关键参数和非关键参数）
        missing_cols = self.df.isnull().sum()[self.df.isnull().sum() > 0].index
        for col in missing_cols:
            if col in self.qar_core_fields.get("engine_params", []):
                # 发动机关键参数：用前5秒滑动平均填充
                if hasattr(self.df[col], 'rolling'):
                    self.df[col] = self.df[col].fillna(self.df[col].rolling(window=5, min_periods=1).mean())
                    log.append(f"发动机参数[{col}]缺失值填充：滑动平均（窗口5秒）")
                else:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    log.append(f"发动机参数[{col}]缺失值填充: 中位数({median_val})")
            else:
                # 非关键参数：保留原始填充逻辑
                if self.df[col].dtype == 'object':
                    mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "未知"
                    self.df[col] = self.df[col].fillna(mode_val)
                    log.append(f"分类变量[{col}]缺失值填充: 众数({mode_val})")
                else:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    log.append(f"数值变量[{col}]缺失值填充: 中位数({median_val})")

        # 4. 异常值处理（替换为QAR行业阈值）
        # 发动机参数阈值（示例）
        engine_thresholds = {
            "n1_speed": (20.0, 100.0),  # N1转速正常范围20%-100%
            "n2_speed": (50.0, 105.0),  # N2转速正常范围50%-105%
            "fuel_flow": (0.0, 5000.0)  # 燃油流量正常范围0-5000kg/h
        }
        # 飞行参数阈值（示例）
        flight_thresholds = {
            "altitude": (-100, 15000),  # 高度正常范围-100至15000米
            "airspeed": (0, 600)        # 空速正常范围0-600km/h
        }
        
        # 应用阈值过滤 - 发动机参数
        for col in self.qar_core_fields.get("engine_params", []):
            if col in engine_thresholds and col in self.df.columns:
                lower, upper = engine_thresholds[col]
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                if not outliers.empty:
                    self.df.loc[self.df[col] < lower, col] = lower
                    self.df.loc[self.df[col] > upper, col] = upper
                    log.append(f"发动机参数[{col}]异常值处理：{len(outliers)} 个值替换为行业阈值({lower}-{upper})")
        
        # 应用阈值过滤 - 飞行参数
        for col in self.qar_core_fields.get("flight_params", []):
            if col in flight_thresholds and col in self.df.columns:
                lower, upper = flight_thresholds[col]
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                if not outliers.empty:
                    self.df.loc[self.df[col] < lower, col] = lower
                    self.df.loc[self.df[col] > upper, col] = upper
                    log.append(f"飞行参数[{col}]异常值处理：{len(outliers)} 个值替换为行业阈值({lower}-{upper})")

        self.cleaned_df = self.df
        log.append(f"清洗后QAR数据规模: {self.cleaned_df.shape[0]} 行 × {self.cleaned_df.shape[1]} 列")
        return {"清洗日志": log, "清洗后数据": self.cleaned_df}

    # 2. 探索性数据分析
    def eda_analysis(self) -> dict:
        if self.cleaned_df is None:
            self.clean_data()
        
        # 检查数据是否为空
        if self.cleaned_df is None or self.cleaned_df.empty:
            print("⚠️  数据为空，无法进行EDA分析")
            return {
                "数值变量描述统计": {},
                "分类变量分布": {},
                "数值变量相关性": None,
                "QAR时间序列滑动窗口统计": {},
                "QAR飞行阶段分组统计": {}
            }
        
        # 基础统计保持不变
        numeric_stats = self.cleaned_df.describe().round(2)
        cat_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns
        cat_dist = {col: self.cleaned_df[col].value_counts().to_dict() for col in cat_cols}
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.cleaned_df[num_cols].corr().round(2) if len(num_cols)>=2 else None

        # QAR专属分析：时间序列滑动窗口统计
        time_series_stats = {}
        if self.qar_core_fields.get("timestamp") and self.qar_core_fields["timestamp"] in self.cleaned_df.columns:
            # 按时间戳排序
            self.cleaned_df = self.cleaned_df.sort_values(by=self.qar_core_fields["timestamp"])
            # 10秒滑动窗口统计（关键参数）
            window_params = []
            if self.qar_core_fields.get("engine_params"):
                window_params.extend(self.qar_core_fields["engine_params"])
            if self.qar_core_fields.get("flight_params"):
                window_params.extend(self.qar_core_fields["flight_params"][:2])
            
            for col in window_params:
                if col in self.cleaned_df.columns:
                    time_series_stats[f"{col}_10s_mean"] = self.cleaned_df[col].rolling(window=10, min_periods=1).mean().describe().round(2)
                    time_series_stats[f"{col}_10s_std"] = self.cleaned_df[col].rolling(window=10, min_periods=1).std().describe().round(2)

        # QAR专属分析：飞行阶段分组统计
        phase_stats = {}
        if self.qar_core_fields.get("flight_phase") and self.qar_core_fields["flight_phase"] in self.cleaned_df.columns:
            phase_col = self.qar_core_fields["flight_phase"]
            if window_params:
                valid_params = [col for col in window_params if col in self.cleaned_df.columns]
                if valid_params:
                    phase_stats = self.cleaned_df.groupby(phase_col)[valid_params].agg(["mean", "std", "max"]).round(2).to_dict()

        return {
            "数值变量描述统计": numeric_stats,
            "分类变量分布": cat_dist,
            "数值变量相关性": corr_matrix,
            "QAR时间序列滑动窗口统计": time_series_stats,  # 新增
            "QAR飞行阶段分组统计": phase_stats  # 新增
        }
    # 3. 自动化可视化
    def generate_visuals(self, save_dir: str = "visuals/") -> list:
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.visualizations = []

        if self.cleaned_df is None:
            self.clean_data()

        # 数值变量分布直方图
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            plt.figure(figsize=(12, 8))
            for i, col in enumerate(num_cols[:4]):  # 最多显示4个变量
                plt.subplot(2, 2, i+1)
                sns.histplot(self.cleaned_df[col], kde=True, bins=20)
                plt.title(f"{col} 分布")
            hist_path = f"{save_dir}/numeric_dist.png"
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.visualizations.append(("数值变量分布", hist_path))

        # 分类变量计数图
        cat_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            plt.figure(figsize=(12, 8))
            for i, col in enumerate(cat_cols[:4]):
                plt.subplot(2, 2, i+1)
                sns.countplot(x=col, data=self.cleaned_df)
                plt.title(f"{col} 分布")
                plt.xticks(rotation=45)
            count_path = f"{save_dir}/cat_count.png"
            plt.savefig(count_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.visualizations.append(("分类变量分布", count_path))

        # 相关性热力图
        if len(num_cols) >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.cleaned_df[num_cols].corr(), annot=True, cmap='coolwarm')
            plt.title("数值变量相关性热力图")
            corr_path = f"{save_dir}/corr_heatmap.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.visualizations.append(("相关性热力图", corr_path))
        
        # QAR时间序列图
        if self.qar_core_fields.get("timestamp") and self.qar_core_fields["timestamp"] in self.cleaned_df.columns:
            time_col = self.qar_core_fields["timestamp"]
            # 选择3个关键参数绘制时间序列
            plot_params = []
            if self.qar_core_fields.get("engine_params") and len(self.qar_core_fields["engine_params"]) >= 2:
                plot_params.extend(self.qar_core_fields["engine_params"][:2])
            if self.qar_core_fields.get("flight_params") and len(self.qar_core_fields["flight_params"]) >= 1:
                plot_params.extend(self.qar_core_fields["flight_params"][:1])
            
            if plot_params:
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(plot_params):
                    if col in self.cleaned_df.columns:
                        plt.subplot(len(plot_params), 1, i+1)
                        plt.plot(self.cleaned_df[time_col], self.cleaned_df[col], linewidth=0.8)
                        plt.title(f"{col} 随时间变化趋势")
                        plt.xticks(rotation=45)
                ts_path = f"{save_dir}/qar_time_series.png"
                plt.tight_layout()
                plt.savefig(ts_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.visualizations.append(("QAR参数时间序列趋势", ts_path))

        # QAR专属可视化：飞行阶段参数箱线图
        if self.qar_core_fields.get("flight_phase") and self.qar_core_fields["flight_phase"] in self.cleaned_df.columns:
            phase_col = self.qar_core_fields["flight_phase"]
            if self.qar_core_fields.get("engine_params") and len(self.qar_core_fields["engine_params"]) >= 2:
                plt.figure(figsize=(12, 8))
                for i, col in enumerate(self.qar_core_fields["engine_params"][:2]):
                    if col in self.cleaned_df.columns:
                        plt.subplot(2, 1, i+1)
                        sns.boxplot(x=phase_col, y=col, data=self.cleaned_df)
                        plt.title(f"{col} 在不同飞行阶段的分布")
                phase_path = f"{save_dir}/qar_phase_boxplot.png"
                plt.tight_layout()
                plt.savefig(phase_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.visualizations.append(("飞行阶段参数分布对比", phase_path))

        # QAR专属可视化：发动机参数相关性散点图
        if self.qar_core_fields.get("engine_params") and len(self.qar_core_fields["engine_params"]) >= 2:
            col1, col2 = self.qar_core_fields["engine_params"][0], self.qar_core_fields["engine_params"][1]
            if col1 in self.cleaned_df.columns and col2 in self.cleaned_df.columns:
                plt.figure(figsize=(10, 8))
                if self.qar_core_fields.get("flight_phase") and self.qar_core_fields["flight_phase"] in self.cleaned_df.columns:
                    sns.scatterplot(x=col1, y=col2, hue=self.qar_core_fields["flight_phase"], data=self.cleaned_df)
                else:
                    sns.scatterplot(x=col1, y=col2, data=self.cleaned_df)
                plt.title(f"{col1} 与 {col2} 的相关性（按飞行阶段分组）")
                corr_scatter_path = f"{save_dir}/qar_engine_corr.png"
                plt.savefig(corr_scatter_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.visualizations.append(("发动机参数相关性散点图", corr_scatter_path))

            
        return self.visualizations

    # 4. 统计检验
    def statistical_tests(self, target_col: str = None) -> dict:
        if self.cleaned_df is None:
            self.clean_data()
        
        # 检查数据是否为空
        if self.cleaned_df is None or self.cleaned_df.empty:
            print("⚠️  数据为空，无法进行统计检验")
            return {}

        results = {}
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        cat_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns

        # Pearson相关性检验
        if len(num_cols) >= 2:
            corr_res = {}
            for i, col1 in enumerate(num_cols):
                for j, col2 in enumerate(num_cols):
                    if i < j:
                        corr, p_val = stats.pearsonr(self.cleaned_df[col1], self.cleaned_df[col2])
                        corr_res[f"{col1} vs {col2}"] = {
                            "相关系数": round(corr, 3),
                            "p值": round(p_val, 5),
                            "显著性": "显著" if p_val < 0.05 else "不显著"
                        }
            results["Pearson相关性检验"] = corr_res

        # 目标变量相关检验
        if target_col and target_col in self.cleaned_df.columns:
            # 卡方检验（分类变量）
            if target_col in cat_cols:
                chi2_res = {}
                for col in cat_cols:
                    if col != target_col:
                        cont_table = pd.crosstab(self.cleaned_df[target_col], self.cleaned_df[col])
                        chi2, p_val, dof, _ = stats.chi2_contingency(cont_table)
                        chi2_res[f"{target_col} vs {col}"] = {
                            "卡方值": round(chi2, 3),
                            "p值": round(p_val, 5),
                            "显著性": "显著" if p_val < 0.05 else "不显著"
                        }
                results["卡方检验"] = chi2_res

                # ANOVA（分类目标 vs 数值特征）
                anova_res = {}
                for col in num_cols:
                    model = ols(f"{col} ~ C({target_col})", data=self.cleaned_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    f_val = anova_table['F'].iloc[0]
                    p_val = anova_table['PR(>F)'].iloc[0]
                    anova_res[f"{col} vs {target_col}"] = {
                        "F值": round(f_val, 3),
                        "p值": round(p_val, 5),
                        "显著性": "显著" if p_val < 0.05 else "不显著"
                    }
                results["方差分析(ANOVA)"] = anova_res
        qar_specific_tests = {}
        
        # 1. 发动机推力与飞行速度的相关性（效率评估）
        if self.qar_core_fields.get("engine_params") and self.qar_core_fields.get("flight_params"):
            engine_params_lower = [col.lower() for col in self.qar_core_fields["engine_params"]]
            flight_params_lower = [col.lower() for col in self.qar_core_fields["flight_params"]]
            
            if "thrust" in engine_params_lower and "airspeed" in flight_params_lower:
                thrust_col = next(col for col in self.qar_core_fields["engine_params"] if col.lower() == "thrust")
                speed_col = next(col for col in self.qar_core_fields["flight_params"] if col.lower() == "airspeed")
                
                if thrust_col in self.cleaned_df.columns and speed_col in self.cleaned_df.columns:
                    corr, p_val = stats.pearsonr(self.cleaned_df[thrust_col], self.cleaned_df[speed_col])
                    qar_specific_tests["推力-速度相关性"] = {
                        "相关系数": round(corr, 3),
                        "p值": round(p_val, 5),
                        "业务解读": "正相关显著（p<0.05）表明推力调节与速度控制匹配性好" if p_val<0.05 else "相关性不显著，需检查推力控制系统"
                    }
        
        # 2. 不同飞行阶段的参数差异检验（ANOVA）
        if self.qar_core_fields.get("flight_phase") and self.qar_core_fields["flight_phase"] in self.cleaned_df.columns:
            phase_col = self.qar_core_fields["flight_phase"]
            if self.qar_core_fields.get("engine_params"):
                for col in self.qar_core_fields["engine_params"]:
                    if col in self.cleaned_df.columns:
                        try:
                            model = ols(f"{col} ~ C({phase_col})", data=self.cleaned_df).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            f_val = anova_table['F'].iloc[0]
                            p_val = anova_table['PR(>F)'].iloc[0]
                            qar_specific_tests[f"{col}的飞行阶段差异"] = {
                                "F值": round(f_val, 3),
                                "p值": round(p_val, 5),
                                "业务解读": "不同阶段参数差异显著（p<0.05），符合正常飞行逻辑" if p_val<0.05 else "阶段参数差异不显著，可能存在传感器异常"
                            }
                        except Exception as e:
                            qar_specific_tests[f"{col}的飞行阶段差异"] = {
                                "错误": f"无法计算: {str(e)}"
                            }
        
        if qar_specific_tests:
            results["QAR专项检验"] = qar_specific_tests  # 加入结果
        self.stats_results = results
        return results
