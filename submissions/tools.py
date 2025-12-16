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

    def parse_qar_data(self, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """解析QAR数据（支持CSV格式）"""
        if file_type == "csv":
            # 标准CSV格式QAR数据解析
            df = pd.read_csv(file_path)
            # 自动识别QAR核心字段（时间戳、飞行阶段等）
            self.qar_core_fields = {
                "timestamp": next((col for col in df.columns if "time" in col.lower()), None),
                "flight_phase": next((col for col in df.columns if "phase" in col.lower()), None),
                "engine_params": [col for col in df.columns if any(p in col.lower() for p in ["n1", "n2", "thrust", "fuel"])],
                "flight_params": [col for col in df.columns if any(p in col.lower() for p in ["altitude", "speed", "angle"])]
            }
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
        
    # 1. 自动数据清洗
    def clean_data(self) -> dict:
        log = []
        log.append(f"原始QAR数据规模: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        # 1. 飞行阶段过滤（保留有效阶段）
        if self.qar_core_fields["flight_phase"]:
            valid_phases = ["takeoff", "cruise", "landing"]  # 有效飞行阶段
            phase_col = self.qar_core_fields["flight_phase"]
            invalid_count = self.df[~self.df[phase_col].isin(valid_phases)].shape[0]
            if invalid_count > 0:
                self.df = self.df[self.df[phase_col].isin(valid_phases)]
                log.append(f"过滤无效飞行阶段数据：{invalid_count} 行（保留：{valid_phases}）")

        # 2. 重复值处理（时间戳+关键参数联合去重）
        if self.qar_core_fields["timestamp"]:
            dup_cols = [self.qar_core_fields["timestamp"]] + self.qar_core_fields["engine_params"][:3]
            dup_count = self.df.duplicated(subset=dup_cols).sum()
            if dup_count > 0:
                self.df = self.df.drop_duplicates(subset=dup_cols)
                log.append(f"删除重复时间戳记录：{dup_count} 行")

        # 3. 缺失值处理（区分关键参数和非关键参数）
        missing_cols = self.df.isnull().sum()[self.df.isnull().sum() > 0].index
        for col in missing_cols:
            if col in self.qar_core_fields["engine_params"]:
                # 发动机关键参数：用前5秒滑动平均填充
                self.df[col] = self.df[col].fillna(self.df[col].rolling(window=5, min_periods=1).mean())
                log.append(f"发动机参数[{col}]缺失值填充：滑动平均（窗口5秒）")
            else:
                # 非关键参数：保留原始填充逻辑
                if self.df[col].dtype == 'object':
                    mode_val = self.df[col].mode()[0]
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
        
        # 应用阈值过滤
        for col in self.qar_core_fields["engine_params"]:
            if col in engine_thresholds:
                lower, upper = engine_thresholds[col]
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                if not outliers.empty:
                    self.df.loc[self.df[col] < lower, col] = lower
                    self.df.loc[self.df[col] > upper, col] = upper
                    log.append(f"发动机参数[{col}]异常值处理：{len(outliers)} 个值替换为行业阈值({lower}-{upper})")
        
        for col in self.qar_core_fields["flight_params"]:
            if col in flight_thresholds:
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
        
        # 基础统计保持不变
        numeric_stats = self.cleaned_df.describe().round(2)
        cat_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns
        cat_dist = {col: self.cleaned_df[col].value_counts().to_dict() for col in cat_cols}
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.cleaned_df[num_cols].corr().round(2) if len(num_cols)>=2 else None

        # QAR专属分析：时间序列滑动窗口统计
        time_series_stats = {}
        if self.qar_core_fields["timestamp"]:
            # 按时间戳排序
            self.cleaned_df = self.cleaned_df.sort_values(by=self.qar_core_fields["timestamp"])
            # 10秒滑动窗口统计（关键参数）
            window_params = self.qar_core_fields["engine_params"] + self.qar_core_fields["flight_params"][:2]
            for col in window_params:
                time_series_stats[f"{col}_10s_mean"] = self.cleaned_df[col].rolling(window=10, min_periods=1).mean().describe().round(2)
                time_series_stats[f"{col}_10s_std"] = self.cleaned_df[col].rolling(window=10, min_periods=1).std().describe().round(2)

        # QAR专属分析：飞行阶段分组统计
        phase_stats = {}
        if self.qar_core_fields["flight_phase"]:
            phase_col = self.qar_core_fields["flight_phase"]
            phase_stats = self.cleaned_df.groupby(phase_col)[window_params].agg(["mean", "std", "max"]).round(2).to_dict()

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
        if self.qar_core_fields["timestamp"]:
                time_col = self.qar_core_fields["timestamp"]
                # 选择3个关键参数绘制时间序列
                plot_params = self.qar_core_fields["engine_params"][:2] + self.qar_core_fields["flight_params"][:1]
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(plot_params):
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
        if self.qar_core_fields["flight_phase"]:
                phase_col = self.qar_core_fields["flight_phase"]
                plt.figure(figsize=(12, 8))
                for i, col in enumerate(self.qar_core_fields["engine_params"][:2]):
                    plt.subplot(2, 1, i+1)
                    sns.boxplot(x=phase_col, y=col, data=self.cleaned_df)
                    plt.title(f"{col} 在不同飞行阶段的分布")
                phase_path = f"{save_dir}/qar_phase_boxplot.png"
                plt.tight_layout()
                plt.savefig(phase_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.visualizations.append(("飞行阶段参数分布对比", phase_path))

            # QAR专属可视化：发动机参数相关性散点图
        if len(self.qar_core_fields["engine_params"]) >= 2:
                plt.figure(figsize=(10, 8))
                col1, col2 = self.qar_core_fields["engine_params"][0], self.qar_core_fields["engine_params"][1]
                sns.scatterplot(x=col1, y=col2, hue=self.qar_core_fields["flight_phase"], data=self.cleaned_df)
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
        if "thrust" in [col.lower() for col in self.qar_core_fields["engine_params"]] and \
        "airspeed" in [col.lower() for col in self.qar_core_fields["flight_params"]]:
                thrust_col = next(col for col in self.qar_core_fields["engine_params"] if "thrust" in col.lower())
                speed_col = next(col for col in self.qar_core_fields["flight_params"] if "airspeed" in col.lower())
                corr, p_val = stats.pearsonr(self.cleaned_df[thrust_col], self.cleaned_df[speed_col])
                qar_specific_tests["推力-速度相关性"] = {
                    "相关系数": round(corr, 3),
                    "p值": round(p_val, 5),
                    "业务解读": "正相关显著（p<0.05）表明推力调节与速度控制匹配性好" if p_val<0.05 else "相关性不显著，需检查推力控制系统"
                }
            
            # 2. 不同飞行阶段的参数差异检验（ANOVA）
        if self.qar_core_fields["flight_phase"]:
                phase_col = self.qar_core_fields["flight_phase"]
                for col in self.qar_core_fields["engine_params"]:
                    model = ols(f"{col} ~ C({phase_col})", data=self.cleaned_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    f_val = anova_table['F'].iloc[0]
                    p_val = anova_table['PR(>F)'].iloc[0]
                    qar_specific_tests[f"{col}的飞行阶段差异"] = {
                        "F值": round(f_val, 3),
                        "p值": round(p_val, 5),
                        "业务解读": "不同阶段参数差异显著（p<0.05），符合正常飞行逻辑" if p_val<0.05 else "阶段参数差异不显著，可能存在传感器异常"
                    }
            
        results["QAR专项检验"] = qar_specific_tests  # 加入结果
        self.stats_results = results
        return results

