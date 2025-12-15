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

    # 1. 自动数据清洗
    def clean_data(self) -> dict:
        log = []
        log.append(f"原始数据规模: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        # 处理重复值
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.df = self.df.drop_duplicates()
            log.append(f"删除重复值: {dup_count} 行")

        # 处理缺失值
        missing_cols = self.df.isnull().sum()[self.df.isnull().sum() > 0].index
        for col in missing_cols:
            if self.df[col].dtype == 'object':
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                log.append(f"分类变量[{col}]缺失值填充: 众数({mode_val})")
            else:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                log.append(f"数值变量[{col}]缺失值填充: 中位数({median_val})")

        # 处理数值异常值（IQR方法）
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1, Q3 = self.df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            bound = 1.5 * IQR
            outliers = self.df[(self.df[col] < Q1-bound) | (self.df[col] > Q3+bound)]
            if not outliers.empty:
                self.df.loc[self.df[col] < Q1-bound, col] = Q1-bound
                self.df.loc[self.df[col] > Q3+bound, col] = Q3+bound
                log.append(f"数值变量[{col}]异常值处理: {len(outliers)} 个值替换为IQR边界")

        # 数据类型优化
        for col in self.df.columns:
            if self.df[col].dtype == 'int64' and self.df[col].nunique() <= 10:
                self.df[col] = self.df[col].astype('category')
                log.append(f"变量[{col}]转换为分类类型")

        self.cleaned_df = self.df
        log.append(f"清洗后数据规模: {self.cleaned_df.shape[0]} 行 × {self.cleaned_df.shape[1]} 列")
        return {"清洗日志": log, "清洗后数据": self.cleaned_df}

    # 2. 探索性数据分析
    def eda_analysis(self) -> dict:
        if self.cleaned_df is None:
            self.clean_data()

        # 数值变量统计
        numeric_stats = self.cleaned_df.describe().round(2)
        
        # 分类变量分布
        cat_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns
        cat_dist = {col: self.cleaned_df[col].value_counts().to_dict() for col in cat_cols}
        
        # 相关性矩阵
        corr_matrix = None
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            corr_matrix = self.cleaned_df[num_cols].corr().round(2)

        return {
            "数值变量描述统计": numeric_stats,
            "分类变量分布": cat_dist,
            "数值变量相关性": corr_matrix
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

        self.stats_results = results
        return results