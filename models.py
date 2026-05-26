import numpy as np
import pandas as pd
import scipy
import statistics
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor



class DataAnalysisModel:

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("数据集的格式必须是 pd.DataFrame")
        self.data = data.copy()
        self.models = {}  # 可用于保存多个模型的结果

    # 最小二乘法线性回归
    def linear_regression(self, y_col: str, x_cols: list, add_constant: bool = True):
        y = self.data[y_col]
        X = self.data[x_cols]
        if add_constant:
            X = sm.add_constant(X)  # statsmodels 默认不包含截距项，需要手动加
        model = sm.OLS(y, X).fit()
        self.models['lr'] = model
        return model

    # Ridge Regression | 惩罚项：Lamda*SUM(Beta^2) --> L2 惩罚 → 系数缩小但不为零 → 适合共线性数据，稳定预测
    def ridge_regression(self, y_col: str, x_cols: list, alpha=1.0, **kwargs):
        y = self.data[y_col]
        X = self.data[x_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=alpha, **kwargs)
        model.fit(X_scaled, y)

        results = {
            'model': model,
            'intercept': model.intercept_,
            'coefficients': pd.Series(model.coef_, index=x_cols),
            'r2_score': model.score(X_scaled, y)  # 获取 R2 操纵
        }

        self.models['ridge'] = results
        return results

    # 使用自举法来进行显著性检测的ridge回归
    def ridge_regression_boot(self, y_col: str, x_cols: list, n_bootstrap=500, alpha=1.0, **kwargs):
        y = self.data[y_col]
        X = self.data[x_cols]

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ridge 模型 (fit_intercept默认=True)
        model = Ridge(alpha=alpha, **kwargs)
        model.fit(X_scaled, y)

        # Bootstrap 自举法，抽样计算显著性
        boot_coefs = []
        n_samples = len(self.data)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X_scaled[indices], y[indices]

            boot_model = Ridge(alpha=alpha, **kwargs)
            boot_model.fit(X_boot, y_boot)
            boot_coefs.append(boot_model.coef_)

        boot_coefs = np.array(boot_coefs)

        # 标准误
        se = np.std(boot_coefs, axis=0)

        # 计算 t 统计量 (原始系数 / 标准误)
        t_stats = np.where(se != 0, model.coef_ / se, 0)

        # 计算双尾 p-value
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

        results = {
            'model': model,
            'intercept': model.intercept_,
            'coefficients': pd.Series(model.coef_, index=x_cols),
            'standard_errors': pd.Series(se, index=x_cols),
            't_values': pd.Series(t_stats, index=x_cols),
            'p_values': pd.Series(p_values, index=x_cols),
            'r2_score': model.score(X_scaled, y)
        }

        self.models['ridge_boot'] = results
        return results

    # Lasso Regression | 惩罚项: SUM(abs(Beta)) --> L1 惩罚 → 系数可精确到零 → 自动特征选择，产生稀疏解。(在多重共线特征明显的时候表现不好)
    def lasso_regression(self, y_col: str, x_cols: list, alpha=1.0, **kwargs):
        y = self.data[y_col].values
        X = self.data[x_cols].values

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. 拟合模型，max_iter设为10000以防不收敛
        model = Lasso(alpha=alpha, max_iter=10000, **kwargs)
        model.fit(X_scaled, y)

        results = {
            'model': model,
            'intercept': model.intercept_,
            'coefficients': pd.Series(model.coef_, index=x_cols),
            'r2_score': model.score(X_scaled, y),
            'selected_features': [col for col, coef in zip(x_cols, model.coef_) if coef != 0]
        }

        self.models['lasso_basic'] = results
        return results

    # 使用自举法来进行显著性检测的lasso回归：
    def lasso_regression_boot(self, y_col: str, x_cols: list, alpha=1.0, n_bootstrap=500, **kwargs):
        """
        Lasso回归 - 进阶版 (使用Bootstrap计算p-value)
        侧重于统计推断与特征显著性评估
        """
        y = self.data[y_col].values
        X = self.data[x_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        base_model = Lasso(alpha=alpha, max_iter=10000, **kwargs)
        base_model.fit(X_scaled, y)

        # Bootstrap 采样
        boot_coefs = []
        n_samples = len(self.data)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X_scaled[indices], y[indices]

            boot_model = Lasso(alpha=alpha, max_iter=10000, **kwargs)
            boot_model.fit(X_boot, y_boot)
            boot_coefs.append(boot_model.coef_)

        boot_coefs = np.array(boot_coefs)  # 形状: (n_bootstrap, n_features)

        # 计算每个特征在Bootstrap中被压缩为0的概率（稀疏稳定性）
        zero_probs = np.mean(boot_coefs == 0, axis=0)

        # 计算标准误 (各特征系数的标准差)
        se = np.std(boot_coefs, axis=0)

        # 计算 t 统计量 (若标准误为0则说明该特征在所有抽样中均被归零，t值设为0)
        t_stats = np.where(se != 0, base_model.coef_ / se, 0)

        # 计算双尾 p-value
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

        results = {
            'model': base_model,
            'intercept': base_model.intercept_,
            'coefficients': pd.Series(base_model.coef_, index=x_cols),
            'standard_errors': pd.Series(se, index=x_cols),
            't_values': pd.Series(t_stats, index=x_cols),
            'p_values': pd.Series(p_values, index=x_cols),
            'zero_probabilities': pd.Series(zero_probs, index=x_cols),
            'r2_score': base_model.score(X_scaled, y)
        }

        self.models['lasso_boot'] = results
        return results


    # ElasticNet | 惩罚项: r * SUM(abs(Beta)) + [(1-r)/2] * SUM(Beta^2) --> 混合L1，L2； 通过调整r来调整L1，L2的权重
    def elastic_net_basic(self, y_col: str, x_cols: list, alpha=1.0, l1_ratio=0.5, **kwargs):
        # 这里的L1的系数设定为0.5
        y = self.data[y_col].values
        X = self.data[x_cols].values

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 拟合模型
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, **kwargs)
        model.fit(X_scaled, y)

        results = {
            'model': model,
            'intercept': model.intercept_,
            'coefficients': pd.Series(model.coef_, index=x_cols),
            'r2_score': model.score(X_scaled, y),
            'selected_features': [col for col, coef in zip(x_cols, model.coef_) if coef != 0]
        }

        self.models['elasticnet'] = results
        return results

    # KNN临近算法：
    # KNN 分类（KNeighborsClassifier）：用于预测离散标签
    # KNN 回归（KNeighborsRegressor）：用于预测连续数值，其原理是取最近K个邻居的平均值或加权平均值
    def knn_analysis(self, y_col: str, x_cols: list, task_type: str = 'classification', n_neighbors: int = 5, weights: str = 'uniform', **kwargs):
        """
        - y_col: 目标变量列名
        - x_cols: 特征变量列名列表
        - task_type: 任务类型, 可选 'classification' (分类) 或 'regression' (回归)
        - n_neighbors: 邻居数量 K
        - weights: 权重类型, 'uniform' (均等权重) 或 'distance' (反比例距离权重)
        """
        y = self.data[y_col].values
        X = self.data[x_cols].values

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 初始化不同的模型
        if task_type == 'classification':
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, **kwargs)
            model.fit(X_scaled, y)

            score_name = 'accuracy_score'
            score_value = model.score(X_scaled, y)

        elif task_type == 'regression':
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, **kwargs)
            model.fit(X_scaled, y)

            score_name = 'r2_score'
            score_value = model.score(X_scaled, y)

        else:
            raise ValueError("task_type 必须是 'classification' 或 'regression'")

        results = {
            'model': model,
            'task_type': task_type,
            'n_neighbors': n_neighbors,
            'weights': weights,
            'features': x_cols,
            'target': y_col,
            score_name: score_value
        }

        self.models[f'knn_{task_type}'] = results
        return results

    # 查询已经储存的模型结果
    def get_model(self, name):
        return self.models.get(name, None)


