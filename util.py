import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from sklearn.manifold import TSNE
from scipy.stats import f_oneway, mannwhitneyu
from sklearn.preprocessing import KBinsDiscretizer

class CreditRiskConformalPredictor:
    def __init__(self, alpha: float = 0.1, method: str = "standard", 
                 class_conditional: bool = False, model = None, scaler = None):
        self.alpha = alpha
        self.method = method
        self.class_conditional = class_conditional
        if model is None:
            print("model is None!")
        self.model = model
        self.scaler = scaler
        self.qhat = None
        self.qhat_class = None  # 类别条件分位数
    
    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        """        
        X_calib: 校准集特征, y_calib: 校准集标签
        """
        X_calib_scaled = self.scaler.transform(X_calib) if self.scaler is not None else X_calib
        calib_probs = self.model.predict_proba(X_calib_scaled)
        if self.class_conditional:
            self._calibrate_class_conditional(calib_probs, y_calib)
        else:
            self._calibrate_standard(calib_probs, y_calib)
    
    def _calibrate_standard(self, calib_probs: np.ndarray, y_calib: np.ndarray) -> None:
        n = len(y_calib)
        if self.method == "standard":
            # 校准函数
            calib_scores = 1 - calib_probs[np.arange(n), y_calib]
        elif self.method == "adaptive":
            calib_scores = self._compute_adaptive_scores(calib_probs, y_calib)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(calib_scores, q_level, method='higher')
    
    def _calibrate_class_conditional(self, calib_probs: np.ndarray, y_calib: np.ndarray) -> None:
        n = len(y_calib)
        classes = np.unique(y_calib)
        self.qhat_class = {}
        for c in classes:
            class_indices = np.where(y_calib == c)[0]
            class_probs = calib_probs[class_indices]
            if self.method == "standard":
                class_scores = 1 - class_probs[np.arange(len(class_indices)), y_calib[class_indices]]
            elif self.method == "adaptive":
                class_scores = self._compute_adaptive_scores(class_probs, y_calib[class_indices])
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            q_level = np.ceil((len(class_indices) + 1) * (1 - self.alpha)) / len(class_indices)
            self.qhat_class[c] = np.quantile(class_scores, q_level, method='higher')

    def _compute_adaptive_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n, K = probs.shape
        scores = np.zeros(n)
        for i in range(n):
            pi = np.argsort(probs[i])[::-1]
            srt = np.cumsum(probs[i, pi])
            k = np.where(pi == labels[i])[0][0]
            scores[i] = srt[k]
        return scores
        # n, K = probs.shape
        # cal_pi = probs.argsort(1)[:,::-1]
        # cal_srt = np.take_along_axis(probs,cal_pi,axis=1).cumsum(axis=1)
        # scores = np.take_along_axis(cal_srt,cal_pi.argsort(axis=1),axis=1)[range(n),labels]
        # return scores
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """预测"""
        X_test_scaled = self.scaler.transform(X_test) if self.scaler is not None else X_test
        test_probs = self.model.predict_proba(X_test_scaled)
        n, K = test_probs.shape
        prediction_sets = np.zeros((n, K), dtype=bool)
        if self.class_conditional:
            for c in self.qhat_class.keys():
                if self.method == "standard":
                    scores = 1 - test_probs[:, c]
                    prediction_sets[:, c] = scores <= self.qhat_class[c]
                elif self.method == "adaptive":
                    for i in range(n):
                        pi = np.argsort(test_probs[i])[::-1]
                        srt = np.cumsum(test_probs[i, pi])
                        k = np.where(pi == c)[0][0]
                        prediction_sets[i, c] = srt[k] <= self.qhat_class[c]
        else:
            if self.method == "standard":
                prediction_sets = test_probs >= (1 - self.qhat)
            elif self.method == "adaptive":
                for i in range(n):
                    pi = np.argsort(test_probs[i])[::-1]
                    srt = np.cumsum(test_probs[i, pi])
                    k_max = np.sum(srt <= self.qhat)
                    prediction_sets[i, pi[:k_max]] = True
        return prediction_sets

    def coverage_avg(self, X_test, y_test, X_cal, y_cal, iteration = 100, plot = False):
        X_combined = np.vstack([X_cal, X_test])
        y_combined = np.hstack([y_cal, y_test])
        print(X_combined.shape)
        print(y_combined.shape)
        probs = self.model.predict_proba(X_combined)
        n, k = X_combined.shape
        cal_len, _ = X_cal.shape
        if self.method == "standard":
            scores = 1 - probs[np.arange(n), y_combined]
        elif self.method == "adaptive":
            scores = self._compute_adaptive_scores(probs, y_combined)
        coverage_metric = []
        
        for i in range(iteration):
            np.random.shuffle(scores)
            calib_scores, val_scores = (scores[:cal_len],scores[cal_len:])
            qhat = np.quantile(calib_scores, np.ceil((cal_len+1)*(1-self.alpha)/cal_len), method='higher')
            coverage_metric.append((val_scores <= qhat).astype(float).mean())
        average_coverage = np.mean(coverage_metric)
        if plot:
            plt.hist(coverage_metric)
        return average_coverage

    def _compute_fsc(self, x_test, y_test, y_pred, prediction_sets, n_bins=10):
        n_samples, n_features = x_test.shape
        feature_fsc = []
        for feature_idx in range(n_features):
            feature_values = x_test[:, feature_idx]
            # 处理连续特征分箱
            if np.issubdtype(feature_values.dtype, np.number):
                kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                try:
                    # 尝试分箱，如果无法分箱（如所有值相同）则跳过该特征
                    feature_bins = kbd.fit_transform(feature_values.reshape(-1, 1)).flatten().astype(int)
                except:
                    continue
            else:
                # 离散特征直接转换为整数
                unique_vals = np.unique(feature_values)
                feature_bins = np.array([np.where(unique_vals == v)[0][0] for v in feature_values])
            unique_groups = np.unique(feature_bins)
            group_coverages = []
            # 计算每个分箱组的覆盖率
            for group in unique_groups:
                group_indices = np.where(feature_bins == group)[0]
                if len(group_indices) == 0:
                    continue  
                # 计算该组内的覆盖率
                y_group = y_test[group_indices]
                pred_sets_group = prediction_sets[group_indices]
                # 检查每个样本的真实标签是否在预测集中
                covered = np.array([pred_sets_group[i, y_group[i]] for i in range(len(group_indices))])
                coverage = covered.mean()
                group_coverages.append(coverage)
            if group_coverages:
                # 取该特征所有分箱组的最小覆盖率
                feature_fsc.append(np.min(group_coverages))
        return feature_fsc

    def _compute_ssc(self, x_test, y_test, y_pred, prediction_sets, n_bins=3):
        set_sizes = np.sum(prediction_sets, axis=1)
        # 等频分箱
        bins = np.quantile(set_sizes, np.linspace(0, 1, n_bins + 1))
        # 确保边界包含所有可能的值
        bins[0] = bins[0] - 0.1
        bins[-1] = bins[-1] + 0.1
        bin_labels = np.digitize(set_sizes, bins, right=False)
        unique_bins = np.unique(bin_labels)
        bin_coverages = []
        # 计算每个大小分箱的覆盖率
        for bin_label in unique_bins:
            bin_indices = np.where(bin_labels == bin_label)[0]
            if len(bin_indices) == 0:
                continue
            # 计算该箱内的覆盖率
            y_bin = y_test[bin_indices]
            pred_sets_bin = prediction_sets[bin_indices]
            covered = np.array([pred_sets_bin[i, y_bin[i]] for i in range(len(bin_indices))])
            coverage = covered.mean()
            bin_coverages.append(coverage)
        return bin_coverages
    
    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        prediction_sets = self.predict(X_test)
        y_pred = self.model.predict(X_test)
        n = len(y_test)
        # 计算覆盖度
        coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(n)])
        # 计算平均集合大小
        prediction_set_len = np.sum(prediction_sets, axis=1)
        set_size = np.mean(prediction_set_len)
        # 按类别计算覆盖度
        class_coverage = {}
        class_average_set_size = {}
        unique_classes = np.unique(y_test)
        for c in unique_classes:
            class_indices = np.where(y_test == c)[0]
            if len(class_indices) > 0:
                class_coverage[c] = np.mean([prediction_sets[i, c] for i in class_indices])
                class_average_set_size[c] = np.mean([prediction_set_len[i] for i in class_indices])

        # 计算 FSC 指标
        feature_FSC = self._compute_fsc(X_test, y_test, y_pred, prediction_sets)
        # 计算 SSC 指标
        ssc = self._compute_ssc(X_test, y_test, y_pred, prediction_sets)
        return {
            "coverage": float(coverage),
            "average_set_size": float(set_size),
            "class_coverage": class_coverage,
            "class_average_set_size": class_average_set_size,
            "avg_FSC":np.mean(feature_FSC),
            "min_FSC": min(feature_FSC),
            "SSC": ssc
        }
    
    def visualize_results(self, X_test: np.ndarray, y_test: np.ndarray, 
                          class_names: Optional[List[str]] = None) -> None:
        
        prediction_sets = self.predict(X_test) 
        predcition_labels = self.model.predict(X_test)

        # 绘制错误样本的示例
        plt.figure(figsize=(12, 8), dpi = 300)        
        sample_indices = np.where(predcition_labels != y_test)[0]
        sample_indices = np.random.choice(sample_indices, min(10,len(sample_indices)), replace=False)
        sample_pred_sets = prediction_sets[sample_indices]
        if class_names:
            yticklabels = [f"True label = {class_names[y_test[i]]}" for i in sample_indices]
            xticklabels = class_names.values()
        else:
            yticklabels = [f"True label = {y_test[i]}" for i in sample_indices]
            xticklabels = np.unique(y_test)
        sns.heatmap(sample_pred_sets, annot=True, cmap='Blues', 
                    yticklabels=yticklabels, xticklabels=xticklabels)
        plt.title('Prediction Sets for False predicted Samples')
        plt.tight_layout()
        plt.savefig("False predicted samples standard.jpg")
        
        # 绘制正确样本的示例
        plt.figure(figsize=(12, 8), dpi = 300)        
        sample_indices = np.where(predcition_labels != y_test)[0]
        sample_indices = np.random.choice(sample_indices, min(10,len(sample_indices)), replace=False)
        sample_pred_sets = prediction_sets[sample_indices]
        if class_names:
            yticklabels = [f"True label = {class_names[y_test[i]]}" for i in sample_indices]
            xticklabels = class_names.values()
        else:
            yticklabels = [f"True label = {y_test[i]}" for i in sample_indices]
            xticklabels = np.unique(y_test)
        sns.heatmap(sample_pred_sets, annot=True, cmap='Blues', 
                    yticklabels=yticklabels, xticklabels=xticklabels)
        plt.title('Prediction Sets for True predicted Samples')
        plt.tight_layout()
        # plt.show()
        plt.savefig("True predicted samples standard.jpg")

    def analyze_prediction_sets(self, X_test: np.ndarray, y_test: np.ndarray, 
                               class_names: Optional[List[str]] = None) -> None:
        prediction_sets = self.predict(X_test)
        n, K = prediction_sets.shape
        # 按预测类别分组，进行方差分析
        self._anova_by_predicted_class(X_test, prediction_sets)
        # 在t-SNE上标记各个类别
        self._visualize_tsne(X_test, prediction_sets, class_names)
        # 3. 比较True≥2与True=1组的自变量差异
        self._compare_multiple_vs_single_true(X_test, prediction_sets)
        # 4. 分析原模型分错样本在预测集中的覆盖情况
        self._analyze_misclassified_samples(X_test, y_test, prediction_sets, class_names)
    
    def _anova_by_predicted_class(self, X: np.ndarray, prediction_sets: np.ndarray, show = 30) -> None:
        """按预测类别分组，对每个特征进行方差分析"""
        print("\n=== 按预测类别分组的方差分析 ===")
        unique_sets = np.unique([tuple(row) for row in prediction_sets], axis=0)
        group_indices = {tuple(s): np.where(np.all(prediction_sets == s, axis=1))[0] for s in unique_sets}
        p_values = []
        for p in range(X.shape[1]):
            groups = []
            for s in group_indices:
                idx = group_indices[s]
                if len(idx) == 0:
                    continue
                groups.append(X[idx, p])
            if len(groups) < 2:
                p_values.append(1.0)
                continue
            f_stat, p_val = f_oneway(*groups)
            p_values.append(p_val)
            # if p_val < 0.05:
            #     print(f"特征 {p+1} 组间差异显著（p={p_val:.4f}）")
        plt.figure(figsize=(12, 5), dpi = 300)
        if show:
            show = min(show, X.shape[1])
            plt.bar(range(1, show + 1), p_values[0:show])
        else:
            plt.bar(range(1, X.shape[1]+1), p_values)
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level: 0.05')
        plt.xlabel('Feature Index')
        plt.ylabel('p-value')
        plt.title('F-one way analysis')
        plt.legend()
        # plt.show()
        plt.savefig("F - one way analysis.jpg")
    
    def _visualize_tsne(self, X: np.ndarray, prediction_sets: np.ndarray, 
                       class_names: Optional[List[str]] = None) -> None:
        # 降维
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        unique_sets = np.unique([tuple(row) for row in prediction_sets], axis=0)
        set_labels = {tuple(s): str(np.where(s)[0]) for s in unique_sets}
        set_colors = {tuple(s): i for i, s in enumerate(unique_sets)}
        plt.figure(figsize=(12, 12), dpi = 300)
        for s in unique_sets:
            idx = np.where(np.all(prediction_sets == s, axis=1))[0]
            label = set_labels[tuple(s)]
            if class_names and len(np.where(s)[0]) > 0:
                label = ", ".join([class_names[c] for c in np.where(s)[0]])
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                        label=label, alpha=0.6, s=50)
        plt.legend(title='Predicted Classes', loc='upper left')
        plt.title('t-SNE Visualization of Prediction Sets')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig("F - one way analysis.jpg")
    
    def _compare_multiple_vs_single_true(self, X: np.ndarray, prediction_sets: np.ndarray) -> None:
        # 分组
        set_sizes = np.sum(prediction_sets, axis=1)
        groupA_idx = np.where(set_sizes >= 2)[0]  # True≥2
        groupB_idx = np.where(set_sizes == 1)[0]  # True=1
        groupA = X[groupA_idx, :]
        groupB = X[groupB_idx, :]
        print(f"True≥2 样本数: {len(groupA)}, True=1 样本数: {len(groupB)}")
        significant_features = []
        p_values_list = []
        for p in range(X.shape[1]):
            a = groupA[:, p]
            b = groupB[:, p]
            _, p_val = mannwhitneyu(a, b)  # 非参数检验
            p_values_list.append(p_val)
            if p_val < 0.05:
                significant_features.append(p)
                print(f"特征 {p+1} 在两组间差异显著（p={p_val:.4f}）")
        np.random.choice(significant_features)
        if significant_features:
            plt.figure(figsize=(15, 5), dpi = 300)
            for i, p in enumerate(significant_features[:min(6, len(significant_features))]):
                plt.subplot(1, min(6, len(significant_features)), i+1)
                sns.boxplot(x=[0]*len(groupA[:, p]) + [1]*len(groupB[:, p]),
                           y=np.concatenate([groupA[:, p], groupB[:, p]]))
                plt.xticks([0, 1], ['True≥2', 'True=1'])
                plt.title(f'Feature {p+1}')
            plt.tight_layout()
            plt.show()

            
    def _analyze_misclassified_samples(self, X_test: np.ndarray, y_test: np.ndarray, 
                                      prediction_sets: np.ndarray, 
                                      class_names: Optional[List[str]] = None) -> None:
        print("\n=== 分析原模型分错样本在预测集中的覆盖情况 ===")
        # 获取原模型预测
        X_test_scaled = self.scaler.transform(X_test) if self.scaler is not None else X_test
        y_pred = self.model.predict(X_test_scaled)

        
        classified_idx = np.where(y_pred == y_test)[0]
        print(f"原模型正确分类样本数: {len(classified_idx)}/{len(y_test)}")
        coverage = []
        for i in classified_idx:
            true_label = y_test[i]
            is_covered = prediction_sets[i, true_label]
            coverage.append(is_covered)
        coverage_rate_true = np.mean(coverage)
        print(f"正确样本中，真实标签被预测集覆盖的比例: {coverage_rate_true:.2%}")
        
        # 找出分错的样本
        misclassified_idx = np.where(y_pred != y_test)[0]
        print(f"原模型错误分类样本数: {len(misclassified_idx)}/{len(y_test)}")
        if len(misclassified_idx) == 0:
            print("原模型没有错误分类的样本！")
            return
        
        # 检查分错样本的真实标签是否在预测集中
        coverage = []
        for i in misclassified_idx:
            true_label = y_test[i]
            is_covered = prediction_sets[i, true_label]
            coverage.append(is_covered)
        coverage_rate = np.mean(coverage)
        print(f"分错样本中，真实标签被预测集覆盖的比例: {coverage_rate:.2%}")
        plt.figure(figsize=(10, 6))
        plt.bar(['Covered', 'Not Covered'], 
                [np.sum(coverage), len(coverage) - np.sum(coverage)])
        plt.title('Coverage of True Labels in Prediction Sets for Misclassified Samples')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        covered_idx = misclassified_idx[np.where(coverage)[0]]
        not_covered_idx = misclassified_idx[np.where(~np.array(coverage))[0]]
        if len(covered_idx) > 0 and len(not_covered_idx) > 0:
            plt.figure(figsize=(15, 6))
            for p in range(min(6, X_test.shape[1])):  # 展示前6个特征
                plt.subplot(2, 3, p+1)
                sns.kdeplot(X_test[covered_idx, p], label='Covered')
                sns.kdeplot(X_test[not_covered_idx, p], label='Not Covered')
                plt.title(f'Feature {p+1}')
                plt.legend()
            plt.tight_layout()
            plt.show()

    

            