import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.datasets import make_classification
import random
from typing import List, Tuple, Dict, Any, Optional
import pytest

class LStarFormula:
    """L*公式的表示类"""

    def __init__(self, formula_type: str, **kwargs):
        """
        formula_type: 'atom', 'and', 'or', 'sum', 'max', 'const'
        根据类型不同，kwargs包含不同参数
        """
        self.type = formula_type
        self.params = kwargs

    def evaluate(self, x: np.ndarray) -> float:
        """在输入x上计算公式的值"""
        if self.type == 'const':
            return self.params['value']

        elif self.type == 'atom':
            feature_idx = self.params['feature_idx']
            threshold = self.params['threshold']
            direction = self.params['direction']  # 'leq' 或 'gt'

            if direction == 'leq':
                return 1.0 if x[feature_idx] <= threshold else 0.0
            else:  # 'gt'
                return 1.0 if x[feature_idx] > threshold else 0.0

        elif self.type == 'and':
            left_val = self.params['left'].evaluate(x)
            right_val = self.params['right'].evaluate(x)
            return left_val * right_val  # 乘积语义

        elif self.type == 'or':
            left_val = self.params['left'].evaluate(x)
            right_val = self.params['right'].evaluate(x)
            return 1 - (1 - left_val) * (1 - right_val)  # 概率或语义

        elif self.type == 'sum':
            total = 0.0
            for formula in self.params['formulas']:
                total += formula.evaluate(x)
            return total

        elif self.type == 'max':
            values = [formula.evaluate(x) for formula in self.params['formulas']]
            return max(values) if values else 0.0

        elif self.type == 'weighted_sum':
            total = 0.0
            for weight, formula in zip(self.params['weights'], self.params['formulas']):
                total += weight * formula.evaluate(x)
            return total

        else:
            raise ValueError(f"未知的公式类型: {self.type}")

    def to_string(self, indent=0) -> str:
        """将公式转换为可读字符串"""
        prefix = "  " * indent

        if self.type == 'const':
            return f"{prefix}Const({self.params['value']})"

        elif self.type == 'atom':
            direction = "<=" if self.params['direction'] == 'leq' else ">"
            return f"{prefix}Atom(x{self.params['feature_idx']} {direction} {self.params['threshold']})"

        elif self.type == 'and':
            left_str = self.params['left'].to_string(indent + 1)
            right_str = self.params['right'].to_string(indent + 1)
            return f"{prefix}And:\n{left_str}\n{right_str}"

        elif self.type == 'or':
            left_str = self.params['left'].to_string(indent + 1)
            right_str = self.params['right'].to_string(indent + 1)
            return f"{prefix}Or:\n{left_str}\n{right_str}"

        elif self.type == 'sum':
            formulas_str = "\n".join([f.to_string(indent + 1) for f in self.params['formulas']])
            return f"{prefix}Sum:\n{formulas_str}"

        elif self.type == 'max':
            formulas_str = "\n".join([f.to_string(indent + 1) for f in self.params['formulas']])
            return f"{prefix}Max:\n{formulas_str}"

        elif self.type == 'weighted_sum':
            formulas_str = []
            for weight, formula in zip(self.params['weights'], self.params['formulas']):
                formulas_str.append(f"  {prefix}Weight {weight}:\n{formula.to_string(indent + 2)}")
            return f"{prefix}WeightedSum:\n" + "\n".join(formulas_str)

        else:
            return f"{prefix}Unknown({self.type})"

    def __str__(self):
        return self.to_string()


class DecisionTreeTranslator:
    """决策树到L*公式的翻译器"""

    def __init__(self, tree: DecisionTreeClassifier, feature_names=None, X_train=None, feature_ranges=None):
        self.tree = tree
        self.feature_names = feature_names or [f"x{i}" for i in range(tree.n_features_in_)]
        self.n_nodes = self.tree.tree_.node_count
        self.children_left = self.tree.tree_.children_left
        self.children_right = self.tree.tree_.children_right
        self.feature = self.tree.tree_.feature
        self.threshold = self.tree.tree_.threshold
        self.value = self.tree.tree_.value  # 节点上的值分布

        # 新增：存储训练数据用于采样
        self.X_train = X_train

        # 新增：设置特征范围
        if feature_ranges is not None:
            self.feature_ranges = feature_ranges
        elif X_train is not None:
            # 从训练数据计算特征范围
            self.feature_ranges = []
            for i in range(X_train.shape[1]):
                min_val = X_train[:, i].min()
                max_val = X_train[:, i].max()
                # 稍微扩展一点范围以避免边界问题
                epsilon = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
                self.feature_ranges.append((min_val - epsilon, max_val + epsilon))
        else:
            # 默认范围（无训练数据时无法推断真实范围，仅作占位）
            import warnings
            warnings.warn(
                "DecisionTreeTranslator initialized without X_train or feature_ranges; "
                "defaulting feature_ranges to [0, 1] per feature. If the features are "
                "standardized (or otherwise outside [0, 1]), pass X_train so that sampling "
                "uses the correct ranges.",
                stacklevel=2,
            )
            self.feature_ranges = [(0, 1) for _ in range(tree.n_features_in_)]

    def extract_all_paths(self) -> List[Tuple[List[Tuple[int, str, float]], int, float]]:
        """提取所有从根节点到叶节点的路径"""
        paths = []

        def dfs(node_id, current_path):
            # current_path: [(feature_idx, direction, threshold), ...]

            # 如果是叶节点
            if self.children_left[node_id] == self.children_right[node_id]:
                # 叶节点：获取类别
                class_dist = self.value[node_id][0]
                predicted_class = np.argmax(class_dist)
                confidence = class_dist[predicted_class] / np.sum(class_dist)
                paths.append((current_path.copy(), predicted_class, confidence))
                return

            # 内部节点：继续递归
            feature_idx = self.feature[node_id]
            threshold = self.threshold[node_id]

            # 左子树：条件 x <= threshold
            left_path = current_path + [(feature_idx, 'leq', threshold)]
            dfs(self.children_left[node_id], left_path)

            # 右子树：条件 x > threshold
            right_path = current_path + [(feature_idx, 'gt', threshold)]
            dfs(self.children_right[node_id], right_path)

        dfs(0, [])
        return paths

    def path_to_formula(self, path: List[Tuple[int, str, float]]) -> LStarFormula:
        """将路径转换为合取公式"""
        if not path:
            return LStarFormula('const', value=1.0)

        # 从最后一个条件开始构建
        formula = None

        for feature_idx, direction, threshold in path:
            atom = LStarFormula('atom',
                                feature_idx=feature_idx,
                                direction=direction,
                                threshold=threshold)

            if formula is None:
                formula = atom
            else:
                formula = LStarFormula('and', left=atom, right=formula)

        return formula

    def translate_to_Lstar(self) -> LStarFormula:
        """将决策树翻译为L*公式（加权和形式）"""
        paths = self.extract_all_paths()

        formulas = []
        weights = []

        for path, predicted_class, confidence in paths:
            path_formula = self.path_to_formula(path)

            # 注意：这里我们使用预测类别作为权重，但实际应该根据任务调整
            # 对于二分类，可以用1表示正类，0表示负类
            formulas.append(path_formula)
            weights.append(float(predicted_class))

        return LStarFormula('weighted_sum', formulas=formulas, weights=weights)

    def find_activated_path(self, x: np.ndarray) -> Tuple[List[Tuple[int, str, float]], int, float]:
        """找到输入x激活的路径"""
        current_node = 0
        path = []

        while self.children_left[current_node] != self.children_right[current_node]:
            feature_idx = self.feature[current_node]
            threshold = self.threshold[current_node]

            if x[feature_idx] <= threshold:
                path.append((feature_idx, 'leq', threshold))
                current_node = self.children_left[current_node]
            else:
                path.append((feature_idx, 'gt', threshold))
                current_node = self.children_right[current_node]

        # 到达叶节点
        class_dist = self.value[current_node][0]
        predicted_class = np.argmax(class_dist)
        confidence = class_dist[predicted_class] / np.sum(class_dist)

        return path, predicted_class, confidence

    def project_pi_x(self, x: np.ndarray) -> LStarFormula:
        """投影函数π_x：返回激活路径的公式"""
        path, _, _ = self.find_activated_path(x)
        return self.path_to_formula(path)

    def generate_samples_satisfying_formula(self, formula: LStarFormula, n_samples=100) -> List[np.ndarray]:
        """高效生成满足合取公式的样本"""
        # 步骤1：从合取公式中提取每个特征的区间约束
        constraints = self._extract_constraints_from_conjunction(formula)

        # 步骤2：对每个特征，根据约束生成值
        samples = []
        for _ in range(n_samples):
            sample = np.zeros(self.tree.n_features_in_)
            for feat_idx in range(self.tree.n_features_in_):
                if feat_idx in constraints:
                    # 有约束：从约束区间生成
                    low, high, include_low, include_high = constraints[feat_idx]
                    # 调整边界以避免浮点误差
                    if not include_low:
                        low += 1e-10
                    if not include_high:
                        high -= 1e-10
                    if low < high:
                        sample[feat_idx] = np.random.uniform(low, high)
                    else:
                        # 边界相等或无效，使用边界值
                        sample[feat_idx] = low
                else:
                    # 无约束：从特征范围生成
                    low, high = self.feature_ranges[feat_idx]
                    sample[feat_idx] = np.random.uniform(low, high)
            samples.append(sample)

        # 验证（可选，用于调试）
        valid_count = 0
        for sample in samples:
            if abs(formula.evaluate(sample) - 1.0) < 1e-6:
                valid_count += 1
        if valid_count < n_samples * 0.9:  # 验证失败
            print(f"警告：只有{valid_count}/{n_samples}个样本满足公式")

        return samples

    def _extract_constraints_from_conjunction(self, formula: LStarFormula) -> Dict:
        """从合取公式中提取特征约束"""
        constraints = {}

        def extract(node):
            if node.type == 'atom':
                feat = node.params['feature_idx']
                direction = node.params['direction']
                threshold = node.params['threshold']

                if feat not in constraints:
                    # 初始化约束：使用特征范围
                    low, high = self.feature_ranges[feat]
                    constraints[feat] = [low, high, True, True]  # [min, max, include_min, include_max]

                if direction == 'leq':
                    # x <= threshold，更新上界
                    constraints[feat][1] = min(constraints[feat][1], threshold)
                    constraints[feat][3] = True  # 包含上界
                else:  # 'gt'
                    # x > threshold，更新下界
                    constraints[feat][0] = max(constraints[feat][0], threshold)
                    constraints[feat][2] = False  # 不包含下界
            elif node.type == 'and':
                extract(node.params['left'])
                extract(node.params['right'])

        extract(formula)
        return constraints


class AxiomVerifier:
    """公理验证器"""

    @staticmethod
    def verify_local_faithfulness(translator: DecisionTreeTranslator,
                                  x: np.ndarray,
                                  n_test_samples=100) -> Tuple[bool, List[float]]:
        """
        验证公理3.1（局部忠实性）：
        所有满足π_x(T(M))的x'应该有相同的预测
        """
        # 获取投影公式
        formula = translator.project_pi_x(x)

        # 生成满足公式的样本
        samples = translator.generate_samples_satisfying_formula(formula, n_test_samples)

        if not samples:
            return False, []  # 无法生成样本

        # 检查所有样本的预测
        predictions = []
        for sample in samples:
            # 使用决策树预测
            pred = translator.tree.predict(sample.reshape(1, -1))[0]
            predictions.append(pred)

        # 检查是否所有预测都相同
        first_pred = predictions[0]
        all_same = all(p == first_pred for p in predictions)

        return all_same, predictions

    @staticmethod
    def verify_decomposability(translator: DecisionTreeTranslator, x: np.ndarray) -> bool:
        """
        验证公理3.2（可分解性）：
        对于决策树，根节点的解释 = 根节点条件 ∧ 子树的解释
        简化验证：检查投影公式是否包含根节点条件
        """
        formula = translator.project_pi_x(x)
        formula_str = str(formula)

        # 提取根节点条件
        root_feature = translator.feature[0]
        root_threshold = translator.threshold[0]

        # 检查x相对于根节点的方向
        if x[root_feature] <= root_threshold:
            root_condition = f"x{root_feature} <= {root_threshold}"
        else:
            root_condition = f"x{root_feature} > {root_threshold}"

        # 检查公式字符串中是否包含根节点条件
        return root_condition in formula_str

    @staticmethod
    def verify_monotonicity(translator: DecisionTreeTranslator,
                            feature_idx: int,
                            n_test_pairs=10) -> float:
        """
        验证公理3.3（单调性）：
        如果特征在更靠近根节点的位置出现，它对最终预测的影响更大
        返回满足单调性的比例
        """
        # 找到特征在树中出现的最小深度
        min_depth = float('inf')

        def find_feature_depth(node_id, depth):
            nonlocal min_depth

            if translator.children_left[node_id] == translator.children_right[node_id]:
                return

            current_feature = translator.feature[node_id]
            if current_feature == feature_idx and depth < min_depth:
                min_depth = depth

            find_feature_depth(translator.children_left[node_id], depth + 1)
            find_feature_depth(translator.children_right[node_id], depth + 1)

        find_feature_depth(0, 0)

        if min_depth == float('inf'):
            return 0.0  # 特征未在树中出现

        # 与其他特征比较：随机选择其他特征
        all_features = list(range(translator.tree.n_features_in_))
        if feature_idx in all_features:
            all_features.remove(feature_idx)

        other_features = random.sample(all_features, min(5, len(all_features)))

        # 计算其他特征的最小深度
        other_depths = []
        for other_feature in other_features:
            other_min_depth = float('inf')

            def find_other_depth(node_id, depth):
                nonlocal other_min_depth

                if translator.children_left[node_id] == translator.children_right[node_id]:
                    return

                current_feature = translator.feature[node_id]
                if current_feature == other_feature and depth < other_min_depth:
                    other_min_depth = depth

                find_other_depth(translator.children_left[node_id], depth + 1)
                find_other_depth(translator.children_right[node_id], depth + 1)

            find_other_depth(0, 0)
            if other_min_depth != float('inf'):
                other_depths.append(other_min_depth)

        if not other_depths:
            return 1.0  # 没有其他特征可比较

        # 计算特征深度小于其他特征的比例
        count_better = sum(1 for d in other_depths if min_depth < d)
        return count_better / len(other_depths)

# xai_verification
def create_example_tree():
    """创建示例决策树（贷款审批示例）"""
    # 手动构建一个简单的决策树
    # 特征0: 收入，特征1: 信用分
    # 规则:
    # 1. 如果收入 > 50000: 批准
    # 2. 如果收入 <= 50000 且 信用分 > 600: 批准
    # 3. 如果收入 <= 50000 且 信用分 <= 600: 拒绝

    # 创建训练数据
    np.random.seed(42)
    X = np.array([
        [40000, 550],  # 拒绝
        [40000, 650],  # 批准
        [60000, 550],  # 批准
        [60000, 650],  # 批准
        [30000, 500],  # 拒绝
        [30000, 700],  # 批准
        [70000, 500],  # 批准
        [70000, 700],  # 批准
    ])

    y = np.array([0, 1, 1, 1, 0, 1, 1, 1])  # 0:拒绝, 1:批准

    # 训练决策树
    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(X, y)

    return tree, X, y


def test_decision_tree_translation():
    """测试决策树翻译"""
    print("=" * 60)
    print("测试决策树翻译")
    print("=" * 60)

    # 创建示例树
    tree, X, y = create_example_tree()
    translator = DecisionTreeTranslator(tree, feature_names=["收入", "信用分"])

    # 1. 提取所有路径
    print("\n1. 所有路径:")
    paths = translator.extract_all_paths()
    for i, (path, cls, conf) in enumerate(paths):
        path_str = " ∧ ".join([f"x{idx} {dir} {thresh}" for idx, dir, thresh in path])
        print(f"  路径 {i}: {path_str} → 类别{cls} (置信度{conf:.2f})")

    # 2. 翻译为L*公式
    print("\n2. 完整L*公式:")
    formula = translator.translate_to_Lstar()
    print(formula)

    # 3. 测试投影函数
    print("\n3. 投影函数测试:")
    test_samples = [
        np.array([40000, 550]),  # 应该走路径: x0 <= 50000 ∧ x1 <= 600
        np.array([40000, 650]),  # 应该走路径: x0 <= 50000 ∧ x1 > 600
        np.array([60000, 550]),  # 应该走路径: x0 > 50000
    ]

    for i, x in enumerate(test_samples):
        path, pred, conf = translator.find_activated_path(x)
        path_str = " ∧ ".join([f"x{idx} {dir} {thresh}" for idx, dir, thresh in path])
        print(f"  样本{i}: {x} → 路径: {path_str}, 预测: {pred}")

        # 获取投影公式
        proj_formula = translator.project_pi_x(x)
        print(f"    投影公式:\n{proj_formula.to_string(2)}")

        # 验证公式计算
        formula_val = proj_formula.evaluate(x)
        print(f"    公式在x上的值: {formula_val} (应为1.0)")

    return translator, tree, X, y

@pytest.fixture
def translator():
    # 初始化一个树并返回 translator 实例
    tree, X, y = create_example_tree()
    return DecisionTreeTranslator(tree)

def test_axioms(translator):
    """测试公理"""
    print("\n" + "=" * 60)
    print("测试公理")
    print("=" * 60)

    # 测试样本
    test_x = np.array([40000, 550])

    # 1. 测试局部忠实性（公理3.1）
    print("\n1. 测试局部忠实性:")
    faithful, predictions = AxiomVerifier.verify_local_faithfulness(
        translator, test_x, n_test_samples=20
    )

    if faithful:
        print(f"  通过! 所有{predictions[:5]}... 等{predictions}个样本预测相同")
    else:
        print(f"  失败! 预测值: {predictions}")

    # 2. 测试可分解性（公理3.2）
    print("\n2. 测试可分解性:")
    decomposable = AxiomVerifier.verify_decomposability(translator, test_x)
    if decomposable:
        print("  通过! 投影公式包含根节点条件")
    else:
        print("  失败!")

    # 3. 测试单调性（公理3.3）
    print("\n3. 测试单调性:")
    for feature_idx in [0, 1]:  # 测试两个特征
        monotonicity_score = AxiomVerifier.verify_monotonicity(
            translator, feature_idx, n_test_pairs=5
        )
        print(f"  特征{feature_idx}单调性分数: {monotonicity_score:.2f}")


def test_with_sklearn_dataset():
    """使用sklearn生成的数据集测试"""
    print("\n" + "=" * 60)
    print("使用合成数据集测试")
    print("=" * 60)

    # 生成合成数据集
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )

    # 训练决策树
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    translator = DecisionTreeTranslator(tree)

    # 随机测试10个样本
    n_test = 10
    faithful_count = 0
    decomposable_count = 0

    for i in range(n_test):
        # 随机选择一个训练样本
        idx = np.random.randint(0, len(X))
        x = X[idx]

        # 测试局部忠实性
        faithful, _ = AxiomVerifier.verify_local_faithfulness(
            translator, x, n_test_samples=10
        )
        if faithful:
            faithful_count += 1

        # 测试可分解性
        decomposable = AxiomVerifier.verify_decomposability(translator, x)
        if decomposable:
            decomposable_count += 1

    print(f"局部忠实性通过率: {faithful_count}/{n_test} ({faithful_count / n_test * 100:.1f}%)")
    print(f"可分解性通过率: {decomposable_count}/{n_test} ({decomposable_count / n_test * 100:.1f}%)")


def advanced_experiment():
    """高级实验：比较投影公式与SHAP解释"""
    print("\n" + "=" * 60)
    print("高级实验：与SHAP比较")
    print("=" * 60)

    try:
        import shap

        # 创建更复杂的数据集
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            random_state=42
        )

        # 训练决策树
        tree = DecisionTreeClassifier(max_depth=4, random_state=42)
        tree.fit(X, y)

        translator = DecisionTreeTranslator(tree)

        # 选择一个测试样本
        test_idx = 0
        x = X[test_idx]

        # 获取投影公式
        proj_formula = translator.project_pi_x(x)

        # 提取特征重要性（简单版本）
        # 从投影公式中提取出现的特征
        formula_str = str(proj_formula)
        feature_importance_proj = {}

        # 简单解析：统计特征在公式中出现的次数
        for i in range(tree.n_features_in_):
            count = formula_str.count(f"x{i}")
            if count > 0:
                feature_importance_proj[i] = count

        print(f"投影公式的特征重要性（出现次数）: {feature_importance_proj}")

        # 计算SHAP值
        explainer = shap.TreeExplainer(tree)
        shap_values = explainer.shap_values(x.reshape(1, -1))

        # 对于二分类，shap_values可能是列表
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 取正类的SHAP值

        print(f"SHAP值: {shap_values[0]}")

        # 比较排序
        proj_ranking = sorted(feature_importance_proj.items(), key=lambda x: x[1], reverse=True)
        shap_ranking = sorted(enumerate(abs(shap_values[0])), key=lambda x: x[1], reverse=True)

        print(f"投影公式特征排序: {[f[0] for f in proj_ranking]}")
        print(f"SHAP特征排序: {[f[0] for f in shap_ranking]}")

        # 计算排序相关性（简单版本）
        proj_features = [f[0] for f in proj_ranking]
        shap_features = [f[0] for f in shap_ranking]

        # 只考虑共同的特征
        common_features = set(proj_features) & set(shap_features)

        if len(common_features) >= 2:
            # 计算秩相关系数
            from scipy.stats import spearmanr

            proj_ranks = {f: i for i, f in enumerate(proj_features)}
            shap_ranks = {f: i for i, f in enumerate(shap_features)}

            common_proj_ranks = [proj_ranks[f] for f in common_features]
            common_shap_ranks = [shap_ranks[f] for f in common_features]

            corr, p_value = spearmanr(common_proj_ranks, common_shap_ranks)
            print(f"Spearman秩相关系数: {corr:.3f} (p={p_value:.3f})")
        else:
            print("共同特征不足，无法计算相关性")

    except ImportError:
        print("SHAP库未安装。请运行: pip install shap")

    # 运行测试
    if __name__ == "__main__":
        # 测试决策树翻译
        translator, tree, X, y = test_decision_tree_translation()

        # 测试公理
        test_axioms(translator)

        # 使用合成数据集测试
        test_with_sklearn_dataset()

        # 高级实验（需要SHAP）
        advanced_experiment()

# xai_experiments
class LinearModelTranslator:
    """线性模型到L*公式的翻译器"""

    def __init__(self, coef, intercept, feature_names=None):
        """
        coef: 线性系数
        intercept: 截距
        """
        self.coef = np.array(coef).flatten()
        self.intercept = float(intercept)
        self.n_features = len(self.coef)
        self.feature_names = feature_names or [f"x{i}" for i in range(self.n_features)]

    def translate_to_Lstar(self) -> LStarFormula:
        """将线性模型翻译为L*公式"""
        # 线性模型: f(x) = sign(w·x + b)
        # 在L*中表示为加权和

        atoms = []
        weights = []

        # 特征项
        for i in range(self.n_features):
            # 为每个特征创建原子公式（总是为真，但权重不同）
            # 注意：线性模型没有阈值条件，所以我们需要不同的表示
            atom = LStarFormula('const', value=1.0)  # 占位符
            atoms.append(atom)
            weights.append(self.coef[i])

        # 创建加权和公式
        weighted_sum = LStarFormula('weighted_sum', formulas=atoms, weights=weights)

        # 添加截距
        intercept_formula = LStarFormula('const', value=self.intercept)

        # 最终公式：加权和 + 截距
        return LStarFormula('sum', formulas=[weighted_sum, intercept_formula])

    def project_pi_x(self, x: np.ndarray) -> LStarFormula:
        """投影函数π_x：对于线性模型，返回特征的线性组合"""
        # 对于线性模型，投影显示每个特征的贡献
        atoms = []

        for i in range(self.n_features):
            # 创建显示特征贡献的公式：coef[i] * x[i]
            # 我们需要扩展L*以支持乘法

            # 简化：只返回最重要的特征
            if abs(self.coef[i] * x[i]) > 0.1:  # 阈值
                atom = LStarFormula('atom',
                                    feature_idx=i,
                                    direction='leq',  # 方向不重要
                                    threshold=0.0)  # 阈值不重要
                # 注意：这里我们简化了，实际应该能表示 coef * x
                atoms.append(atom)

        if atoms:
            return LStarFormula('and', left=atoms[0], right=atoms[1]) if len(atoms) > 1 else atoms[0]
        else:
            return LStarFormula('const', value=1.0)

    def predict(self, x: np.ndarray) -> int:
        """线性模型预测"""
        linear_output = np.dot(self.coef, x) + self.intercept
        return 1 if linear_output >= 0 else 0


def test_linear_model():
    """测试线性模型翻译"""
    print("\n" + "=" * 60)
    print("测试线性模型翻译")
    print("=" * 60)

    # 创建一个简单的线性模型
    coef = np.array([0.8, -0.5, 0.3])  # 3个特征
    intercept = -0.2

    translator = LinearModelTranslator(coef, intercept, ["收入", "负债", "信用分"])

    # 翻译为L*公式
    formula = translator.translate_to_Lstar()
    print("线性模型的L*公式:")
    print(formula)

    # 测试预测
    test_x = np.array([0.7, 0.3, 0.9])
    pred = translator.predict(test_x)
    print(f"\n测试样本: {test_x}")
    print(f"线性模型预测: {pred}")

    # 计算公式值
    formula_val = formula.evaluate(test_x)
    print(f"公式值 (w·x + b): {formula_val:.3f}")
    print(f"sign(公式值): {1 if formula_val >= 0 else 0}")

    # 投影
    proj_formula = translator.project_pi_x(test_x)
    print(f"\n投影公式: {proj_formula}")


# 运行所有测试
if __name__ == "__main__":
    # 运行基本测试
    translator, tree, X, y = test_decision_tree_translation()
    test_axioms(translator)
    test_with_sklearn_dataset()
    test_linear_model()
    # 如果需要SHAP测试，可以取消注释下面这行
    # advanced_experiment()