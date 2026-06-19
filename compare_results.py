import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

yours = pd.read_csv("cnn_result.csv", sep=';')
author = pd.read_csv("data/1/cnn_result.csv", sep=';', header=None, names=['prediction', 'test_label', 'test_price'])

print("=" * 70)
print("         你的结果 vs 作者结果 (data/1) 详细对比分析")
print("=" * 70)

# 1. 基本信息对比
print("\n【1. 基本信息】")
print(f"  你的结果样本数: {len(yours)}")
print(f"  作者结果样本数: {len(author)}")
print(f"  test_price 是否一致: {np.allclose(yours['test_price'].values, author['test_price'].values, atol=0.01)}")
print(f"  test_label 是否一致: {np.array_equal(yours['test_label'].values, author['test_label'].values)}")

# 2. 整体准确率
print("\n【2. 整体准确率】")
acc_yours = accuracy_score(yours['test_label'], yours['prediction'])
acc_author = accuracy_score(author['test_label'], author['prediction'])
print(f"  你的准确率: {acc_yours:.4f} ({int(acc_yours * len(yours))}/{len(yours)})")
print(f"  作者准确率: {acc_author:.4f} ({int(acc_author * len(author))}/{len(author)})")
print(f"  差距: {acc_author - acc_yours:+.4f}")

# 3. 真实标签分布
print("\n【3. 真实标签分布 (test_label)】")
label_dist = yours['test_label'].value_counts().sort_index()
for label in [0, 1, 2]:
    name = {0: 'Hold(0)', 1: 'Buy(1)', 2: 'Sell(2)'}[label]
    count = label_dist.get(label, 0)
    print(f"  {name}: {count} ({count / len(yours) * 100:.1f}%)")

# 4. 预测分布对比
print("\n【4. 预测分布对比】")
pred_dist_yours = yours['prediction'].value_counts().sort_index()
pred_dist_author = author['prediction'].value_counts().sort_index()
print(f"  {'类别':<10} {'你的预测数':>10} {'你的占比':>8} {'作者预测数':>10} {'作者占比':>8}")
for label in [0, 1, 2]:
    name = {0: 'Hold(0)', 1: 'Buy(1)', 2: 'Sell(2)'}[label]
    y_count = pred_dist_yours.get(label, 0)
    a_count = pred_dist_author.get(label, 0)
    print(
        f"  {name:<10} {y_count:>10} {y_count / len(yours) * 100:>7.1f}% {a_count:>10} {a_count / len(author) * 100:>7.1f}%")

# 5. 逐类别准确率
print("\n【5. 逐类别准确率】")
print(f"  {'类别':<10} {'你的正确数/总数':>15} {'你的准确率':>10} {'作者正确数/总数':>15} {'作者准确率':>10}")
for label in [0, 1, 2]:
    name = {0: 'Hold(0)', 1: 'Buy(1)', 2: 'Sell(2)'}[label]
    mask = yours['test_label'] == label
    total = mask.sum()
    y_correct = (yours.loc[mask, 'prediction'] == label).sum()
    a_correct = (author.loc[mask, 'prediction'] == label).sum()
    print(
        f"  {name:<10} {y_correct:>5}/{total:<5} {y_correct / total * 100:>9.2f}% {a_correct:>5}/{total:<5} {a_correct / total * 100:>9.2f}%")

# 6. 混淆矩阵
print("\n【6. 你的混淆矩阵】")
cm_yours = confusion_matrix(yours['test_label'], yours['prediction'])
print(f"  {'':>12} 预测Hold(0) 预测Buy(1) 预测Sell(2)")
for i, name in enumerate(['真实Hold(0)', '真实Buy(1)', '真实Sell(2)']):
    print(f"  {name:>12} {cm_yours[i, 0]:>10} {cm_yours[i, 1]:>10} {cm_yours[i, 2]:>10}")

print("\n【7. 作者混淆矩阵】")
cm_author = confusion_matrix(author['test_label'], author['prediction'])
print(f"  {'':>12} 预测Hold(0) 预测Buy(1) 预测Sell(2)")
for i, name in enumerate(['真实Hold(0)', '真实Buy(1)', '真实Sell(2)']):
    print(f"  {name:>12} {cm_author[i, 0]:>10} {cm_author[i, 1]:>10} {cm_author[i, 2]:>10}")

# 7.5 分类报告
print("\n【8. 你的分类报告】")
print(classification_report(yours['test_label'], yours['prediction'], target_names=['Hold(0)', 'Buy(1)', 'Sell(2)'],
                            digits=4))

print("【9. 作者分类报告】")
print(classification_report(author['test_label'], author['prediction'], target_names=['Hold(0)', 'Buy(1)', 'Sell(2)'],
                            digits=4))

# 8. 两份预测的一致性分析
print("【10. 预测一致性分析】")
agree = (yours['prediction'].values == author['prediction'].values)
print(f"  预测完全一致的样本数: {agree.sum()}/{len(yours)} ({agree.sum() / len(yours) * 100:.1f}%)")
kappa = cohen_kappa_score(yours['prediction'].values, author['prediction'].values)
print(f"  Cohen's Kappa 一致性系数: {kappa:.4f}")
print(f"  (Kappa>0.6表示较好一致, >0.8表示高度一致)")

# 9. 不一致样本分析
print("\n【11. 不一致样本详细分析】")
disagree_mask = yours['prediction'].values != author['prediction'].values
disagree_count = disagree_mask.sum()
if disagree_count > 0:
    y_pred_dis = yours.loc[disagree_mask, 'prediction']
    a_pred_dis = author.loc[disagree_mask, 'prediction']
    true_label_dis = yours.loc[disagree_mask, 'test_label']

    # 不一致样本中谁更准
    y_correct_dis = (y_pred_dis.values == true_label_dis.values).sum()
    a_correct_dis = (a_pred_dis.values == true_label_dis.values).sum()
    both_wrong = ((y_pred_dis.values != true_label_dis.values) & (a_pred_dis.values != true_label_dis.values)).sum()

    print(f"  不一致样本总数: {disagree_count}")
    print(f"  其中你预测正确: {y_correct_dis} ({y_correct_dis / disagree_count * 100:.1f}%)")
    print(f"  其中作者预测正确: {a_correct_dis} ({a_correct_dis / disagree_count * 100:.1f}%)")
    print(f"  两者都错: {both_wrong} ({both_wrong / disagree_count * 100:.1f}%)")

    # 不一致模式
    print(f"\n  不一致模式分布:")
    for y_p in [0, 1, 2]:
        for a_p in [0, 1, 2]:
            if y_p != a_p:
                cnt = ((y_pred_dis == y_p) & (a_pred_dis == a_p)).sum()
                if cnt > 0:
                    y_name = {0: 'Hold', 1: 'Buy', 2: 'Sell'}[y_p]
                    a_name = {0: 'Hold', 1: 'Buy', 2: 'Sell'}[a_p]
                    print(f"    你预测{y_name}({y_p}) vs 作者预测{a_name}({a_p}): {cnt}个")

# 10. 按真实标签分组的不一致性
print("\n【12. 按真实标签分组的不一致性】")
for label in [0, 1, 2]:
    name = {0: 'Hold(0)', 1: 'Buy(1)', 2: 'Sell(2)'}[label]
    mask = yours['test_label'] == label
    sub_agree = (yours.loc[mask, 'prediction'].values == author.loc[mask, 'prediction'].values)
    total = mask.sum()
    print(
        f"  {name}: 一致{sub_agree.sum()}/{total} ({sub_agree.sum() / total * 100:.1f}%), 不一致{total - sub_agree.sum()}/{total}")

print("\n" + "=" * 70)