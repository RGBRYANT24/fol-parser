import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('all_result.csv')

# 重命名算法名称以匹配表格需求
method_mapping = {
    'DFS': 'DFS',
    'MCTS': 'MCTS',
    'NeuralHeuristic_1': '第一阶段加速',
    'NeuralHeuristic_2': '第二阶段加速',
    'NeuralHeuristic_ALL': '两阶段加速(本文)'
}

df['Method'] = df['Method'].map(method_mapping)

# 将持续时间从毫秒转换为秒
df['Duration(s)'] = df['Duration(ms)'] / 1000

# 根据图的规模进行分类
df['SizeCategory'] = pd.cut(
    df['GraphSize'], 
    bins=[0, 4, 7, 10], 
    labels=['小规模(3-4节点)', '中规模(5-7节点)', '大规模(8-10节点)']
)

# 表1: 各算法在全部测试集上的平均性能对比
table1 = pd.DataFrame()

for method in method_mapping.values():
    method_df = df[df['Method'] == method]
    
    # 计算平均求解时间
    avg_time = method_df['Duration(s)'].mean()
    
    # 计算平均节点展开数
    avg_nodes = method_df['VisitedStates'].mean()
    
    # 计算成功率
    success_rate = (method_df['Success'] == 'Yes').mean() * 100
    
    # 计算相对于DFS的时间加速比
    dfs_avg_time = df[df['Method'] == 'DFS']['Duration(s)'].mean()
    time_speedup = dfs_avg_time / avg_time if avg_time > 0 else float('inf')
    
    # 添加到结果表格
    table1 = table1.append({
        '算法': method,
        '平均求解时间(s)': round(avg_time, 2),
        '平均节点展开数': int(avg_nodes),
        '成功率(%)': round(success_rate, 2),
        '时间加速比': round(time_speedup, 2)
    }, ignore_index=True)

# 表2: 各算法在不同规模图上的平均求解时间
table2 = pd.DataFrame()

# 根据算法和规模分组计算平均求解时间
size_time = df.pivot_table(
    index='Method', 
    columns='SizeCategory', 
    values='Duration(s)', 
    aggfunc='mean'
)

# 计算相对于DFS的提升
for method in method_mapping.values():
    small_time = size_time.loc[method, '小规模(3-4节点)'] if method in size_time.index and '小规模(3-4节点)' in size_time.columns else np.nan
    medium_time = size_time.loc[method, '中规模(5-7节点)'] if method in size_time.index and '中规模(5-7节点)' in size_time.columns else np.nan
    large_time = size_time.loc[method, '大规模(8-10节点)'] if method in size_time.index and '大规模(8-10节点)' in size_time.columns else np.nan
    
    # 计算相对DFS的提升（只为"两阶段加速(本文)"计算）
    improvement = '-'
    if method == '两阶段加速(本文)':
        dfs_small = size_time.loc['DFS', '小规模(3-4节点)'] if 'DFS' in size_time.index and '小规模(3-4节点)' in size_time.columns else np.nan
        dfs_medium = size_time.loc['DFS', '中规模(5-7节点)'] if 'DFS' in size_time.index and '中规模(5-7节点)' in size_time.columns else np.nan
        dfs_large = size_time.loc['DFS', '大规模(8-10节点)'] if 'DFS' in size_time.index and '大规模(8-10节点)' in size_time.columns else np.nan
        
        # 计算平均提升
        small_imp = dfs_small / small_time if not np.isnan(small_time) and small_time > 0 else 0
        medium_imp = dfs_medium / medium_time if not np.isnan(medium_time) and medium_time > 0 else 0
        large_imp = dfs_large / large_time if not np.isnan(large_time) and large_time > 0 else 0
        
        # 计算加权平均提升（给大规模图更高权重）
        weighted_avg = (small_imp * 1 + medium_imp * 1 + large_imp * 1) / 3
        improvement = round(weighted_avg, 2)
    
    # 添加到结果表格
    table2 = table2.append({
        '算法': method,
        '小规模(3-4节点)': round(small_time, 2) if not np.isnan(small_time) else '-',
        '中规模(5-7节点)': round(medium_time, 2) if not np.isnan(medium_time) else '-',
        '大规模(8-10节点)': round(large_time, 2) if not np.isnan(large_time) else '-',
        '相对DFS提升': improvement
    }, ignore_index=True)

# 输出结果
print("表1: 各算法在全部测试集上的平均性能对比")
print(table1.to_string(index=False))
print("\n表2: 各算法在不同规模图上的平均求解时间(秒)")
print(table2.to_string(index=False))

# # 将结果保存到LaTeX格式
# with open('table1_latex.txt', 'w') as f:
#     f.write("\\begin{table}[h]\n")
#     f.write("\\centering\n")
#     f.write("\\caption{各算法在全部测试集上的平均性能对比}\n")
#     f.write("\\label{tab:overall_performance}\n")
#     f.write("\\begin{tabular}{c c c c c}\n")
#     f.write("\\hline\n")
#     f.write("\\textbf{算法} & \\textbf{平均求解时间(s)} & \\textbf{平均节点展开数} & \\textbf{成功率(\\%)} & \\textbf{时间加速比} \\\\\n")
#     f.write("\\hline\n")
    
#     for _, row in table1.iterrows():
#         f.write(f"{row['算法']} & {row['平均求解时间(s)']} & {row['平均节点展开数']} & {row['成功率(%)']} & {row['时间加速比']} \\\\\n")
    
#     f.write("\\hline\n")
#     f.write("\\end{tabular}\n")
#     f.write("\\end{table}\n")

# with open('table2_latex.txt', 'w') as f:
#     f.write("\\begin{table}[h]\n")
#     f.write("\\centering\n")
#     f.write("\\caption{各算法在不同规模图上的平均求解时间(秒)}\n")
#     f.write("\\label{tab:solving_time_by_size}\n")
#     f.write("\\begin{tabular}{c c c c c}\n")
#     f.write("\\hline\n")
#     f.write("\\textbf{算法} & \\textbf{小规模(3-4节点)} & \\textbf{中规模(5-7节点)} & \\textbf{大规模(8-10节点)} & \\textbf{相对DFS提升} \\\\\n")
#     f.write("\\hline\n")
    
#     for _, row in table2.iterrows():
#         f.write(f"{row['算法']} & {row['小规模(3-4节点)']} & {row['中规模(5-7节点)']} & {row['大规模(8-10节点)']} & {row['相对DFS提升']} \\\\\n")
    
#     f.write("\\hline\n")
#     f.write("\\end{tabular}\n")
#     f.write("\\end{table}\n")

# print("\n已将LaTeX代码保存到table1_latex.txt和table2_latex.txt")