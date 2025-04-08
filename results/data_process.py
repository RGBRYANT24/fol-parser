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

# 根据图的规模进行分类 - 只关注6-7节点和8-10节点
df['SizeCategory'] = pd.cut(
    df['GraphSize'], 
    bins=[5, 7, 10], 
    labels=['中小规模(6-7节点)', '大规模(8-10节点)']
)

# 过滤掉小于6节点的数据
df = df[df['GraphSize'] >= 6]

# 表1: 各算法在全部测试集上的平均性能对比 (只考虑6-10节点的图)
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
    row_data = {
        '算法': method,
        '平均求解时间(s)': round(avg_time, 2),
        '平均节点展开数': int(avg_nodes),
        '成功率(%)': round(success_rate, 2),
        '时间加速比': round(time_speedup, 2)
    }
    
    table1 = pd.concat([table1, pd.DataFrame([row_data])], ignore_index=True)

# 表2: 各算法在不同规模图上的平均求解时间
table2 = pd.DataFrame()

# 根据算法和规模分组计算平均求解时间
size_time = df.pivot_table(
    index='Method', 
    columns='SizeCategory', 
    values='Duration(s)', 
    aggfunc='mean'
)

# 计算相对于DFS的提升 - 所有方法都计算相对DFS的提升
for method in method_mapping.values():
    medium_time = size_time.loc[method, '中小规模(6-7节点)'] if method in size_time.index and '中小规模(6-7节点)' in size_time.columns else np.nan
    large_time = size_time.loc[method, '大规模(8-10节点)'] if method in size_time.index and '大规模(8-10节点)' in size_time.columns else np.nan
    
    # 计算相对DFS的提升（为所有方法计算）
    dfs_medium = size_time.loc['DFS', '中小规模(6-7节点)'] if 'DFS' in size_time.index and '中小规模(6-7节点)' in size_time.columns else np.nan
    dfs_large = size_time.loc['DFS', '大规模(8-10节点)'] if 'DFS' in size_time.index and '大规模(8-10节点)' in size_time.columns else np.nan
    
    # 计算每个规模的提升比
    if method == 'DFS':
        improvement = 1.00  # DFS对比自身，提升比为1
    else:
        # 特别关注大规模(8-10节点)的提升
        medium_imp = dfs_medium / medium_time if not np.isnan(medium_time) and medium_time > 0 else 0
        large_imp = dfs_large / large_time if not np.isnan(large_time) and large_time > 0 else 0
        
        # 计算提升比 (主要关注大规模)
        improvement = round(large_imp, 2)
    
    # 添加到结果表格
    row_data = {
        '算法': method,
        '中小规模(6-7节点)': round(medium_time, 2) if not np.isnan(medium_time) else '-',
        '大规模(8-10节点)': round(large_time, 2) if not np.isnan(large_time) else '-',
        '相对DFS提升': improvement
    }
    
    table2 = pd.concat([table2, pd.DataFrame([row_data])], ignore_index=True)

# 输出结果
print("表1: 各算法在全部测试集上的平均性能对比 (6-10节点)")
print(table1.to_string(index=False))
print("\n表2: 各算法在不同规模图上的平均求解时间(秒)")
print(table2.to_string(index=False))