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

# 根据图的密度进行分类
df['DensityCategory'] = pd.cut(
    df['EdgeDensity'], 
    bins=[0, 0.15, 0.4, 1.0], 
    labels=['稀疏图(密度<0.15)', '中等密度(0.15-0.4)', '稠密图(密度>0.4)']
)

# 根据图的规模进行分类
df['SizeCategory'] = pd.cut(
    df['GraphSize'], 
    bins=[0, 4, 7, 10], 
    labels=['小规模(3-4节点)', '中规模(5-7节点)', '大规模(8-10节点)']
)

# 为表4创建更详细的密度分类
df['DensityDetailedCategory'] = pd.cut(
    df['EdgeDensity'], 
    bins=[0, 0.4, 0.7, 1.0], 
    labels=['<0.4', '0.4-0.7', '>0.7']
)

# 表3: 各算法在不同密度图上的平均节点展开数
table3 = pd.DataFrame()

# 根据算法和密度分组计算平均节点展开数
density_nodes = df.pivot_table(
    index='Method', 
    columns='DensityCategory', 
    values='VisitedStates', 
    aggfunc='mean'
)

# 填充表格并计算两阶段相对DFS减少率
for method in method_mapping.values():
    sparse_nodes = density_nodes.loc[method, '稀疏图(密度<0.15)'] if method in density_nodes.index and '稀疏图(密度<0.15)' in density_nodes.columns else np.nan
    medium_nodes = density_nodes.loc[method, '中等密度(0.15-0.4)'] if method in density_nodes.index and '中等密度(0.15-0.4)' in density_nodes.columns else np.nan
    dense_nodes = density_nodes.loc[method, '稠密图(密度>0.4)'] if method in density_nodes.index and '稠密图(密度>0.4)' in density_nodes.columns else np.nan
    
    # 计算相对DFS的节点减少比例（只为"两阶段加速(本文)"计算）
    reduction = '-'
    if method == '两阶段加速(本文)':
        dfs_sparse = density_nodes.loc['DFS', '稀疏图(密度<0.15)'] if 'DFS' in density_nodes.index and '稀疏图(密度<0.15)' in density_nodes.columns else np.nan
        dfs_medium = density_nodes.loc['DFS', '中等密度(0.15-0.4)'] if 'DFS' in density_nodes.index and '中等密度(0.15-0.4)' in density_nodes.columns else np.nan
        dfs_dense = density_nodes.loc['DFS', '稠密图(密度>0.4)'] if 'DFS' in density_nodes.index and '稠密图(密度>0.4)' in density_nodes.columns else np.nan
        
        # 计算平均减少比例 (1 - 两阶段/DFS)
        sparse_red = 1 - sparse_nodes / dfs_sparse if not np.isnan(sparse_nodes) and not np.isnan(dfs_sparse) and dfs_sparse > 0 else 0
        medium_red = 1 - medium_nodes / dfs_medium if not np.isnan(medium_nodes) and not np.isnan(dfs_medium) and dfs_medium > 0 else 0
        dense_red = 1 - dense_nodes / dfs_dense if not np.isnan(dense_nodes) and not np.isnan(dfs_dense) and dfs_dense > 0 else 0
        
        # 计算加权平均减少比例
        weighted_avg = (sparse_red + medium_red + dense_red) / 3
        reduction = f"{round(weighted_avg * 100, 2)}%"
    
    row_data = {
        '算法': method,
        '稀疏图(密度<0.15)': int(round(sparse_nodes)) if not np.isnan(sparse_nodes) else '-',
        '中等密度(0.15-0.4)': int(round(medium_nodes)) if not np.isnan(medium_nodes) else '-',
        '稠密图(密度>0.4)': int(round(dense_nodes)) if not np.isnan(dense_nodes) else '-',
        '两阶段相对DFS减少': reduction
    }
    
    table3 = pd.concat([table3, pd.DataFrame([row_data])], ignore_index=True)

# 表4: 两阶段方法相对DFS的性能提升比率
table4 = pd.DataFrame(columns=['图分类', '节点数', '密度', '求解时间提升比', '节点展开数减少比'])

# 获取两阶段加速和DFS的数据
two_stage_df = df[df['Method'] == '两阶段加速(本文)']
dfs_df = df[df['Method'] == 'DFS']

# 图分类列表
graph_types = ['稀疏图', '中等密度', '稠密图']
size_categories = ['小规模(3-4节点)', '中规模(5-7节点)', '大规模(8-10节点)']
density_categories = ['<0.4', '0.4-0.7', '>0.7']

# 遍历每种图类型、大小和密度组合
for i, graph_type in enumerate(graph_types):
    density_cat = density_categories[i]
    
    for size_cat in size_categories:
        # 筛选两阶段加速和DFS在当前组合下的数据
        two_stage_subset = two_stage_df[(two_stage_df['SizeCategory'] == size_cat) & 
                                        (two_stage_df['DensityDetailedCategory'] == density_cat)]
        
        dfs_subset = dfs_df[(dfs_df['SizeCategory'] == size_cat) & 
                           (dfs_df['DensityDetailedCategory'] == density_cat)]
        
        # 计算平均求解时间和节点展开数
        if not two_stage_subset.empty and not dfs_subset.empty:
            avg_two_stage_time = two_stage_subset['Duration(s)'].mean()
            avg_dfs_time = dfs_subset['Duration(s)'].mean()
            
            avg_two_stage_nodes = two_stage_subset['VisitedStates'].mean()
            avg_dfs_nodes = dfs_subset['VisitedStates'].mean()
            
            # 计算提升比和减少比
            time_improvement_ratio = avg_dfs_time / avg_two_stage_time if avg_two_stage_time > 0 else float('inf')
            nodes_reduction_ratio = 1 - (avg_two_stage_nodes / avg_dfs_nodes) if avg_dfs_nodes > 0 else 0
            
            # 添加到表格
            row_data = {
                '图分类': graph_type,
                '节点数': size_cat.split('(')[0],
                '密度': density_cat,
                '求解时间提升比': round(time_improvement_ratio, 2),
                '节点展开数减少比': f"{round(nodes_reduction_ratio * 100, 2)}%"
            }
            
            table4 = pd.concat([table4, pd.DataFrame([row_data])], ignore_index=True)
        else:
            # 如果没有数据，添加空行
            row_data = {
                '图分类': graph_type,
                '节点数': size_cat.split('(')[0],
                '密度': density_cat,
                '求解时间提升比': '-',
                '节点展开数减少比': '-'
            }
            
            table4 = pd.concat([table4, pd.DataFrame([row_data])], ignore_index=True)

# 输出结果
print("表3: 各算法在不同密度图上的平均节点展开数")
print(table3.to_string(index=False))
print("\n表4: 两阶段方法相对DFS的性能提升比率")
print(table4.to_string(index=False))