import networkx as nx
import igraph as ig
import leidenalg

# 创建一个简单图
G = nx.Graph()

# 添加5个节点
G.add_nodes_from([1, 2, 3, 4, 5])

# 添加边，带权重（越大表示越相似）
edges = [
    (1, 2, 3),  # 节点1和2相连，权重3
    (2, 3, 2),
    (3, 4, 4),
    (4, 5, 1),
    (5, 1, 1),
    (2, 4, 2)
]
G.add_weighted_edges_from(edges)

print("图的节点：", G.nodes())
print("图的边：", G.edges(data=True))

# 转换为 igraph
# networkx 的节点会变成 igraph 的 0-based 索引
mapping = {node: idx for idx, node in enumerate(G.nodes())}
edges_igraph = [(mapping[u], mapping[v], d['weight']) for u, v, d in G.edges(data=True)]

g = ig.Graph()
g.add_vertices(len(G.nodes()))
g.add_edges([(u, v) for u, v, w in edges_igraph])
g.es['weight'] = [w for u, v, w in edges_igraph]

print("igraph 图的边和权重：")
for e in g.es:
    print(e.tuple, e['weight'])


# 使用Leiden算法检测社区
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight')

# 输出结果
print("每个节点所属社区：")
for idx, comm in enumerate(partition.membership):
    print(f"节点 {list(G.nodes())[idx]} -> 社区 {comm}")

print("\n总共社区数：", len(set(partition.membership)))