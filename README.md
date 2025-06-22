《三体》中的女性宇宙：社会网络分析方法复现与解读
本分析报告主要对《科幻中的女性宇宙：《三体》的社会网络研究 》这篇论文钟数据分析的方法进行总结和复现，正文包含三部分内容：数据分析方法概述，实验流程，实验流程总结。
一、数据分析方法概述
主要通过自然语言处理（NLP）和社会网络分析（SNA）的方法，对《三体》系列小说中的人物关系进行量化研究，并结合女性主义理论进行解读。分析的核心包括以下几个方面：
（1）文本数据预处理
1.深度阅读文本，构建人物词典
先手工整理人物名单，建立两份词典
正式姓名词典（如：罗辑、程心、叶文洁）
别称映射词典（如：罗辑 -> 小罗、罗兄、罗教授）
2.实体识别
使用Python的NLP工具（如jieba、spaCy等）进行分词和实体识别，提取小说中的人物名称。
（2）人物关系构建
1.共现分析（Co-occurrence Analysis）
定义人物关系： 两个人物在小说同一行同时出现，视为存在一次联系。
建立共现矩阵： 统计小说文本中人物的共现次数，形成人物-人物共现矩阵（即邻接矩阵）。
2.数据清洗
过滤出现次数较少的人物，最终确定180名人物和368对共现关系。
（3）社会网络分析
1.使用 Gephi 进行可视化
导入清洗后的人物-人物共现矩阵，构建社交网络。
节点（Node）： 代表小说中的角色。
边（Edge）： 代表人物间的联系（共现次数）。
过滤掉共现次数小于5的弱联系，确保主要人物关系清晰可见。
2.计算社会网络中心性指标
1.度中心性（Degree Centrality）
计算角色的直接连接数，代表人物的影响力。
统计度中心性最高的角色，发现智子、程心等女性角色占据重要位置。
算法实现原理：
 def degree_centrality(G):
    n = G.number_of_nodes()
    return {n: d / (n-1) for n, d in G.degree()}
数据结构：利用图的邻接表存储结构（G.adj）
计算流程：
1.遍历所有节点获取度数（G.degree()）
2.归一化处理（除以n-1）

2.中介中心性（Betweenness Centrality）
计算角色在其他人物间的最短路径上出现的次数，衡量其在网络中的“桥梁”作用。
发现智子在整个社交网络中是最核心的节点，连接多个主要角色。
Brandes算法实现原理：
def brandes_shortest_paths(G, source):
    # 初始化
    S = []
    P = defaultdict(list)
    sigma = defaultdict(int)
    dist = {n: -1 for n in G}
    
    # BFS计算最短路径
    q = deque([source])
    dist[source] = 0
    sigma[source] = 1
    while q:
        v = q.popleft()
        S.append(v)
        for w in G.neighbors(v):
            if dist[w] < 0:
                q.append(w)
                dist[w] = dist[v] + 1
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                P[w].append(v)
    
    # 依赖累积
    delta = defaultdict(float)
    while S:
        w = S.pop()
        for v in P[w]:
            delta[v] += (sigma[v]/sigma[w]) * (1 + delta[w])
        if w != source:
            bc[w] += delta[w]
    return bc
核心数据结构组成：
​​队列（Queue）​​
​​用途：执行广度优先搜索（BFS）遍历图
​​功能特性：
按层次遍历节点（保证最短路径性质）
维护待处理的节点序列
​​前驱节点集合（Predecessors）​​
​​数据结构：字典嵌套集合{node: set(predecessors)}
​​作用：记录最短路径中每个节点的前驱节点
​​距离数组（Distance）​​
​​数据结构：字典{node: distance}
​​初始化：{n: -1 for n in G}（-1表示未访问）
​​更新规则：dist[w] = dist[v] + 1
​​路径计数数组（Sigma）​​
​​数据结构：字典{node: count}
​​初始化：{n: 0 for n in G}
​​更新规则：σ[w] += σ[v]
​​依赖值数组（Dependency）​​
​​数据结构：字典{node: dependency}
​​作用：累积节点对中介中心性的贡献值
​​计算公式：

3.特征向量中心性（Eigenvector Centrality）
计算角色是否与其他重要角色有较多联系，衡量人物的社会影响力。
发现智子、程心、叶文洁等女性角色在小说中具有很强的网络影响力。
幂迭代法实现：
def power_iteration(A, num_simulations=100):
    n = A.shape[0]
    b_k = np.random.rand(n)
    
    for _ in range(num_simulations):
        # 矩阵乘法
        b_k1 = np.dot(A, b_k)
        # 归一化
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
return eigenvalue, b_k
数据结构：邻接矩阵
计算流程：
1.​​矩阵表示​​：邻接矩阵转置处理（入度特征向量）
2.​​收敛加速​​：采用Rayleigh商迭代
3.​​内存优化​​：稀疏矩阵存储（CSR格式）
