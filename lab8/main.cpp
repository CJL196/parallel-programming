#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iomanip>

// 定义无穷大
const float INF = std::numeric_limits<float>::infinity();

// 图结构
struct Graph {
    int num_vertices;
    std::vector<std::vector<float>> dist; // 距离矩阵
    std::unordered_map<int, int> id_to_index; // 顶点ID到索引的映射
    std::vector<int> index_to_id; // 索引到顶点ID的映射
};

// 读取邻接表文件
Graph readGraph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    Graph graph;
    std::string line;
    
    // 跳过标题行
    std::getline(file, line);
    
    // 读取边并构建图
    std::unordered_map<int, std::vector<std::pair<int, float>>> edges;
    std::unordered_set<int> vertices;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string source_str, target_str, distance_str;
        
        // 格式: source,target,distance
        std::getline(ss, source_str, ',');
        std::getline(ss, target_str, ',');
        std::getline(ss, distance_str, ',');
        
        int source = std::stoi(source_str);
        int target = std::stoi(target_str);
        float distance = std::stof(distance_str);
        
        // 添加顶点
        vertices.insert(source);
        vertices.insert(target);
        
        // 添加边 (无向图，所以两个方向都添加)
        edges[source].push_back({target, distance});
        edges[target].push_back({source, distance});
    }
    
    // 创建顶点ID到索引的映射
    std::vector<int> vertex_ids(vertices.begin(), vertices.end());
    std::sort(vertex_ids.begin(), vertex_ids.end());
    
    graph.num_vertices = vertex_ids.size();
    graph.index_to_id.resize(graph.num_vertices);
    
    for (int i = 0; i < graph.num_vertices; ++i) {
        graph.id_to_index[vertex_ids[i]] = i;
        graph.index_to_id[i] = vertex_ids[i];
    }
    
    // 初始化距离矩阵
    graph.dist.resize(graph.num_vertices, std::vector<float>(graph.num_vertices, INF));
    
    // 设置自身到自身的距离为0
    for (int i = 0; i < graph.num_vertices; ++i) {
        graph.dist[i][i] = 0.0f;
    }
    
    // 填充距离矩阵
    for (const auto& [vertex_id, neighbors] : edges) {
        int u = graph.id_to_index[vertex_id];
        for (const auto& [neighbor_id, weight] : neighbors) {
            int v = graph.id_to_index[neighbor_id];
            graph.dist[u][v] = weight;
        }
    }
    
    return graph;
}

// 使用Floyd-Warshall算法计算多源最短路径 - 不同的并行策略
void floydWarshall(Graph& graph, int num_threads, int parallel_strategy = 0) {
    int n = graph.num_vertices;
    auto& dist = graph.dist;
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    
    // 根据不同的并行策略执行Floyd-Warshall算法
    switch (parallel_strategy) {
        case 0: // 默认策略 - 并行化i循环，动态调度
            for (int k = 0; k < n; ++k) {
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (dist[i][k] != INF && dist[k][j] != INF) {
                            dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                        }
                    }
                }
            }
            break;
            
        case 1: // 并行化i循环，静态调度
            for (int k = 0; k < n; ++k) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (dist[i][k] != INF && dist[k][j] != INF) {
                            dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                        }
                    }
                }
            }
            break;
            
        case 2: // 并行化j循环，动态调度
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < n; ++i) {
                    #pragma omp parallel for schedule(dynamic)
                    for (int j = 0; j < n; ++j) {
                        if (dist[i][k] != INF && dist[k][j] != INF) {
                            dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                        }
                    }
                }
            }
            break;
            
        case 3: // 并行化j循环，静态调度
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < n; ++i) {
                    #pragma omp parallel for schedule(static)
                    for (int j = 0; j < n; ++j) {
                        if (dist[i][k] != INF && dist[k][j] != INF) {
                            dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                        }
                    }
                }
            }
            break;
            
        case 4: // 使用任务并行
            for (int k = 0; k < n; ++k) {
                #pragma omp parallel
                {
                    #pragma omp single
                    {
                        for (int i = 0; i < n; ++i) {
                            #pragma omp task
                            {
                                for (int j = 0; j < n; ++j) {
                                    if (dist[i][k] != INF && dist[k][j] != INF) {
                                        dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
    }
}

// 处理测试文件并输出结果
void processTestFile(const std::string& filename, const Graph& graph) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开测试文件: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int source, target;
        ss >> source >> target;
        
        // 检查顶点是否存在
        if (graph.id_to_index.find(source) == graph.id_to_index.end() ||
            graph.id_to_index.find(target) == graph.id_to_index.end()) {
            std::cout << "INF" << std::endl;
            continue;
        }
        
        int u = graph.id_to_index.at(source);
        int v = graph.id_to_index.at(target);
        
        float distance = graph.dist[u][v];
        if (distance == INF) {
            std::cout << "INF" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(6) << distance << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <邻接表文件> <测试文件> [线程数] [并行策略]" << std::endl;
        std::cerr << "并行策略选项:" << std::endl;
        std::cerr << "  0: 并行化i循环，动态调度 (默认)" << std::endl;
        std::cerr << "  1: 并行化i循环，静态调度" << std::endl;
        std::cerr << "  2: 并行化j循环，动态调度" << std::endl;
        std::cerr << "  3: 并行化j循环，静态调度" << std::endl;
        std::cerr << "  4: 使用任务并行" << std::endl;
        return 1;
    }
    
    std::string adj_file = argv[1];
    std::string test_file = argv[2];
    int num_threads = (argc > 3) ? std::stoi(argv[3]) : omp_get_max_threads();
    int parallel_strategy = (argc > 4) ? std::stoi(argv[4]) : 0;
    
    // 读取图
    Graph graph = readGraph(adj_file);
    std::cout << "图加载完成，顶点数: " << graph.num_vertices << std::endl;
    
    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();
    
    // 使用Floyd-Warshall算法计算多源最短路径
    floydWarshall(graph, num_threads, parallel_strategy);
    
    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // 输出计算时间
    std::cout << "计算时间: " << elapsed.count() << " 秒" << std::endl;
    std::cout << "使用线程数: " << num_threads << std::endl;
    std::cout << "并行策略: " << parallel_strategy << std::endl;
    
    // 处理测试文件
    std::cout << "测试结果:" << std::endl;
    processTestFile(test_file, graph);
    
    return 0;
}
