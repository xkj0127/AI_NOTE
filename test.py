import json
import os
import re
import time
import sys
import networkx as nx
from openai import OpenAI
import warnings
from pyvis.network import Network
import tiktoken
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import chromadb
from collections import defaultdict
from functools import wraps
warnings.filterwarnings('ignore')
sys.path.append('../')
os.environ['API_KEY'] = 'sk-xx'
api_key = os.environ.get("API_KEY")
client = chromadb.PersistentClient(path="./chroma_db")
collection_relation = client.get_or_create_collection(name="my_collection_relation")
embeddings = SentenceTransformer(
    r'D:\Models_Home\Huggingface\models--BAAI--bge-base-zh\snapshots\0e5f83d4895db7955e4cb9ed37ab73f7ded339b6'
)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 更高精度的计时
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.6f} 秒")
        return result
    return wrapper


def build_bidirectional_mapping(data):
    mapping = {
        "entity_to_label": {},
        "label_to_entities": defaultdict(list)
    }
    seen_entities = set()

    for entity, label in data:
        if entity not in seen_entities:
            mapping["entity_to_label"][entity] = label
            mapping["label_to_entities"][label].append(entity)
            seen_entities.add(entity)

    return mapping


def get_entity_label(knowledge_graph,entity):
    return knowledge_graph["entity_to_label"].get(entity, "未知标签")

def get_entities_by_label(knowledge_graph,label):
    return knowledge_graph["label_to_entities"].get(label, [])

def normalize_text(text):
    """去除首尾空格、合并多余空格、换行转换为空格"""
    return re.sub(r'\s+', ' ', text.strip())
def replace_blocks_and_find_changes(original_blocks, new_text,split_text_fun):
    """用原文块替换未变部分，找出新增和删除的块"""
    normalized_new_text = normalize_text(new_text)  # 归一化新文本
    replaced_text = normalized_new_text  # 复制新文本
    matched_blocks = set()  # 记录匹配的块文本

    # **第一步**：替换未变的部分
    for bid, text in original_blocks:
        norm_text = normalize_text(text)
        if norm_text in replaced_text:
            replaced_text = replaced_text.replace(norm_text, bid, 1)
            matched_blocks.add(norm_text)

    # **第二步**：计算删除的块（原文本中未出现在新文本中的部分）
    deleted_blocks = [(bid, text) for bid, text in original_blocks if normalize_text(text) not in normalized_new_text]

    # **第三步**：用 `block_id` 作为分隔符，分割出变动部分
    split_pattern = '|'.join(re.escape(bid) for bid, _ in original_blocks)
    unmatched_parts = re.split(split_pattern, replaced_text)  # 只保留变动部分
    unmatched_parts = [normalize_text(part) for part in unmatched_parts if part.strip()]  # 清理空格

    # **第四步**：用你的 `split_text()` 切割新增内容
    added_texts = []
    for part in unmatched_parts:
        added_texts.extend([t for t in split_text_fun(part) if t])

    # **第五步**：分配新增块 ID
    added_blocks = [text for i, text in enumerate(added_texts)]

    return replaced_text, deleted_blocks, added_blocks


class SemanticTextSplitter:
    """
    增强版文本分割器，结合语义分析和实体边界检测

    功能特点：
    1. 语义连贯性分析
    2. 实体边界保护
    3. 动态调整分割阈值
    4. 内容类型自适应
    5. 预加载模型提高效率
    """

    SPLIT_PUNCTUATION = [
        '\n\n', '\n', '。', '！', '？', '；', '…',
        '. ', '! ', '? ', '; ', '... ', ', ', '、'
    ]

    def __init__(self,
                 max_tokens: int = 2000,
                 min_tokens: int = 500,
                 overlap_tokens: int = 0,
                 semantic_threshold: float = 0.80,
                 enforce_entity_boundary: bool = False):
        """
        初始化文本分割器

        参数:
            max_tokens: 每个块的推荐最大token数(实际可能根据语义调整)
            min_tokens: 寻找断点的最小token数
            overlap_tokens: 块之间重叠的token数
            semantic_threshold: 语义分割的相似度阈值(0-1)
            enforce_entity_boundary: 是否强制不在实体中间分割
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.semantic_threshold = semantic_threshold
        self.enforce_entity_boundary = enforce_entity_boundary

        # 预加载模型（单例）
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("zh_core_web_sm")

        # 参数校验
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens 必须小于 max_tokens")
        if self.overlap_tokens >= self.min_tokens:
            raise ValueError("overlap_tokens 应小于 min_tokens")

    def split_text(self, text: str, doc_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        分割文本方法

        参数:
            text: 要分割的文本
            doc_id: 文档级标识符

        返回:
            元组列表，格式为 (block_id, chunk_text)
        """
        # 1. 预处理阶段
        text_clean, tags = self._preprocess_text(text)

        # 2. 实体边界检测
        entity_boundaries = self._get_entity_boundaries(text_clean) if self.enforce_entity_boundary else set()

        # 3. 主分割逻辑
        tokens = self.encoder.encode(text_clean)
        chunks = []
        start_idx = 0
        chunk_counter = 1

        while start_idx < len(tokens):
            # 动态调整实际max_tokens（±20%）
            adjusted_max = min(
                int(self.max_tokens * (1 + 0.2 * (chunk_counter % 3 - 1))),  # 波动调整
                len(tokens) - start_idx
            )
            end_idx = min(start_idx + adjusted_max, len(tokens))

            # 短内容直接作为最后一块
            if len(tokens) - start_idx <= self.min_tokens + self.overlap_tokens:
                chunk_tokens = tokens[start_idx:]
                chunk_text = self._reconstruct_text(chunk_tokens, tags)
                bid = self._generate_block_id(chunk_text, chunk_counter, doc_id)
                chunks.append((bid, chunk_text))
                break

            # 获取候选分割区间
            candidate_text = self.encoder.decode(tokens[start_idx:end_idx])
            best_split_pos = self._find_best_split_position(
                candidate_text, start_idx, end_idx, entity_boundaries
            )

            # 生成当前块
            chunk_tokens = tokens[start_idx:best_split_pos]
            chunk_text = self._reconstruct_text(chunk_tokens, tags)

            # 设置下一个块的起始位置（考虑重叠）
            next_start = max(start_idx, best_split_pos - self.overlap_tokens)

            # 添加到结果
            bid = self._generate_block_id(chunk_text, chunk_counter, doc_id)
            chunks.append((bid, chunk_text))
            chunk_counter += 1
            start_idx = next_start

        return chunks

    def _preprocess_text(self, text: str) -> Tuple[str, list]:
        """处理特殊内容并分析文档结构"""
        tag_pattern = re.compile(r'(<(img|table|code)[^>]*>.*?</\2>)', re.DOTALL)
        tags = []

        def replace_tag(match):
            tags.append(match.group(1))
            return f"__TAG_{len(tags) - 1}__"

        text_clean = tag_pattern.sub(replace_tag, text)
        return text_clean, tags

    def _analyze_semantic_breaks(self, text: str) -> List[int]:
        """使用语义相似度检测自然断点"""
        sentences = [sent.text for sent in self.nlp(text).sents]
        if len(sentences) < 2:
            return []

        embeddings = self.semantic_model.encode(sentences)
        similarities = []
        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i - 1], embeddings[i])
            similarities.append(sim)

        # 找出语义变化大的位置
        break_points = [i for i, sim in enumerate(similarities)
                        if sim < self.semantic_threshold]
        return break_points

    def _get_entity_boundaries(self, text: str) -> set:
        """获取实体边界位置"""
        doc = self.nlp(text)
        boundaries = set()
        for ent in doc.ents:
            start = len(self.encoder.encode(text[:ent.start_char]))
            end = len(self.encoder.encode(text[:ent.end_char]))
            boundaries.update(range(start, end))
        return boundaries

    def _find_best_split_position(self, candidate_text: str, start_idx: int, end_idx: int,
                                  entity_boundaries: set) -> int:
        """寻找最佳分割位置"""
        # 优先级1：语义断点
        semantic_breaks = self._analyze_semantic_breaks(candidate_text)
        if semantic_breaks:
            semantic_pos = start_idx + len(self.encoder.encode(candidate_text[:semantic_breaks[0]]))
            if semantic_pos - start_idx >= self.min_tokens:
                return semantic_pos

        # 优先级2：标点断点
        for punct in self.SPLIT_PUNCTUATION:
            punct_pos = candidate_text.rfind(punct)
            if punct_pos != -1:
                punct_token_pos = len(self.encoder.encode(candidate_text[:punct_pos + len(punct)]))
                candidate_pos = start_idx + punct_token_pos

                # 检查是否满足最小长度且不破坏实体
                if (punct_token_pos >= self.min_tokens and
                        not any(pos in entity_boundaries for pos in range(candidate_pos - 3, candidate_pos + 3))):
                    return candidate_pos

        # 优先级3：安全位置（避开实体）
        for pos in range(end_idx, start_idx + self.min_tokens - 1, -1):
            if pos not in entity_boundaries:
                return pos

        # 最终回退策略
        return end_idx

    def _reconstruct_text(self, tokens: list, tags: list) -> str:
        """重建文本并恢复被替换的标签"""
        text = self.encoder.decode(tokens)
        for j, tag in enumerate(tags):
            text = text.replace(f"__TAG_{j}__", tag)
        return text

    def _generate_block_id(self, text: str, counter: int, doc_id: Optional[str]) -> str:
        """生成块ID"""
        prefix = f"{doc_id}_" if doc_id else ""
        return f"{prefix}block_{counter}_{hash(text[:50])}"


class DeepSeekAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )


    def temp_sleep(self,seconds=0.1):
        time.sleep(seconds)

    def ollama_safe_generate_response(self,prompt,input_parameter,repeat=3):
        for i in range(repeat):
            # print(f"repeat:{i}")
            try:
                curr_gpt_response = self.ollama_request(prompt,input_parameter)
                # print(curr_gpt_response,type(curr_gpt_response))
                x = ""
                if 'json' in curr_gpt_response:
                    pattern = r"```json\s*({.*?})\s*```"
                    match = re.search(pattern, curr_gpt_response, re.DOTALL)
                    if match:
                        json_content = match.group(1)
                        # print(json_content,222222222222222222)
                        x = json.loads(json_content)
                    else:
                        print("未找到匹配的 JSON 内容")
                        continue
                return x
            except:
                print("ERROR")
        return -1

    def ollama_request(self,prompt,input_parameter):
        self.temp_sleep()

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content":prompt},
                {'role': 'user', 'content': input_parameter}],
            temperature=1
        )
        output = response.choices[0].message.content

        return output

    @timing_decorator
    def rag_local(self,query,knowledge_base):
        prompt = open("./prompt/v2/rag_v1.txt", encoding='utf-8').read()
        input_parameter = open("./prompt/v2/rag_v1_query.txt", encoding='utf-8').read()
        knowledges = "\n".join(knowledge_base)
        input_parameter = input_parameter.replace("{{query}}", query)
        input_parameter = input_parameter.replace("{{context}}", knowledges)
        output = self.ollama_safe_generate_response(prompt, input_parameter)
        print(input_parameter, 1111111111111111111111111)
        return output


    @timing_decorator
    def hybrid_rag(self,query,knowledge_base):
        embedding111 = embeddings.encode(query)
        rag = collection_relation.query(embedding111, n_results=3)
        prompt = open("./prompt/v2/rag_v1_hybrid.txt", encoding='utf-8').read()
        input_parameter = open("./prompt/v2/rag_v1_query_hy.txt", encoding='utf-8').read()
        knowledges = "\n".join(knowledge_base)
        input_parameter = input_parameter.replace("{{query}}", query)
        input_parameter = input_parameter.replace("{{relation}}", knowledges)
        input_parameter = input_parameter.replace("{{context}}", "\n".join(*rag['documents']))
        print(input_parameter,1111111111111111111111111)
        output = self.ollama_safe_generate_response(prompt, input_parameter)

        return output

    @timing_decorator
    def 创建(self,Bolts):
        xx = []
        entitie_label = []
        prompt = open("./prompt/v2/entity_extraction.txt", encoding='utf-8').read()
        prompt2 = open("./prompt/v2/relationship_extraction.txt", encoding='utf-8').read()
        for bid,Bolt in Bolts:
            input_parameter = Bolt
            output = self.ollama_safe_generate_response(prompt, input_parameter)
            entitie_label+=output["entities"]

            # print(output)

            output2 = self.ollama_safe_generate_response(prompt2,
                                                          "笔记内容：" + input_parameter + "\n实体列表：" + json.dumps(
                                                              output['entities']))
            # print(output2['relations'])

            xx += [{"bid":bid,"relation":output2['relations']}]
        entitie_label = {"entities":entitie_label}
        open("实体-实体类型-存储.json", mode="w", encoding='utf-8').write(
            json.dumps(entitie_label, ensure_ascii=False, indent=4))
        return xx

    @timing_decorator
    def 知识融合(self,relations):
        prompt = open("./prompt/v2/tune_graph.txt", encoding='utf-8').read()
        output = self.ollama_safe_generate_response(prompt, json.dumps(relations))
        relations_tuning = output
        # print(relations_tuning['output'])
        return relations_tuning['output']

    @timing_decorator
    def 增量更新(self,original_blocks,new_text,relations):

        replaced_new_text, deleted_blocks, added_blocks = replace_blocks_and_find_changes(original_blocks, new_text,
                                                                                          splitter.split_text)

        # 输出
        print("新文本替换后的文本:")
        print(replaced_new_text)

        print("\n被删除的块:")
        for bid, text in deleted_blocks:
            print(f"{bid}: {text}")

        print("\n新增的块:")
        for bid, text in added_blocks:
            print(f"{bid}: {text}")
        bids_to_remove=[]

        # 被删除的块
        for bid, text in deleted_blocks:
            bids_to_remove.append(bid)
            # print(bids_to_remove)

        filtered_data = [item for item in relations if item['bid'] not in bids_to_remove]


        add_data = []
        # 新增的块
        for bid, text in added_blocks:
            add_data.append((bid,text))
        # print(add_data,2222222)

        relations_new = self.创建(add_data)
        out = relations_new+filtered_data
        # print(out)
        return out

    def 绘制知识图谱(self, relations, name):
        # 读取实体数据
        with open("实体-实体类型-存储.json", mode="r", encoding='utf-8') as f:
            entity_obj = json.load(f)['entities']

        knowledge_graph = build_bidirectional_mapping(entity_obj)

        # 创建有向图
        G = nx.DiGraph()

        # 添加节点和边
        for relation in relations:
            for rel in relation['relation']:
                source = rel['source']
                target = rel['target']
                context = rel['context']
                relation_type = rel['relation']

                # 添加节点
                G.add_node(source,
                           title=get_entity_label(knowledge_graph, source),
                           color='#97c2fc',
                           group=get_entity_label(knowledge_graph, source))
                G.add_node(target,
                           title=get_entity_label(knowledge_graph, target),
                           color='#97c2fc',
                           group=get_entity_label(knowledge_graph, target))

                # 添加边（初始状态）
                G.add_edge(source, target,
                           title=context,
                           label=relation_type,
                           font={"size": 0},  # 初始标签隐藏
                           color='#97c2fc',
                           width=2,
                           hoverWidth=4,
                           chosen={  # 点击选中样式
                               "edge": {
                                   "color": "#00FF00",
                                   "width": 4
                               }
                           })

        # 使用pyvis可视化
        net = Network(notebook=True, height="900px", width="100%",
                      bgcolor="#ffffff", font_color="black", directed=True)

        # 配置选项
        options = {
            "edges": {
                "font": {
                    "size": 0,
                    "face": "arial",
                    "align": "middle"
                },
                "color": {
                    "inherit": False,
                    "highlight": "#FFA500",
                    "hover": "#FFA500"
                },
                "selectionWidth": 1.5,
                "smooth": {"type": "continuous"}
            },
            "interaction": {
                "hover": True,
                "tooltipDelay": 150,
                "hideEdgesOnDrag": False,
                "multiselect": True  # 允许多选
            }
        }
        net.set_options(json.dumps(options, indent=2))

        # 从NetworkX图导入数据
        net.from_nx(G)

        # 生成HTML文件
        html_file = f"{name}.html"
        net.show(html_file)

        # 增强交互功能
        with open(html_file, "r+", encoding="utf-8") as f:
            content = f.read()

            js_injection = """
            <style>
                .control-panel {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    z-index: 1000;
                    background: rgba(255,255,255,0.9);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                }
                .control-btn {
                    padding: 8px 12px;
                    margin: 5px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: all 0.3s;
                }
                .control-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }
                #showAllBtn {
                    background-color: #4CAF50;
                    color: white;
                }
                #hideAllBtn {
                    background-color: #f44336;
                    color: white;
                }
                #toggleBtn {
                    background-color: #2196F3;
                    color: white;
                }
                #resetBtn {
                    background-color: #9E9E9E;
                    color: white;
                }
                .status-indicator {
                    margin-top: 10px;
                    font-size: 12px;
                    color: #555;
                }
            </style>

            <script>
            // 全局状态管理
            const edgeStates = {};
            let globalHideMode = true;

            document.addEventListener("DOMContentLoaded", function() {
                // 初始化所有边状态
                network.body.data.edges.get().forEach(edge => {
                    edgeStates[edge.id] = {
                        clicked: false,
                        labelVisible: false
                    };
                });

                // 创建控制面板
                const container = document.getElementById("mynetwork");
                const panel = document.createElement("div");
                panel.className = "control-panel";
                panel.innerHTML = `
                    <button id="showAllBtn" class="control-btn">显示所有标签</button>
                    <button id="hideAllBtn" class="control-btn">隐藏未点击标签</button>
                    <button id="toggleBtn" class="control-btn">切换显示状态</button>
                    <button id="resetBtn" class="control-btn">重置所有状态</button>
                    <div class="status-indicator">已复习: <span id="counter">0</span>/${network.body.data.edges.get().length}</div>
                `;
                container.parentNode.insertBefore(panel, container);

                // 更新计数器
                function updateCounter() {
                    const count = Object.values(edgeStates).filter(s => s.clicked).length;
                    document.getElementById("counter").innerText = count;
                }

                // 显示所有标签
                document.getElementById("showAllBtn").onclick = function() {
                    network.body.data.edges.get().forEach(edge => {
                        edge.font = {size: 14};
                        edge.color = {color: "#97c2fc"};
                        network.body.data.edges.update(edge);
                        edgeStates[edge.id].labelVisible = true;
                    });
                    globalHideMode = false;
                    updateCounter();
                };

                // 隐藏未点击标签
                document.getElementById("hideAllBtn").onclick = function() {
                    network.body.data.edges.get().forEach(edge => {
                        if (!edgeStates[edge.id].clicked) {
                            edge.font = {size: 0};
                            edge.color = {color: "#97c2fc"};
                            network.body.data.edges.update(edge);
                            edgeStates[edge.id].labelVisible = false;
                        }
                    });
                    globalHideMode = true;
                    updateCounter();
                };

                // 切换显示状态
                document.getElementById("toggleBtn").onclick = function() {
                    globalHideMode = !globalHideMode;
                    network.body.data.edges.get().forEach(edge => {
                        edge.font = {size: globalHideMode && !edgeStates[edge.id].clicked ? 0 : 14};
                        network.body.data.edges.update(edge);
                        edgeStates[edge.id].labelVisible = !globalHideMode || edgeStates[edge.id].clicked;
                    });
                    updateCounter();
                };

                // 重置所有状态
                document.getElementById("resetBtn").onclick = function() {
                    network.body.data.edges.get().forEach(edge => {
                        edge.font = {size: 0};
                        edge.color = {color: "#97c2fc"};
                        network.body.data.edges.update(edge);
                        edgeStates[edge.id] = {
                            clicked: false,
                            labelVisible: false
                        };
                    });
                    globalHideMode = true;
                    updateCounter();
                };

                // 点击边持久化显示
                network.on("selectEdge", function(params) {
                    const edge = network.body.data.edges.get(params.edges[0]);
                    edgeStates[edge.id].clicked = true;
                    edge.font = {size: 14};
                    edge.color = {color: "#00FF00", highlight: "#00FF00"};
                    network.body.data.edges.update(edge);
                    updateCounter();
                });

                // 悬停边时高亮
                network.on("hoverEdge", function(params) {
                    const edge = network.body.data.edges.get(params.edge);
                    if (!edgeStates[edge.id].clicked) {
                        edge.color = {color: "#FFA500", highlight: "#FFA500"};
                        network.body.data.edges.update(edge);
                    }
                });

                // 移出边时恢复
                network.on("blurEdge", function(params) {
                    const edge = network.body.data.edges.get(params.edge);
                    if (!edgeStates[edge.id].clicked) {
                        edge.color = {color: "#97c2fc", highlight: "#97c2fc"};
                        network.body.data.edges.update(edge);
                    }
                });

                updateCounter();
            });
            </script>
            """
            content = content.replace("</body>", js_injection + "</body>")
            f.seek(0)
            f.write(content)
            f.truncate()

        print(f"知识图谱已生成，保存为 {html_file}")
        return G

    @timing_decorator
    def text2entity(self,text, entity=None):
        prompt = open("./prompt/v2/entity_q2merge.txt", encoding='utf-8').read()
        entity111 = [str(i) for i in entity]
        input_parameter = f"实体列表：{entity111}\n问题：{text}"
        # print(input_parameter)
        output = self.ollama_safe_generate_response(prompt, input_parameter)
        return output['entities']

@timing_decorator
def compare_and_visualize(G1, G2, output_file="diff_graph.html"):
    """比较两个有向图并用pyvis高亮差异"""
    # 创建合并图（包含G1和G2的所有节点和边）
    G_diff = nx.DiGraph()

    # 记录差异
    diff = {
        "added_nodes": set(G2.nodes()) - set(G1.nodes()),
        "removed_nodes": set(G1.nodes()) - set(G2.nodes()),
        "added_edges": set(G2.edges()) - set(G1.edges()),
        "removed_edges": set(G1.edges()) - set(G2.edges()),
        "node_attr_changes": {},
        "edge_attr_changes": {}
    }

    # 检查节点属性变化
    common_nodes = set(G1.nodes()) & set(G2.nodes())
    for node in common_nodes:
        if G1.nodes[node] != G2.nodes[node]:
            diff["node_attr_changes"][node] = {
                "old": G1.nodes[node],
                "new": G2.nodes[node]
            }

    # 检查边属性变化
    common_edges = set(G1.edges()) & set(G2.edges())
    for u, v in common_edges:
        if G1.edges[u, v] != G2.edges[u, v]:
            diff["edge_attr_changes"][(u, v)] = {
                "old": G1.edges[u, v],
                "new": G2.edges[u, v]
            }

    # 将差异信息添加到图
    for node in G1.nodes() | G2.nodes():
        G_diff.add_node(node)

        # 设置节点颜色和标题（悬停显示详情）
        if node in diff["added_nodes"]:
            G_diff.nodes[node]["color"] = "green"
            G_diff.nodes[node]["title"] = f"新增节点: {node}"
        elif node in diff["removed_nodes"]:
            G_diff.nodes[node]["color"] = "red"
            G_diff.nodes[node]["title"] = f"删除节点: {node}"
        elif node in diff["node_attr_changes"]:
            G_diff.nodes[node]["color"] = "yellow"
            changes = diff["node_attr_changes"][node]
            G_diff.nodes[node]["title"] = (
                f"节点属性修改: {node}\n"
                f"旧值: {changes['old']}\n"
                f"新值: {changes['new']}"
            )
        else:
            G_diff.nodes[node]["color"] = "skyblue"

    for u, v in G1.edges() | G2.edges():
        if (u, v) in diff["added_edges"]:
            G_diff.add_edge(u, v, color="green", title=f"新增边: ({u}→{v})")
        elif (u, v) in diff["removed_edges"]:
            G_diff.add_edge(u, v, color="red", title=f"删除边: ({u}→{v})")
        elif (u, v) in diff["edge_attr_changes"]:
            changes = diff["edge_attr_changes"][(u, v)]
            G_diff.add_edge(u, v, color="yellow",
                            title=f"边属性修改: ({u}→{v})\n旧值: {changes['old']}\n新值: {changes['new']}")
        else:
            G_diff.add_edge(u, v, color="gray")

    # 用pyvis绘制动态图
    nt = Network( height="900px", width="100%", notebook=True)
    nt.from_nx(G_diff)


    # 保存并显示
    nt.show(output_file)




from community import community_louvain
@timing_decorator
def community_louvain_G(G, entity_names):
    knowledge_base = []
    # 执行社区检测（在整个图上）
    partition = community_louvain.best_partition(G.to_undirected())

    # 输出每个节点所属的社区编号
    # print("Community partition:")
    for node, community_id in partition.items():
        pass
        # print(f"Node {node} belongs to community {community_id}")

    # 获取每个输入实体的社区编号
    community_ids = set()
    for entity in entity_names:
        if entity in partition:
            community_ids.add(partition[entity])

    # 提取特定社区内的所有节点
    community_nodes = [node for node, comm_id in partition.items() if comm_id in community_ids]

    # 构建包含选定社区内所有节点的子图
    subgraph = G.subgraph(community_nodes)

    # 输出结果
    # print("Selected Community Nodes and Edges:")

    # 打印每个节点及其社区编号和属性
    for node in subgraph.nodes(data=True):
        node_name = node[0]
        node_attributes = node[1]
        community_id = partition.get(node_name, 'No Community')

        # print(f"Node: {node_name}, Attributes: {node_attributes}, Community ID: {community_id}")

    # 打印边的信息，检查是否存在'title'属性，如果不存在则使用默认值
    for edge in subgraph.edges(data=True):
        relation = edge[2].get('title', 'No Title')  # 如果没有'title'属性，则返回默认值'No Title'
        # print(edge,111111111111111)
        print(f"Edge from {edge[0]} to {edge[1]}, Relation: {relation}")
        knowledge_base.append(f"Edge from {edge[0]} to {edge[1]}, Relation: {relation}, context:{edge[2].get('title', 'No Title')}")

    # 如果需要更详细的社区信息，可以计算模块度等
    modularity = community_louvain.modularity(partition, G.to_undirected())
    print(f"\nModularity of the entire graph: {modularity}")

    return knowledge_base





if __name__ == '__main__':
    splitter = SemanticTextSplitter(2045,1024)
    agent = DeepSeekAgent()
    input_parameter = open("./docs/test_text.txt", encoding='utf-8').read()
    Bolts = splitter.split_text(input_parameter)
    documents = []
    embed = []

    ids = []
    for bid, Bolt in Bolts:
        ids.append(bid)
        documents.append(Bolt)
        embed.append(embeddings.encode(Bolt))



    collection_relation.add(
        documents=documents,
        embeddings=embed,
        ids=ids,
    )
    # print(Bolts,111111111)
    relations = agent.创建(Bolts)

    g1 = agent.绘制知识图谱(relations,"gra1")
    # relations2 = agent.知识融合(relations)
    # g1t = agent.绘制知识图谱(relations2,"gra1tune")
    input_parameter2 = open("./docs/test_text4.txt", encoding='utf-8').read()
    relations2 = agent.增量更新(Bolts,input_parameter2,relations)

    g2 = agent.绘制知识图谱(relations2,"gra2")
    # compare_and_visualize(g1,g1t)


    for i in range(3):
        print("\n")
        q = input("请输入问题")
        entitys = agent.text2entity(q,g1)
                             # print(entitys,1111111111111111)
        knowledges = community_louvain_G(g1, entitys)
        # print(knowledges,1111)
        # print(g1,111111)
        anw = agent.rag_local(q,knowledges)
        print(f"回答：{"\n".join(anw['answer'])}\n参考资料：\n{"\n".join(anw['material'])}")

        anw2 = agent.hybrid_rag(q, knowledges)
        print(f"回答：{"\n".join(anw2['answer'])}\n参考资料：\n{"\n".join(anw2['material'])}")








