import re
import time
import warnings
import gradio as gr
from pathlib import Path
import json
import os
import webbrowser
from typing import List, Dict, Tuple, Optional
import networkx as nx
import numpy as np
import spacy
import tiktoken
from openai import OpenAI
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

os.environ['API_KEY'] = 'sk-506f969de9e444c8b445e91d542127bc'
api_key = os.environ.get("API_KEY")


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

splitter = SemanticTextSplitter(512, 128)

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


    def 创建(self,Bolts):
        xx = []
        prompt = open("./prompt/v2/entity_extraction.txt", encoding='utf-8').read()
        prompt2 = open("./prompt/v2/relationship_extraction.txt", encoding='utf-8').read()
        for bid,Bolt in Bolts:
            input_parameter = Bolt
            output = self.ollama_safe_generate_response(prompt, input_parameter)
            # print(output)

            output2 = self.ollama_safe_generate_response(prompt2,"笔记内容：" + input_parameter + "\n实体列表：" + json.dumps(output['entities']))
            # print(output2['relations'])
            xx += [{"bid":bid,"relation":output2['relations']}]
        return xx


    def 知识融合(self,relations):
        pass
        relations_tuning = relations
        return relations_tuning


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







    def 绘制知识图谱(self,relations,name):
        # 创建有向图
        G = nx.DiGraph()
        # 添加节点和边
        for relation in relations:
            for rel in relation['relation']:
                source = rel['source']
                target = rel['target']
                relation = rel['relation']

                # 添加节点
                G.add_node(source, title=source, color='#97c2fc')  # 浅蓝色
                G.add_node(target, title=target, color='#97c2fc')  # 浅蓝色

                # 添加边
                G.add_edge(source, target, title=relation, label=relation)

        # 使用pyvis可视化
        net = Network(notebook=True, height="900px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)

        # 保存并显示
        net.show(f"{name}.html")
        print(f"知识图谱已生成，保存为 {name}.html")
        return G


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

# 这里保留您原有的所有类定义 (SemanticTextSplitter, DeepSeekAgent等)
# 为了简洁，我假设它们已经在同一个文件中定义或已导入

# --------------------------
# Gradio界面核心功能
# --------------------------

class GradioApp:
    def __init__(self):
        self.agent = DeepSeekAgent()
        self.current_relations = []
        self.current_blocks = []
        self.graphs = {"原始": None, "更新后": None}
        self.original_text = ""

    def process_initial_files(self, files: List[str]) -> Dict:
        """处理初始文件"""
        filepaths = [f.name for f in files]
        text_content = ""
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                text_content += f.read() + "\n\n"

        self.original_text = text_content
        self.current_blocks = splitter.split_text(text_content)
        self.current_relations = self.agent.创建(self.current_blocks)

        # 生成初始知识图谱
        self.graphs["原始"] = self.agent.绘制知识图谱(self.current_relations, "原始图谱")

        return {
            "原始文本": text_content,
            "块数量": len(self.current_blocks),
            "关系数量": sum(len(r['relation']) for r in self.current_relations)
        }

    def process_update(self, new_text: str) -> Dict:
        """处理增量更新"""
        if not self.current_blocks:
            return {"错误": "请先上传初始文件"}

        # 执行增量更新
        updated_relations = self.agent.增量更新(
            self.current_blocks,
            new_text,
            self.current_relations
        )

        # 更新当前状态
        self.current_relations = updated_relations
        self.graphs["更新后"] = self.agent.绘制知识图谱(updated_relations, "更新图谱")

        # 生成差异图谱
        if self.graphs["原始"] and self.graphs["更新后"]:
            compare_and_visualize(
                self.graphs["原始"],
                self.graphs["更新后"],
                "差异图谱.html"
            )

        return {
            "更新结果": "增量更新完成",
            "新关系数量": sum(len(r['relation']) for r in updated_relations)
        }

    def get_html_report(self, report_type: str) -> str:
        """获取HTML报告内容"""
        filename = {
            "原始图谱": "原始图谱.html",
            "更新图谱": "更新图谱.html",
            "差异图谱": "差异图谱.html"
        }.get(report_type, "")

        if filename and Path(filename).exists():
            with open(filename, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # 创建一个适应Gradio的iframe包装
            return f"""
            <div style="width:100%; height:600px;">
                <iframe srcdoc='{html_content.replace("'", "&apos;")}' 
                        style="width:100%; height:100%; border:none;"></iframe>
            </div>
            """
        return f"<p>{report_type}尚未生成</p>"

    def open_in_browser(self, report_type: str):
        """在浏览器中打开报告"""
        filename = {
            "原始图谱": "原始图谱.html",
            "更新图谱": "更新图谱.html",
            "差异图谱": "差异图谱.html"
        }.get(report_type, "")

        if filename and Path(filename).exists():
            webbrowser.open(f"file://{Path(filename).absolute()}")
            return f"已在浏览器中打开 {report_type}"
        return f"无法打开 {report_type}"

    def get_current_text(self) -> str:
        """获取当前所有块的文本内容"""
        return "\n".join([text for _, text in self.current_blocks])


# --------------------------
# Gradio界面布局
# --------------------------

def create_interface():
    app = GradioApp()
    # 设置默认文件路径和默认文本
    DEFAULT_FILE = "test_text.txt"  # 你可以改为你想要的默认文件路径
    DEFAULT_TEXT = """
    # 计算机科学
1. 哈希表
   - 哈希表（Hash Table）是一种数据结构，通过键值对存储数据。
   - 它使用哈希函数计算键的索引，实现O(1)时间复杂度的查找。
   - 冲突解决方法：链地址法（Chaining）、开放寻址法（Open Addressing）。
   - 哈希表在数据库索引和缓存系统（如Redis）中广泛应用。

2. 深度学习
   - 神经网络由输入层、隐藏层、输出层组成。
   - 反向传播用于优化模型参数。
   - 常见框架：PyTorch 2.0、TensorFlow、JAX。

3. 红黑树
   - 一种自平衡二叉搜索树，保证O(log n)时间复杂度。
   - 规则：节点是红/黑，根节点是黑，红色节点的子节点必须是黑。

# 数学
1. 线性代数
   - 矩阵乘法不满足交换律：A×B ≠ B×A。
   - 特征值和特征向量：Av = λv。
   - 奇异值分解（SVD）用于降维和推荐系统。
    """

    # 检查并创建默认文件（如果不存在）
    if not Path(DEFAULT_FILE).exists():
        with open(DEFAULT_FILE, "w", encoding="utf-8") as f:
            f.write("""这是默认的示例文本文件。
    包含一些初始内容用于构建知识图谱。
    你可以上传自己的文件替换它。""")
    with gr.Blocks(title="知识图谱构建系统", theme="soft") as demo:
        gr.Markdown("# 知识图谱构建与增量更新系统")

        with gr.Tabs():
            with gr.Tab("初始处理"):
                with gr.Row():
                    file_input = gr.File(
                        file_count="multiple",
                        file_types=[".txt"],
                        label="上传文本文件",
                        value=[DEFAULT_FILE]  # 设置默认文件
                    )
                    process_btn = gr.Button("处理文件", variant="primary")

                initial_stats = gr.JSON(label="处理结果")

                with gr.Row():
                    original_graph = gr.HTML(label="原始知识图谱")
                    with gr.Column():
                        gr.Markdown("### 原始图谱操作")
                        show_original_btn = gr.Button("显示原始图谱")
                        open_original_btn = gr.Button("在浏览器中打开")

            with gr.Tab("增量更新"):
                with gr.Row():
                    # 左侧列 - 原文显示
                    with gr.Column(scale=1):
                        original_text_display = gr.Textbox(
                            lines=20,
                            label="当前原文内容",
                            interactive=False
                        )
                        refresh_btn = gr.Button("刷新原文显示")

                    # 右侧列 - 更新输入
                    with gr.Column(scale=1):
                        text_update = gr.Textbox(
                            lines=10,
                            label="输入更新后的文本",
                            placeholder="粘贴更新后的文本内容...",
                            value=DEFAULT_TEXT  # 设置默认文本
                        )
                        update_btn = gr.Button("执行增量更新", variant="primary")


                update_stats = gr.JSON(label="更新结果")

                with gr.Tabs():
                    with gr.Tab("更新后图谱"):
                        updated_graph = gr.HTML(label="更新后知识图谱")
                        with gr.Row():
                            show_updated_btn = gr.Button("显示更新图谱")
                            open_updated_btn = gr.Button("在浏览器中打开")

                    with gr.Tab("差异对比"):
                        diff_graph = gr.HTML(label="差异图谱")
                        with gr.Row():
                            show_diff_btn = gr.Button("显示差异图谱")
                            open_diff_btn = gr.Button("在浏览器中打开")

        # 事件处理
        process_btn.click(
            fn=app.process_initial_files,
            inputs=file_input,
            outputs=initial_stats
        )

        refresh_btn.click(
            fn=lambda: "\n".join([text for _, text in app.current_blocks]),
            outputs=original_text_display
        )

        update_btn.click(
            fn=app.process_update,
            inputs=text_update,
            outputs=update_stats
        )

        # 图谱显示控制
        show_original_btn.click(
            fn=lambda: app.get_html_report("原始图谱"),
            outputs=original_graph
        )

        show_updated_btn.click(
            fn=lambda: app.get_html_report("更新图谱"),
            outputs=updated_graph
        )

        show_diff_btn.click(
            fn=lambda: app.get_html_report("差异图谱"),
            outputs=diff_graph
        )

        # 浏览器打开控制
        open_original_btn.click(
            fn=lambda: app.open_in_browser("原始图谱"),
            outputs=gr.Textbox(visible=False)
        )

        open_updated_btn.click(
            fn=lambda: app.open_in_browser("更新图谱"),
            outputs=gr.Textbox(visible=False)
        )

        open_diff_btn.click(
            fn=lambda: app.open_in_browser("差异图谱"),
            outputs=gr.Textbox(visible=False)
        )

    return demo


# --------------------------
# 运行应用
# --------------------------

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )