import camelot
import re
import fitz
import os
from typing import Union, List, Dict, Optional

# 兼容无tqdm环境
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


class PDFProcessor:
    def __init__(self, output_dir: str = "output", image_dir: str = "images"):
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.results = {}  # 存储结构：{filename: content_list}
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip() if isinstance(text, str) else text

    def _process_table_optimized(self, data):
        data = [row for row in data if any(cell.strip() for cell in row)]
        if not data:
            return data

        def process_row(row, is_header=False):
            new_row = []
            i = 0
            while i < len(row):
                cell = str(row[i]).strip()
                if is_header and '  ' in cell:
                    parts = re.split(r'\s{2,}', cell, maxsplit=1)
                    new_row.append(parts[0])
                    found = False
                    for j in range(i + 1, len(row)):
                        if str(row[j]).strip() == '':
                            new_row.append(parts[1])
                            i = j
                            found = True
                            break
                    if not found:
                        new_row.append(parts[1])
                else:
                    new_row.append(cell)
                i += 1
            return new_row

        processed_header = process_row(data[0], is_header=True)
        expected_columns = len(processed_header)
        final_data = [processed_header]

        for row in data[1:]:
            processed_row = process_row(row)
            if len(processed_row) < expected_columns:
                processed_row += [''] * (expected_columns - len(processed_row))
            else:
                processed_row = processed_row[:expected_columns]
            final_data.append(processed_row)

        for row in final_data:
            for i in range(len(row)):
                row[i] = re.sub(r'\s+', '', str(row[i]))
        return final_data

    @staticmethod
    def _is_empty_table(table_data):
        for row in table_data:
            for cell in row:
                if cell != '':
                    return False
        return True

    def _filter_empty_rows_cols(self, data):
        filtered_rows = [row for row in data if any(cell != '' for cell in row)]
        max_cols = max(len(row) for row in filtered_rows) if filtered_rows else 0
        uniform_data = [row + [''] * (max_cols - len(row)) for row in filtered_rows]
        transposed = list(zip(*uniform_data))
        filtered_cols = [col for col in transposed if any(cell != '' for cell in col)]
        result = list(zip(*filtered_cols)) if filtered_cols else []
        return [list(row) for row in result]

    def _process_tables(self, detailed_tables):
        filtered = self._filter_empty_rows_cols(detailed_tables)
        if not filtered:
            return "<table></table>"

        markdown = "| " + " | ".join(filtered[0]) + " |\n"
        for row in filtered[1:]:
            markdown += "| " + " | ".join(row) + " |\n"
        return f"<table>\n{markdown}</table>"

    def _extract_images(self, page, page_num):
        image_info_list = []
        image_blocks = page.get_images(full=True)

        for img in image_blocks:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_ext = base_image["ext"]
            image_data = base_image["image"]
            # TODO 可能多文件图片名会冲突，建议再加上文件名
            image_filename = f"page_{page_num + 1}_img_{xref}.{image_ext}"
            image_path = os.path.join(self.image_dir, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(image_data)

            image_rects = page.get_image_rects(xref)
            for rect in image_rects:
                adj_bbox = (
                    max(0, rect.x0 - 2),
                    max(0, rect.y1 - 2),
                    min(page.rect.width, rect.x1 + 2),
                    min(page.rect.height, rect.y0 + 2)
                )
                image_info_list.append({
                    "path": image_path,
                    "bbox": adj_bbox
                })

        return image_info_list

    def _process_pdf(self, pdf_path: str) -> List[str]:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        doc = fitz.open(pdf_path)
        full_content = []
        table_meta = []

        for idx, table in enumerate(tables):
            page_num = int(table.parsing_report['page']) - 1
            x1, y1, x2, y2 = table._bbox
            table_meta.append({
                "page": page_num,
                "bbox": (x1, y1, x2, y2),
                "index": idx
            })

        # 添加页面处理进度条
        with tqdm(total=len(doc), desc=f"处理 {os.path.basename(pdf_path)}",
                  leave=False, unit='page') as page_pbar:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_rect = page.rect
                page_height = page_rect.height

                current_tables = []
                for table in table_meta:
                    if table["page"] == page_num:
                        x1, y1, x2, y2 = table["bbox"]
                        fitz_bbox = (
                            x1,
                            page_height - y2,
                            x2,
                            page_height - y1
                        )
                        current_tables.append({
                            "fitz_bbox": fitz_bbox,
                            "index": table["index"]
                        })

                text_blocks = page.get_text("blocks")
                image_blocks = self._extract_images(page, page_num)

                all_blocks = []
                for block in text_blocks:
                    all_blocks.append({
                        "type": "text",
                        "bbox": (block[0], block[1], block[2], block[3]),
                        "content": self._clean_text(block[4])
                    })
                for img in image_blocks:
                    all_blocks.append({
                        "type": "image",
                        "bbox": img["bbox"],
                        "content": f"<image>{img['path']}</image>"
                    })

                all_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

                processed_tables = set()
                for block in all_blocks:
                    block_bbox = block["bbox"]

                    if block_bbox[1] < page_rect.height * 0.07:
                        continue

                    in_table = False
                    for table in sorted(current_tables, key=lambda t: t["fitz_bbox"][1]):
                        t_bbox = table["fitz_bbox"]
                        if (block_bbox[0] >= t_bbox[0] and
                                block_bbox[2] <= t_bbox[2] and
                                block_bbox[1] >= t_bbox[1] and
                                block_bbox[3] <= t_bbox[3]):
                            if table["index"] not in processed_tables:
                                full_content.append({"type": "table", "index": table["index"]})
                                processed_tables.add(table["index"])
                            in_table = True
                            break

                    if not in_table:
                        if block["type"] == "text" and block["content"]:
                            full_content.append({
                                "type": "text",
                                "content": block["content"]
                            })
                        elif block["type"] == "image":
                            full_content.append({
                                "type": "image",
                                "content": block["content"]
                            })
                page_pbar.update(1)

        final_output = []
        table_index = 0
        for item in full_content:
            if item["type"] == "text":
                final_output.append(item["content"])
            elif item["type"] == "image":
                final_output.append(item["content"])
            elif item["type"] == "table":
                if table_index < len(tables):
                    table = tables[table_index]
                    processed = self._process_table_optimized(table.data)
                    if not self._is_empty_table(processed):
                        final_output.append(self._process_tables(processed))
                    else:
                        final_output.append("<table>表格解析失败</table>")
                    table_index += 1

        return final_output

    def process(self, pdf_files: Union[str, List[str]]):
        """处理多个PDF文件"""
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]

        valid_files = []
        missing_files = []
        for path in pdf_files:
            if os.path.exists(path):
                if path.lower().endswith('.pdf'):
                    valid_files.append(path)
                else:
                    tqdm.write(f"文件类型错误: {path} (非PDF文件)")
            else:
                missing_files.append(path)

        if missing_files:
            tqdm.write("\n以下文件不存在，已跳过处理:")
            for f in missing_files:
                tqdm.write(f"  - {f}")

        with tqdm(total=len(valid_files), desc="总进度", unit="file") as main_pbar:
            for pdf_path in valid_files:
                try:
                    if not os.path.exists(pdf_path):
                        tqdm.write(f"文件突然不可访问: {pdf_path}")
                        continue

                    try:
                        with open(pdf_path, 'rb') as f:
                            pass
                    except PermissionError:
                        tqdm.write(f"文件无读取权限: {pdf_path}")
                        continue

                    content = self._process_pdf(pdf_path)
                    filename = os.path.basename(pdf_path)
                    self.results[filename] = content
                    main_pbar.set_postfix(file=filename[:15])
                except Exception as e:
                    tqdm.write(f"\n处理文件 {pdf_path} 时出错: {str(e)}")
                finally:
                    main_pbar.update(1)

    def save_as_txt(self, combine: bool = False, output_path: Optional[str] = None):
        """
        保存处理结果到文本文件
        :param combine: 是否合并所有文件结果
        :param output_path: 自定义输出路径（仅combine=True时有效）
        """
        if not self.results:
            print("没有可保存的结果")
            return

        if combine:
            # 合并保存逻辑
            combined = []
            for filename, content in self.results.items():
                combined.append(f"\n{'=' * 20} {filename} {'=' * 20}\n")
                combined.extend(content)

            final_path = output_path or os.path.join(self.output_dir, "combined_output.txt")
            with open(final_path, "w", encoding="utf-8") as f:
                f.write("\n".join(combined))
            print(f"合并保存到: {final_path}")
        else:
            # 分文件保存逻辑
            for filename, content in self.results.items():
                final_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.txt")
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(content))
                print(f"保存到: {final_path}")

    def get_output(self, combine: bool = False) -> Union[str, Dict[str, str]]:
        """
        获取处理结果
        :param combine: 是否合并所有文件结果
        :return: 合并时返回字符串，否则返回{文件名: 内容}字典
        """
        if not self.results:
            return "" if combine else {}

        if combine:
            combined = []
            for filename, content in self.results.items():
                combined.append(f"\n{'=' * 20} {filename} {'=' * 20}\n")
                combined.extend(content)
            return "\n".join(combined)
        else:
            return {filename: "\n".join(content) for filename, content in self.results.items()}