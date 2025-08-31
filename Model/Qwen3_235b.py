import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


# ===== Qwen API 配置 =====
ROOT_DIR = ""  # 数据集路径
OUTPUT_FILE = ""  # 当前目录下保存结果
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY = ""
MODEL = "qwen3-235b-a22b-thinking-2507"  # Qwen3 235B 模型 

# 并发配置
MAX_WORKERS = 1    

# 需要分析的文件扩展名
ALLOWED_EXTS = [".sol"]

# 文件选择控制（0 表示不限）
MAX_FILES = 0              # 全局最多处理的文件数
SKIP_FILES = 0            # 跳过前 N 个
MAX_FILES_PER_CLASS = 0    # 每个漏洞类别最多处理的文件数
# ===========================


class VulnerabilityAnalyzer:
    """漏洞分析器 - Qwen API 版本

    通过 HTTP 调用 Qwen API 服务，对输入的智能合约文本进行九类漏洞的二分类判定。

    参数说明：
    - api_url: Qwen API 地址
    - api_key: API 密钥
    - model_name: 模型名称
    - temperature: 采样温度（越低越确定，越高越发散）
    - max_tokens: 最大输出token数
    - request_timeout/request_retries: HTTP 超时与重试设置
    - max_workers: 并发线程数
    """
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        request_timeout: int = 120,
        request_retries: int = 3,
        max_workers: int = 1,
    ):
        self.api_url = api_url or os.environ.get("QWEN_API_URL", QWEN_API_URL)
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", QWEN_API_KEY)
        self.model_name = model_name or os.environ.get("QWEN_MODEL", MODEL)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout = int(request_timeout)
        self.request_retries = int(request_retries)
        self.max_workers = max(1, int(max_workers))

        # 使用官方推荐的 OpenAI SDK
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.vulnerability_types = [
            "Reentrancy",
            "Access Control",
            "Arithmetic Issues",
            "Unchecked Return Values For Low Level Calls",
            "Denial of Service",
            "Bad Randomness",
            "Front-Running",
            "Time manipulation",
            "Short Address Attack",
        ]

        # 严格遵循"漏洞名: 0/1"输出风格（文档输入风格）
        self.system_prompt = (
            "You are a semantic analyzer of text. Here are nine common vulnerabilities.\n"
            "1. Reentrancy\n"
            "also known as or related to race to empty, recursive call vulnerability, call to the unknown\n"
            "2. Access Control\n"
            "3. Arithmetic Issues\n"
            "also known as integer overflow and integer underflow\n"
            "4. Unchecked Return Values For Low Level Calls\n"
            "also known as or related to silent failing sends, unchecked-send\n"
            "5. Denial of Service\n"
            "including gas limit reached, unexpected throw, unexpected kill, access control breached\n"
            "6. Bad Randomness\n"
            "also known as nothing is secret\n"
            "7. Front-Running\n"
            "also known as time-of-check vs time-of-use (TOCTOU), race condition, transaction ordering dependence (TOD)\n"
            "8. Time manipulation\n"
            "also known as timestamp dependence\n"
            "9. Short Address Attack\n"
            "also known as or related to off-chain issues, client vulnerabilities.\n\n"
            "Provide a detailed analysis with explanations, then MUST end your response with a summary in this EXACT format:\n\n"
            "Reentrancy: 0\n"
            "Access Control: 0\n"
            "Arithmetic Issues: 0\n"
            "Unchecked Return Values For Low Level Calls: 0\n"
            "Denial of Service: 0\n"
            "Bad Randomness: 0\n"
            "Front-Running: 0\n"
            "Time manipulation: 0\n"
            "Short Address Attack: 0\n\n"
            "The following text is a vulnerability detection result for a smart contract. Use 0 or 1 to indicate whether there are specific types of vulnerabilities. For example: \"Reentrancy: 1\". Think step by step, carefully."
        )

    # ---------- Helpers ----------

    def _ensure_int_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保输出 DataFrame 中与计数相关的列为整型，时间列为浮点型。

        说明：Excel 读写后可能出现类型漂移，这里做一次统一转换，避免后续处理时报错。
        """
        # 只处理漏洞类型列为整型
        int_cols = self.vulnerability_types
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        
        # 确保花费时间列为浮点型
        if "花费时间" in df.columns:
            df["花费时间"] = pd.to_numeric(df["花费时间"], errors="coerce").fillna(0.0).astype(float)
        
        return df

    def _read_file_text(self, path: str) -> str:
        """以多编码尝试读取文本，避免编码不兼容导致失败。"""
        for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                with open(path, "r", encoding=enc, errors="ignore") as f:
                    return f.read()
            except Exception:
                continue
        return ""

    def _normalize_label(self, name: str) -> str:
        """将目录名（可能大小写/分隔符不同）规范化为标准漏洞名。"""
        name = (name or "").strip()
        for t in self.vulnerability_types:
            if name.lower() == t.lower():
                return t
        aliases = {
            "reentrancy": "Reentrancy",
            "accesscontrol": "Access Control",
            "arithmetic": "Arithmetic Issues",
            "uncheckedlowlevelcalls": "Unchecked Return Values For Low Level Calls",
            "denialofservice": "Denial of Service",
            "badrandomness": "Bad Randomness",
            "frontrunning": "Front-Running",
            "timemanipulation": "Time manipulation",
            "shortaddresses": "Short Address Attack",
        }
        import re as _re
        key = _re.sub(r"[\s_-]+", "", name.lower())
        return aliases.get(key, "")

    # ---------- Qwen API ----------

    def call_qwen_api(self, contract_content: str) -> Tuple[str, str]:
        """调用 Qwen API，使用官方推荐的 OpenAI SDK。
        
        返回: (response_content, reasoning_content)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please analyze the following smart contract code:\n\n{contract_content}"}
        ]

        for attempt in range(1, self.request_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                
                response_content = response.choices[0].message.content
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
                
                # 输出AI的回答
                if response_content:
                    print("=" * 60)
                    print("🤖 AI分析结果:")
                    print("=" * 60)
                    print(response_content)
                    print("=" * 60)
                elif reasoning_content:
                    print("=" * 60)
                    print("🤖 AI思考过程:")
                    print("=" * 60)
                    print(reasoning_content[-2000:])  # 显示最后2000字符
                    print("=" * 60)
                    # 尝试从思考过程中提取
                    response_content = self._extract_from_reasoning(reasoning_content)
                    if response_content:
                        print("📝 提取的分析结果:")
                        print(response_content)
                        print("=" * 60)
                
                return response_content if response_content else "", reasoning_content
                    
            except Exception as e:
                if attempt < self.request_retries:
                    wait = min(2**attempt, 8)
                    print(f"调用Qwen API失败(第 {attempt}/{self.request_retries} 次)，重试前等待 {wait}s：{e}")
                    time.sleep(wait)
                else:
                    print(f"调用Qwen API失败(最终失败)：{e}")
                    
        return "", ""

    def _extract_from_reasoning(self, reasoning_content: str) -> str:
        """从AI的思考过程中提取最终答案"""
        if not reasoning_content:
            return ""
        
        # 尝试从思考过程中提取最终的漏洞分析结果
        lines = reasoning_content.split('\n')
        for line in reversed(lines):  # 从后往前找，通常结果在最后
            line = line.strip()
            if any(vuln in line for vuln in self.vulnerability_types):
                return line
        return reasoning_content

    def parse_vulnerability_response(self, response: str) -> Dict[str, int]:
        """从模型文本输出中解析九类漏洞的 0/1 结果。

        解析策略：对每一类漏洞尝试多种分隔符匹配（:、=、空格），未命中默认 0。
        """
        # 按"漏洞名: 0/1"解析；默认 0
        results = {k: 0 for k in self.vulnerability_types}
        if not response:
            return results

        text = response
        found_any = False
        
        for vuln_type in self.vulnerability_types:
            # 匹配多种格式的漏洞输出
            patterns = [
                # 标准格式: "Reentrancy: 0" 或 "**Reentrancy**: 0"
                rf"\*\*{re.escape(vuln_type)}\*\*\s*:\s*([01])\b",
                rf"{re.escape(vuln_type)}\s*:\s*([01])\b",
                rf"{re.escape(vuln_type)}\s*=\s*([01])\b",
                rf"{re.escape(vuln_type)}\s+([01])\b",
                # 编号格式: "1. **Reentrancy**: 0"
                rf"\d+\.\s*\*\*{re.escape(vuln_type)}\*\*\s*:\s*([01])\b",
                rf"\d+\.\s*{re.escape(vuln_type)}\s*[:=]\s*([01])\b",
                # 大小写不敏感
                rf"(?i)\*\*{re.escape(vuln_type.lower())}\*\*\s*:\s*([01])\b",
                rf"(?i){re.escape(vuln_type.lower())}\s*[:=]\s*([01])\b",
                # 更灵活的匹配
                rf"{re.escape(vuln_type)}\s*[-:=]\s*([01])\b",
            ]
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        results[vuln_type] = int(m.group(1))
                        found_any = True
                        print(f"✅ 解析到: {vuln_type} = {results[vuln_type]}")
                    except Exception:
                        results[vuln_type] = 0
                    break

        return results

    def compute_results_metric(self, df_output: pd.DataFrame) -> pd.DataFrame:
        """基于结果表计算每类 TP/FP/TN/FN，作为指标表。

        规则：对每个分析记录与每个漏洞：
        - 每行记录代表一次分析结果（0或1）
        - 若实际标签为该漏洞且预测为1：TP += 1
        - 若实际标签为该漏洞且预测为0：FN += 1
        - 若实际标签不为该漏洞且预测为1：FP += 1
        - 若实际标签不为该漏洞且预测为0：TN += 1
        """
        if df_output is None or df_output.empty:
            return pd.DataFrame(columns=["漏洞类型", "TP", "FP", "TN", "FN"])

        df = self._ensure_int_columns(df_output.copy())

        counts: Dict[str, Dict[str, int]] = {
            v: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for v in self.vulnerability_types
        }

        for _, row in df.iterrows():
            actual_label = str(row.get("实际漏洞类型", ""))
            for vuln in self.vulnerability_types:
                predicted = int(pd.to_numeric(row.get(vuln, 0), errors="coerce"))
                if predicted < 0:
                    predicted = 0
                
                if actual_label == vuln:
                    if predicted == 1:
                        counts[vuln]["TP"] += 1
                    else:
                        counts[vuln]["FN"] += 1
                else:
                    if predicted == 1:
                        counts[vuln]["FP"] += 1
                    else:
                        counts[vuln]["TN"] += 1

        data = []
        for vuln in self.vulnerability_types:
            c = counts[vuln]
            data.append({
                "漏洞类型": vuln,
                "TP": c["TP"],
                "FP": c["FP"],
                "TN": c["TN"],
                "FN": c["FN"],
            })
        df = pd.DataFrame(data)

        # 计算 Precision / Recall / Specificity / F1（分母为 0 时记为 0.0）
        def safe_div(n: float, d: float) -> float:
            try:
                return float(n) / float(d) if float(d) != 0.0 else 0.0
            except Exception:
                return 0.0

        df["Precision"] = df.apply(lambda r: safe_div(r["TP"], r["TP"] + r["FP"]), axis=1)
        df["Recall"] = df.apply(lambda r: safe_div(r["TP"], r["TP"] + r["FN"]), axis=1)
        df["Specificity"] = df.apply(lambda r: safe_div(r["TN"], r["TN"] + r["FP"]), axis=1)
        df["F1"] = df.apply(lambda r: safe_div(2 * r["Precision"] * r["Recall"], r["Precision"] + r["Recall"]), axis=1)

        return df

    def analyze_single_contract(self, contract_content: str) -> Tuple[Dict[str, int], str]:
        """对单个合约文本进行一次分析，返回九类漏洞的 0/1 结果和AI思考内容。"""
        resp, reasoning = self.call_qwen_api(contract_content)
        if not resp:
            return {k: 0 for k in self.vulnerability_types}, reasoning
        return self.parse_vulnerability_response(resp), reasoning

    # ---------- Folder mode ----------

    def _gather_files(self, root_dir: str, allowed_exts: Tuple[str, ...]) -> List[Tuple[str, str, str]]:
        """遍历根目录，按子目录名映射漏洞标签，收集合规扩展名的文件列表。"""
        items = []
        root_dir = os.path.abspath(root_dir)
        for dirpath, _, filenames in os.walk(root_dir):
            vuln_label_raw = os.path.basename(dirpath)
            norm_label = self._normalize_label(vuln_label_raw)
            if dirpath == root_dir or not norm_label:
                continue

            for fn in filenames:
                if allowed_exts and not fn.lower().endswith(allowed_exts):
                    continue
                file_path = os.path.join(dirpath, fn)
                rel_id = os.path.relpath(file_path, root_dir).replace("\\", "/")
                items.append((rel_id, file_path, norm_label))
        return items

    def _analyze_one_file(self, rel_id: str, file_path: str, norm_label: str) -> Optional[Dict[str, any]]:
        """读取文件内容并调用模型分析，返回单行结果记录。"""
        content = self._read_file_text(file_path)
        if not content.strip():
            print(f"空文件，跳过: {rel_id}")
            return None
        
        # 记录分析开始时间
        start_time = time.time()
        vuln_results, ai_reasoning = self.analyze_single_contract(content)
        # 计算花费时间（秒）
        analysis_time = time.time() - start_time
        
        row = {
            "文件ID": rel_id,
            "实际漏洞类型": norm_label,
            "花费时间": round(analysis_time, 2),  # 保留2位小数
            "AI思考内容": ai_reasoning,  # 新增AI思考内容列
            #"AI思考内容": "",  # 临时设置为空，不显示AI思考内容
            **{k: int(vuln_results.get(k, 0)) for k in self.vulnerability_types},
        }
        print(f"完成: {rel_id} (耗时: {analysis_time:.2f}s)")
        return row

    def process_folder(
        self,
        root_dir: str,
        output_file: str,
        allowed_exts: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        skip_files: int = 0,
        max_files_per_class: Optional[int] = None,
    ):
        """目录模式主流程：收集、分析并将结果写入 Excel。

        重要说明：
        - 每次分析都会作为新的一行记录，不会累加相同文件ID的结果
        - `MAX_WORKERS`>1 时使用线程池并发请求，以提升 API 调用效率
        """
        if not os.path.isdir(root_dir):
            print(f"目录不存在: {root_dir}")
            return

        if not allowed_exts:
            allowed_exts = [".sol"]
        allowed_exts_t = tuple(x.lower() for x in allowed_exts)

        output_columns = [
            "文件ID",
            "实际漏洞类型",
            "花费时间",
            "AI思考内容",
            "Reentrancy",
            "Access Control",
            "Arithmetic Issues",
            "Unchecked Return Values For Low Level Calls",
            "Denial of Service",
            "Bad Randomness",
            "Front-Running",
            "Time manipulation",
            "Short Address Attack",
        ]

        df_output = pd.DataFrame(columns=output_columns)
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_excel(output_file, engine="openpyxl")
                if all(col in existing_df.columns for col in output_columns):
                    df_output = existing_df
                    print(f"读取现有输出文件，已有 {len(df_output)} 行数据")
                else:
                    print("现有输出文件格式不正确，将创建新文件")
            except Exception as e:
                print(f"读取输出文件时出错，创建新文件: {e}")

        items = self._gather_files(root_dir, allowed_exts_t)
        if not items:
            print("未发现可处理的文件")
            return

        # -------- 文件数量控制 --------
        if max_files_per_class and max_files_per_class > 0:
            grouped = {}
            for rel_id, file_path, label in items:
                grouped.setdefault(label, []).append((rel_id, file_path, label))
            items_limited = []
            for label, lst in grouped.items():
                items_limited.extend(lst[:max_files_per_class])
            items = items_limited

        if skip_files and skip_files > 0:
            items = items[skip_files:]
        if max_files and max_files > 0:
            items = items[:max_files]

        print(f"将处理 {len(items)} 个文件"
              + (f"（每类上限 {max_files_per_class}）" if max_files_per_class else "")
              )

        print(f"开始分析...")

        results = []
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = [ex.submit(self._analyze_one_file, rel_id, file_path, norm_label)
                        for rel_id, file_path, norm_label in items]
                for fut in as_completed(futs):
                    r = fut.result()
                    if r:
                        results.append(r)
        else:
            for rel_id, file_path, norm_label in items:
                r = self._analyze_one_file(rel_id, file_path, norm_label)
                if r:
                    results.append(r)

        # 每次分析都作为新的一行添加
        if results:
            # 创建新的DataFrame包含所有结果
            new_df = pd.DataFrame(results)
            # 如果现有数据为空，直接使用新数据；否则连接
            if df_output.empty:
                df_output = new_df
            else:
                df_output = pd.concat([df_output, new_df], ignore_index=True)

        # 将当前批次的结果保存/更新到结果表
        df_output = self._ensure_int_columns(df_output)
        
        # 按漏洞类型和文件ID排序，优先按漏洞类型集中，后按文件ID集中
        if not df_output.empty and "文件ID" in df_output.columns and "实际漏洞类型" in df_output.columns:
            df_output = df_output.sort_values(by=["实际漏洞类型", "文件ID"], ascending=[True, True]).reset_index(drop=True)
        
        try:
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                df_output.to_excel(writer, index=False)
            print(f"\n目录模式完成！共处理 {len(results)} 个文件，结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

        # 仅输出：累计（按次数重算）的指标表到一张 Excel
        try:
            base = os.path.splitext(output_file)[0]
            metrics_path = base + "_metrics.xlsx"  # 输出为 results_metrics.xlsx

            metrics_df = self.compute_results_metric(df_output)
            with pd.ExcelWriter(metrics_path, engine="openpyxl", mode="w") as writer:
                metrics_df.to_excel(writer, index=False)

            print(f"指标表(累计，按次数)已保存到: {metrics_path}")
        except Exception as e:
            print(f"保存混淆矩阵时出错: {e}")


def main():
    """程序入口：构造分析器并按配置运行目录模式。"""
    analyzer = VulnerabilityAnalyzer(
        api_url=QWEN_API_URL,
        api_key=QWEN_API_KEY,
        model_name=MODEL,
        max_workers=MAX_WORKERS,
    )

    if not os.path.isdir(ROOT_DIR):
        print(f"错误：数据集目录不存在: {ROOT_DIR}")
        print("请确保数据集已正确放置在指定目录下")
        return

    analyzer.process_folder(
        root_dir=ROOT_DIR,
        output_file=OUTPUT_FILE,
        allowed_exts=ALLOWED_EXTS,
        max_files=MAX_FILES,
        skip_files=SKIP_FILES,
        max_files_per_class=MAX_FILES_PER_CLASS,
    )


if __name__ == "__main__":
    main()
