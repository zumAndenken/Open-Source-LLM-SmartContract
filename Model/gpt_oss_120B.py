import os
import re
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


# ===== GPUGeek H20 优化配置 =====
ROOT_DIR = ""
OUTPUT_FILE = ""  # 当前目录下保存结果
OLLAMA_URL = "http://localhost:11434"
MODEL = "gpt-oss:120b"  # 使用120B模型

# 并发配置
MAX_WORKERS = 1    

# 需要分析的文件扩展名
ALLOWED_EXTS = [".sol"]

# 文件选择控制（0 表示不限）
MAX_FILES = 1              # 全局最多处理的文件数
SKIP_FILES = 0            # 跳过前 N 个
MAX_FILES_PER_CLASS = 0    # 每个漏洞类别最多处理的文件数
# ===========================


class VulnerabilityAnalyzer:
    """漏洞分析器 - H20 优化版本

    通过 HTTP 调用本地 Ollama 服务，对输入的智能合约文本进行九类漏洞的二分类判定。
    针对 NVIDIA H20 96GB 显存进行优化。

    参数说明：
    - ollama_url: Ollama 服务地址
    - model_name: 模型名称
    - temperature: 采样温度（越低越确定，越高越发散）
    - num_ctx: 上下文长度，H20大显存可支持更大值
    - request_timeout/request_retries: HTTP 超时与重试设置
    - max_workers: 并发线程数，H20可支持更高并发
    """
    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        num_ctx: int = 8192,          # 调低默认上下文，避免显存不足导致 GPU 回退
        request_timeout: int = 3600,
        request_retries: int = 1,
        max_workers: int = 1,
    ):
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model_name = model_name or os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self.request_timeout = int(request_timeout)
        self.request_retries = int(request_retries)
        self.max_workers = max(1, int(max_workers))

        self.session = requests.Session()

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

        # 严格遵循“漏洞名: 0/1”输出风格（文档输入风格）
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
            "The following text is a vulnerability detection result for a smart contract. Use 0 or 1 to indicate whether there are specific types of vulnerabilities. For example: \"Reentrancy: 1\". Think step by step, carefully. The input is [INPUT]."
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

    # ---------- LLM ----------

    def call_ollama(self, prompt: str) -> str:
        """调用 Ollama /api/generate 接口，返回纯文本响应。

        注意：
        - num_ctx 设得越大显存占用越高；如显存不足，Ollama 可能回退到 CPU
        - num_batch 影响提示吸收阶段吞吐与峰值显存，需与 num_ctx、显存余量共同权衡
        """
        url = f"{self.ollama_url}/api/generate"
        # 为避免显存不足，限制上下文长度与批量
        safe_ctx = int(min(self.num_ctx, 8192))
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                # 上下文窗口（越大越占显存；过大可能触发 CPU 回退）
                "num_ctx": safe_ctx,
                # 提示吸收阶段的批量大小（长提示加速；会提高峰值显存）
                "num_batch": 256, #千万别调高，电脑会直接死机
            },
        }

        for attempt in range(1, self.request_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.request_timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            except requests.exceptions.RequestException as e:
                if attempt < self.request_retries:
                    wait = min(2**attempt, 8)
                    print(f"调用Ollama失败(第 {attempt}/{self.request_retries} 次)，重试前等待 {wait}s：{e}")
                    time.sleep(wait)
                else:
                    print(f"调用Ollama失败(最终失败)：{e}")
        return ""

    def parse_vulnerability_response(self, response: str) -> Dict[str, int]:
        """从模型文本输出中解析九类漏洞的 0/1 结果。

        解析策略：对每一类漏洞尝试多种分隔符匹配（:、=、空格），未命中默认 0。
        """
        # 按“漏洞名: 0/1”解析；默认 0
        results = {k: 0 for k in self.vulnerability_types}
        if not response:
            return results

        text = response

        for vuln_type in self.vulnerability_types:
            # 匹配：Vuln: 0/1 或 Vuln = 0/1 或 Vuln 0/1
            patterns = [
                rf"{re.escape(vuln_type)}\s*:\s*([01])\b",
                rf"{re.escape(vuln_type)}\s*=\s*([01])\b",
                rf"{re.escape(vuln_type)}\s+([01])\b",
            ]
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        results[vuln_type] = int(m.group(1))
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
        df["F1"] = df.apply(lambda r: safe_div(2 * r["Precision"] * r["Specificity"], r["Precision"] + r["Specificity"]), axis=1)

        return df

    def analyze_single_contract(self, contract_content: str) -> Dict[str, int]:
        """对单个合约文本进行一次分析，返回九类漏洞的 0/1 结果。"""
        full_prompt = self.system_prompt.replace("[INPUT]", contract_content)
        resp = self.call_ollama(full_prompt)
        if not resp:
            return {k: 0 for k in self.vulnerability_types}
        return self.parse_vulnerability_response(resp)

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

    def _analyze_one_file(self, rel_id: str, file_path: str, norm_label: str) -> Optional[Dict[str, int]]:
        """读取文件内容并调用模型分析，返回单行结果记录。"""
        content = self._read_file_text(file_path)
        if not content.strip():
            print(f"空文件，跳过: {rel_id}")
            return None
        
        # 记录分析开始时间
        start_time = time.time()
        vuln_results = self.analyze_single_contract(content)
        # 计算花费时间（秒）
        analysis_time = time.time() - start_time
        
        row = {
            "文件ID": rel_id,
            "实际漏洞类型": norm_label,
            "花费时间": round(analysis_time, 2),  # 保留2位小数
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
        - `MAX_WORKERS`>1 时使用线程池并发请求，以提升 GPU 利用率
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
        ollama_url=OLLAMA_URL,
        model_name=MODEL,
        max_workers=MAX_WORKERS,
    )

    if not os.path.isdir(ROOT_DIR):
        print(f"错误：数据集目录不存在: {ROOT_DIR}")
        print("请确保数据集已正确放置在 /gz-data/dataset/ 目录下")
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