"""Model-authored Matplotlib plots, executed with bounded retries and recorded attempts."""

from __future__ import annotations

import ast
import base64
import csv
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from ..config import get_settings
from .model_adapters import get_llm_client
from .research_workbench import analyze_csv

MAX_ATTEMPTS = 4
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "input", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "__import__", "exit", "quit"}
FORBIDDEN_ATTRIBUTES = {"imread", "imsave", "savefig", "load", "read_csv", "read_table", "to_csv"}


def _check_code(code: str) -> None:
    if not code or len(code) > 16_000:
        raise ValueError("绘图代码为空或超过长度上限")
    tree = ast.parse(code)
    if not any(isinstance(node, ast.Name) and node.id == "rows" and isinstance(node.ctx, ast.Load)
               for node in ast.walk(tree)):
        raise ValueError("绘图代码未使用上传的 CSV 数据 rows")
    forbidden_nodes = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                       ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.Try, ast.Raise)
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"绘图代码包含不允许的语句：{type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            raise ValueError(f"绘图代码包含不允许的名称：{node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise ValueError(f"绘图代码包含不允许的属性：{node.attr}")


def _remove_preloaded_imports(code: str) -> str:
    """Accept common model boilerplate for modules already supplied by the runner."""
    allowed = {"import matplotlib.pyplot as plt", "import numpy as np", "import math",
               "from matplotlib import pyplot as plt"}
    lines = []
    for line in code.splitlines():
        if line.strip() in allowed:
            continue
        lines.append(line)
    return "\n".join(lines)


def _run_plot(code: str, csv_text: str) -> bytes:
    _check_code(code)
    with tempfile.TemporaryDirectory(prefix="research-plot-") as directory:
        root = Path(directory)
        (root / "input.csv").write_text(csv_text, encoding="utf-8")
        (root / "plot_code.py").write_text(code, encoding="utf-8")
        runner = Path(__file__).with_name("csv_plot_runner.py")
        env = {key: value for key, value in os.environ.items()
               if not any(secret in key.upper() for secret in ("KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL"))}
        env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(root / "mplconfig"), PYTHONNOUSERSITE="1")
        try:
            completed = subprocess.run([sys.executable, "-I", str(runner), str(root)], cwd=root,
                                       env=env, capture_output=True, text=True, timeout=30, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("绘图代码运行超过 30 秒") from exc
        image = root / "figure.png"
        if completed.returncode or not image.is_file():
            raise ValueError((completed.stderr or completed.stdout or "绘图脚本没有生成 PNG")[-1500:])
        content = image.read_bytes()
        if not content.startswith(b"\x89PNG\r\n\x1a\n") or len(content) < 1000 or len(content) > 3_000_000:
            raise ValueError("绘图脚本输出的 PNG 无效或超过 3 MB")
        return content


def generate_csv_plot(filename: str, csv_text: str, goal: str, *,
                      on_attempt: Callable[[dict], None] | None = None) -> dict:
    if len(csv_text.encode("utf-8")) > 2_000_000:
        raise ValueError("CSV 超过 2 MB")
    # This deterministic pass validates rows and supplies a factual caption. It does not draw the figure.
    chart_type = ("stacked_bar" if re.search(r"堆叠|堆积|stacked|相对丰度|relative abundance", goal, re.I) else
                  "radar3d" if re.search(r"雷达|radar", goal, re.I) else
                  "line" if re.search(r"折线|趋势|line chart", goal, re.I) else
                  "scatter" if re.search(r"散点|scatter", goal, re.I) else "bar")
    try:
        result = analyze_csv(filename, csv_text, chart_type=chart_type)
    except ValueError:
        # Chart rules are only metadata for the workbench; they must not reject a valid model-designed figure.
        result = analyze_csv(filename, csv_text, chart_type="bar")
        result["caption"] = ("图 1. 根据上传 CSV 和用户需求绘制的数据图。坐标轴、数据汇总及变换方式见图中标注与"
                             "可查看的 Python 绘图代码；图形展示描述性结果，未进行显著性检验。")
    result["method_note"] = "绘图代码由模型生成并在本地对全部 CSV 行运行；图注中的统计描述由 CSV 计算。请核对图形与研究设计。"
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    preview = csv_text if len(csv_text) <= 35_000 else csv_text[:35_000] + "\n[后续数据省略；本地运行时仍使用全部 CSV 行]"
    settings = get_settings()
    attempts = []
    feedback = ""
    for number in range(1, MAX_ATTEMPTS + 1):
        prompt = (f"用户绘图需求：{goal}\n文件：{filename}\n列：{result['columns']}\n数据行数：{len(rows)}"
                  f"\nCSV 数据{'（预览；本地绘图运行使用全部行）' if len(csv_text) > 35_000 else ''}：\n{preview}\n"
                  "请只输出 Python 代码，不要 Markdown 代码块或解释。代码在本地运行，已有变量 rows（全部 CSV 行的字典列表）、"
                  "columns、plt（matplotlib.pyplot）、np（numpy）、math。必须从 rows 读取绘图数值，不得硬编码样本值。"
                  "不得 import、读写文件、调用 savefig。请创建 fig, ax，并根据用户需求绘图。"
                  "没有指定格式则输出 PNG，由系统保存。使用真实数值；相对丰度须正确计算分母；设置坐标轴、图例和布局。"
                  "避免调用 numpy 的 array.sum 方法，可用 Python sum 或 np.sum。允许循环、列表推导和数学计算。"
                  + (f"\n上次运行反馈，请据此修改代码后重新输出完整代码：{feedback}" if feedback else ""))
        code = ""
        try:
            response = get_llm_client().with_options(timeout=180).chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "system", "content": "You write executable Matplotlib plotting code for a research assistant. Follow the user request and the provided data exactly. Return code only."},
                          {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=4500, extra_body={"thinking": {"type": "disabled"}},
            )
            code = (response.choices[0].message.content or "").strip()
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code, flags=re.I).strip()
            code = _remove_preloaded_imports(code)
            image = _run_plot(code, csv_text)
            attempt = {"attempt": number, "status": "success", "code": code,
                       "model": response.model or settings.llm_model, "png_bytes": len(image)}
            attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            result.update(generated_png=base64.b64encode(image).decode("ascii"), generated_code=code,
                          plot_attempts=attempts, plot_model=attempt["model"], plot_format="png")
            return result
        except Exception as exc:
            feedback = f"{type(exc).__name__}: {exc}"[-1500:]
            attempt = {"attempt": number, "status": "failed", "code": code, "error": feedback}
            attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt)
    raise ValueError(f"模型绘图连续 {MAX_ATTEMPTS} 次未成功。最后错误：{feedback}")
