"""Bounded, auditable ReAct coordinator over existing read-only research tools."""

from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from ..config import get_settings
from .connections import redis_client
from .model_adapters import get_llm_client

TTL = 7 * 86400
MAX_STEPS = 8
TOOLS = {"search_papers", "answer_with_evidence", "citation_graph", "concept_graph", "graph_review", "future_work", "research_draft", "writing_sources", "writing_draft", "draft_alignment", "general_answer", "architecture_topology", "csv_analysis"}
ALIGNMENT_INTENT = re.compile(r"核查草稿|草稿.*文献|文献.*草稿|draft alignment|align.*draft", re.I)
WRITING_INTENT = re.compile(r"论文框架|写.*论文|撰写|写作|摘要.*引言|references|related work|paper outline|draft a paper|write a paper", re.I)
TOPOLOGY_INTENT = re.compile(r"架构图|架构拓扑|拓扑图|流程图|Mermaid|Graphviz|TikZ|Matplotlib|architecture diagram|topology diagram|flowchart", re.I)
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research-agent")
_local_lock = Lock()
_running: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(task_id: str) -> str:
    return f"research:agent:{task_id}"


def _csv_key(task_id: str) -> str:
    return f"research:agent:csv:{task_id}"


def _save(state: dict) -> None:
    state["updated_at"] = _now()
    redis_client().set(_key(state["task_id"]), json.dumps(state, ensure_ascii=False), ex=TTL)


def get_agent_task(task_id: str) -> dict | None:
    try:
        uuid.UUID(task_id)
    except ValueError:
        return None
    raw = redis_client().get(_key(task_id))
    return json.loads(raw) if raw else None


def _event(state: dict, kind: str, **data: Any) -> None:
    state["events"].append({"seq": len(state["events"]) + 1, "at": _now(), "kind": kind, **data})
    _save(state)


def _model_json(system: str, payload: dict) -> dict:
    response = get_llm_client().chat.completions.create(
        model=get_settings().llm_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
        temperature=0, max_tokens=1200,
    )
    content = (response.choices[0].message.content or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Agent model output must be a JSON object")
    return value


def _tool(name: str, query: str, paper_ids: list[str]) -> dict:
    if name == "writing_sources":
        from .research_workbench import _source_records
        return {"references": [item for item in _source_records(query, 10) if item.get("verified")]}
    if name == "general_answer":
        response = get_llm_client().chat.completions.create(
            model=get_settings().llm_model,
            messages=[{"role": "system", "content": "Answer the user's general question briefly. This tool has no paper evidence; never invent citations or claim source verification. For scholarly factual claims, recommend evidence-grounded retrieval."},
                      {"role": "user", "content": query}], temperature=0, max_tokens=700)
        return {"answer": (response.choices[0].message.content or "").strip(), "evidence_status": "未使用论文证据"}
    if name == "search_papers":
        from .scholarly_discovery import search_external_papers
        return search_external_papers(query, 5)
    if name == "answer_with_evidence":
        from .hybrid_retrieval import answer_with_evidence
        return answer_with_evidence(query, top_k=6, paper_id=paper_ids[0] if len(paper_ids) == 1 else None)
    if name == "citation_graph":
        from .research_workbench import build_citation_graph
        return build_citation_graph(paper_ids, min(20, max(1, len(paper_ids))))
    if name == "concept_graph":
        from .concept_graph import build_concept_graph
        return build_concept_graph(paper_ids, 20)
    if name == "draft_alignment":
        from .draft_alignment import align_draft_to_library
        return align_draft_to_library(query, paper_ids, 12)
    if name == "graph_review":
        from .research_workbench import generate_graph_review
        return generate_graph_review(paper_ids, min(20, max(1, len(paper_ids))))
    if name == "future_work":
        from .research_workbench import assess_future_work_coverage
        return assess_future_work_coverage(paper_ids, min(20, max(1, len(paper_ids))))
    if name == "research_draft":
        from .research_workbench import generate_research_draft
        return generate_research_draft(query, 6)
    if name == "architecture_topology":
        from .research_workbench import mermaid_architecture
        return mermaid_architecture(query)
    raise ValueError("Tool is not allowed")


def _write_research_draft(goal: str, sources: dict) -> dict:
    references = [{key: item.get(key) for key in ("ref_id", "title", "authors", "year", "doi", "arxiv_id", "abstract", "source_url")}
                  for item in sources.get("references", [])]
    common = ("你是科研写作助手。只可引用所给文献编号，不能捏造论文、作者或实验结果。"
              "这些元数据只证明条目存在，具体论文结论若未查全文须标注待核查。"
              "无法保证一个 idea 从未有人提出，必须给出新颖性检索和对比方案。用中文 Markdown 写具体内容，不要 JSON。")
    payload = json.dumps({"request": goal, "allowed_references": references}, ensure_ascii=False)
    parts = []
    for instruction in (
        "第一部分：提出一个具体且可证伪的研究 idea。写明问题、与 GRPO 的差异、算法步骤、失败条件和新颖性核查；再设计详细实验，包括数据集选择、基线、指标、训练预算、消融、统计检验及复现设置。不要空泛占位。",
        "第二部分：依据前一部分方案写论文初稿，包含标题、Abstract、Introduction、Related Work、Method、Experiments（仅写实验计划，结果不得杜撰）、Limitations、References。正文要有实质段落和具体方法；引用仅使用给定条目编号。",
    ):
        response = get_llm_client().with_options(timeout=180).chat.completions.create(
            model=get_settings().llm_model,
            messages=[{"role": "system", "content": common + instruction},
                      {"role": "user", "content": payload + ("\n已确定的研究方案：\n" + parts[0] if parts else "")}],
            temperature=0.4, max_tokens=8500,
            extra_body={"thinking": {"type": "disabled"}},
        )
        part = (response.choices[0].message.content or "").strip()
        if len(part) < 350:
            raise ValueError(f"模型正文过短（{len(part)} 字，结束原因 {response.choices[0].finish_reason}）；推理模型可能耗尽输出 token，请调整模型或输出上限")
        parts.append(part)
    markdown = "\n\n".join(parts)
    if len(markdown) < 900:
        raise ValueError("模型返回的写作内容过短；请检查模型输出上限后重试")
    required = ("实验", "Abstract", "Introduction", "Related Work", "Method", "References")
    missing = [name for name in required if name.lower() not in markdown.lower()]
    if missing:
        raise ValueError("模型写作内容缺少必要章节：" + "、".join(missing))
    allowed_ids = {str(item["ref_id"]) for item in references}
    cited_ids = set(re.findall(r"\[(P\d+)\]", markdown))
    if cited_ids - allowed_ids:
        raise ValueError("模型引用了未核查的文献编号：" + "、".join(sorted(cited_ids - allowed_ids)))
    bibliography = "\n".join(
        f"[{item['ref_id']}] {item['title']} ({item.get('year') or '年份待核查'}). "
        f"{('DOI: ' + item['doi']) if item.get('doi') else ('arXiv: ' + item['arxiv_id']) if item.get('arxiv_id') else '标识符待核查'}"
        for item in references)
    markdown = re.split(r"(?im)^#{1,3}\s*References\b", markdown, maxsplit=1)[0].rstrip() + "\n\n## References\n" + bibliography
    return {"markdown": markdown, "references": references, "evidence_status": "参考条目有可定位标识符；正文论断与新颖性仍需核查",
            "missing_inputs": ["新颖性系统检索", "真实实验结果"], "model": response.model or get_settings().llm_model}


def _summary(name: str, result: dict) -> dict:
    """Keep detailed output in artifacts; only compact facts enter model context."""
    if name == "answer_with_evidence":
        verified = [claim for claim in result.get("claims", []) if claim.get("verified")]
        return {"answer_status": result.get("answer_status"), "verified_count": len(verified),
                "unverified_count": len(result.get("claims", [])) - len(verified),
                "sources": [{"source_id": cite.get("source_id"), "page": cite.get("page"),
                             "source_filename": cite.get("source_filename"), "quote": cite.get("quote", "")[:240]}
                            for claim in verified[:8] for cite in claim.get("citations", []) if cite.get("quote_location", {}).get("found")]}
    if name == "citation_graph":
        return {"summary": result.get("summary"), "node_titles": [x.get("title") for x in result.get("nodes", [])[:20]]}
    if name == "architecture_topology":
        return {"title": result.get("title"), "nodes": len(result.get("nodes", [])),
                "formats": list(result.get("scripts", {})), "note": str(result.get("note", ""))[:300]}
    if name == "csv_analysis":
        return {"filename": result.get("filename"), "row_count": result.get("row_count"),
                "chart_type": result.get("chart_type"), "caption": str(result.get("caption", ""))[:300]}
    if name == "search_papers":
        records = result.get("results", result.get("papers", []))
        return {"papers": [{"title": x.get("title"), "doi": x.get("doi"), "source": x.get("source")}
                           for x in records[:8]]} if isinstance(records, list) else {"keys": list(result)[:12]}
    return {"keys": list(result)[:12], "status": result.get("status"), "note": str(result.get("note", ""))[:300]}


def _plan(state: dict, revise: bool = False) -> None:
    previous = state["plan"][:state["cursor"]] if revise else []
    remaining = MAX_STEPS - state.get("action_count", 0)
    if remaining < 1:
        raise ValueError("Agent action budget exhausted")
    if not revise and ALIGNMENT_INTENT.search(state["goal"]):
        steps = [{"tool": "draft_alignment", "query": state["goal"],
                  "success_criterion": "Return source-located candidate passages for each draft section without asserting unsupported stance",
                  "status": "pending"}]
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    if state.get("version_id") and not revise:
        steps = [{"tool": "answer_with_evidence", "query": state["goal"],
                  "success_criterion": "Answer only from verified excerpts in the selected paper version or state evidence is insufficient",
                  "status": "pending"}]
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    if not revise and state.get("csv_filename"):
        steps = [{"tool": "csv_analysis", "query": state["goal"][:1000],
                  "success_criterion": "Ask the model for Matplotlib code, run it on the attached data, retry errors, and return a verified PNG",
                  "status": "pending"}]
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    if not revise and TOPOLOGY_INTENT.search(state["goal"]):
        steps = [{"tool": "architecture_topology", "query": state["goal"][:11000],
                  "success_criterion": "Generate a diagram and four scripts from the same nodes and edges",
                  "status": "pending"}]
        if state.get("task_type") == "writing":
            steps.append({"tool": "research_draft", "query": state["goal"][:1000],
                          "success_criterion": "Produce a draft with verifiable references", "status": "pending"})
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    if not revise and state.get("task_type") == "writing":
        steps = [{"tool": "writing_sources", "query": state["goal"][:1000],
                  "success_criterion": "Collect real, identifiable literature metadata for the writing task", "status": "pending"},
                 {"tool": "writing_draft", "query": state["goal"][:1000],
                  "success_criterion": "Generate a concrete idea, experiments, outline, and substantive draft with grounded references", "status": "pending"}]
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    if not revise and len(state["paper_ids"]) == 1 and state.get("task_type") != "writing":
        steps = [{"tool": "answer_with_evidence", "query": state["goal"],
                  "success_criterion": "Answer with verified excerpts from the attached paper or state evidence is insufficient",
                  "status": "pending"}]
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    if not revise and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?\s*(?:=\s*\??|\?)?\s*", state["goal"]):
        steps = [{"tool": "general_answer", "query": state["goal"],
                  "success_criterion": "Answer the arithmetic question without claiming paper evidence", "status": "pending"}]
        state["plan"] = steps
        _event(state, "plan", steps=steps)
        return
    instruction = ("For writing, include research_draft as the final tool and pass the research idea as its query. "
                   if state.get("task_type") == "writing" else "")
    response = _model_json(
        instruction + "Plan a research task using only these tools: " + ", ".join(sorted(TOOLS)) +
        ". For ordinary short questions without a paper evidence request, use general_answer. "
        ". Return JSON {\"steps\":[{\"tool\":\"...\",\"query\":\"...\",\"success_criterion\":\"...\"}]}.",
        {"goal": state["goal"], "paper_ids": state["paper_ids"], "limit": remaining,
         "previous_observations": [x for x in state["events"] if x["kind"] in {"observe", "reflect"}][-6:]},
    )
    steps = response.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= remaining:
        raise ValueError("Agent plan must contain 1 to 8 steps")
    for step in steps:
        if not isinstance(step, dict) or step.get("tool") not in TOOLS:
            raise ValueError("Agent plan contains an unsupported tool")
        if not isinstance(step.get("query"), str) or not step["query"].strip():
            raise ValueError("Agent plan has an empty query")
        if not isinstance(step.get("success_criterion"), str) or not step["success_criterion"].strip():
            raise ValueError("Agent plan lacks a completion criterion")
        step["query"] = step["query"][:1000]
        step["status"] = "pending"
    if state.get("task_type") == "writing" and not any(x["tool"] == "research_draft" for x in previous + steps):
        if len(steps) >= remaining:
            steps.pop()
        steps.append({"tool": "research_draft", "query": state["goal"][:1000],
                      "success_criterion": "Produce a draft with verifiable references", "status": "pending"})
    if state.get("task_type") == "writing":
        drafts = [step for step in steps if step["tool"] == "research_draft"]
        steps = [step for step in steps if step["tool"] != "research_draft"] + drafts[:1]
    state["plan"] = previous + steps
    _event(state, "replan" if revise else "plan", steps=steps)


def _reflect(state: dict, observation: dict) -> dict:
    if state["plan"][state["cursor"]]["tool"] in {"writing_sources", "writing_draft"}:
        success = observation.get("success", False)
        return {"hypothesis_check": "文献元数据已收集，写作产物需按引用逐条核查。" if success else "写作步骤未成功。",
                "lesson": "新颖性不能保证；未做的实验不得写成结果。" if success else str(observation.get("message", "失败"))[:300],
                "next_step": "继续生成并展示可核查的写作产物。" if success else "检查模型 API 或文献服务后重试。",
                "decision": "continue" if success else "stop"}
    if state["plan"][state["cursor"]]["tool"] == "research_draft":
        success = observation.get("success", False)
        return {"hypothesis_check": "论文框架已生成，参考文献需逐条回查；新颖性和实验结果尚未验证。" if success else "论文框架生成失败。",
                "lesson": "不能保证想法从未有人提出，也不能把拟议实验写成已完成结果。" if success else str(observation.get("message", "生成失败"))[:300],
                "next_step": "展示框架及参考文献，提示用户核验新颖性。" if success else "检查工具错误后重试。",
                "decision": "continue" if success else "stop"}
    if state["plan"][state["cursor"]]["tool"] == "answer_with_evidence" and observation.get("success"):
        summary = observation.get("summary", {})
        sources = summary.get("sources", [])
        anchors = "；".join(f"{item.get('source_filename') or item.get('source_id')} 第{item.get('page')}页：{item.get('quote', '')[:90]}" for item in sources[:4])
        count = summary.get("verified_count", 0)
        return {"hypothesis_check": f"通过引文原文、数字单位及语义核验的陈述 {count} 条。{anchors}" if count else "未找到可完整核验的原文依据。",
                "lesson": "只输出可定位并核验的陈述；未通过核验的候选不作为答案。",
                "next_step": "展示已核验陈述及来源锚点。" if count else "明确提示证据不足或待核查。",
                "decision": "continue"}
    if state.get("version_id") and state["plan"][state["cursor"]]["tool"] == "answer_with_evidence":
        return {"hypothesis_check": "原文检索或核验未完成。", "lesson": "不能在缺少核验结果时回答论文细节。",
                "next_step": "检查检索服务和论文索引后重试。", "decision": "stop"}
    if state["plan"][state["cursor"]]["tool"] in {"architecture_topology", "csv_analysis"}:
        success = observation.get("success", False)
        is_csv = state["plan"][state["cursor"]]["tool"] == "csv_analysis"
        return {"hypothesis_check": ("模型绘图代码已对所附 CSV 运行，PNG 已通过校验。" if is_csv else "架构脚本已由同一组节点和边生成。") if success else "工具未成功返回。",
                "lesson": ("图注由 CSV 数值计算；模型绘图逻辑可在执行轨迹核查。" if is_csv else "四种脚本共享统一拓扑，具体节点仍需用户核对。") if success else str(observation.get("message", "生成失败"))[:300],
                "next_step": "继续执行剩余任务或展示脚本。" if success else "检查任务输入并重试。",
                "decision": "continue" if success else "stop"}
    try:
        response = _model_json(
        "Reflect on the observed research tool result. Do not invent evidence. Return JSON with "
        "nonempty strings for hypothesis_check, lesson, next_step, and decision "
        "(continue|stop|revise). Stop when evidence is insufficient or a repeated failure has no new strategy.",
        {"goal": state["goal"], "step": state["plan"][state["cursor"]],
         "observation": observation, "recent_events": state["events"][-4:]},
        )
    except (ValueError, IndexError, KeyError, TypeError):
        success = observation.get("success", False)
        return {"hypothesis_check": "工具已返回结果。" if success else "工具执行失败。",
                "lesson": "模型反思输出不可解析；按工具的实际执行状态记录，不补造证据。",
                "next_step": "继续处理已取得的结果。" if success else "检查工具错误并重试。",
                "decision": "continue" if success else "stop"}
    for field in ("hypothesis_check", "lesson", "next_step"):
        if not isinstance(response.get(field), str) or not response[field].strip():
            raise ValueError(f"Agent reflection lacks {field}")
    if response.get("decision") not in {"continue", "stop", "revise"}:
        raise ValueError("Agent reflection has an invalid decision")
    return {key: response[key][:600] for key in ("hypothesis_check", "lesson", "next_step", "decision")}


def submit_agent_task(goal: str, paper_ids: list[str] | None = None, *, task_type: str = "research",
                      csv_filename: str | None = None, csv_text: str | None = None,
                      version_id: str | None = None) -> dict:
    if task_type not in {"research", "writing"}:
        raise ValueError("Unsupported task type")
    if task_type == "research" and WRITING_INTENT.search(goal):
        task_type = "writing"
    if (csv_filename is None) != (csv_text is None):
        raise ValueError("CSV filename and content must be supplied together")
    if version_id and len(paper_ids or []) != 1:
        raise ValueError("A version_id requires exactly one paper_id")
    if csv_text is not None and (not csv_filename or not csv_filename.lower().endswith(".csv") or not csv_text or len(csv_text.encode("utf-8")) > 2_000_000):
        raise ValueError("CSV must be nonempty, end in .csv, and be at most 2 MB")
    state = {"task_id": str(uuid.uuid4()), "goal": goal, "paper_ids": paper_ids or [], "version_id": version_id,
             "task_type": task_type, "csv_filename": csv_filename,
             "status": "queued", "cursor": 0, "plan": [], "artifacts": {}, "events": [],
             "action_count": 0, "failure_counts": {}, "output": "",
             "created_at": _now(), "updated_at": _now()}
    if csv_text is not None:
        redis_client().set(_csv_key(state["task_id"]), csv_text, ex=TTL)
    _event(state, "created", goal=goal)
    _launch(state["task_id"])
    return state


def resume_agent_task(task_id: str) -> dict | None:
    state = get_agent_task(task_id)
    if state is None:
        return None
    if state["status"] == "completed":
        return state
    if state["status"] == "running":
        age = datetime.now(timezone.utc) - datetime.fromisoformat(state["updated_at"])
        if age.total_seconds() < 300:
            raise ValueError("Task is already running; stale tasks can resume after five minutes")
    state["status"] = "queued"
    _event(state, "resume_requested")
    _launch(task_id)
    return state


def _launch(task_id: str) -> None:
    with _local_lock:
        if task_id in _running:
            raise ValueError("Task is already running")
        _running.add(task_id)
    try:
        _pool.submit(_run, task_id)
    except Exception:
        with _local_lock:
            _running.discard(task_id)
        raise


def _run(task_id: str) -> None:
    try:
        state = get_agent_task(task_id)
        if state is None:
            return
        state["status"] = "running"
        _event(state, "started")
        if not state["plan"] or state.pop("replan_requested", False):
            _plan(state, revise=bool(state["plan"]))
        while state["cursor"] < len(state["plan"]):
            step = state["plan"][state["cursor"]]
            if step["status"] == "completed":
                state["cursor"] += 1
                continue
            index = state["cursor"]
            if any(key.startswith(f"{step['tool']}:{step['query']}:") and count >= 2
                   for key, count in state["failure_counts"].items()):
                state["status"] = "failed"
                _event(state, "failed", step=index, reason="Repeated identical tool failure; change strategy")
                return
            _event(state, "think", step=index, intention=step["success_criterion"])
            _event(state, "act", step=index, tool=step["tool"], query=step["query"])
            state["action_count"] = state.get("action_count", 0) + 1
            if state["action_count"] > MAX_STEPS:
                raise ValueError("Agent action budget exhausted")
            try:
                if step["tool"] == "csv_analysis":
                    from .csv_plot_agent import generate_csv_plot
                    csv_text = redis_client().get(_csv_key(task_id))
                    if not csv_text:
                        raise ValueError("CSV attachment expired or is unavailable")
                    if isinstance(csv_text, bytes):
                        csv_text = csv_text.decode("utf-8")
                    result = generate_csv_plot(state["csv_filename"], csv_text, step["query"],
                                               on_attempt=lambda attempt: _event(state, "plot_attempt", step=index,
                                                                                 attempt=attempt))
                elif step["tool"] == "writing_draft":
                    result = _write_research_draft(step["query"], state["artifacts"].get(str(index - 1), {}))
                elif step["tool"] == "answer_with_evidence" and state.get("version_id"):
                    from .hybrid_retrieval import answer_with_evidence
                    result = answer_with_evidence(step["query"], top_k=6,
                                                  paper_id=state["paper_ids"][0], version_id=state["version_id"])
                else:
                    result = _tool(step["tool"], step["query"], state["paper_ids"])
                if not isinstance(result, dict):
                    raise ValueError("Tool returned a non-object result")
                if step["tool"] == "answer_with_evidence":
                    verified_claims = [claim for claim in result.get("claims", []) if claim.get("verified")]
                    state["artifacts"][str(index)] = {**result, "claims": verified_claims,
                        "rejected_claim_count": len(result.get("claims", [])) - len(verified_claims)}
                else:
                    state["artifacts"][str(index)] = result
                rendered = result.get("markdown") if step["tool"] in {"research_draft", "writing_draft"} else None
                if step["tool"] == "general_answer":
                    rendered = "（未使用论文证据）\n" + (result.get("answer") or "模型未返回文字。")
                if not rendered and step["tool"] == "answer_with_evidence":
                    rendered = "\n\n".join(
                        claim.get("text", "") + "\n" + "\n".join(
                            f"来源：{cite.get('source_filename') or cite.get('title') or cite.get('source_id')}，第 {cite.get('page')} 页；原文：{cite.get('quote')}"
                            for cite in claim.get("citations", []) if cite.get("quote_location", {}).get("found"))
                        for claim in result.get("claims", []) if claim.get("verified"))
                    if not rendered:
                        rendered = "证据不足或待核查：当前论文没有可同时通过原文定位、数字单位和语义支持校验的陈述。"
                if not rendered and step["tool"] == "search_papers":
                    records = result.get("results", result.get("papers", []))
                    if isinstance(records, list):
                        rendered = "\n".join(f"{item.get('title', '未知题名')} · {item.get('doi') or item.get('source') or '来源待核查'}" for item in records[:8]) or "未找到匹配文献。"
                if not rendered and step["tool"] == "graph_review":
                    rendered = str(result.get("overview") or "图谱综述已生成，请在执行轨迹中查看完整结果。")
                if not rendered and step["tool"] == "future_work":
                    rendered = str(result.get("note") or "Future Work 分析已完成，请在执行轨迹中查看证据。")
                if not rendered and step["tool"] == "citation_graph":
                    rendered = "引用图谱已生成。" + json.dumps(result.get("summary", {}), ensure_ascii=False)
                if not rendered and step["tool"] == "concept_graph":
                    rendered = f"概念视图已生成：{result.get('summary', {}).get('term_count', 0)} 个术语；请到引用图谱页面查看来源。"
                if not rendered and step["tool"] == "draft_alignment":
                    rendered = "\n\n".join(section.get("section", "草稿") + "：\n" + "\n".join(
                        f"候选：{match['title']}，第 {match['page']} 页；原文：{match['quote']}"
                        for match in section.get("matches", [])) for section in result.get("sections", [])) or "未找到足够相关的已入库原文段落。"
                if not rendered and step["tool"] == "architecture_topology":
                    rendered = f"架构拓扑图已生成：{result.get('title', '研究流程草图')}。可预览并下载 Mermaid、Graphviz、TikZ、Matplotlib 脚本。"
                if not rendered and step["tool"] == "csv_analysis":
                    rendered = str(result.get("caption") or "CSV 图表已生成，请查看统计结果。")
                if isinstance(rendered, str):
                    state["output"] = ""
                    for offset in range(0, len(rendered), 160):
                        delta = rendered[offset:offset + 160]
                        state["output"] += delta
                        _event(state, "output_delta", step=index, text=delta)
                observation = {"success": True, "summary": _summary(step["tool"], result)}
                step["status"] = "completed"
            except Exception as exc:
                observation = {"success": False, "error_type": type(exc).__name__, "message": str(exc)[:250]}
                step["status"] = "failed"
            _event(state, "observe", step=index, observation=observation)
            reflection = _reflect(state, observation)
            _event(state, "reflect", step=index, reflection=reflection)
            if not observation["success"]:
                fingerprint = f"{step['tool']}:{step['query']}:{observation['error_type']}"
                counts = state["failure_counts"]
                counts[fingerprint] = counts.get(fingerprint, 0) + 1
                if reflection["decision"] == "revise" and counts[fingerprint] < 2:
                    state["replan_requested"] = True
                    state["status"] = "needs_review"
                    _event(state, "needs_review", reason=reflection["next_step"])
                else:
                    state["status"] = "failed"
                    _event(state, "failed", step=index, reason=observation["error_type"])
                return
            state["cursor"] += 1
            _save(state)
            if reflection["decision"] == "stop":
                break
            if reflection["decision"] == "revise":
                state["replan_requested"] = True
                state["status"] = "needs_review"
                _event(state, "needs_review", reason=reflection["next_step"])
                return
        state["status"] = "completed"
        _event(state, "completed", finished_steps=state["cursor"])
    except Exception as exc:
        state = get_agent_task(task_id)
        if state:
            state["status"] = "failed"
            _event(state, "failed", error_type=type(exc).__name__, message=str(exc)[:250])
    finally:
        with _local_lock:
            _running.discard(task_id)
