"""Offline state-machine tests: no model calls or external services."""

import json
import unittest
from unittest.mock import patch

from backend.app.services import research_agent as agent
from backend.app.services import hybrid_retrieval


class MemoryRedis:
    def __init__(self):
        self.values = {}

    def set(self, key, value, ex=None):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.redis = MemoryRedis()
        self.redis_patch = patch.object(agent, "redis_client", return_value=self.redis)
        self.redis_patch.start()
        self.addCleanup(self.redis_patch.stop)

    def test_plan_observe_reflect_complete(self):
        def model(system, payload):
            if "Plan a research task" in system:
                return {"steps": [{"tool": "search_papers", "query": "transformer", "success_criterion": "Find metadata"}]}
            return {"hypothesis_check": "A source was found", "lesson": "Use source metadata",
                    "next_step": "Finish", "decision": "continue"}

        with patch.object(agent, "_model_json", side_effect=model), patch.object(
            agent, "_tool", return_value={"results": [{"title": "Paper", "doi": "10.1/x", "source": "Crossref"}]}
        ), patch.object(agent, "_launch"):
            task = agent.submit_agent_task("Find papers on transformer")
            agent._run(task["task_id"])
        state = agent.get_agent_task(task["task_id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["cursor"], 1)
        self.assertEqual([e["kind"] for e in state["events"] if e["kind"] != "output_delta"],
                         ["created", "started", "plan", "think", "act", "observe", "reflect", "completed"])
        self.assertIn("0", state["artifacts"])

    def test_invalid_tool_rejected_before_execution(self):
        with patch.object(agent, "_model_json", return_value={"steps": [
            {"tool": "execute_code", "query": "rm", "success_criterion": "done"}]}), patch.object(
            agent, "_tool") as tool, patch.object(agent, "_launch"):
            task = agent.submit_agent_task("Please inspect the literature")
            agent._run(task["task_id"])
        self.assertEqual(agent.get_agent_task(task["task_id"])["status"], "failed")
        tool.assert_not_called()

    def test_writing_adds_draft_and_streams_chunks(self):
        with patch.object(agent, "_model_json") as model, patch.object(
            agent, "_tool", return_value={"references": []}
        ), patch.object(agent, "_write_research_draft", return_value={
            "markdown": "# Draft\n" + "A" * 320, "references": []
        }), patch.object(agent, "_launch"):
            task = agent.submit_agent_task("Qwen code reinforcement learning", task_type="writing")
            agent._run(task["task_id"])
        state = agent.get_agent_task(task["task_id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["plan"][-1]["tool"], "writing_draft")
        self.assertEqual(state["output"], "# Draft\n" + "A" * 320)
        self.assertEqual(len([e for e in state["events"] if e["kind"] == "output_delta" and e["step"] == 1]), 3)
        model.assert_not_called()

    def test_single_input_detects_writing_intent(self):
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("根据这个 Idea 生成论文框架与 References")
        self.assertEqual(task["task_type"], "writing")

    def test_short_question_can_be_queued(self):
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("1+1=?")
        self.assertEqual(task["goal"], "1+1=?")
        with patch.object(agent, "_model_json") as model:
            agent._plan(task)
        self.assertEqual(task["plan"][0]["tool"], "general_answer")
        model.assert_not_called()

    def test_attached_paper_forces_scoped_evidence_qa(self):
        paper_id = "11111111-1111-1111-1111-111111111111"
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("What was the BLEU score?", [paper_id])
        with patch.object(agent, "_model_json") as model:
            agent._plan(task)
        self.assertEqual(task["plan"][0]["tool"], "answer_with_evidence")
        model.assert_not_called()
        from backend.app.services import hybrid_retrieval
        with patch.object(hybrid_retrieval, "answer_with_evidence", return_value={"answer_status": "insufficient_evidence", "claims": []}) as answer:
            agent._tool("answer_with_evidence", task["goal"], [paper_id])
        answer.assert_called_once_with(task["goal"], top_k=6, paper_id=paper_id)

    def test_single_paper_answer_hides_unverified_claims_and_keeps_source_page(self):
        paper_id = "11111111-1111-1111-1111-111111111111"
        version_id = "22222222-2222-2222-2222-222222222222"
        result = {"answer_status": "needs_review", "claims": [
            {"text": "论文使用 8 张 GPU。", "verified": True, "citations": [{"source_id": "chunk-1",
                "paper_id": paper_id, "version_id": version_id, "source_filename": "paper.pdf", "page": 7,
                "quote": "trained on 8 GPUs", "quote_location": {"found": True}}]},
            {"text": "论文使用 64 张 GPU。", "verified": False, "citations": []},
        ]}
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("使用了多少 GPU？", [paper_id], version_id=version_id)
        from backend.app.services import hybrid_retrieval
        with patch.object(hybrid_retrieval, "answer_with_evidence", return_value=result) as answer, patch.object(agent, "_model_json") as model:
            agent._run(task["task_id"])
        state = agent.get_agent_task(task["task_id"])
        self.assertEqual(state["status"], "completed")
        self.assertIn("论文使用 8 张 GPU。", state["output"])
        self.assertIn("paper.pdf，第 7 页；原文：trained on 8 GPUs", state["output"])
        self.assertNotIn("64 张", state["output"])
        self.assertEqual(state["artifacts"]["0"]["rejected_claim_count"], 1)
        self.assertNotIn("64 张", str(state["artifacts"]))
        answer.assert_called_once_with(task["goal"], top_k=6, paper_id=paper_id, version_id=version_id)
        model.assert_not_called()

    def test_evidence_requires_pdf_page_anchor(self):
        source = {"source_id": "chunk-1", "paper_id": "paper-1", "version_id": "version-1",
                  "source_filename": "paper.pdf", "title": "Paper", "page": None,
                  "text": "The model used 8 GPUs."}
        generated = {"claims": [{"text": "The model used 8 GPUs.",
                                  "citations": [{"source_id": "chunk-1", "quote": "The model used 8 GPUs."}]}]}
        with patch.object(hybrid_retrieval, "hybrid_search", return_value={"results": [source], "retrieval": {}}), \
             patch.object(hybrid_retrieval, "_llm_json", return_value=generated), \
             patch.object(hybrid_retrieval, "_semantic_judgments", return_value={0: {"verdict": "supported", "reason": "exact"}}):
            answer = hybrid_retrieval.answer_with_evidence("How many GPUs?", paper_id="paper-1", version_id="version-1")
        self.assertFalse(answer["claims"][0]["verified"])
        self.assertFalse(answer["claims"][0]["checks"]["source_location_available"])
        source["page"] = 7
        source["version_id"] = "other-version"
        with patch.object(hybrid_retrieval, "hybrid_search", return_value={"results": [source], "retrieval": {}}), \
             patch.object(hybrid_retrieval, "_llm_json", return_value=generated), \
             patch.object(hybrid_retrieval, "_semantic_judgments", return_value={0: {"verdict": "supported", "reason": "exact"}}):
            answer = hybrid_retrieval.answer_with_evidence("How many GPUs?", paper_id="paper-1", version_id="version-1")
        self.assertFalse(answer["claims"][0]["verified"])

    def test_architecture_request_uses_topology_tool(self):
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("根据这个 Qwen RL idea 生成架构拓扑图")
        with patch.object(agent, "_model_json") as model:
            agent._plan(task)
        self.assertEqual([step["tool"] for step in task["plan"]], ["architecture_topology"])
        model.assert_not_called()

    def test_writing_and_topology_request_keeps_both_outputs(self):
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("根据这个 idea 生成论文框架和架构图")
        with patch.object(agent, "_model_json") as model:
            agent._plan(task)
        self.assertEqual([step["tool"] for step in task["plan"]], ["architecture_topology", "research_draft"])
        model.assert_not_called()

    def test_topology_task_completes_without_model_api(self):
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("为轻量 Qwen RL 开放代码问题绘制架构拓扑图")
        with patch.object(agent, "_model_json") as model:
            agent._run(task["task_id"])
        state = agent.get_agent_task(task["task_id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(set(state["artifacts"]["0"]["scripts"]), {"mermaid", "graphviz", "tikz", "matplotlib"})
        self.assertIn("架构拓扑图已生成", state["output"])
        model.assert_not_called()

    def test_csv_attachment_records_model_plot_attempt(self):
        csv_text = "group,value\nA,1\nA,2\nB,3\nB,4\n"
        with patch.object(agent, "_launch"):
            task = agent.submit_agent_task("根据 CSV 绘制柱状图和图注", csv_filename="sample.csv", csv_text=csv_text)
        self.assertNotIn("csv_text", task)
        from backend.app.services import csv_plot_agent
        def generated(filename, text, goal, *, on_attempt):
            on_attempt({"attempt": 1, "status": "success", "code": "fig, ax = plt.subplots()"})
            return {"filename": filename, "row_count": 4, "caption": "sample.csv 图注", "generated_png": "cG5n"}
        with patch.object(csv_plot_agent, "generate_csv_plot", side_effect=generated), patch.object(agent, "_model_json") as model:
            agent._run(task["task_id"])
        state = agent.get_agent_task(task["task_id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["plan"][0]["tool"], "csv_analysis")
        self.assertEqual(state["artifacts"]["0"]["row_count"], 4)
        self.assertIn("sample.csv", state["output"])
        self.assertIn("plot_attempt", [event["kind"] for event in state["events"]])
        self.assertIn("reflect", [event["kind"] for event in state["events"]])
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
