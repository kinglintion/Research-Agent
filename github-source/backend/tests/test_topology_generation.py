import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.services import research_workbench as workbench


class TopologyGenerationTests(unittest.TestCase):
    def test_chinese_nodes_and_branches_share_four_scripts(self):
        labels = ["科学问题", "男性队列", "临床病理", "RNA 测序", "胚系检测", "ER 活性", "AR 活性",
                  "解耦指数", "分型聚类", "预后验证", "类器官验证", "治疗映射"]
        design = {"title": "男性乳腺癌分型技术路线", "nodes": [
            {"label": label, "layer": min(index // 2, 5)} for index, label in enumerate(labels)],
            "edges": [[index, index + 1] for index in range(11)] + [[2, 5], [3, 6], [4, 8]]}
        client = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(design, ensure_ascii=False)))])))))
        with patch.object(workbench, "get_llm_client", create=True, return_value=client):
            # The function imports the adapter locally, so patch its defining module.
            with patch("backend.app.services.model_adapters.get_llm_client", return_value=client):
                result = workbench.mermaid_architecture("男性乳腺癌多组学分型架构拓扑图")
        self.assertEqual(len(result["nodes"]), 12)
        self.assertEqual(len(result["edges"]), 14)
        for script in result["scripts"].values():
            self.assertIn("男性队列", script)
        self.assertIn("ctex", result["tikz"])

    def test_invalid_edges_are_retried_and_named_edges_are_resolved(self):
        labels = [f"研究节点{i}" for i in range(10)]
        nodes = [{"id": f"step{i}", "label": label, "layer": i // 2} for i, label in enumerate(labels)]
        outputs = [json.dumps({"title": "研究路线", "nodes": nodes, "edges": [["missing", "step1"]] * 9}, ensure_ascii=False),
                   json.dumps({"title": "研究路线", "nodes": nodes, "edges": [
                       {"source": f"step{i}", "target": f"step{i+1}"} for i in range(9)]}, ensure_ascii=False)]
        def completion(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outputs.pop(0)))])
        client = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=completion))))
        with patch("backend.app.services.model_adapters.get_llm_client", return_value=client):
            result = workbench.mermaid_architecture("生成研究架构拓扑图")
        self.assertEqual(len(result["edges"]), 9)
        self.assertFalse(outputs)

    def test_repeated_invalid_edges_keep_nodes_and_mark_layout_links(self):
        nodes = [{"id": f"node{i}", "label": f"中文步骤{i}", "layer": i // 2} for i in range(10)]
        output = json.dumps({"title": "研究路线", "nodes": nodes,
                             "edges": [{"source": "unknown", "target": "missing"} for _ in range(9)]}, ensure_ascii=False)
        calls = []
        def completion(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=output))])
        client = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=completion))))
        with patch("backend.app.services.model_adapters.get_llm_client", return_value=client):
            result = workbench.mermaid_architecture("生成研究架构拓扑图")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result["nodes"]), 10)
        self.assertTrue(any(edge["inferred"] for edge in result["edges"]))
        self.assertIn("人工核对", result["note"])


if __name__ == "__main__":
    unittest.main()
