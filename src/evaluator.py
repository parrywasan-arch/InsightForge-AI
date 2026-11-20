import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


class InsightForgeEvaluator:
    def __init__(self, llm: Any = None, model_name: Optional[str] = None):
        self.llm = llm
        self.model_name = model_name or "gpt-4o-mini"
        if self.llm is None:
            self._init_llm()

    def _init_llm(self):
        load_dotenv()
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            self.llm = None
            return
        try:
            from langchain_openai import ChatOpenAI

            base_url = (os.getenv("OPENAI_API_BASE") or "").strip() or None
            self.model_name = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "api_key": api_key,
                "temperature": 0.0,
                "timeout": 20,
                "max_retries": 1,
            }
            if base_url:
                kwargs["base_url"] = base_url

            self.llm = ChatOpenAI(**kwargs)
        except Exception:
            self.llm = None

    @staticmethod
    def _heuristic_score(
        question: str,
        answer: str,
        context: str = "",
        reason: str = "Local heuristic evaluation (no LLM key or LLM call skipped)",
    ) -> Dict[str, Any]:
        q_tokens = {w for w in re.findall(r"\w{3,}", (question or "").lower())}
        a_text = (answer or "").strip()
        a_tokens = {w for w in re.findall(r"\w{3,}", a_text.lower())}

        if not a_text or a_text.lower().startswith("**llm error"):
            return {
                "groundedness": 0.0,
                "relevance": 0.0,
                "fairness": 1.0,
                "overall": 0.0,
                "mode": "Local Heuristic",
                "verdict": "Error / Empty Answer",
                "notes": reason,
            }

        overlap = (len(q_tokens & a_tokens) / len(q_tokens)) if q_tokens else 0.6
        relevance = round(min(1.0, 0.45 + overlap * 0.65), 2)

        has_context = bool((context or "").strip())
        if has_context:
            ctx_tokens = {w for w in re.findall(r"\w{4,}", context.lower())}
            ctx_overlap = (len(a_tokens & ctx_tokens) / max(1, min(len(a_tokens), 25))) if ctx_tokens else 0.5
            groundedness = round(min(1.0, 0.5 + ctx_overlap * 0.5), 2)
        else:
            groundedness = 0.65

        fairness = 1.0
        overall = round((groundedness + relevance + fairness) / 3.0, 2)

        return {
            "groundedness": groundedness,
            "relevance": relevance,
            "fairness": fairness,
            "overall": overall,
            "mode": "Local Heuristic",
            "verdict": "Pass" if overall >= 0.6 else "Review",
            "notes": reason,
        }

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str = "",
        llm: Any = None,
    ) -> Dict[str, Any]:
        active_llm = llm if llm is not None else self.llm
        if active_llm is None:
            return self._heuristic_score(
                question=question,
                answer=answer,
                context=context,
                reason="Running without LLM key; scored using local heuristic checks.",
            )

        prompt = (
            "You are an objective BI answer quality evaluator. Score the response from 0.0 to 1.0 on:\n"
            "- groundedness (supported by provided context without hallucination)\n"
            "- relevance (directly answers the user question)\n"
            "- fairness (neutral, non-biased tone)\n"
            "Reply strictly in 4 short lines:\n"
            "groundedness: <0.0-1.0>\n"
            "relevance: <0.0-1.0>\n"
            "fairness: <0.0-1.0>\n"
            "notes: <one brief sentence>\n\n"
            f"Question: {question}\n"
            f"Context: {(context or 'None')[:2000]}\n"
            f"Answer: {(answer or '')[:2000]}"
        )

        try:
            resp = active_llm.invoke(prompt)
            raw_text = getattr(resp, "content", "") or ""

            def _extract_score(label: str, default: float) -> float:
                m = re.search(rf"{label}\s*[:=]\s*([0-1](?:\.\d+)?)", raw_text, flags=re.IGNORECASE)
                if not m:
                    return default
                val = float(m.group(1))
                return max(0.0, min(1.0, round(val, 2)))

            groundedness = _extract_score("groundedness", 0.75)
            relevance = _extract_score("relevance", 0.75)
            fairness = _extract_score("fairness", 1.0)
            notes_match = re.search(r"notes\s*[:=]\s*(.+)", raw_text, flags=re.IGNORECASE)
            notes = notes_match.group(1).strip() if notes_match else f"Evaluated via {self.model_name}."
            overall = round((groundedness + relevance + fairness) / 3.0, 2)

            return {
                "groundedness": groundedness,
                "relevance": relevance,
                "fairness": fairness,
                "overall": overall,
                "mode": f"LLM Judge ({self.model_name})",
                "verdict": "Pass" if overall >= 0.65 else "Review",
                "notes": notes,
            }
        except Exception as exc:
            return self._heuristic_score(
                question=question,
                answer=answer,
                context=context,
                reason=f"LLM judge skipped ({type(exc).__name__}); used local fallback score.",
            )

    def evaluate_batch(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in items or []:
            results.append(
                self.evaluate_single(
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    context=item.get("context", ""),
                )
            )
        return results