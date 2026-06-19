"""
chatbot.py - Groq-powered maintenance chatbot with session history.
"""

from __future__ import annotations
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from diagnosis_engine import DiagnosisResult, _make_groq_client
from prompts import CHATBOT_SYSTEM_PROMPT

load_dotenv()


class FaultChatbot:
    MODEL       = "llama-3.3-70b-versatile"
    HISTORY_KEY = "chatbot_history"

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.client = _make_groq_client(api_key)
        if self.HISTORY_KEY not in st.session_state:
            st.session_state[self.HISTORY_KEY] = []

    @staticmethod
    def _diagnosis_context(diagnosis: Optional[DiagnosisResult]) -> str:
        if not diagnosis or diagnosis.error:
            return "No diagnosis available yet."
        lines = [
            f"Health Score: {diagnosis.health_score}/100",
            f"Severity: {diagnosis.severity}",
            f"Summary: {diagnosis.summary}",
            "Faults:",
        ]
        for f in diagnosis.faults_detected:
            lines.append(f"  - {f.get('fault_name','?')} [{f.get('confidence','?')}]: {f.get('description','')}")
        lines.append("Top fixes:")
        for fx in diagnosis.recommended_fixes[:3]:
            lines.append(f"  - [{fx.get('priority','?')}] {fx.get('action','')}")
        fp = diagnosis.failure_prediction
        if fp:
            lines.append(f"Time to failure: {fp.get('estimated_time_to_failure','?')}")
        return "\n".join(lines)

    def chat(self, user_message: str, diagnosis: Optional[DiagnosisResult], data_context: str = "") -> str:
        system = CHATBOT_SYSTEM_PROMPT.format(
            diagnosis_context = self._diagnosis_context(diagnosis),
            data_context      = data_context[:800],
        )
        history = st.session_state[self.HISTORY_KEY][-16:]  # last 8 pairs
        messages = [{"role": "system", "content": system}]
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            resp  = self.client.chat.completions.create(
                model      = self.MODEL,
                temperature= 0.4,
                max_tokens = 800,
                messages   = messages,
            )
            reply = resp.choices[0].message.content or "No response."
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate_limit" in err.lower():
                reply = "Rate limit reached. Please wait a moment and try again."
            elif "401" in err or "authentication" in err.lower():
                reply = "Invalid API key. Run python setup_env.py"
            else:
                reply = f"Error: {err[:150]}"

        st.session_state[self.HISTORY_KEY].append({"role": "user",      "content": user_message})
        st.session_state[self.HISTORY_KEY].append({"role": "assistant", "content": reply})
        return reply

    def clear_history(self) -> None:
        st.session_state[self.HISTORY_KEY] = []

    def get_history(self) -> list[dict]:
        return st.session_state.get(self.HISTORY_KEY, [])
