"""
diagnosis_engine.py - Core fault diagnosis using Groq API (LLaMA 3.3 70B).
"""

from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from prompts import DIAGNOSIS_SYSTEM_PROMPT, DIAGNOSIS_USER_PROMPT, REPORT_USER_PROMPT
from utils import now_str, safe_parse_json

load_dotenv()


def _make_groq_client(api_key: str):
    """
    Build a Groq client in a way that is resilient to httpx version
    mismatches (older groq SDKs pass an 'proxies' kwarg that newer
    httpx releases removed). Falls back to a manually-built httpx
    client if the default constructor fails.
    """
    from groq import Groq

    try:
        return Groq(api_key=api_key)
    except TypeError as exc:
        if "proxies" in str(exc):
            # Newer httpx removed the 'proxies' kwarg; build our own client.
            import httpx
            http_client = httpx.Client()
            return Groq(api_key=api_key, http_client=http_client)
        raise


@dataclass
class DiagnosisResult:
    faults_detected:        list[dict]    = field(default_factory=list)
    root_causes:            list[dict]    = field(default_factory=list)
    severity:               str           = "Unknown"
    severity_justification: str           = ""
    recommended_fixes:      list[dict]    = field(default_factory=list)
    maintenance_actions:    list[dict]    = field(default_factory=list)
    risk_assessment:        dict          = field(default_factory=dict)
    failure_prediction:     dict          = field(default_factory=dict)
    health_score:           int           = 0
    summary:                str           = ""
    preventive_actions:     list[str]     = field(default_factory=list)
    maintenance_schedule:   list[dict]    = field(default_factory=list)
    raw_response:           str           = ""
    error:                  Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, raw: str = "") -> "DiagnosisResult":
        try:
            hs = int(data.get("health_score", 0))
        except (TypeError, ValueError):
            hs = 0
        return cls(
            faults_detected        = data.get("faults_detected") or [],
            root_causes            = data.get("root_causes") or [],
            severity               = str(data.get("severity", "Unknown")),
            severity_justification = str(data.get("severity_justification", "")),
            recommended_fixes      = data.get("recommended_fixes") or [],
            maintenance_actions    = data.get("maintenance_actions") or [],
            risk_assessment        = data.get("risk_assessment") or {},
            failure_prediction     = data.get("failure_prediction") or {},
            health_score           = max(0, min(100, hs)),
            summary                = str(data.get("summary", "")),
            preventive_actions     = data.get("preventive_actions") or [],
            maintenance_schedule   = data.get("maintenance_schedule") or [],
            raw_response           = raw,
        )

    def to_dict(self) -> dict:
        return {
            "faults_detected":        self.faults_detected,
            "root_causes":            self.root_causes,
            "severity":               self.severity,
            "severity_justification": self.severity_justification,
            "recommended_fixes":      self.recommended_fixes,
            "maintenance_actions":    self.maintenance_actions,
            "risk_assessment":        self.risk_assessment,
            "failure_prediction":     self.failure_prediction,
            "health_score":           self.health_score,
            "summary":                self.summary,
            "preventive_actions":     self.preventive_actions,
            "maintenance_schedule":   self.maintenance_schedule,
        }


class DiagnosisEngine:
    """Groq-backed fault diagnosis engine using LLaMA 3.3 70B."""

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key or api_key == "your_groq_api_key_here":
            raise EnvironmentError(
                "GROQ_API_KEY not configured.\n"
                "Run: python setup_env.py\n"
                "Or get a free key at https://console.groq.com"
            )
        try:
            self.client = _make_groq_client(api_key)
        except Exception as exc:
            raise EnvironmentError(
                "Could not initialise the Groq client. This is usually caused "
                "by a version mismatch between the 'groq' and 'httpx' packages.\n"
                "Fix: pip install -r requirements.txt --force-reinstall\n"
                f"Original error: {exc}"
            )

    def diagnose(self, data_summary: str, statistics: str, anomaly_summary: str) -> DiagnosisResult:
        prompt = DIAGNOSIS_USER_PROMPT.format(
            data_summary = data_summary[:4000],
            statistics   = statistics[:1500],
            anomalies    = anomaly_summary[:1000],
        )
        try:
            resp = self.client.chat.completions.create(
                model       = self.MODEL,
                temperature = 0.1,
                max_tokens  = 4096,
                messages    = [
                    {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
            parsed = safe_parse_json(raw)
            if parsed is None:
                # Second attempt: ask model to fix its own output
                fix_resp = self.client.chat.completions.create(
                    model      = self.MODEL,
                    temperature= 0.0,
                    max_tokens = 4096,
                    messages   = [
                        {"role": "system",    "content": "Return only valid JSON. No prose."},
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": raw},
                        {"role": "user",      "content": "Your response was not valid JSON. Return ONLY the JSON object, nothing else."},
                    ],
                )
                raw = fix_resp.choices[0].message.content or ""
                parsed = safe_parse_json(raw)

            if parsed is None:
                return DiagnosisResult(
                    error="Could not parse model response as JSON. Try again.",
                    raw_response=raw[:500],
                )
            return DiagnosisResult.from_dict(parsed, raw=raw)

        except Exception as exc:
            err = str(exc)
            if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
                return DiagnosisResult(error="Invalid Groq API key. Run python setup_env.py")
            if "429" in err or "rate_limit" in err.lower():
                return DiagnosisResult(error="Rate limit hit. Wait a moment and retry.")
            if "model_not_found" in err.lower() or "decommissioned" in err.lower():
                return DiagnosisResult(error=f"Model unavailable: {self.MODEL}. Check Groq's current model list.")
            return DiagnosisResult(error=f"Groq API error: {err[:200]}")

    def generate_markdown_report(self, diagnosis: DiagnosisResult, filename: str) -> str:
        prompt = REPORT_USER_PROMPT.format(
            diagnosis_json = json.dumps(diagnosis.to_dict(), indent=2)[:4000],
            filename       = filename,
            date           = now_str(),
        )
        try:
            resp = self.client.chat.completions.create(
                model      = self.MODEL,
                temperature= 0.3,
                max_tokens = 3000,
                messages   = [{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or "Report generation failed."
        except Exception as exc:
            return f"# Report Error\n\n{exc}"
