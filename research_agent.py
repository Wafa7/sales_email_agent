"""
LLM-powered ResearchAgent

Features:
- Fetch website text (simple scraper)
- Run Serper search to gather extra context
- Call OpenAI (configurable model) asking for structured JSON output:
    { summary, pain_points, recommended_roles, fit_score, confidence, follow_up_queries }
- If fit_score or confidence is low, optionally perform 1 extra retrieval iteration using follow_up_queries
- Returns a structured dict with insights and sources.

Requires:
- SERPER_API_KEY in env (or pass to constructor) for SerperSearchTool
- OPENAI_API_KEY in env (or pass to constructor) for OpenAI
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# Import your SerperSearchTool from search_api.py
from search_api import SerperSearchTool

# --- 🛠️ UPDATED IMPORTS ---
from openai import OpenAI, APIError 


MAX_CONTEXT_CHARS = 3000  # keep LLM prompt reasonably sized


def _truncate(text: str, n: int):
    if not text:
        return ""
    return text[:n]


def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        # attempt to salvage common mistakes: look for first { ... }
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
        return None


class ResearchAgent:
    def __init__(self, serper_api_key: str = None, openai_api_key: str = None,
                 model: str = "gpt-4o-mini", max_iters: int = 1):
        """
        serper_api_key/openai_api_key: optional, taken from environment if not provided
        model: LLM model to use (e.g. gpt-4o-mini). Change as needed.
        max_iters: number of total LLM-driven retrieval iterations (1 = no retry, 2 = one retry)
        """
        self.serper_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_iters = max_iters if max_iters >= 1 else 1

        if not self.openai_key:
            raise RuntimeError("OPENAI_API_KEY required for ResearchAgent")
        
        # --- 🛠️ UPDATED: Initialize OpenAI client in __init__ ---
        self.openai_client = OpenAI(api_key=self.openai_key)

        if not self.serper_key:
            # Serper is optional: agent will still run with only website scraping
            self.search_tool = None
        else:
            self.search_tool = SerperSearchTool(api_key=self.serper_key)

    # --- 🛠️ REMOVED: _init_openai method is no longer needed ---
    # def _init_openai(self):
    #     openai.api_key = self.openai_key

    # ----------------------
    # Retrieval helpers
    # ----------------------
    def _fetch_website_text(self, url: str, max_chars: int = 3000) -> str:
        if not url:
            return ""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for t in soup(["script", "style", "noscript"]):
                t.decompose()
            text = " ".join(soup.stripped_strings)
            return _truncate(text, max_chars)
        except Exception:
            return ""

    def _serper_search(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        if not self.search_tool:
            return []
        try:
            return self.search_tool.search(query, num_results=num_results)
        except Exception:
            return []

    # ----------------------
    # LLM interaction
    # ----------------------
    def _ask_llm_for_insights(self, company_name: str, website_text: str,
                              snippets: List[Dict[str, str]], offering: str = "",
                              tone: str = "Professional") -> Dict[str, Any]:
        """
        Instruct the LLM to output strict JSON with keys:
        - summary (short)
        - pain_points (list of short strings)
        - recommended_roles (list of strings, e.g., ['VP Sales', 'Head of Partnerships'])
        - fit_score (int 0-100)
        - confidence (float 0.0-1.0)
        - follow_up_queries (list of strings)
        """
        # Build context
        context_parts = []
        if website_text:
            context_parts.append("Website snippet:\n" + _truncate(website_text, 1200))
        for s in snippets:
            title = s.get("title") or ""
            link = s.get("link") or ""
            snip = s.get("snippet") or ""
            context_parts.append(f"Source: {link}\n{snip}")
        context = "\n\n".join(context_parts)
        context = _truncate(context, MAX_CONTEXT_CHARS)

        system_msg = (
            "You are a research assistant specialized in GTM and SDR outreach. "
            "You must respond with strict JSON (no surrounding commentary) using the exact keys: "
            "summary, pain_points, recommended_roles, fit_score, confidence, follow_up_queries. "
            "Values:\n"
            "- summary: short (max 80 words) summary of the company's offering and positioning based only on the provided context.\n"
            "- pain_points: array of 0..6 short strings describing problems this company likely faces or signals indicating need.\n"
            "- recommended_roles: array of roles/titles to contact for outreach.\n"
            "- fit_score: integer 0..100 indicating how good a fit (100 = perfect fit) relative to the offering.\n"
            "- confidence: float 0.0..1.0 expressing confidence in these outputs.\n"
            "- follow_up_queries: array of up to 4 search queries to run if additional info is needed.\n\n"
            "DO NOT HALLUCINATE facts. If context is insufficient, be conservative: lower fit_score and include follow_up_queries.\n"
        )

        user_msg = (
            f"Company: {company_name}\n"
            f"Offering to pitch: {offering}\n"
            f"Tone: {tone}\n\n"
            f"Context:\n{context}\n\n"
            "Provide the JSON now."
        )

        # --- 🛠️ UPDATED: Call the OpenAI ChatCompletion via the client ---
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                max_tokens=700
            )
            # --- 🛠️ UPDATED: Access the response content (uses dot notation) ---
            text = resp.choices[0].message.content
            parsed = _safe_json_load(text)
            if parsed is None:
                # return full raw text fallback
                return {"__raw": text}
            return parsed
        # --- 🛠️ UPDATED: Catch the new OpenAI exception class (APIError) ---
        except APIError as e:
            return {"__error": f"OpenAI API Error: {str(e)}"}
        except Exception as e:
            return {"__error": str(e)}

    # ----------------------
    # Main run logic
    # ----------------------
    def run(self, company: Dict[str, Any], offering: str = "", tone: str = "Professional") -> Dict[str, Any]:
        """
        company: { 'name': ..., 'website': ..., 'description': ... }
        Returns: structured dict containing insights, raw LLM output, sources, final_score, etc.
        """
        name = company.get("name") or ""
        website = company.get("website") or company.get("site") or ""
        result = {
            "company": name,
            "website": website,
            "offering": offering,
            "iterations": [],
            "final": None,
            "sources": []
        }

        # 1) initial retrieval
        site_text = self._fetch_website_text(website)
        if site_text:
            result["sources"].append({"source": website, "type": "website"})

        # run an initial web search (if available)
        initial_query = f"{name} reviews OR reddit OR news OR funding"
        snippets = self._serper_search(initial_query, num_results=3)
        for s in snippets:
            result["sources"].append({"source": s.get("link"), "type": "search"})

        # 2) call LLM for analysis
        iter_count = 0
        last_parsed = None
        while iter_count < self.max_iters:
            iter_count += 1
            llm_output = self._ask_llm_for_insights(name, site_text, snippets, offering, tone)
            iteration_record = {"iteration": iter_count, "llm_raw": llm_output}
            result["iterations"].append(iteration_record)

            # If LLM returned __raw or error, stop and return raw
            if "__error" in llm_output or "__raw" in llm_output:
                # capture and finish
                result["final"] = {
                    "summary": None,
                    "pain_points": [],
                    "recommended_roles": [],
                    "fit_score": 0,
                    "confidence": 0.0,
                    "note": "LLM failed or returned non-JSON"
                }
                result["final"]["llm_raw"] = llm_output.get("__raw") or llm_output.get("__error")
                break

            # Normalize fields
            summary = llm_output.get("summary", "")
            pain_points = llm_output.get("pain_points") or []
            recommended_roles = llm_output.get("recommended_roles") or []
            fit_score = int(llm_output.get("fit_score") or 0)
            confidence = float(llm_output.get("confidence") or 0.0)
            follow_up_queries = llm_output.get("follow_up_queries") or []

            last_parsed = {
                "summary": summary,
                "pain_points": pain_points,
                "recommended_roles": recommended_roles,
                "fit_score": fit_score,
                "confidence": confidence,
                "follow_up_queries": follow_up_queries
            }

            # If the agent is confident enough or fit_score is high, finish
            if confidence >= 0.65 or fit_score >= 70 or iter_count >= self.max_iters:
                result["final"] = last_parsed
                break

            # Otherwise, perform one more retrieval step using follow_up_queries
            if follow_up_queries:
                # fetch more snippets by running the first 2 follow up queries
                extra_snips = []
                for q in follow_up_queries[:2]:
                    extra_snips.extend(self._serper_search(q, num_results=2))
                # append to existing snippets and continue
                snippets = (snippets or []) + extra_snips
                # also expand site_text by re-fetching (no-op in many cases)
                site_text = site_text + " " + " ".join([_truncate(s.get("snippet",""), 800) for s in extra_snips])
                # small sleep to avoid rapid-fire calls
                time.sleep(0.5)
                continue
            else:
                # no follow up queries -> end
                result["final"] = last_parsed
                break

        # attach raw llm record too
        result["llm_last_raw"] = result["iterations"][-1]["llm_raw"] if result["iterations"] else None
        return result