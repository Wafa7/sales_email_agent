"""
LLM-Powered CompanyFinderAgent

Capabilities:
- Uses LLM to generate targeted search queries based on user intent
- Uses Serper.dev search tool to retrieve companies
- LLM evaluates search results and extracts:
    - company name
    - website
    - description
    - relevance score (0–100)
    - confidence group
- If results are weak → LLM generates improved queries → runs a second iteration
"""

import os
import json
import time
# --- 🛠️ UPDATED IMPORT ---
from openai import OpenAI
from search_api import SerperSearchTool


def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
        return None


class CompanyFinderAgent:
    def __init__(self,
                 serper_api_key=None,
                 openai_api_key=None,
                 model="gpt-4o-mini",
                 max_iterations=2):

        self.serper_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_iterations = max_iterations

        if not self.serper_key:
            raise RuntimeError("❌ SERPER_API_KEY missing")

        if not self.openai_key:
            raise RuntimeError("❌ OPENAI_API_KEY missing")

        # --- 🛠️ UPDATED: Initialize the OpenAI client ---
        self.client = OpenAI(api_key=self.openai_key)
        self.search_tool = SerperSearchTool(api_key=self.serper_key)

    # -----------------------------
    # LLM generates smart queries
    # -----------------------------
    def llm_generate_queries(self, criteria: str):
        prompt = f"""
You are an intelligent B2B research agent.

User wants to find companies matching:
"{criteria}"

Generate 3–5 high-value Google search queries that will return:
- real companies
- SaaS companies (if relevant)
- founders / startup profiles
- websites
- company lists

Return strict JSON:
{{"queries": ["...", "..."]}}
"""

        # --- 🛠️ UPDATED LLM CALL ---
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            # --- 🛠️ UPDATED RESPONSE ACCESS ---
            text = resp.choices[0].message.content
        except Exception:
            return [criteria]
            
        parsed = safe_json_load(text)

        if parsed:
            return parsed.get("queries", [])

        return [criteria]

    # ------------------------------------------
    # LLM evaluates results → extracts companies
    # ------------------------------------------
    def llm_extract_companies(self, criteria, results):
        prompt = f"""
You are an intelligent GTM analyst.

User criteria:
"{criteria}"

Search results (JSON array):
{json.dumps(results, indent=2)}

Extract *real companies* by reading titles, URLs and descriptions.

Return strict JSON in this format:
{{
  "companies": [
    {{
      "name": "Company Name",
      "website": "https://...",
      "description": "short summary",
      "relevance_score": 0-100
    }}
  ],
  "overall_confidence": 0-1
}}
"""
        # --- 🛠️ UPDATED LLM CALL ---
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            # --- 🛠️ UPDATED RESPONSE ACCESS ---
            text = resp.choices[0].message.content
        except Exception:
            return {"companies": [], "overall_confidence": 0.2}

        parsed = safe_json_load(text)

        if parsed:
            return parsed

        return {"companies": [], "overall_confidence": 0.2}

    # ------------------------
    # MAIN AGENT LOGIC
    # ------------------------
    def run(self, criteria: str):
        print("🤖 [CompanyFinderAgent] Starting intelligent search...")

        iteration = 0
        all_companies = []

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔍 Iteration {iteration}/{self.max_iterations}")

            # Step 1: LLM generates smart queries
            queries = self.llm_generate_queries(criteria)
            print(f"   🧠 LLM queries: {queries}")

            # Step 2: Run Serper searches
            combined_results = []
            for q in queries[:3]:  # limit API usage
                hits = self.search_tool.search(q, num_results=5)
                combined_results.extend(hits)
                time.sleep(0.3)

            # Step 3: LLM extracts companies from noisy search results
            evaluation = self.llm_extract_companies(criteria, combined_results)
            companies = evaluation.get("companies", [])
            confidence = evaluation.get("overall_confidence", 0.3)

            print(f"   📊 Found {len(companies)} companies | Confidence = {confidence}")

            all_companies = companies

            # Stop early if results are good
            if confidence >= 0.65:
                print("   ✅ Confidence sufficient. Stopping iterations.")
                break

            print("   ⚠️ Low confidence → attempting improved search...")

        print("\n🎉 Final company list ready.")
        return all_companies