"""
LLM-Powered EmailWriterAgent

Capabilities:
- Writes personalized outreach emails based on:
  - Contact persona (name, role, seniority)
  - Company details
  - ResearchAgent insights
  - GTM offering
  - Chosen tone
- Ensures the output is clean, personalized, and non-hallucinated.
"""

import os
import json
# --- 🛠️ UPDATED IMPORT ---
from openai import OpenAI


def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                return json.loads(s[start:end+1])
        except:
            pass
    return None


class EmailWriterAgent:
    def __init__(self, openai_api_key=None, model="gpt-4o-mini"):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise RuntimeError("❌ OPENAI_API_KEY missing in environment")

        # --- 🛠️ UPDATED: Initialize the OpenAI client ---
        self.client = OpenAI(api_key=self.api_key)

    def _build_insights_text(self, insights):
        """Convert insights list (strings OR dicts) into bullet points."""
        if not insights:
            return "No specific insights available."

        bullets = ""

        for item in insights[:3]: # limit to 3
            # Case 1: Item is a dict (from ResearchAgent)
         if isinstance(item, dict):
                snip = (
                    item.get("snippet")
                    or item.get("text")
                    or item.get("pain_point")
                    or item.get("summary")
                    or str(item)
                )

            # Case 2: Item is already a string
        else:
                snip = str(item)

                bullets += f"- {snip}\n"

        return bullets.strip()


    def run(self, company: dict, contact: dict, research: dict,
            offering: str, tone: str = "Professional"):

        print(f"✉️  [EmailWriterAgent] Generating email for {contact['name']} at {company['name']}...")

        # Extract data
        company_name = company.get("name")
        website = company.get("website")
        contact_name = contact.get("name")
        role = contact.get("role")
        seniority = contact.get("seniority")
        # Ensure 'insights' key exists in research dict before accessing
        final_research = research.get("final", {}) 
        # Check if research structure is correct, using a safe fallback
        insights_list = final_research.get("pain_points") or research.get("insights", [])

        insights_text = self._build_insights_text(insights_list)

        # Tone prompt configurations
        tone_map = {
            "Professional": "Write a concise, polished B2B professional sales email.",
            "Casual": "Write a friendly, human, lightly casual outreach email.",
            "Cold": "Write a short, direct cold email with a strong value prop.",
            "Consultative": "Write a consultative email focused on diagnosing problems with thoughtful questions."
        }

        tone_instruction = tone_map.get(tone, tone_map["Professional"])

        # -----------------------------
        # LLM PROMPT FOR EMAIL CREATION
        # -----------------------------
        prompt = f"""
You are an expert SDR & email copywriter.

Write a personalized outreach email following these rules:

Recipient:
- Name: {contact_name}
- Role: {role}
- Seniority: {seniority}
- Company: {company_name}
- Website: {website}

Research Insights (Pain Points/Context):
{insights_text}

Offering:
"{offering}"

Tone:
{tone_instruction}

Rules:
- Email length: 120–180 words
- Must reference insights (if any) WITHOUT hallucinating facts
- Clear CTA: propose a 12–15 minute intro call
- Avoid generic fluff
- Keep paragraphs short and readable
- Start with the recipient’s name
- Do NOT invent metrics, results, or customer names
- Use concise, professional language

Return strict JSON:
{{
  "email_text": "...",
  "confidence": 0-1
}}
"""

        # --- 🛠️ UPDATED LLM CALL ---
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            # --- 🛠️ UPDATED RESPONSE ACCESS ---
            text = response.choices[0].message.content
        except Exception as e:
            # Handle potential API errors gracefully
            print(f"❌ LLM API Error: {e}")
            text = "Error generating email."
            parsed = {"email_text": text, "confidence": 0.0, "error": str(e)}
            return parsed


        parsed = safe_json_load(text)

        if parsed is None:
            # Fallback if LLM does not return strict JSON
            parsed = {"email_text": text, "confidence": 0.5}

        print("   ✅ Email generated")
        return parsed