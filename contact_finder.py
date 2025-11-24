"""
LLM-Powered ContactFinderAgent

Capabilities:
- Uses LLM to identify best contact roles for outreach
- Uses LLM to generate realistic contact personas (name + role)
- Infers email patterns based on domain
- Generates probable email addresses
- LLM evaluates confidence, relevance & seniority
- Supports follow-up iterations (optional)
"""

import os
import json
import time
from openai import OpenAI
from utils import extract_domain, generate_email_candidates


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


class ContactFinderAgent:
    def __init__(self,
                 openai_api_key=None,
                 model="gpt-4o-mini",
                 max_personas=3,
                 max_iters=1):

        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_personas = max_personas
        self.max_iters = max_iters

        if not self.api_key:
            raise RuntimeError("❌ OPENAI_API_KEY missing in .env")

        self.client = OpenAI(api_key=self.api_key)

    # ---------------------------------------------------------
    # Step 1: LLM generates persona profiles
    # ---------------------------------------------------------
    def llm_generate_personas(self, company_name, description, domain):

        prompt = f"""
You are an expert SDR contact research agent.

Your responsibilities:
1. Identify the MOST relevant decision-makers for B2B outreach.
2. Generate realistic persona names (avoid repeating same name).
3. Ensure job titles match the company type and size.
4. Vary names and seniority levels.
5. Output STRICT JSON ONLY.

Company Name: {company_name}
Description: {description}
Domain: {domain}

Return STRICT JSON EXACTLY like this:

{{
  "contacts": [
    {{
      "name": "Realistic Full Name",
      "role": "Job Title",
      "seniority": "exec | senior | mid",
      "email_pattern_hint": "first.last | first | first_last | firstlast"
    }}
  ],
  "confidence": 0-1
}}

Rules:
- Generate **{self.max_personas}** unique personas.
- Do NOT reuse the same first name across personas.
- For SaaS / AI workflow / automation companies, preferred roles include:
    • Head of Sales
    • Director of Sales
    • VP Sales
    • Chief Revenue Officer
    • Growth Lead
    • Partnerships Lead
    • Product Lead
    • CTO (if outbound is technical)
- Names must be realistic, diverse, and fit US/EU naming patterns.
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            text = resp.choices[0].message.content
        except Exception:
            return {"contacts": [], "confidence": 0.0}

        parsed = safe_json_load(text)
        return parsed or {"contacts": [], "confidence": 0.3}

    # ---------------------------------------------------------
    # Step 2: Email inference
    # ---------------------------------------------------------
    def infer_email(self, name, domain, pattern_hint=None):
        parts = name.lower().split()
        first = parts[0] if parts else ""
        last = parts[-1] if len(parts) > 1 else ""

        candidates = generate_email_candidates(first, last, domain)

        if pattern_hint == "first.last":
            return f"{first}.{last}@{domain}"
        if pattern_hint == "first":
            return f"{first}@{domain}"
        if pattern_hint == "first_last":
            return f"{first}_{last}@{domain}"

        return candidates[0] if candidates else f"{first}@{domain}"

    # ---------------------------------------------------------
    # MAIN RUN LOGIC
    # ---------------------------------------------------------
    def run(self, company: dict):
        print(f"🤖 [ContactFinderAgent] Finding contacts for {company['name']}...")

        domain = extract_domain(company.get("website", "")) or ""
        description = company.get("description", "")

        iteration = 0
        final_contacts = []

        while iteration < self.max_iters:
            iteration += 1
            print(f"   🔍 Iteration {iteration}/{self.max_iters}")

            output = self.llm_generate_personas(
                company_name=company["name"],
                description=description,
                domain=domain
            )

            contacts = output.get("contacts", [])
            confidence = output.get("confidence", 0.3)

            processed = []
            for c in contacts:
                name = c.get("name")
                role = c.get("role")
                seniority = c.get("seniority")
                pattern_hint = c.get("email_pattern_hint")

                email = self.infer_email(name, domain, pattern_hint)

                processed.append({
                    "name": name,
                    "role": role,
                    "seniority": seniority,
                    "domain": domain,
                    "email": email,
                    "source": "AI-generated",
                    "confidence": confidence
                })

            final_contacts = processed

            if confidence >= 0.6:
                print("   ✅ Confidence high enough. Stopping.")
                break

            print("   ⚠️ Confidence low. Retrying...")
            time.sleep(0.5)

        print(f"   🎯 Found {len(final_contacts)} contacts.")
        return final_contacts
