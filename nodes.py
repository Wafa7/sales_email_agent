# nodes.py
import logging
from state_model import OutreachState, Company, Contact

# Import agent files — FIXED to use local module paths
from company_finder import CompanyFinderAgent
from contact_finder import ContactFinderAgent
from research_agent import ResearchAgent
from email_writer import EmailWriterAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NODE 1 — Discover companies
# ---------------------------------------------------------
def discover_companies_node(state: OutreachState, config, runtime):
    print("NODE RUN: discover_companies")
    try:
        agent = CompanyFinderAgent()
        raw = agent.run(state.query)

        # limit to max companies
        raw = raw[: state.max_companies]

        companies = [
            Company(
                name=c.get("name"),
                domain=c.get("website"),
                score=c.get("relevance_score"),
                metadata={"description": c.get("description", "")}
            )
            for c in raw
        ]

        # DIRECT state update
        return {"companies": companies}

    except Exception as e:
        logger.exception("discover_companies_node failed")
        return {"errors": state.errors + [str(e)]}


# ---------------------------------------------------------
# NODE 2 — Find contacts
# ---------------------------------------------------------
def find_contacts_node(state: OutreachState, config, runtime):
    print("NODE RUN: find_contacts")
    try:
        agent = ContactFinderAgent()
        updated_companies = []

        for comp in state.companies:

            company_dict = {
                "name": comp.name,
                "website": comp.domain,
                "description": comp.metadata.get("description", "")
            }

            found = agent.run(company_dict)

            mapped_contacts = [
                Contact(
                    name=c.get("name"),
                    title=c.get("role"),
                    email=c.get("email"),
                    confidence=c.get("confidence"),
                    source=c.get("source", "AI")
                )
                for c in found
            ]

            comp.contacts = mapped_contacts
            updated_companies.append(comp)

        # DIRECT state update
        return {"companies": updated_companies}

    except Exception as e:
        logger.exception("find_contacts_node failed")
        return {"errors": state.errors + [str(e)]}


# ---------------------------------------------------------
# NODE 3 — Collect research
# ---------------------------------------------------------
def collect_research_node(state: OutreachState, config, runtime):
    print("NODE RUN: collect_research")
    try:
        agent = ResearchAgent()
        research_map = {}

        for comp in state.companies:

            company_dict = {
                "name": comp.name,
                "website": comp.domain,
                "description": comp.metadata.get("description", "")
            }

            insights = agent.run(
                company=company_dict,
                offering=state.offering or "",
                tone=state.email_style or "Professional"
            )

            research_map[comp.name] = insights

        # DIRECT state update
        return {"research": research_map}

    except Exception as e:
        logger.exception("collect_research_node failed")
        return {"errors": state.errors + [str(e)]}


# ---------------------------------------------------------
# NODE 4 — Generate emails
# ---------------------------------------------------------
def draft_emails_node(state: OutreachState, config, runtime):
    print("NODE RUN: draft_emails")
    try:
        agent = EmailWriterAgent()
        emails = {}

        for comp in state.companies:

            if not comp.contacts:
                emails[comp.name] = "No valid contact found."
                continue

            contact = comp.contacts[0]  # First contact only

            company_dict = {
                "name": comp.name,
                "website": comp.domain
            }

            contact_dict = {
                "name": contact.name,
                "role": contact.title,
                "seniority": "senior"
            }

            research_data = state.research.get(comp.name, {})

            email_result = agent.run(
                company=company_dict,
                contact=contact_dict,
                research=research_data,
                offering=state.offering or "",
                tone=state.email_style
            )

            email_text = email_result.get("email_text") or ""

            emails[comp.name] = email_text

        # DIRECT state update
        return {"emails": emails}

    except Exception as e:
        logger.exception("draft_emails_node failed")
        return {"errors": state.errors + [str(e)]}
