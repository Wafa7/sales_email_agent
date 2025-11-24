# run_app.py
import asyncio
from dotenv import load_dotenv
load_dotenv()

from graph_builder import build_graph
from state_model import OutreachState


async def main():
    graph = build_graph()

    init_state = OutreachState(
        query="AI workflow automation companies",
        email_style="Professional",
        offering="We help automate sales outreach using AI.",
        max_companies=5
    )

    result = await graph.ainvoke(input=init_state)

    print("\n=== RESULTS ===")

    print("\nCompanies:")
    for c in result.get("companies", []):
        print("-", c.name)

    print("\nEmails:")
    for company_name, email_text in result.get("emails", {}).items():
        print(f"\n--- {company_name} ---\n{email_text}")

    print("\nErrors:", result.get("errors", []))


if __name__ == "__main__":
    asyncio.run(main())

