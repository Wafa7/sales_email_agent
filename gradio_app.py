# gradio_app.py
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import asyncio
from graph_builder import build_graph
from state_model import OutreachState


async def run_pipeline(query, offering, email_style, max_companies):

    graph = build_graph()

    init_state = OutreachState(
        query=query,
        email_style=email_style,
        offering=offering,
        max_companies=max_companies,
    )

    # RUN THE GRAPH
    result = await graph.ainvoke(init_state)

    # -----------------------------
    # FORMAT OUTPUT FOR GRADIO UI
    # -----------------------------
    companies_list = ""
    for c in result.get("companies", []):
        companies_list += f"- **{c.name}** ({c.domain})\n"

    contacts_block = ""
    for c in result.get("companies", []):
        contacts_block += f"### {c.name}\n"
        if not c.contacts:
            contacts_block += "_No contacts found._\n\n"
            continue

        for con in c.contacts:
            contacts_block += (
                f"- **{con.name}** — {con.title}\n"
                f"  - Email: `{con.email}`\n"
                f"  - Confidence: {con.confidence}\n\n"
            )

        contacts_block += "\n"

    research_block = ""
    research = result.get("research", {})
    for comp_name, data in research.items():
        research_block += f"## {comp_name}\n"
        final = data.get("final", {})

        research_block += f"**Summary:** {final.get('summary', '')}\n\n"
        research_block += "**Pain Points:**\n"
        for p in final.get("pain_points", []):
            research_block += f"- {p}\n"
        research_block += "\n"

    emails_block = ""
    emails = result.get("emails", {})
    for comp_name, email in emails.items():
        emails_block += f"## {comp_name}\n\n"
        emails_block += f"```\n{email}\n```\n\n"

    errors_block = ""
    for e in result.get("errors", []):
        errors_block += f"- {e}\n"

    return companies_list, contacts_block, research_block, emails_block, errors_block


def launch_app():

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💼 AI SDR Outreach Agent\n### Multi-Agent Pipeline (Company → Contacts → Research → Email)")

        with gr.Row():
            query = gr.Textbox(
                label="What type of companies are you looking for?",
                placeholder="Example: AI workflow automation companies",
            )

        with gr.Row():
            offering = gr.Textbox(
                label="Your Offering (Pitch)",
                placeholder="Example: We help automate sales outreach using AI."
            )

        with gr.Row():
            email_style = gr.Dropdown(
                ["Professional", "Casual", "Cold", "Consultative"],
                value="Professional",
                label="Email Tone"
            )

            max_companies = gr.Slider(
                1, 10, value=5, step=1, label="Max Companies"
            )

        btn = gr.Button("Run Outreach Pipeline 🚀", variant="primary")

        gr.Markdown("## 📊 Results")

        companies_out = gr.Markdown()
        contacts_out = gr.Markdown()
        research_out = gr.Markdown()
        emails_out = gr.Markdown()
        errors_out = gr.Markdown()

        btn.click(
            fn=lambda q, off, tone, mc: asyncio.run(run_pipeline(q, off, tone, mc)),
            inputs=[query, offering, email_style, max_companies],
            outputs=[companies_out, contacts_out, research_out, emails_out, errors_out]
        )

    demo.launch()


if __name__ == "__main__":
    launch_app()