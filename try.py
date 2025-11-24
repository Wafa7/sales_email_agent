from dotenv import load_dotenv

# This function searches for .env file and loads variables into environment
load_dotenv()
from email_writer import EmailWriterAgent

agent = EmailWriterAgent()

company = {"name": "Gong", "website": "https://gong.io"}
contact = {
    "name": "Emily Chen",
    "role": "Chief Revenue Officer",
    "seniority": "exec"
}
research = {
    "insights": [
        {"source": "https://gong.io", "snippet": "Gong focuses on AI-powered revenue intelligence."}
    ]
}

offering = "We help revenue teams automate sales workflows using AI."

email = agent.run(company, contact, research, offering, tone="Professional")

print("\n--- EMAIL ---\n")
print(email["email_text"])
print("\nConfidence:", email["confidence"])
