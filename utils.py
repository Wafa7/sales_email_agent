import re
import tldextract
from urllib.parse import urlparse

# Common email patterns
EMAIL_PATTERNS = [
    "{first}.{last}@{domain}",
    "{first}{last}@{domain}",
    "{f}{last}@{domain}",
    "{first}{l}@{domain}",
    "{first}@{domain}",
    "{first}_{last}@{domain}",
]

def extract_domain(url_or_host: str) -> str:
    """Extract clean domain name from URL."""
    if not url_or_host:
        return ""
    if "http" not in url_or_host:
        url_or_host = "http://" + url_or_host
    try:
        parsed = urlparse(url_or_host)
        ext = tldextract.extract(parsed.netloc)
        if ext.domain:
            return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        return parsed.netloc
    except Exception:
        return url_or_host

def generate_email_candidates(first: str, last: str, domain: str) -> list:
    """Generate multiple possible emails based on common patterns."""
    first = (first or "").lower()
    last = (last or "").lower()
    f = first[:1]
    l = last[:1]
    domain = domain.lower()

    candidates = []
    for pattern in EMAIL_PATTERNS:
        email = pattern.format(first=first, last=last, f=f, l=l, domain=domain)
        email = re.sub(r"\s+", "", email)
        candidates.append(email)

    return candidates
