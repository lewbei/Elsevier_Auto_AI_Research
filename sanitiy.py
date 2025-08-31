import os, requests
from dotenv import load_dotenv; load_dotenv()
k = os.environ["ELSEVIER_KEY"]

u = "https://api.elsevier.com/content/search/author"
q = "AUTHFIRST(John) AND AUTHLASTNAME(Kitchin) AND SUBJAREA(COMP)"
r = requests.get(u, params={"query": q, "count": 5, "view": "STANDARD"},
                 headers={"X-ELS-APIKey": k, "Accept": "application/json"}, timeout=30)

print("status:", r.status_code, "| X-ELS-Status:", r.headers.get("X-ELS-Status"))
print(r.text[:500])
