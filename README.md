# LLM Brand Visibility Audit
Analyzes brand perception in LLM responses, with competitor benchmarking and citation analysis.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env`
3. Run: `uvicorn main:app --host 0.0.0.0 --port 10000`
4. Configure n8n workflow: `audit_workflow.json`

## Usage
- Endpoint: `/brand-visibility-audit`
- Input: `{"brand": "YourBrand", "industry": "SaaS", "queries": ["What is YourBrand?"], "desired_description": "YourBrand is a UK-based innovator"}`
- Output: JSON with perception, competitors, citations, and framing suggestions.
