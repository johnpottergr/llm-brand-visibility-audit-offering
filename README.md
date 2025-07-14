# LLM Brand Visibility Audit
Analyzes brand perception, competitor benchmarking, and citations in LLM responses.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env`
3. Run API: `uvicorn main:app --host 0.0.0.0 --port 10000`
4. Run visualization: `python visualization.py`
5. Configure n8n: Import `audit_workflow.json`

## Usage
- Endpoint: `/brand-visibility-audit`
- Input: `{"brand": "YourBrand", "industry": "SaaS", "queries": ["What is YourBrand?"], "desired_description": "YourBrand is a UK-based innovator", "competitors": ["CompA", "CompB"]}`
- Outputs:
  - JSON: Perception, competitors, citations
  - CSV: `brand_audit_output.csv`
  - Visualization: `brand_visualization.html`
