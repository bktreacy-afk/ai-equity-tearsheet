# AI Equity Research Tearsheet

A Bloomberg-style equity research tool that generates professional tearsheets for any stock ticker. Built with Python and Flask, powered by the Claude AI API.

## What it does

- Fetches live financial data (price, P/E, EPS, market cap, revenue, 52-week range) via yfinance
- Pulls recent investor-relevant news headlines via NewsAPI
- Generates a 3-sentence AI analyst summary using the Claude API
- Renders a professional dark-themed HTML tearsheet with interactive Chart.js charts including YTD performance vs S&P 500 and quarterly revenue/earnings
- Served via a Flask web app with a search interface at localhost:5000

## Tech stack

- Python, Flask
- yfinance, NewsAPI, Anthropic Claude API
- Chart.js, HTML/CSS

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

4. Run: `python financials.py`
5. Open http://localhost:5000
