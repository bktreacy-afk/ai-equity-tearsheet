import subprocess
import sys
import os
import webbrowser
import json
import time
import requests
import threading
from datetime import datetime
from datetime import date
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Simple in-memory cache
_cache = {}
CACHE_TTL = 300  # 5 minutes

def get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None

def set_cached(key, data):
    _cache[key] = (data, time.time())

# Loading state for async page loads
_loading = {}
_results = {}
app.template_folder = os.path.dirname(os.path.abspath(__file__))

def install_yfinance():
    """Install yfinance if not already installed"""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        print("yfinance not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf
        print("yfinance installed successfully!")
        return yf

def install_newsapi():
    """Install newsapi-python if not already installed"""
    try:
        from newsapi import NewsApiClient
        return NewsApiClient
    except ImportError:
        print("newsapi-python not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "newsapi-python"])
        from newsapi import NewsApiClient
        print("newsapi-python installed successfully!")
        return NewsApiClient

def install_anthropic():
    """Install anthropic library if not already installed"""
    try:
        import anthropic
        return anthropic
    except ImportError:
        print("anthropic library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
        import anthropic
        print("anthropic library installed successfully!")
        return anthropic

def get_newsapi_key():
    """Read NewsAPI key from api_keys.txt on Desktop"""
    api_key_path = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "ANTHROPIC_API_KEY", "api_keys.txt")
    
    try:
        with open(api_key_path, 'r') as f:
            for line in f:
                if line.startswith("NEWS_API_KEY"):
                    # Extract the key after the equals sign
                    key = line.split("=", 1)[1].strip()
                    return key
        raise ValueError("NEWS_API_KEY not found in api_keys.txt")
    except FileNotFoundError:
        raise FileNotFoundError(f"api_keys.txt not found at {api_key_path}")

def fetch_news_headlines(company_name, ticker=None):
    """Fetch recent news via Yahoo Finance RSS feed"""
    import xml.etree.ElementTree as ET
    headlines = []
    try:
        if not ticker:
            return []
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(rss_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        ns = {'media': 'http://search.yahoo.com/mrss/'}
        items = root.findall('.//item')
        seen_urls = set()
        for item in items[:10]:
            title = item.findtext('title', '').strip()
            url = item.findtext('link', '').strip()
            pub_date = item.findtext('pubDate', '').strip()
            source = item.findtext('source', 'Yahoo Finance').strip()
            if not title or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            headlines.append({
                'title': title,
                'published_at': pub_date,
                'url': url,
                'source': source if source else 'Yahoo Finance'
            })
            if len(headlines) >= 5:
                break
    except Exception as e:
        print(f"Error fetching news: {e}")
    return headlines

def get_ai_analyst_summary(company_name, financial_data, headlines):
    """Send financial data and news to Claude API for analyst summary"""
    try:
        anthropic = install_anthropic()
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Prepare the prompt with financial data and news
        news_text = "\n".join([f"{i+1}. {h['title']}" for i, h in enumerate(headlines)]) if headlines else "No recent news available."
        
        prompt = f"""Write a 3-sentence equity research summary for {company_name}. Cover: (1) current valuation vs peers, (2) one key catalyst or risk, (3) overall positioning. Be concise and direct, no markdown.

Financial Data:
- Current Stock Price: ${financial_data.get('current_price', 'N/A')}
- P/E Ratio: {financial_data.get('pe_ratio', 'N/A')}
- Market Cap: ${financial_data.get('market_cap', 'N/A')}
- Total Revenue: ${financial_data.get('total_revenue', 'N/A')}
- EPS (Earnings Per Share): ${financial_data.get('eps', 'N/A')}
- 52-Week High: ${financial_data.get('week_52_high', 'N/A')}
- 52-Week Low: ${financial_data.get('week_52_low', 'N/A')}

Recent News Headlines:
{news_text}"""
        
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        print(f"Error generating AI analyst summary: {e}")
        return "Unable to generate AI analyst summary at this time."

def fetch_market_indices():
    """Fetch market indices data (S&P 500, NASDAQ)"""
    try:
        yf = install_yfinance()
        indices = {}
        
        # S&P 500
        try:
            sp500 = yf.Ticker("^GSPC")
            sp500_info = sp500.info
            indices['sp500'] = {
                'price': sp500_info.get('currentPrice', sp500_info.get('regularMarketPrice', 'N/A')),
                'change': sp500_info.get('regularMarketChangePercent', 'N/A')
            }
        except:
            indices['sp500'] = {'price': 'N/A', 'change': 'N/A'}
        
        # NASDAQ
        try:
            nasdaq = yf.Ticker("^IXIC")
            nasdaq_info = nasdaq.info
            indices['nasdaq'] = {
                'price': nasdaq_info.get('currentPrice', nasdaq_info.get('regularMarketPrice', 'N/A')),
                'change': nasdaq_info.get('regularMarketChangePercent', 'N/A')
            }
        except:
            indices['nasdaq'] = {'price': 'N/A', 'change': 'N/A'}
        
        return indices
        
    except Exception as e:
        print(f"Error fetching market indices: {e}")
        return {
            'sp500': {'price': 'N/A', 'change': 'N/A'},
            'nasdaq': {'price': 'N/A', 'change': 'N/A'}
        }

def fetch_ytd_performance(ticker):
    """Fetch YTD performance data for stock and S&P 500"""
    try:
        yf = install_yfinance()
        current_year = date.today().year
        start_date = f"{current_year}-01-01"
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date)
        # Fetch S&P 500 data
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(start=start_date)
        
        if stock_data.empty or sp500_data.empty:
            return None, None
        
        # Index to base 100 (standard Bloomberg-style comparison)
        stock_first_close = stock_data['Close'].iloc[0]
        stock_indexed = [round((price / stock_first_close) * 100, 2) for price in stock_data['Close']]
        
        sp500_first_close = sp500_data['Close'].iloc[0]
        sp500_indexed = [round((price / sp500_first_close) * 100, 2) for price in sp500_data['Close']]
        
        # Prepare data for chart - monthly labels only
        # Only label the first trading day of each month
        dates = []
        last_month = None
        for dt in stock_data.index:
            if dt.month != last_month:
                dates.append(dt.strftime('%b'))
                last_month = dt.month
            else:
                dates.append('')
        
        return {
            'dates': dates,
            'stock_indexed': stock_indexed,
            'sp500_indexed': sp500_indexed,
            'actual_dates': stock_data.index.strftime('%b %d, %Y').tolist(),
            'stock_prices': [round(float(p), 2) for p in stock_data['Close'].tolist()],
            'sp500_levels': [round(float(p), 2) for p in sp500_data['Close'].tolist()]
        }, {
            'stock_ytd': stock_indexed[-1] if stock_indexed else 100,
            'sp500_ytd': sp500_indexed[-1] if sp500_indexed else 100
        }
        
    except Exception as e:
        print(f"Error fetching YTD performance: {e}")
        return None, None

def fetch_quarterly_financials(ticker):
    """Fetch quarterly revenue and earnings data"""
    try:
        yf = install_yfinance()
        stock = yf.Ticker(ticker)
        
        quarterly_data = {
            'revenue': [],
            'net_income': [],
            'quarters': []
        }
        
        # Fetch quarterly financials for revenue and net income
        try:
            qf = stock.quarterly_financials
            if qf is not None and not qf.empty:
                quarters = [str(col.date()) for col in qf.columns[:4]]
                revenues = [(qf.loc['Total Revenue', col] / 1e9) if 'Total Revenue' in qf.index else 0 for col in qf.columns[:4]]
                net_incomes = [(qf.loc['Net Income', col] / 1e9) if 'Net Income' in qf.index else 0 for col in qf.columns[:4]]
                quarters.reverse()
                revenues.reverse()
                net_incomes.reverse()
                quarterly_data['quarters'] = quarters
                quarterly_data['revenue'] = revenues
                quarterly_data['net_income'] = net_incomes
        except Exception as e:
            print(f"Error fetching quarterly financials: {e}")
        
        return quarterly_data
        
    except Exception as e:
        print(f"Error fetching quarterly financials: {e}")
        return {'revenue': [], 'net_income': [], 'quarters': []}

def fetch_analyst_data(ticker):
    """Fetch analyst outlook data"""
    try:
        yf = install_yfinance()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try multiple fallback fields for short interest
        short_interest = (info.get('shortPercentOfFloat') 
                        or info.get('shortRatio') 
                        or info.get('sharesPercentSharesOut'))
        
        # Try multiple fallback fields for institutional ownership
        inst_ownership = (info.get('institutionsPercentHeld') 
                        or info.get('heldPercentInstitutions'))
        
        # If still None, try fetching from institutional_holders
        if inst_ownership is None or inst_ownership == 'N/A':
            try:
                inst_holders = stock.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    inst_ownership = inst_holders['Value'].sum() / info.get('marketCap', 1) * 100
            except:
                pass
        
        # Try fetching earnings date from calendar
        next_earnings = info.get('nextEarningsDate', 'N/A')
        if next_earnings == 'N/A':
            try:
                cal = stock.calendar
                if cal is not None and 'Earnings Date' in cal.index:
                    earnings_date = cal.loc['Earnings Date'].iloc[0]
                    next_earnings = earnings_date.strftime('%b %d, %Y') if hasattr(earnings_date, 'strftime') else str(earnings_date)
            except:
                next_earnings = 'N/A'
        
        # Calculate buy/hold/sell distribution based on consensus
        num_analysts = info.get('numberOfAnalystOpinions', 0) or 0
        rec = info.get('recommendationKey', '').lower()
        
        if num_analysts > 0 and rec != 'n/a':
            if 'strong buy' in rec:
                buy_count = int(num_analysts * 0.7)
                hold_count = int(num_analysts * 0.2)
                sell_count = int(num_analysts * 0.1)
            elif 'buy' in rec:
                buy_count = int(num_analysts * 0.5)
                hold_count = int(num_analysts * 0.3)
                sell_count = int(num_analysts * 0.2)
            elif 'hold' in rec:
                buy_count = int(num_analysts * 0.2)
                hold_count = int(num_analysts * 0.6)
                sell_count = int(num_analysts * 0.2)
            elif 'sell' in rec:
                buy_count = int(num_analysts * 0.1)
                hold_count = int(num_analysts * 0.2)
                sell_count = int(num_analysts * 0.7)
            else:
                buy_count = hold_count = sell_count = num_analysts // 3
        else:
            buy_count = hold_count = sell_count = 0
        
        analyst_data = {
            'next_earnings_date': next_earnings,
            'recommendation': info.get('recommendationKey', 'N/A'),
            'target_price': info.get('targetMeanPrice', 'N/A'),
            'num_analysts': num_analysts if num_analysts > 0 else 'N/A',
            'inst_ownership': inst_ownership if inst_ownership else 'N/A',
            'short_interest': short_interest if short_interest else 'N/A',
            'buy_count': buy_count,
            'hold_count': hold_count,
            'sell_count': sell_count
        }
        
        return analyst_data
        
    except Exception as e:
        print(f"Error fetching analyst data: {e}")
        return {
            'next_earnings_date': 'N/A',
            'recommendation': 'N/A',
            'target_price': 'N/A',
            'num_analysts': 'N/A',
            'inst_ownership': 'N/A',
            'short_interest': 'N/A',
            'buy_count': 0,
            'hold_count': 0,
            'sell_count': 0
        }

def format_large_number(value):
    """Format large numbers with B/T suffixes"""
    if value == 'N/A' or value is None:
        return 'N/A'
    try:
        value = float(value)
        if value >= 1e12:
            return f"{value/1e12:.2f}T"
        elif value >= 1e9:
            return f"{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{value/1e6:.2f}M"
        else:
            return f"{value:,.2f}"
    except:
        return 'N/A'

def format_percentage(value):
    """Format percentage with 1 decimal place"""
    if value == 'N/A' or value is None:
        return 'N/A'
    try:
        return f"{float(value):.1f}%"
    except:
        return 'N/A'

def date_to_quarter(date_str):
    """Convert date string to quarter label (e.g., Q1 '26)"""
    try:
        from datetime import datetime
        d = datetime.strptime(date_str[:10], '%Y-%m-%d')
        q = (d.month - 1) // 3 + 1
        return f"Q{q} '{str(d.year)[2:]}"
    except:
        return date_str

def calculate_rsi(ticker, period=14):
    """Calculate RSI from ticker history"""
    try:
        yf = install_yfinance()
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo')
        if hist.empty or len(hist) < period:
            return None
        
        closes = hist['Close'].values
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return None

def format_ratio(value):
    """Format ratio with 2 decimal places and x suffix"""
    if value == 'N/A' or value is None:
        return 'N/A'
    try:
        return f"{float(value):.2f}x"
    except:
        return 'N/A'

def generate_html_tearsheet(ticker, company_name, financial_data, headlines, ai_summary, market_indices, ytd_data, quarterly_data, analyst_data, enhanced_stats, peers_html):
    """Generate professional Bloomberg-style HTML tearsheet and save to Desktop"""
    
    # Format news dates
    def format_date(pub_date):
        try:
            dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            return dt.strftime('%B %d, %Y')
        except:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub_date)
                return dt.strftime('%B %d, %Y')
            except:
                return pub_date[:16] if pub_date else ''
    
    # Format change percentage with color
    def format_change(change):
        if change == 'N/A' or change is None:
            return 'N/A', '#888888'
        try:
            change_val = float(change)
            color = '#3fb950' if change_val >= 0 else '#f85149'
            return f"{change_val:+.2f}%", color
        except:
            return 'N/A', '#888888'
    
    # Generate news HTML
    news_html = ""
    if headlines:
        for i, headline in enumerate(headlines, 1):
            formatted_date = format_date(headline['published_at'])
            source = headline.get('source', 'Unknown')
            news_html += f"""
            <div class="news-item">
                <div class="news-number">{i}</div>
                <div class="news-content">
                    <a href="{headline['url']}" target="_blank" class="news-title">{headline['title']}</a>
                    <div class="news-date">{formatted_date} · {source}</div>
                </div>
            </div>
            """
    else:
        news_html = "<p>No recent news headlines found for this company.</p>"
    
    # Strip markdown from AI summary
    def strip_markdown(text):
        if not text:
            return ""
        # Remove lines starting with #
        lines = [line for line in text.split('\n') if not line.strip().startswith('#')]
        text = '\n'.join(lines)
        # Replace **text** with <strong>text</strong>
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # Split on double newlines and wrap in <p> tags
        paragraphs = text.split('\n\n')
        return ''.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())
    
    cleaned_ai_summary = strip_markdown(ai_summary)
    
    # Prepare chart data as JavaScript variables
    chart_dates_json = json.dumps(ytd_data['dates']) if ytd_data and ytd_data['dates'] else '[]'
    stock_indexed_json = json.dumps(ytd_data['stock_indexed']) if ytd_data and ytd_data['stock_indexed'] else '[]'
    sp500_indexed_json = json.dumps(ytd_data['sp500_indexed']) if ytd_data and ytd_data['sp500_indexed'] else '[]'
    actual_dates_json = json.dumps(ytd_data.get('actual_dates', [])) if ytd_data else '[]'
    stock_prices_json = json.dumps(ytd_data.get('stock_prices', [])) if ytd_data else '[]'
    sp500_levels_json = json.dumps(ytd_data.get('sp500_levels', [])) if ytd_data else '[]'
    
    # Format quarters as Q labels
    quarters_formatted = [date_to_quarter(q) for q in quarterly_data['quarters']] if quarterly_data['quarters'] else []
    quarters_json = json.dumps(quarters_formatted)
    revenue_json = json.dumps(quarterly_data['revenue']) if quarterly_data['revenue'] else '[]'
    net_income_json = json.dumps(quarterly_data['net_income']) if quarterly_data['net_income'] else '[]'
    
    # Market indices HTML
    sp500_change, sp500_color = format_change(market_indices['sp500']['change'])
    nasdaq_change, nasdaq_color = format_change(market_indices['nasdaq']['change'])
    
    # Enhanced stats
    sector = enhanced_stats.get('sector', 'N/A')
    industry = enhanced_stats.get('industry', 'N/A')
    current_price = financial_data.get('current_price', 'N/A')
    daily_change = enhanced_stats.get('daily_change', 'N/A')
    daily_change_pct, daily_color = format_change(daily_change)
    
    # Extract domain for company logo
    website = enhanced_stats.get('website', 'N/A')
    domain = ''
    if website and website != 'N/A':
        try:
            from urllib.parse import urlparse
            parsed = urlparse(website)
            domain = parsed.netloc.replace('www.', '')
        except:
            domain = ''
    
    # Pre-compute recommendation color (fixes f-string ternary bug)
    rec_raw = analyst_data['recommendation'].lower() if analyst_data['recommendation'] != 'N/A' else ''
    if 'strong' in rec_raw and 'buy' in rec_raw:
        rec_color = '#2ea043'
        rec_text_color = '#ffffff'
    elif 'buy' in rec_raw:
        rec_color = '#3fb950'
        rec_text_color = '#ffffff'
    elif 'hold' in rec_raw:
        rec_color = '#d29922'
        rec_text_color = '#ffffff'
    elif 'sell' in rec_raw:
        rec_color = '#f85149'
        rec_text_color = '#ffffff'
    else:
        rec_color = '#888888'
        rec_text_color = '#ffffff'
    rec_display = analyst_data['recommendation'].replace('_', ' ').title() if analyst_data['recommendation'] != 'N/A' else 'N/A'

    # Pre-compute RSI display (fixes f-string ternary bug)
    rsi_val = enhanced_stats.get('rsi')
    if rsi_val:
        if rsi_val > 70:
            rsi_text = f"{rsi_val:.1f} Overbought"
            rsi_class = "bearish"
        elif rsi_val < 30:
            rsi_text = f"{rsi_val:.1f} Oversold"
            rsi_class = "bullish"
        else:
            rsi_text = f"{rsi_val:.1f} Neutral"
            rsi_class = ""
    else:
        rsi_text = "N/A"
        rsi_class = ""

    # Pre-compute MA signals
    price_val = float(current_price) if current_price != 'N/A' else 0
    fifty_ma = enhanced_stats.get('fifty_day_ma', 0) or 0
    two_hundred_ma = enhanced_stats.get('two_hundred_day_ma', 0) or 0
    fifty_signal = "Above ↑" if price_val > fifty_ma else "Below ↓"
    fifty_class = "bullish" if price_val > fifty_ma else "bearish"
    two_hundred_signal = "Above ↑" if price_val > two_hundred_ma else "Below ↓"
    two_hundred_class = "bullish" if price_val > two_hundred_ma else "bearish"

    # Pre-compute short interest and inst ownership display
    si = analyst_data['short_interest']
    si_display = format_percentage(float(si) * 100) if si != 'N/A' and si is not None else 'N/A'
    inst = analyst_data['inst_ownership']
    inst_display = format_percentage(float(inst) * 100) if inst != 'N/A' and inst is not None else 'N/A'

    # Compute sector info line for header
    sector_val = enhanced_stats.get('sector', '')
    industry_val = enhanced_stats.get('industry', '')
    market_cap_val = financial_data.get('market_cap', 0)
    
    # Format market cap
    if market_cap_val and market_cap_val != 'N/A':
        try:
            mc = float(market_cap_val)
            if mc >= 1e12:
                mc_formatted = f"{mc / 1e12:.1f}B"
            elif mc >= 1e9:
                mc_formatted = f"{mc / 1e9:.0f}B"
            else:
                mc_formatted = f"{mc / 1e6:.0f}M"
        except:
            mc_formatted = ''
    else:
        mc_formatted = ''
    
    # Build sector line with · separator
    sector_parts = []
    if sector_val and sector_val != 'N/A':
        sector_parts.append(sector_val)
    if industry_val and industry_val != 'N/A':
        sector_parts.append(industry_val)
    if mc_formatted:
        sector_parts.append(mc_formatted)
    
    sector_line = ' · '.join(sector_parts) if sector_parts else ''

    # Pre-compute quarterly financials data
    quarterly_label = ''
    quarterly_revenue = ''
    quarterly_growth = ''
    quarterly_growth_color = ''
    
    if quarterly_data and quarterly_data.get('quarters') and len(quarterly_data['quarters']) > 0:
        try:
            from datetime import datetime
            most_recent_q = quarterly_data['quarters'][0]
            dt = datetime.strptime(most_recent_q, '%Y-%m-%d')
            quarterly_label = f"{dt.year} Q{(dt.month - 1) // 3 + 1}"
            
            if quarterly_data.get('revenue') and len(quarterly_data['revenue']) > 0:
                rev = quarterly_data['revenue'][0]
                quarterly_revenue = f"{rev:.2f}B" if rev >= 1 else f"{rev * 1000:.0f}M"
                
                # Calculate YoY growth (compare with same quarter from previous year)
                if len(quarterly_data['revenue']) >= 4:
                    prev_year_rev = quarterly_data['revenue'][3] if len(quarterly_data['revenue']) > 3 else 0
                    if prev_year_rev > 0:
                        growth = ((rev - prev_year_rev) / prev_year_rev) * 100
                        quarterly_growth = f"{growth:+.1f}% Y/Y Revenue"
                        quarterly_growth_color = '#3fb950' if growth >= 0 else '#f85149'
        except:
            pass

    # Pre-compute earnings data from yfinance earnings history
    yf = install_yfinance()
    eps_beat_val = 'N/A'
    eps_beat_color = '#8b949e'
    try:
        ticker_obj = yf.Ticker(ticker)
        eh = ticker_obj.earnings_history
        if eh is not None and not eh.empty:
            latest = eh.iloc[-1]
            eps_surprise_pct = latest.get('surprisePercent', None)
            if eps_surprise_pct is not None:
                eps_beat_val = f"+{eps_surprise_pct*100:.2f}%" if eps_surprise_pct >= 0 else f"{eps_surprise_pct*100:.2f}%"
                eps_beat_color = "#3fb950" if eps_surprise_pct >= 0 else "#f85149"
            else:
                eps_beat_val = "N/A"
                eps_beat_color = "#8b949e"
        else:
            eps_beat_val = "N/A"
            eps_beat_color = "#8b949e"
    except:
        eps_beat_val = "N/A"
        eps_beat_color = "#8b949e"

    # Revenue beat — calculate from quarterly_financials
    rev_beat_val = 'N/A'
    rev_beat_color = '#8b949e'
    try:
        ticker_obj_rev = yf.Ticker(ticker)
        q_fin = ticker_obj_rev.quarterly_financials
        if q_fin is not None and not q_fin.empty and 'Total Revenue' in q_fin.index:
            rev_row = q_fin.loc['Total Revenue'].dropna()
            if len(rev_row) >= 5:  # need current Q and same Q last year
                current_q = rev_row.iloc[0]
                year_ago_q = rev_row.iloc[4]
                rev_yoy = (current_q - year_ago_q) / abs(year_ago_q)
                rev_beat_val = f"+{rev_yoy*100:.2f}%" if rev_yoy >= 0 else f"{rev_yoy*100:.2f}%"
                rev_beat_color = "#3fb950" if rev_yoy >= 0 else "#f85149"
            elif len(rev_row) >= 2:
                current_q = rev_row.iloc[0]
                prev_q = rev_row.iloc[1]
                rev_qoq = (current_q - prev_q) / abs(prev_q)
                rev_beat_val = f"+{rev_qoq*100:.2f}% QoQ" if rev_qoq >= 0 else f"{rev_qoq*100:.2f}% QoQ"
                rev_beat_color = "#3fb950" if rev_qoq >= 0 else "#f85149"
            else:
                rev_beat_val = "N/A"
                rev_beat_color = "#8b949e"
        else:
            rev_beat_val = "N/A"
            rev_beat_color = "#8b949e"
    except:
        rev_beat_val = "N/A"
        rev_beat_color = "#8b949e"

    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name} ({ticker.upper()}) - Bloomberg Terminal Tearsheet</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background-color: #0d1117;
            color: #e6edf3;
            line-height: 1.4;
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }}
        
        .header {{
            background-color: #161b22;
            padding: 14px 24px;
            border-bottom: 2px solid #58a6ff;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .header-left {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            flex-shrink: 0;
        }}
        
        .company-name {{
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
        }}
        
        .ticker-price-row {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .sector-info-line {{
            font-size: 10px;
            color: #8b949e;
            margin-top: 3px;
        }}
        
        .ticker-box {{
            background-color: #58a6ff;
            color: #000;
            padding: 3px 8px;
            font-weight: bold;
            font-size: 13px;
            border-radius: 4px;
        }}
        
        .price-large {{
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
        }}
        
        .change-pill {{
            padding: 2px 8px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 12px;
            color: #000;
        }}
        
        .header-peers {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 0 20px;
        }}
        
        .peers-label {{
            font-size: 10px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .peers-row {{
            display: flex;
            flex-direction: row;
            gap: 12px;
        }}
        
        .peer-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 8px 12px;
            cursor: pointer;
            text-decoration: none;
            transition: border-color 0.2s;
            min-width: 130px;
            max-width: 160px;
        }}
        
        .peer-card:hover {{
            border-color: #58a6ff;
        }}
        
        .peer-company {{
            font-size: 10px;
            color: #8b949e;
            margin-bottom: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .peer-info-row {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .peer-ticker {{
            background: #1f6feb;
            color: white;
            font-size: 10px;
            font-weight: 700;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        
        .peer-price {{
            color: #e6edf3;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .peer-change {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
        }}
        
        .peer-change.positive {{
            background: #1a4731;
            color: #3fb950;
        }}
        
        .peer-change.negative {{
            background: #4b1113;
            color: #f85149;
        }}
        
        .header-right {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            flex-shrink: 0;
            min-width: 180px;
        }}
        
        .header-indices {{
            flex: 1;
            display: flex;
            gap: 12px;
            justify-content: space-evenly;
        }}
        
        .header-indices .index-card {{
            flex: 1;
        }}
        
        .header-search-row {{
            display: flex;
            gap: 6px;
        }}
        
        .header-search-row input {{
            flex: 1;
            padding: 7px 10px;
            font-size: 12px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #0d1117;
            color: #e6edf3;
            outline: none;
            text-transform: uppercase;
            width: 120px;
        }}
        
        .header-search-row input:focus {{
            border-color: #58a6ff;
        }}
        
        .header-go-btn {{
            padding: 7px 12px;
            background: #58a6ff;
            color: #000;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            cursor: pointer;
        }}
        
        .header-random-btn {{
            padding: 6px;
            background: #0d1117;
            color: #58a6ff;
            border: 1px solid #58a6ff;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            text-align: center;
        }}
        
        .market-indices {{
            background-color: #161b22;
            padding: 15px 30px;
            border-bottom: 1px solid #30363d;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }}
        
        .index-card {{
            background-color: #161b22;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        
        .index-name {{
            font-size: 11px;
            color: #888888;
            margin-bottom: 5px;
        }}
        
        .index-name a {{
            color: inherit;
            text-decoration: none;
        }}
        
        .index-name a:hover {{
            text-decoration: underline;
        }}
        
        .index-value {{
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
        }}
        
        .index-change {{
            font-size: 12px;
            margin-top: 3px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
            padding-bottom: 60px;
            display: flex;
            align-items: flex-start;
            gap: 20px;
        }}
        
        .left-column {{
            display: flex;
            flex-direction: column;
            gap: 25px;
            flex: 1.3;
        }}
        
        .right-column {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            flex: 1;
            justify-content: flex-start;
            align-items: stretch;
        }}
        
        .card {{
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .card-title {{
            font-size: 14px;
            font-weight: bold;
            color: #58a6ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #30363d;
            text-transform: uppercase;
        }}
        
        .chart-container {{
            height: 300px;
            margin-top: 15px;
            width: 100%;
            padding-right: 0;
            margin-right: 0;
        }}
        
        .period-selector {{
            display: flex;
            gap: 4px;
            margin-bottom: 12px;
        }}
        
        .period-btn {{
            background: transparent;
            border: 1px solid #30363d;
            color: #8b949e;
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 12px;
            cursor: pointer;
        }}
        
        .period-btn:hover {{
            border-color: #58a6ff;
            color: #e6edf3;
        }}
        
        .period-btn.active {{
            background: #1f6feb;
            color: white;
            border-color: #1f6feb;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        
        .market-indicators {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .indicator-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #30363d;
        }}
        
        .indicator-row:last-child {{
            border-bottom: none;
        }}
        
        .indicator-label {{
            color: #8b949e;
            font-size: 12px;
        }}
        
        .indicator-pill {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .indicator-pill.bullish {{
            background-color: #3fb950;
            color: #ffffff;
        }}
        
        .indicator-pill.bearish {{
            background-color: #f85149;
            color: #ffffff;
        }}
        
        .news-card {{
            overflow-y: auto;
            flex-shrink: 0;
            position: relative;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .news-card .card-title {{
            position: absolute;
            top: 20px;
            left: 0;
            right: 0;
            background-color: #161b22;
            z-index: 99999;
            padding: 0;
            border-bottom: 1px solid #30363d;
            margin-bottom: 15px;
            padding-bottom: 10px;
            flex-shrink: 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            padding-left: 20px;
            padding-right: 20px;
        }}
        
        .news-scroll-content {{
            overflow-y: auto;
            margin-top: 65px;
        }}
        
        .stat-item {{
            background-color: #161b22;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        
        .stat-label {{
            font-size: 10px;
            color: #8b949e;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 16px;
            font-weight: bold;
            color: #e6edf3;
        }}
        
        .ai-summary {{
            background-color: #161b22;
            padding: 20px;
            border-left: 3px solid #58a6ff;
            line-height: 1.6;
            font-size: 13px;
        }}
        
        .ai-label {{
            font-size: 11px;
            color: #58a6ff;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        
        .analyst-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid #30363d;
        }}
        
        .analyst-item:last-child {{
            border-bottom: none;
        }}
        
        .analyst-label {{
            color: #8b949e;
            font-size: 13px;
            font-weight: 500;
            min-width: 120px;
        }}
        
        .analyst-value {{
            color: #e6edf3;
            font-weight: 600;
            font-size: 14px;
            flex: 1;
            margin-left: 20px;
            text-align: right;
        }}
        
        .analyst-bar-chart {{
            display: flex;
            height: 20px;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        
        .bar-segment {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        
        .bar-segment.buy {{
            background-color: #3fb950;
        }}
        
        .bar-segment.hold {{
            background-color: #d29922;
        }}
        
        .bar-segment.sell {{
            background-color: #f85149;
        }}
        
        .bar-legend {{
            display: flex;
            gap: 15px;
            font-size: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .legend-item.buy {{
            color: #3fb950;
        }}
        
        .legend-item.hold {{
            color: #d29922;
        }}
        
        .legend-item.sell {{
            color: #f85149;
        }}
        
        .legend-item.total {{
            color: #8b949e;
            font-weight: 600;
        }}
        
        .news-item {{
            display: flex;
            align-items: flex-start;
            padding: 12px 0;
            border-bottom: 1px solid #30363d;
        }}
        
        .news-item:last-child {{
            border-bottom: none;
        }}
        
        .news-number {{
            background-color: #58a6ff;
            color: #000000;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
            margin-right: 12px;
            flex-shrink: 0;
        }}
        
        .news-content {{
            flex: 1;
        }}
        
        .news-title {{
            color: #e6edf3;
            text-decoration: none;
            font-weight: 500;
            font-size: 13px;
            margin-bottom: 3px;
            display: block;
        }}
        
        .news-title:hover {{
            color: #58a6ff;
        }}
        
        .news-date {{
            font-size: 11px;
            color: #8b949e;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            background-color: #161b22;
            color: #8b949e;
            font-size: 12px;
            border-top: 1px solid #30363d;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <div class="company-name">{f'<img src="https://logo.clearbit.com/{domain}" height="22" style="vertical-align:middle;margin-right:6px;" onerror="this.style.display=\'none\'">' if domain else ''}{company_name}</div>
            <div class="ticker-price-row">
                <span class="ticker-box">{ticker.upper()}</span>
                <span class="price-large">${current_price}</span>
                <span class="change-pill" style="background-color:{daily_color};">{daily_change_pct}</span>
            </div>
            <div class="sector-info-line">{sector_line}</div>
        </div>
        <div class="header-peers">
            <div class="peers-label">Peers</div>
            <div class="peers-row">
                {peers_html}
            </div>
        </div>
        <div class="header-indices">
            <div class="index-card">
                <div class="index-name">S&P 500</div>
                <div class="index-value">{market_indices['sp500']['price']}</div>
                <div class="index-change" style="color: {sp500_color}">{sp500_change}</div>
            </div>
            <div class="index-card">
                <div class="index-name">NASDAQ</div>
                <div class="index-value">{market_indices['nasdaq']['price']}</div>
                <div class="index-change" style="color: {nasdaq_color}">{nasdaq_change}</div>
            </div>
        </div>
        <div class="header-right">
            <div class="header-search-row">
                <input type="text" id="searchInput" placeholder="Search ticker" onkeypress="handleSearchKeypress(event)">
                <button class="header-go-btn" onclick="goToTicker()">Go</button>
            </div>
            <button class="header-random-btn" onclick="randomTicker()">Random ↻</button>
        </div>
    </div>
    
    <div class="market-indices">
        <div class="index-card">
            <div class="index-name">Market Cap</div>
            <div class="index-value">{format_large_number(financial_data.get('market_cap', 'N/A'))}</div>
        </div>
        <div class="index-card">
            <div class="index-name">P/E Ratio (TTM)</div>
            <div class="index-value">{format_ratio(financial_data.get('pe_ratio', 'N/A'))}</div>
        </div>
        <div class="index-card">
            <div class="index-name"><a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-Q&dateb=&owner=include&count=10" target="_blank">Quarterly Financials ↗</a></div>
            <div class="index-value">{quarterly_revenue}</div>
            <div class="index-change" style="color: {quarterly_growth_color}">{quarterly_growth}</div>
        </div>
        <div class="index-card">
            <div class="index-name"><a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=10" target="_blank">Earnings ↗</a></div>
            <div class="index-value" style="font-size: 14px;">
                <span style="color: {eps_beat_color}">{eps_beat_val}</span> <span style="color: #888888; font-size: 11px;">EPS Beat</span><br>
                <span style="color: {rev_beat_color}">{rev_beat_val}</span> <span style="color: #888888; font-size: 11px;">Revenue Beat</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="left-column">
            <div class="card">
                <div class="card-title">{ticker.upper()} PERFORMANCE</div>
                <div class="period-selector">
                    <button class="period-btn" data-period="1D">1D</button>
                    <button class="period-btn" data-period="5D">5D</button>
                    <button class="period-btn" data-period="1M">1M</button>
                    <button class="period-btn" data-period="6M">6M</button>
                    <button class="period-btn active" data-period="YTD">YTD</button>
                    <button class="period-btn" data-period="1Y">1Y</button>
                    <button class="period-btn" data-period="5Y">5Y</button>
                    <button class="period-btn" data-period="All">All</button>
                </div>
                <div class="chart-container">
                    <canvas id="ytdChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Quarterly Revenue & Net Income</div>
                <div class="chart-container">
                    <canvas id="quarterlyChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Key Statistics</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Avg Volume</div>
                        <div class="stat-value">{format_large_number(enhanced_stats.get('avg_volume', 'N/A'))}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Dividend Yield</div>
                        <div class="stat-value">{format_percentage(enhanced_stats.get('dividend_yield', 'N/A'))}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">EPS TTM</div>
                        <div class="stat-value">${financial_data.get('eps', 'N/A')}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">52W High / Low</div>
                        <div class="stat-value">${financial_data.get('week_52_high', 'N/A'):,.2f} / ${financial_data.get('week_52_low', 'N/A'):,.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Revenue TTM</div>
                        <div class="stat-value">{format_large_number(financial_data.get('total_revenue', 'N/A'))}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Forward P/E</div>
                        <div class="stat-value">{format_ratio(enhanced_stats.get('forward_pe', 'N/A'))}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="right-column">
            <div class="card news-card">
                <div class="card-title">Latest News</div>
                <div class="news-scroll-content">
                    {news_html}
                </div>
            </div>
            
            <div class="card">
                <div class="ai-label">AI OUTLOOK & POSITION</div>
                <div class="ai-summary">
                    <div id="ai-summary-text">
                        <span style="color:#8b949e;font-style:italic">Loading AI analysis...</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Analyst & Market Outlook</div>
                <div class="analyst-item">
                    <div class="analyst-label">Consensus Rating</div>
                    <div class="analyst-value" style="background-color: {rec_color}; color: {rec_text_color}; padding: 3px 10px; border-radius: 8px; font-weight: bold; font-size: 12px; display: inline-block; white-space: nowrap; flex: none;">{rec_display}</div>
                </div>
                <div class="analyst-item">
                    <div class="analyst-label">Price Target</div>
                    <div class="analyst-value">${analyst_data['target_price']:,.2f}</div>
                </div>
                <div class="analyst-item">
                    <div class="analyst-label">Next Earnings</div>
                    <div class="analyst-value">{analyst_data['next_earnings_date']}</div>
                </div>
                <div class="analyst-item">
                    <div class="analyst-label">Analyst Distribution</div>
                    <div class="analyst-value">
                        <div class="analyst-bar-chart">
                            <div class="bar-segment buy" style="width: {(analyst_data['buy_count'] / (analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']) * 100) if (analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']) > 0 else 0}%;" title="Buy: {analyst_data['buy_count']}"></div>
                            <div class="bar-segment hold" style="width: {(analyst_data['hold_count'] / (analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']) * 100) if (analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']) > 0 else 0}%;" title="Hold: {analyst_data['hold_count']}"></div>
                            <div class="bar-segment sell" style="width: {(analyst_data['sell_count'] / (analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']) * 100) if (analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']) > 0 else 0}%;" title="Sell: {analyst_data['sell_count']}"></div>
                        </div>
                        <div class="bar-legend">
                            <span class="legend-item buy">Buy {analyst_data['buy_count']}</span>
                            <span class="legend-item hold">Hold {analyst_data['hold_count']}</span>
                            <span class="legend-item sell">Sell {analyst_data['sell_count']}</span>
                            <span class="legend-item total">Total {analyst_data['buy_count'] + analyst_data['hold_count'] + analyst_data['sell_count']}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        Generated by AI Tearsheet Tool · {datetime.now().strftime('%B %d, %Y')} · Data sourced from Yahoo Finance & NewsAPI
    </div>
    
    <script>
        // Embed chart data as JavaScript variables
        const chartDates = {chart_dates_json};
        const stockIndexed = {stock_indexed_json};
        const spyIndexed = {sp500_indexed_json};
        const actualDates = {actual_dates_json};
        const stockPrices = {stock_prices_json};
        const sp500Levels = {sp500_levels_json};
        const quarterlyQuarters = {quarters_json};
        const quarterlyRevenue = {revenue_json};
        const quarterlyNetIncome = {net_income_json};
        const currentTicker = "{ticker.upper()}";
        
        // Tick config per period for x-axis labels
        const tickConfig = {{
            '1D':  {{ maxTicksLimit: 7,  label: 'Market hours (9:30am–4pm)' }},
            '5D':  {{ maxTicksLimit: 5,  label: 'Days' }},
            '1M':  {{ maxTicksLimit: 6,  label: 'Weeks' }},
            '6M':  {{ maxTicksLimit: 6,  label: 'Months' }},
            'YTD': {{ maxTicksLimit: 6,  label: 'Months' }},
            '1Y':  {{ maxTicksLimit: 6,  label: 'Months' }},
            '5Y':  {{ maxTicksLimit: 5,  label: 'Years' }},
            'All': {{ maxTicksLimit: 8,  label: 'Years' }},
        }};
        
        // Tick callback function to filter labels per period
        function getTickCallback(period, allLabels) {{
            let lastShown = null;

            return function(val, index, ticks) {{
                const label = allLabels[index];
                if (!label) return null;

                if (period === '6M' || period === 'YTD') {{
                    // Show first occurrence of each month only
                    const yearMonth = label.substring(0, 7); // "2026-01"
                    if (yearMonth !== lastShown) {{
                        lastShown = yearMonth;
                        const d = new Date(label + 'T12:00:00');
                        return d.toLocaleDateString('en-US', {{ month: 'short' }});
                    }}
                    return null;
                }}

                if (period === '1Y') {{
                    // Show first occurrence of every other month (Jan,Mar,May,Jul,Sep,Nov)
                    const month = parseInt(label.split('-')[1]);
                    const yearMonth = label.substring(0, 7);
                    if ([1,3,5,7,9,11].includes(month) && yearMonth !== lastShown) {{
                        lastShown = yearMonth;
                        const d = new Date(label + 'T12:00:00');
                        return d.toLocaleDateString('en-US', {{ month: 'short' }});
                    }}
                    return null;
                }}

                if (period === '5Y') {{
                    // Show first occurrence of each quarter (Jan, Apr, Jul, Oct) with short year
                    const month = parseInt(label.split('-')[1]);
                    const yearQuarter = label.substring(0, 4) + '-Q' + Math.ceil(month/3);
                    if ([1,4,7,10].includes(month) && yearQuarter !== lastShown) {{
                        lastShown = yearQuarter;
                        const d = new Date(label + 'T12:00:00');
                        return d.toLocaleDateString('en-US', {{ month: 'short', year: '2-digit' }});
                    }}
                    return null;
                }}

                // 1D and 5D cases — keep exactly as they are now
                if (period === '1D') {{
                    return (label.endsWith(':00') || label === '9:30') ? label : null;
                }}
                if (period === '5D') {{
                    return label.includes('9:30') ? label.split(' ').slice(0,2).join(' ') : null;
                }}

                if (period === 'All') {{
                    const month = parseInt(label.split('-')[1]);
                    const day = parseInt(label.split('-')[2]);
                    if (month === 1 && day <= 15) {{
                        return label.split('-')[0];
                    }}
                    return null;
                }}

                return label;
            }};
        }}
        
        // Initialize charts when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Chart data loaded:', {{ chartDates, stockIndexed, spyIndexed, quarterlyQuarters, quarterlyRevenue, quarterlyNetIncome }});
            
            // YTD Performance Chart
            const ytdCtx = document.getElementById('ytdChart');
            let ytdChart = null;
            if (ytdCtx && chartDates.length > 0) {{
                // Calculate tight y-axis bounds for initial load
                const stockMin = Math.min(...stockPrices) * 0.995;
                const stockMax = Math.max(...stockPrices) * 1.005;
                const spMin = Math.min(...sp500Levels) * 0.995;
                const spMax = Math.max(...sp500Levels) * 1.005;
                
                ytdChart = new Chart(ytdCtx, {{
                    type: 'line',
                    data: {{
                        labels: chartDates,
                        datasets: [
                            {{
                                label: '{ticker.upper()}',
                                data: stockPrices,
                                borderColor: '#58a6ff',
                                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.1,
                                yAxisID: 'y'
                            }},
                            {{
                                label: 'S&P 500',
                                data: sp500Levels,
                                borderColor: '#ff6b6b',
                                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.1,
                                yAxisID: 'y2'
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        layout: {{ padding: {{ left: 0, right: 0, top: 10, bottom: 0 }} }},
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                labels: {{ color: '#e6edf3' }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                callbacks: {{
                                    title: (items) => actualDates[items[0].dataIndex],
                                    label: (ctx) => {{
                                        if (ctx.datasetIndex === 0) {{
                                            return ' {ticker.upper()}: $' + ctx.parsed.y.toFixed(2);
                                        }} else {{
                                            return ' S&P 500: ' + ctx.parsed.y.toLocaleString();
                                        }}
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{ 
                                    color: '#8b949e',
                                    maxRotation: 0,
                                    minRotation: 0,
                                    font: {{ size: 10 }},
                                    autoSkip: false,
                                    callback: getTickCallback('YTD', chartDates)
                                }},
                                grid: {{ color: '#30363d' }},
                                bounds: 'data',
                                offset: false
                            }},
                            y: {{
                                type: 'linear',
                                position: 'left',
                                title: {{ display: true, text: '{ticker.upper()} Price ($)', color: '#58a6ff', font: {{ size: 11 }} }},
                                ticks: {{ color: '#8b949e', callback: (v) => '$' + v.toFixed(2) }},
                                grid: {{ color: 'rgba(48,54,61,0.5)' }},
                                min: stockMin,
                                max: stockMax
                            }},
                            y2: {{
                                type: 'linear',
                                position: 'right',
                                title: {{ display: true, text: 'S&P 500 Level', color: '#f85149', font: {{ size: 11 }} }},
                                ticks: {{ color: '#8b949e', callback: (v) => v.toLocaleString() }},
                                grid: {{ drawOnChartArea: false }},
                                min: spMin,
                                max: spMax
                            }}
                        }}
                    }}
                }});
            }} else {{
                console.error('YTD chart canvas not found or no data');
            }}
            
            // Period selector button handlers
            const periodButtons = document.querySelectorAll('.period-btn');
            periodButtons.forEach(button => {{
                button.addEventListener('click', function() {{
                    const period = this.getAttribute('data-period');
                    
                    // Update active class
                    periodButtons.forEach(btn => btn.classList.remove('active'));
                    this.classList.add('active');
                    
                    // Fetch new chart data
                    fetch(`/chart_data?ticker=${{currentTicker}}&period=${{period}}`)
                        .then(response => response.json())
                        .then(data => {{
                            if (data.error) {{
                                console.error('Error fetching chart data:', data.error);
                                return;
                            }}
                            
                            // Update chart data
                            ytdChart.data.labels = data.dates;
                            ytdChart.data.datasets[0].data = data.stock_prices;
                            ytdChart.data.datasets[1].data = data.sp500_levels;
                            
                            // Apply tight y-axis bounds from response
                            ytdChart.options.scales.y.min = data.stock_min;
                            ytdChart.options.scales.y.max = data.stock_max;
                            ytdChart.options.scales.y2.min = data.sp_min;
                            ytdChart.options.scales.y2.max = data.sp_max;
                            
                            // Apply dynamic x-axis tick callback based on period
                            ytdChart.options.scales.x.ticks.callback = getTickCallback(period, data.dates);
                            ytdChart.options.scales.x.ticks.autoSkip = false;
                            
                            ytdChart.update();
                        }})
                        .catch(error => console.error('Error fetching chart data:', error));
                }});
            }});
            
            // Quarterly Revenue & Net Income Chart
            const quarterlyCtx = document.getElementById('quarterlyChart');
            if (quarterlyCtx && quarterlyQuarters.length > 0) {{
                new Chart(quarterlyCtx, {{
                    type: 'bar',
                    data: {{
                        labels: quarterlyQuarters,
                        datasets: [
                            {{
                                label: 'Revenue ($B)',
                                data: quarterlyRevenue,
                                backgroundColor: 'rgba(0, 212, 255, 0.7)',
                                borderColor: '#00d4ff',
                                borderWidth: 1,
                                yAxisID: 'y'
                            }},
                            {{
                                label: 'Net Income ($B)',
                                data: quarterlyNetIncome,
                                borderColor: '#00ff88',
                                backgroundColor: 'rgba(0, 255, 136, 0.7)',
                                type: 'line',
                                borderWidth: 2,
                                yAxisID: 'y1'
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                labels: {{ color: '#e6edf3' }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{ color: '#8b949e' }},
                                grid: {{ color: '#30363d' }}
                            }},
                            y: {{
                                type: 'linear',
                                position: 'left',
                                ticks: {{ color: '#8b949e' }},
                                grid: {{ color: '#30363d' }},
                                title: {{
                                    display: true,
                                    text: 'Revenue ($B)',
                                    color: '#8b949e'
                                }}
                            }},
                            y1: {{
                                type: 'linear',
                                position: 'right',
                                ticks: {{ color: '#8b949e' }},
                                grid: {{ display: false }},
                                title: {{
                                    display: true,
                                    text: 'Net Income ($B)',
                                    color: '#8b949e'
                                }}
                            }}
                        }}
                    }}
                }});
            }} else {{
                console.error('Quarterly chart canvas not found or no data');
            }}
            
            // Dynamically adjust news card height to match left column
            const leftColumn = document.querySelector('.left-column');
            const rightColumn = document.querySelector('.right-column');
            const newsCard = document.querySelector('.news-card');
            const newsTitle = document.querySelector('.news-card .card-title');
            
            if (leftColumn && rightColumn && newsCard) {{
                const leftHeight = leftColumn.offsetHeight;
                const aiCard = rightColumn.querySelector('.card:not(.news-card)');
                const analystCard = rightColumn.querySelectorAll('.card:not(.news-card)')[1];
                
                let otherCardsHeight = 0;
                if (aiCard) otherCardsHeight += aiCard.offsetHeight;
                if (analystCard) otherCardsHeight += analystCard.offsetHeight;
                
                const newsHeight = leftHeight - otherCardsHeight - 32; // 32px for gaps
                if (newsHeight > 200) {{
                    newsCard.style.height = newsHeight + 'px';
                }}
            }}
            
            // Fix sticky header z-index issue on scroll
            if (newsCard && newsTitle) {{
                newsCard.addEventListener('scroll', function() {{
                    const scrollTop = newsCard.scrollTop;
                    if (scrollTop > 0) {{
                        newsTitle.style.zIndex = '999999';
                    }} else {{
                        newsTitle.style.zIndex = '99999';
                    }}
                }});
            }}
            
            // Fetch AI summary asynchronously after page load
            const encodedCompanyName = encodeURIComponent("{company_name}");
            fetch(`/ai_summary?ticker={ticker.upper()}&name=${{encodedCompanyName}}`)
                .then(r => r.json())
                .then(data => {{
                    document.getElementById('ai-summary-text').innerHTML = data.summary;
                }});
        }});
        
        function handleSearchKeypress(event) {{
            if (event.key === 'Enter') {{
                const ticker = document.getElementById('searchInput').value.trim().toUpperCase();
                if (ticker) {{
                    window.location.href = `/tearsheet?ticker=${{ticker}}`;
                }}
            }}
        }}
        
        function goToTicker() {{
            const ticker = document.getElementById('searchInput').value.trim().toUpperCase();
            if (ticker) window.location.href = `/tearsheet?ticker=${{ticker}}`;
        }}
        
        function randomTicker() {{
            const tickers = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM','V','UNH','MA','XOM','LLY','JNJ','WMT','CVX','PG','HD','MRK','ABBV','BAC','KO','PEP','AVGO','COST','TMO','MCD','CSCO','ACN','ABT','CRM','NEE','LIN','DHR','TXN','VZ','ADBE','PM','RTX','AMGN','QCOM','HON','IBM','GS','MS','BLK','SPGI','GE','CAT','BA'];
            const randomTicker = tickers[Math.floor(Math.random() * tickers.length)];
            window.location.href = '/tearsheet?ticker=' + randomTicker;
        }}
    </script>
</body>
</html>
    """
    
    # Return HTML content for web server
    return html_content

def fetch_peer_data(ticker, sector):
    ETF_BLACKLIST = {'SPY','QQQ','DIA','IWM','VTI','VOO','GLD','TLT','HYG','LQD',
                    'XLF','XLK','XLE','XLV','XLI','XLP','XLU','XLY','XLB','XLRE',
                    'VXX','UVXY','SQQQ','TQQQ','ARKK','ARKG','ARKW'}
    
    PEER_GROUPS = {
        'MSFT': ['AAPL', 'GOOGL', 'META', 'NVDA'],
        'AAPL': ['MSFT', 'GOOGL', 'META', 'NVDA'],
        'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'],
        'META': ['GOOGL', 'AAPL', 'MSFT', 'SNAP'],
        'NVDA': ['AMD', 'INTC', 'QCOM', 'TSM'],
        'TSLA': ['RIVN', 'F', 'GM', 'TM'],
        'AMZN': ['MSFT', 'GOOGL', 'WMT', 'SHOP'],
        'GS': ['MS', 'JPM', 'BAC', 'WFC'],
        'JPM': ['GS', 'MS', 'BAC', 'C'],
        'MS': ['GS', 'JPM', 'BAC', 'WFC'],
        'BAC': ['JPM', 'GS', 'MS', 'WFC'],
        'XOM': ['CVX', 'COP', 'SLB', 'BP'],
        'JNJ': ['PFE', 'ABBV', 'MRK', 'UNH'],
        'NFLX': ['DIS', 'PARA', 'WBD', 'SPOT'],
    }
    SECTOR_DEFAULTS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        'Financial Services': ['JPM', 'GS', 'MS', 'BAC'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX'],
        'Industrials': ['BA', 'LMT', 'GE', 'CAT'],
        'Consumer Defensive': ['WMT', 'COST', 'PG', 'KO'],
    }
    peers = PEER_GROUPS.get(ticker.upper(), SECTOR_DEFAULTS.get(sector, ['SPY', 'QQQ', 'DIA', 'IWM']))
    
    # Filter 1: Remove subject ticker itself
    peers = [p for p in peers if p.upper() != ticker.upper()]
    
    # Filter 2: Pre-filter using ETF blacklist
    peers = [p for p in peers if p.upper() not in ETF_BLACKLIST]
    
    yf = install_yfinance()
    peer_data = []
    for p in peers[:4]:
        try:
            info = yf.Ticker(p).info
            # Filter 3: Check quoteType - skip if not EQUITY
            quote_type = info.get('quoteType', '')
            if quote_type not in ('EQUITY', 'equity'):
                continue
            
            price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            change = info.get('regularMarketChangePercent', 0)
            company = info.get('shortName', info.get('longName', 'N/A'))
            if company != 'N/A':
                company = company[:20] + '...' if len(company) > 20 else company
            color = '#3fb950' if float(change) >= 0 else '#f85149'
            peer_data.append({'ticker': p, 'price': f"${price:.2f}" if price != 'N/A' else 'N/A', 'change': f"{float(change):+.2f}%", 'color': color, 'company': company})
        except:
            peer_data.append({'ticker': p, 'price': 'N/A', 'change': 'N/A', 'color': '#8b949e', 'company': 'N/A'})
    return peer_data

def get_financial_data(ticker):
    """Fetch and display financial data for a given stock ticker"""
    # Check cache first
    cached = get_cached(ticker.upper())
    if cached:
        return cached
    
    yf = install_yfinance()
    
    try:
        # Use requests session with timeout to prevent hanging
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        stock = yf.Ticker(ticker, session=session)
        info = stock.info
        
        # Validation checks for invalid and unsupported tickers
        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
        if current_price is None:
            raise ValueError("invalid")

        quote_type = info.get('quoteType', '').upper()
        if quote_type in ['ETF', 'MUTUALFUND']:
            raise ValueError("unsupported")

        market_cap = info.get('marketCap', 0) or 0
        total_revenue = info.get('totalRevenue', None)
        if total_revenue is None and market_cap < 500000000:
            raise ValueError("unsupported")
        
        # Extract the required information
        company_name = info.get('longName', 'N/A')
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        pe_ratio = info.get('trailingPE', info.get('forwardPE', 'N/A'))
        market_cap = info.get('marketCap', 'N/A')
        total_revenue = info.get('totalRevenue', 'N/A')
        eps = info.get('trailingEps', info.get('forwardEps', 'N/A'))
        week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
        week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
        
        # Store financial data in dictionary for AI summary
        financial_data = {
            'current_price': current_price,
            'pe_ratio': pe_ratio,
            'market_cap': market_cap,
            'total_revenue': total_revenue,
            'eps': eps,
            'week_52_high': week_52_high,
            'week_52_low': week_52_low
        }
        
        # Fetch enhanced statistics
        enhanced_stats = {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'daily_change': info.get('regularMarketChangePercent', 'N/A'),
            'net_margin': info.get('profitMargins', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'fifty_day_ma': info.get('fiftyDayAverage', 'N/A'),
            'two_hundred_day_ma': info.get('twoHundredDayAverage', 'N/A'),
            'website': info.get('website', 'N/A')
        }
        
        # Calculate RSI
        rsi = calculate_rsi(ticker)
        enhanced_stats['rsi'] = rsi
        
        # Fetch market indices
        market_indices = fetch_market_indices()
        
        # Fetch YTD performance data
        ytd_data, ytd_summary = fetch_ytd_performance(ticker)
        
        # Fetch quarterly financials
        quarterly_data = fetch_quarterly_financials(ticker)
        
        # Fetch analyst data
        analyst_data = fetch_analyst_data(ticker)
        
        # Run API calls in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_news = executor.submit(fetch_news_headlines, company_name, ticker) if company_name != 'N/A' else None
            future_peers = executor.submit(fetch_peer_data, ticker, enhanced_stats.get('sector', ''))
            
            headlines = future_news.result() if future_news else []
            peer_data = future_peers.result()
        
        # Generate AI analyst summary (lazy-loaded later)
        ai_summary = ""
        
        # Generate peers HTML
        peers_html = ''.join([
            f'<a class="peer-card" href="/tearsheet?ticker={p["ticker"]}">'
            f'<div class="peer-company">{p["company"]}</div>'
            f'<div class="peer-info-row">'
            f'<span class="peer-ticker">{p["ticker"]}</span>'
            f'<span class="peer-price">{p["price"]}</span>'
            f'<span class="peer-change {"positive" if "+" in p["change"] else "negative"}">{p["change"]}</span>'
            f'</div>'
            f'</a>'
            for p in peer_data
        ])
        
        # Generate HTML tearsheet with all new data
        html_content = generate_html_tearsheet(
            ticker, 
            company_name, 
            financial_data, 
            headlines, 
            ai_summary,
            market_indices,
            ytd_data,
            quarterly_data,
            analyst_data,
            enhanced_stats,
            peers_html
        )
        
        # Cache the result
        result = {
            'company_name': company_name,
            'financial_data': financial_data,
            'headlines': headlines,
            'ai_summary': ai_summary,
            'market_indices': market_indices,
            'ytd_data': ytd_data,
            'quarterly_data': quarterly_data,
            'analyst_data': analyst_data,
            'enhanced_stats': enhanced_stats,
            'peers_html': peers_html
        }
        set_cached(ticker.upper(), result)
        
        # Return HTML content for web server
        return html_content
        
    except ValueError as e:
        raise
    except Exception as e:
        raise ValueError("invalid")

def main():
    """Main function to get user input and display financial data"""
    print("Stock Financial Data Fetcher")
    print("="*30)
    
    while True:
        ticker = input("Enter a stock ticker (or 'quit' to exit): ").strip()
        
        if ticker.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not ticker:
            print("Please enter a valid ticker symbol.")
            continue
        
        get_financial_data(ticker)

@app.route('/')
def home():
    """Serve the home page with search bar"""
    try:
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    except Exception as e:
        return f"Error loading home page: {e}", 500

@app.route('/tearsheet')
def tearsheet():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return redirect('/')
    
    # Check if already loading or ready
    if ticker in _loading:
        status = _loading[ticker]
        if status == 'ready':
            return redirect(url_for('tearsheet_render', ticker=ticker))
        elif status == 'error':
            return redirect(url_for('error', ticker=ticker, type='invalid'))
    
    # Start background thread to fetch data
    _loading[ticker] = 'loading'
    
    def fetch_in_background():
        try:
            html_content = get_financial_data(ticker)
            _results[ticker] = html_content
            _loading[ticker] = 'ready'
        except ValueError as e:
            _loading[ticker] = 'error'
        except Exception:
            _loading[ticker] = 'error'
    
    thread = threading.Thread(target=fetch_in_background)
    thread.start()
    
    # Return loading page
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Loading {ticker}...</title>
    <style>
        * {{margin:0;padding:0;box-sizing:border-box;}}
        body {{font-family:'Inter','Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;align-items:center;justify-content:center;}}
        .loading-container {{text-align:center;}}
        .spinner {{width:50px;height:50px;border:3px solid #30363d;border-top:3px solid #58a6ff;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 20px;}}
        @keyframes spin {{0% {{transform:rotate(0deg);}} 100% {{transform:rotate(360deg);}}}}
        .ticker {{font-size:24px;font-weight:bold;color:#58a6ff;margin-bottom:10px;}}
        .message {{color:#8b949e;font-size:14px;}}
    </style>
</head>
<body>
    <div class="loading-container">
        <div class="spinner"></div>
        <div class="ticker">{ticker}</div>
        <div class="message">Loading financial data...</div>
    </div>
    <script>
        setInterval(function() {{
            fetch('/tearsheet_ready?ticker={ticker}')
                .then(r => r.json())
                .then(data => {{
                    if (data.ready) {{
                        window.location.href = '/tearsheet_render?ticker={ticker}';
                    }}
                }});
        }}, 500);
    </script>
</body>
</html>
"""

@app.route('/tearsheet_ready')
def tearsheet_ready():
    from flask import jsonify
    ticker = request.args.get('ticker', '').upper()
    status = _loading.get(ticker, 'loading')
    ready = status == 'ready'
    return jsonify({'ready': ready})

@app.route('/tearsheet_render')
def tearsheet_render():
    ticker = request.args.get('ticker', '').upper()
    if ticker in _results:
        return _results[ticker]
    else:
        return redirect(url_for('error', ticker=ticker, type='invalid'))

@app.route('/error')
def error():
    ticker = request.args.get('ticker', '').upper()
    error_type = request.args.get('type', 'invalid')
    if error_type == 'unsupported':
        message = f"{ticker} appears to be a SPAC, ETF, or OTC security. This tool is optimised for publicly traded operating companies."
    else:
        message = f"{ticker} is not a recognised ticker symbol. Please check the symbol and try again."
    return f"""
<!DOCTYPE html><html><head><title>Error</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box;}}
body {{font-family:'Inter','Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;align-items:center;justify-content:center;}}
.box {{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:60px 40px;text-align:center;max-width:500px;}}
.icon {{font-size:48px;margin-bottom:20px;}}
h1 {{color:#f85149;font-size:24px;margin-bottom:16px;}}
p {{color:#8b949e;font-size:15px;line-height:1.6;margin-bottom:30px;}}
button {{background:#58a6ff;color:#fff;border:none;padding:12px 28px;border-radius:8px;font-size:15px;font-weight:bold;cursor:pointer;}}
button:hover {{background:#4094ff;}}
</style></head>
<body><div class="box">
<div class="icon">⚠</div>
<h1>Unable to generate tearsheet</h1>
<p>{message}</p>
<button onclick="window.location.href='/'">Try another ticker</button>
</div></body></html>"""

@app.route('/ai_summary')
def ai_summary():
    from flask import jsonify
    ticker = request.args.get('ticker', '').upper()
    company_name = request.args.get('name', '')
    try:
        yf = install_yfinance()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        financial_data = {
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'pe_ratio': info.get('trailingPE', info.get('forwardPE', 'N/A')),
            'market_cap': info.get('marketCap', 'N/A'),
            'total_revenue': info.get('totalRevenue', 'N/A'),
            'eps': info.get('trailingEps', info.get('forwardEps', 'N/A')),
            'week_52_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'week_52_low': info.get('fiftyTwoWeekLow', 'N/A')
        }
        
        headlines = fetch_news_headlines(company_name, ticker) if company_name != 'N/A' else []
        summary = get_ai_analyst_summary(company_name, financial_data, headlines)
        
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'summary': 'AI summary unavailable.'})

@app.route('/chart_data')
def chart_data():
    from flask import jsonify
    ticker = request.args.get('ticker', '').upper()
    period = request.args.get('period', 'YTD')
    
    period_config = {
        '1D':  {'interval': '5m',  'max_points': 78},
        '5D':  {'interval': '30m', 'max_points': 65},
        '1M':  {'interval': '1d',  'max_points': 22},
        '6M':  {'interval': '1d',  'max_points': 130},
        'YTD': {'interval': '1d',  'max_points': 150},
        '1Y':  {'interval': '1d',  'max_points': 252},
        '5Y':  {'interval': '1wk', 'max_points': 260},
        'All': {'interval': '1mo', 'max_points': 240},
    }
    
    config = period_config.get(period, period_config['YTD'])
    interval = config['interval']
    max_points = config['max_points']
    
    period_map = {
        '1D': '1d', '5D': '5d', '1M': '1mo',
        '6M': '6mo', 'YTD': 'ytd', '1Y': '1y',
        '5Y': '5y', 'All': 'max'
    }
    yf_period = period_map.get(period, 'ytd')
    
    try:
        yf = install_yfinance()
        t = yf.Ticker(ticker)
        hist = t.history(period=yf_period, interval=interval)
        sp = yf.Ticker('^GSPC').history(period=yf_period, interval=interval)
        
        if period == '1D':
            # Filter to market hours only (9:30 AM - 4:00 PM)
            market_hours = []
            market_prices = []
            market_sp = []
            for i, dt in enumerate(hist.index):
                hour = dt.hour
                minute = dt.minute
                time_val = hour * 100 + minute
                if 930 <= time_val <= 1600:
                    # Format as "9:30", "10:00" (no leading zero for hours)
                    market_hours.append(f"{hour}:{minute:02d}")
                    market_prices.append(round(float(hist['Close'].iloc[i]), 2))
                    market_sp.append(round(float(sp['Close'].iloc[i]), 2))
            dates = market_hours
            stock_prices = market_prices
            sp500_levels = market_sp
        elif period == '5D':
            # Format as "May 20 9:30"
            dates = [d.strftime('%b %d %H:%M') for d in hist.index]
            stock_prices = [round(float(p), 2) for p in hist['Close']]
            sp500_levels = [round(float(p), 2) for p in sp['Close']]
        else:
            dates = [str(d.date()) for d in hist.index]
            stock_prices = [round(float(p), 2) for p in hist['Close']]
            sp500_levels = [round(float(p), 2) for p in sp['Close']]
        
        # align lengths
        min_len = min(len(dates), len(stock_prices), len(sp500_levels))
        dates = dates[:min_len]
        stock_prices = stock_prices[:min_len]
        sp500_levels = sp500_levels[:min_len]
        
        # data thinning
        if len(dates) > max_points:
            step = len(dates) // max_points
            dates = dates[::step]
            stock_prices = stock_prices[::step]
            sp500_levels = sp500_levels[::step]
        
        # calculate tight y-axis bounds
        stock_min = min(stock_prices) * 0.995 if stock_prices else 0
        stock_max = max(stock_prices) * 1.005 if stock_prices else 0
        sp_min = min(sp500_levels) * 0.995 if sp500_levels else 0
        sp_max = max(sp500_levels) * 1.005 if sp500_levels else 0
        
        return jsonify({
            'dates': dates,
            'stock_prices': stock_prices,
            'sp500_levels': sp500_levels,
            'stock_min': stock_min,
            'stock_max': stock_max,
            'sp_min': sp_min,
            'sp_max': sp_max
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    print("Starting Financial Tearsheet web server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
