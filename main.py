"""
NEURAL CORTEX v5.1 — PRODUCTION BUILD
=======================================
brain.py    = mechanical backbone (probabilities, scoring, learning)
reasoning.py = reasoning audit trail (HOW it thinks, track record, meta-learning)
main.py     = data pipeline + 6 adversarial agents + logging + dashboard

Every run:
1. Fetch 45+ data sources
2. Run mechanical brain (regime, probabilities, pattern matching)
3. Reconcile prior reasoning against current data
4. Feed agents their OWN track record + reconciliation report
5. Run 6 adversarial agents (each sees prior agents + challenges them)
6. Extract structured reasoning chains + disagreements
7. Score resolved reasoning + update agent scoreboards
8. Convert LLM theses into brain-tracked predictions
9. Log EVERYTHING (raw data + reasoning + outcomes)
10. Render dashboard
"""

import os, json, re, datetime, time, traceback, hashlib
import requests
from xml.etree import ElementTree as ET

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

from brain import run_brain, BUCKETS
from reasoning import (
    run_reasoning_pass, get_pre_agent_context, build_agent_context,
    extract_theses_from_synthesis, reconcile_timeframes,
    predict_regime_transition, load_reasoning_db, load_agent_scores,
)

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
FRED_KEY = os.environ.get("FRED_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
UA = {"User-Agent": "Mozilla/5.0 (NeuralCortex/5.1)"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/raw_logs", exist_ok=True)

SOURCE_HEALTH = {}
def log_source(n, ok, d=""): SOURCE_HEALTH[n] = {"ok":ok,"detail":d[:200],"ts":datetime.datetime.utcnow().isoformat()}
def load_json(p, d):
    if not os.path.exists(p): return d
    try:
        with open(p) as f: return json.load(f)
    except: return d
def save_json(p, d):
    with open(p,"w") as f: json.dump(d,f,indent=2,default=str)
def safe_get(url, timeout=14, headers=None, source_name=""):
    try:
        r = requests.get(url, headers=headers or UA, timeout=timeout)
        if source_name: log_source(source_name, r.ok, f"HTTP {r.status_code}")
        return r
    except Exception as e:
        if source_name: log_source(source_name, False, str(e))
        print(f"  [WARN] {source_name or url}: {e}")
        return None
def trunc(s,n=300): return str(s)[:n]

def save_raw_log(ts, data):
    fn = ts.replace(" ","_").replace(":","")
    save_json(f"{DATA_DIR}/raw_logs/{fn}.json", data)
    logs = sorted(os.listdir(f"{DATA_DIR}/raw_logs"))
    for old in logs[:-200]:
        try: os.remove(f"{DATA_DIR}/raw_logs/{old}")
        except: pass


# ══════════════════════════════════════════════════════════════════
# ALL DATA FETCHERS (comprehensive, all v4 bugs fixed)
# ══════════════════════════════════════════════════════════════════

def fetch_warn_act():
    signals = []
    try:
        r = safe_get("https://www.dol.gov/agencies/eta/layoffs/warn", source_name="WARN_DOL")
        if r and r.ok:
            for co,ct in re.findall(r"([A-Z][A-Za-z\s&,\.]+)\s+(\d{2,4})\s+(?:worker|employee|position)",re.sub(r"<[^>]+>"," ",r.text))[:15]:
                signals.append(f"{co.strip()}: {ct} workers")
    except Exception as e: print(f"  [WARN] WARN: {e}")
    try:
        r = safe_get("https://www.edd.ca.gov/jobs_and_training/Layoff_Services_WARN.htm", source_name="WARN_CA")
        if r and r.ok:
            for co,ct in re.findall(r"([A-Z][A-Za-z\s&,\.]+)\s+(\d{2,4})",re.sub(r"<[^>]+>"," ",r.text))[:10]:
                if int(ct)>50: signals.append(f"CA: {co.strip()}: {ct}")
    except: pass
    return signals if signals else ["WARN unavailable"]

def fetch_ofac():
    try:
        r = safe_get("https://www.treasury.gov/ofac/downloads/sdnlist.txt",timeout=20,source_name="OFAC")
        if r and r.ok:
            ents = [trunc(l,200) for l in [l.strip() for l in r.text.split("\n")[:80] if l.strip() and not l.startswith("-")][:30] if any(t in l for t in ["SDN","SDGT","IFSR","IRAN","RUSSIA","CHINA","DPRK"])]
            return ents[:15] if ents else ["OFAC: no flags"]
    except: pass
    return ["OFAC unavailable"]

def fetch_contracts():
    out = []
    try:
        r = requests.post("https://api.usaspending.gov/api/v2/search/spending_by_award/",json={"filters":{"time_period":[{"start_date":(datetime.date.today()-datetime.timedelta(days=7)).isoformat(),"end_date":datetime.date.today().isoformat()}],"award_type_codes":["A","B","C","D"]},"fields":["Recipient Name","Award Amount","Awarding Agency Name","Description"],"sort":"Award Amount","order":"desc","limit":20},timeout=15)
        log_source("USASPENDING",r.ok)
        if r.ok:
            for a in r.json().get("results",[])[:15]:
                amt = a.get("Award Amount",0) or 0
                if amt>100_000_000: out.append(f"${amt/1e9:.2f}B|{a.get('Recipient Name','?')}|{a.get('Awarding Agency Name','?')}|{trunc(a.get('Description',''),80)}")
    except Exception as e: log_source("USASPENDING",False,str(e))
    return out if out else ["USASpending unavailable"]

def fetch_patents():
    pats = []
    for term in ["quantum computing","hypersonic","solid state battery","large language model","semiconductor fabrication","nuclear fusion","mRNA","CRISPR","rare earth","directed energy"]:
        try:
            dt = (datetime.date.today()-datetime.timedelta(days=30)).isoformat()
            r = safe_get(f"https://api.patentsview.org/patents/query?q={{\"_and\":[{{\"_gte\":{{\"patent_date\":\"{dt}\"}}}},{{\"_text_any\":{{\"patent_abstract\":\"{term}\"}}}}]}}&f=[\"patent_title\",\"assignee_organization\",\"patent_date\"]&o={{\"per_page\":3}}",source_name=f"USPTO_{term[:10]}")
            if r and r.ok:
                for p in r.json().get("patents",[])[:2]:
                    asn = p.get("assignees",[{}]); org = asn[0].get("assignee_organization","?") if asn else "?"
                    pats.append(f"[{term}] {org}: {p.get('patent_title','')[:100]}")
            time.sleep(0.2)
        except: pass
    log_source("USPTO",len(pats)>0)
    return pats[:15] if pats else ["Patents unavailable"]

def fetch_pboc():
    if HAS_YF:
        try:
            h = yf.Ticker("CNH=X").history(period="5d")
            if not h.empty and len(h)>=2:
                s,p = round(float(h["Close"].iloc[-1]),4),round(float(h["Close"].iloc[-2]),4)
                log_source("PBOC",True)
                return {"spot":s,"1d":round(s-p,4),"signal":"DEFENDING" if (s-p)<-0.005 else ("RELEASING" if (s-p)>0.01 else "STABLE")}
        except Exception as e: log_source("PBOC",False,str(e))
    return {"note":"unavailable"}

def fetch_shipping():
    intel = {}
    try:
        r = safe_get("https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key=DEMO_KEY&frequency=weekly&data[0]=value&facets[product][]=EPC0&facets[area][]=NUS&sort[0][column]=period&sort[0][direction]=desc&length=4",timeout=12,source_name="EIA")
        if r and r.ok:
            d = r.json().get("response",{}).get("data",[])
            if len(d)>=2:
                c,p = float(d[0]["value"]),float(d[1]["value"])
                intel["crude_storage"]=c; intel["storage_chg"]=round(c-p,1)
    except: pass
    return intel

def fetch_leaders():
    sigs = []
    for tag,url,lim in [("FED","https://www.federalreserve.gov/feeds/speeches.xml",5),("ECB","https://www.ecb.europa.eu/rss/press.html",4),("RBI","https://www.rbi.org.in/rss/RBIRSSFeed.aspx?Id=PRS",4),("WH","https://www.whitehouse.gov/feed/",5),("TREASURY","https://home.treasury.gov/system/files/136/rss.xml",4),("XINHUA","http://www.xinhuanet.com/english/rss/chinesegovernment.xml",4),("KREMLIN","http://en.kremlin.ru/events/president/news/rss",3),("MEA","https://www.mea.gov.in/rss/press-releases.xml",4),("BIS","https://www.bis.org/rss/all_speeches.xml",3)]:
        try:
            r = safe_get(url,timeout=10,source_name=tag)
            if r and r.ok:
                for item in list(ET.fromstring(r.content).iter("item"))[:lim]:
                    t = (item.findtext("title") or "").strip()
                    if t: sigs.append(f"[{tag}] {t[:120]}")
        except: pass
    log_source("LEADERS",len(sigs)>3)
    return sigs if sigs else ["Leaders unavailable"]

def fetch_policy():
    ch = []
    try:
        r = safe_get(f"https://www.federalregister.gov/api/v1/documents.json?conditions[publication_date][gte]={(datetime.date.today()-datetime.timedelta(days=7)).isoformat()}&conditions[type][]=RULE&conditions[type][]=PRESDOCU&per_page=15&order=newest",timeout=12,source_name="FEDREG")
        if r and r.ok:
            kws=["trade","tariff","export","sanction","defense","semiconductor","energy","financial","bank","tax","AI","crypto","climate"]
            for d in r.json().get("results",[])[:10]:
                t=d.get("title","")
                if any(k.lower() in t.lower() for k in kws): ch.append(f"[{d.get('type','')}] {t[:120]}")
    except: pass
    try:
        r = safe_get("https://www.sebi.gov.in/rss/all-news.xml",timeout=10,source_name="SEBI")
        if r and r.ok:
            for item in list(ET.fromstring(r.content).iter("item"))[:5]:
                t=(item.findtext("title") or "").strip()
                if t: ch.append(f"[SEBI] {t[:120]}")
    except: pass
    return ch if ch else ["Policy unavailable"]

def fetch_fed_liquidity():
    liq = {}
    try:
        r = safe_get("https://api.fiscaldata.treasury.gov/services/api/v1/accounting/dts/deposits_withdrawals_operating_cash?fields=record_date,open_today_bal&sort=-record_date&limit=5",timeout=12,source_name="TGA")
        if r and r.ok:
            d=r.json().get("data",[])
            if len(d)>=2:
                n,p=float(d[0]["open_today_bal"])/1000,float(d[1]["open_today_bal"])/1000
                liq["tga_bn"]=round(n,1);liq["tga_chg"]=round(n-p,1)
                liq["tga_signal"]="DRAINING" if (n-p)<-10 else ("BUILDING" if (n-p)>10 else "STABLE")
    except: pass
    if FRED_KEY:
        for s,l in [("RRPONTSYD","RRP"),("M2SL","M2"),("T10Y2Y","YC_10Y2Y"),("BAMLH0A0HYM2","HY_Spread"),("DRCCLACBS","CC_Delinq"),("WALCL","Fed_Assets"),("MORTGAGE30US","Mortgage_30Y")]:
            try:
                r=safe_get(f"https://api.stlouisfed.org/fred/series/observations?series_id={s}&api_key={FRED_KEY}&sort_order=desc&limit=3&file_type=json",source_name=f"FRED_{s}")
                if r and r.ok:
                    obs=[o for o in r.json().get("observations",[]) if o["value"]!="."]
                    if obs:
                        c=float(obs[0]["value"]);p=float(obs[1]["value"]) if len(obs)>1 else c
                        liq[l]={"val":round(c,3),"dt":obs[0]["date"],"delta":round(c-p,4)}
            except: pass
    return liq

def fetch_sofr():
    if not HAS_YF: return {}
    curve={};now=datetime.date.today();mc={3:"H",6:"M",9:"U",12:"Z"}
    for m in [3,6,12,24]:
        td=now+datetime.timedelta(days=m*30);qm=((td.month-1)//3+1)*3;yr=td.year%100
        if qm>12:qm,yr=3,(td.year+1)%100
        sym=f"SR3{mc.get(qm,'Z')}{yr}.CBT"
        try:
            h=yf.Ticker(sym).history(period="3d")
            if not h.empty: curve[f"SOFR_{m}M"]={"price":round(float(h["Close"].iloc[-1]),3),"rate":round(100-float(h["Close"].iloc[-1]),3)}
        except: pass
    log_source("SOFR",len(curve)>0)
    return curve

def fetch_legislative():
    bills=[]
    try:
        dt=(datetime.datetime.utcnow()-datetime.timedelta(days=14)).strftime("%Y-%m-%dT00:00:00Z")
        r=safe_get(f"https://api.congress.gov/v3/bill?format=json&limit=20&fromDateTime={dt}",timeout=12,source_name="CONGRESS")
        if r and r.ok:
            kws=["semiconductor","export control","sanctions","tariff","defense","energy","AI","artificial intelligence","supply chain","critical mineral","cryptocurrency","climate","infrastructure"]
            for b in r.json().get("bills",[])[:15]:
                t=b.get("title","")
                if any(k.lower() in t.lower() for k in kws): bills.append(f"{b.get('type','')}{b.get('number','')}: {t[:120]}")
    except: pass
    return bills[:10] if bills else ["Congress unavailable"]

def fetch_insider():
    try:
        r=safe_get("http://openinsider.com/rss",source_name="INSIDER")
        if r and r.ok:
            trades=[]
            for item in ET.fromstring(r.content).iter("item"):
                t=(item.findtext("title") or "").strip()
                d=re.sub(r"<[^>]+>","",item.findtext("description") or "").strip()[:120]
                if t: trades.append(f"{t}: {d}")
                if len(trades)>=12: break
            return trades
    except: pass
    return ["Insider unavailable"]

def fetch_fii_dii():
    try:
        s=requests.Session();s.headers.update({"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36","Accept":"application/json","Referer":"https://www.nseindia.com/"})
        s.get("https://www.nseindia.com",timeout=10);time.sleep(0.5)
        r=s.get("https://www.nseindia.com/api/fiidiiTradeReact",timeout=12)
        if r.status_code==200: log_source("FII_DII",True); return r.json()[:6] if isinstance(r.json(),list) else {}
    except Exception as e: log_source("FII_DII",False,str(e))
    return {"note":"FII/DII unavailable"}

def fetch_etf_flows():
    if not HAS_YF: return {}
    out={}
    for sym,nm in {"SPY":"SP500","QQQ":"Nasdaq","IWM":"SmallCap","GLD":"Gold","TLT":"LongBond","HYG":"HY","LQD":"IG","EEM":"EM","ITA":"Defense","XLE":"Energy","SOXX":"Semis","GDX":"GoldMiners","XLF":"Fins","XLU":"Utils"}.items():
        try:
            h=yf.Ticker(sym).history(period="10d")
            if len(h)<3: continue
            av,nv=float(h["Volume"].iloc[:-1].mean()),float(h["Volume"].iloc[-1])
            pc=round((float(h["Close"].iloc[-1])-float(h["Close"].iloc[-2]))/float(h["Close"].iloc[-2])*100,2)
            vr=round(nv/av,2) if av>0 else 1.0
            sig="NEUTRAL"
            if vr>1.8 and pc>0.3:sig="ACCUMULATION"
            elif vr>1.8 and pc<-0.3:sig="DISTRIBUTION"
            elif vr>1.8:sig="UNUSUAL"
            out[nm]={"vol_ratio":vr,"chg":pc,"signal":sig}
        except: pass
    log_source("ETF_FLOWS",len(out)>3)
    return out

def fetch_options():
    if not HAS_YF: return {}
    sk={}
    try:
        spy=yf.Ticker("SPY")
        for exp in (spy.options[:2] if spy.options else []):
            ch=spy.option_chain(exp);pv,cv=int(ch.puts["volume"].sum()),int(ch.calls["volume"].sum())
            pcr=round(pv/cv,3) if cv>0 else 0
            sk[f"SPY_{exp}"]={"pcr":pcr,"put_vol":pv,"call_vol":cv,"signal":"HEAVY_PUTS" if pcr>1.3 else ("HEAVY_CALLS" if pcr<0.5 else "NEUTRAL")}
        log_source("OPTIONS",len(sk)>0)
    except Exception as e: log_source("OPTIONS",False,str(e))
    return sk

TICKERS={"SP500":"^GSPC","NASDAQ":"^IXIC","DOW":"^DJI","NIFTY50":"^NSEI","SENSEX":"^BSESN","BANKNIFTY":"^NSEBANK","DAX":"^GDAXI","FTSE":"^FTSE","CAC40":"^FCHI","HANGSENG":"^HSI","NIKKEI":"^N225","KOSPI":"^KS11","BRAZIL":"^BVSP","EM_ETF":"EEM","VIX":"^VIX","VIX9D":"^VIX9D","VVIX":"^VVIX","INDIA_VIX":"^INDIAVIX","US10Y":"^TNX","US2Y":"^TWYX","US30Y":"^TYX","TIP":"TIP","HYG":"HYG","JNK":"JNK","LQD":"LQD","EMB":"EMB","TLT":"TLT","GOLD":"GC=F","SILVER":"SI=F","PLATINUM":"PL=F","COPPER":"HG=F","PALLADIUM":"PA=F","OIL_WTI":"CL=F","OIL_BRENT":"BZ=F","NATGAS":"NG=F","WHEAT":"ZW=F","CORN":"ZC=F","SOYBEAN":"ZS=F","SUGAR":"SB=F","COFFEE":"KC=F","COTTON":"CT=F","LITHIUM":"LIT","URANIUM":"URA","RARE_EARTH":"REMX","COPPER_MINERS":"COPX","WATER":"PHO","STEEL":"SLX","DXY":"DX-Y.NYB","USDINR":"INR=X","EURUSD":"EURUSD=X","USDJPY":"JPY=X","USDCNH":"CNH=X","GBPUSD":"GBPUSD=X","USDTRY":"TRY=X","USDBRL":"BRL=X","USDZAR":"ZAR=X","BITCOIN":"BTC-USD","ETHEREUM":"ETH-USD","SOLANA":"SOL-USD","SEMIS":"SOXX","DEFENSE":"ITA","ENERGY_ETF":"XLE","FINANCIALS":"XLF","HEALTHCARE":"XLV","UTILITIES":"XLU","REALESTATE":"XLRE","CONSUMER_D":"XLY","CONSUMER_S":"XLP","MATERIALS":"XLB","INDUSTRIALS":"XLI","BERKSHIRE":"BRK-B","BLACKROCK":"BLK","GOLD_MINERS":"GDX","BALTIC_DRY":"BDRY","SHORT_BOND":"SHY"}

def fetch_markets():
    if not HAS_YF: return {}
    out={}
    try:
        print(f"  Bulk: {len(TICKERS)} tickers...")
        data=yf.download(list(TICKERS.values()),period="10d",group_by="ticker",progress=False)
        for nm,sym in TICKERS.items():
            try:
                h=data[sym] if sym in data.columns.get_level_values(0) else None
                if h is None or h.empty: out[nm]={"error":"no data"};continue
                h=h.dropna(subset=["Close"])
                if len(h)<2: out[nm]={"error":"insufficient"};continue
                c,p,w=float(h["Close"].iloc[-1]),float(h["Close"].iloc[-2]),float(h["Close"].iloc[0])
                hi,lo=float(h["High"].max()),float(h["Low"].min())
                av=float(h["Volume"].iloc[:-1].mean()) if "Volume" in h.columns else 0
                nv=float(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                out[nm]={"price":round(c,4),"chg_1d":round((c-p)/p*100,3) if p else 0,"chg_10d":round((c-w)/w*100,3) if w else 0,"hi_10d":round(hi,4),"lo_10d":round(lo,4),"vol_ratio":round(nv/av,2) if av>0 else 1.0,"pct_range":round((c-lo)/(hi-lo)*100,1) if hi!=lo else 50}
            except Exception as e: out[nm]={"error":str(e)[:60]}
    except Exception as e:
        print(f"  [ERROR] Bulk fail ({e}), fallback...")
        for nm,sym in TICKERS.items():
            try:
                h=yf.Ticker(sym).history(period="10d")
                if len(h)<2: out[nm]={"error":"no data"};continue
                c,p,w=float(h["Close"].iloc[-1]),float(h["Close"].iloc[-2]),float(h["Close"].iloc[0])
                hi,lo=float(h["High"].max()),float(h["Low"].min())
                av=float(h["Volume"].iloc[:-1].mean()) if "Volume" in h.columns else 0
                nv=float(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                out[nm]={"price":round(c,4),"chg_1d":round((c-p)/p*100,3) if p else 0,"chg_10d":round((c-w)/w*100,3) if w else 0,"hi_10d":round(hi,4),"lo_10d":round(lo,4),"vol_ratio":round(nv/av,2) if av>0 else 1.0,"pct_range":round((c-lo)/(hi-lo)*100,1) if hi!=lo else 50}
            except: out[nm]={"error":"failed"}
            time.sleep(0.05)
    live=len([k for k,v in out.items() if isinstance(v,dict) and "error" not in v])
    log_source("MARKETS",live>20,f"{live}/{len(TICKERS)}")
    return out

def compute_correlations(m):
    pairs=[("GOLD","SP500","Gold/Equity"),("HYG","SP500","Credit/Equity: HYG leads 2-6wk"),("COPPER","SP500","Dr Copper/Equity"),("DXY","GOLD","Dollar/Gold"),("VIX","SP500","Vol/Equity"),("OIL_WTI","USDINR","Oil/INR"),("US10Y","GOLD","Rates/Gold"),("BITCOIN","NASDAQ","Crypto/Tech"),("TLT","SP500","Bonds/Equity"),("COPPER","OIL_WTI","Copper/Oil")]
    res=[]
    for a,b,interp in pairs:
        ca,cb=m.get(a,{}).get("chg_1d"),m.get(b,{}).get("chg_1d")
        if not isinstance(ca,(int,float)) or not isinstance(cb,(int,float)): continue
        same=(ca>0)==(cb>0);div=abs(ca-cb)
        res.append({"pair":f"{a}/{b}","a_chg":ca,"b_chg":cb,"same":same,"div":round(div,3),"interp":interp,"flag":"DIVERGENCE" if not same and div>1.5 else "normal"})
    return res

def detect_anomalies(mkts,fg,corrs):
    fl=[]
    vix=mkts.get("VIX",{}).get("price")
    if vix:
        if vix>35:fl.append(f"VIX {vix}: ACUTE STRESS")
        elif vix>25:fl.append(f"VIX {vix}: ELEVATED")
        elif vix<13:fl.append(f"VIX {vix}: COMPLACENCY")
    fgs=fg.get("score")
    if isinstance(fgs,(int,float)):
        if fgs>80:fl.append(f"F&G {fgs}: EXTREME GREED")
        elif fgs<20:fl.append(f"F&G {fgs}: EXTREME FEAR")
    for c in corrs:
        if c.get("flag")=="DIVERGENCE":fl.append(f"BREAK {c['pair']}: A{c['a_chg']:+.2f}% B{c['b_chg']:+.2f}%")
    for nm in ["HYG","TLT","GLD","SPY","GDX","ITA"]:
        vr=mkts.get(nm,{}).get("vol_ratio")
        if isinstance(vr,(int,float)) and vr>2.2:fl.append(f"{nm} vol {vr}x avg")
    return fl if fl else ["No major anomalies"]

RSS_FEEDS=[("Reuters","https://feeds.reuters.com/reuters/businessNews"),("Reuters World","https://feeds.reuters.com/Reuters/worldNews"),("FT","https://www.ft.com/rss/home/uk"),("IMF","https://www.imf.org/en/News/rss?category=newsglobal"),("NBER","https://www.nber.org/rss/new_working_papers.xml"),("World Bank","https://feeds.worldbank.org/worldbank/news"),("BBC","https://feeds.bbci.co.uk/news/business/rss.xml"),("AJ","https://www.aljazeera.com/xml/rss/all.xml"),("ET","https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),("Wolf St","https://wolfstreet.com/feed/"),("CalcRisk","https://feeds.feedburner.com/CalculatedRisk"),("Mint","https://www.livemint.com/rss/markets"),("ZH","https://feeds.feedburner.com/zerohedge/feed"),("NakedCap","https://www.nakedcapitalism.com/feed"),("CFR","https://www.cfr.org/rss/global"),("FP","https://foreignpolicy.com/feed/"),("Def1","https://www.defenseone.com/rss/")]

def fetch_news():
    items=[];ok=0
    for src,url in RSS_FEEDS:
        try:
            r=requests.get(url,headers=UA,timeout=10)
            if not r.ok: continue
            ok+=1
            for item in ET.fromstring(r.content).iter("item"):
                t=(item.findtext("title") or "").strip();d=re.sub(r"<[^>]+>","",item.findtext("description") or "").strip()[:150]
                hd=bool(re.search(r'\d+\.?\d*%|\$\d+|\d+\s*(billion|million|trillion|bps)',f"{t} {d}",re.I))
                items.append({"src":src,"title":t[:120],"desc":d,"data_rich":hd})
        except: pass
    log_source("NEWS",ok>5,f"{ok}/{len(RSS_FEEDS)}")
    dr=[i for i in items if i["data_rich"]]; nr=[i for i in items if not i["data_rich"]]
    out="[DATA-RICH]\n"+"\n".join(f"[{i['src']}] {i['title']}: {i['desc']}" for i in dr[:40])
    out+="\n[NARRATIVE]\n"+"\n".join(f"[{i['src']}] {i['title']}" for i in nr[:20])
    return out[:14000]

def fetch_gdelt():
    try:
        r=safe_get("https://api.gdeltproject.org/api/v2/doc/doc?query=economy+sanctions+central+bank+military+geopolitical&mode=artlist&maxrecords=20&format=json&timespan=12h",source_name="GDELT")
        if r and r.ok: return "\n".join(f"[{a.get('sourcecountry','?')}] {a.get('title','')}" for a in r.json().get("articles",[])[:15])
    except: pass
    return "GDELT unavailable"

def fetch_macro_cal():
    try:
        r=safe_get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",source_name="MACRO_CAL")
        if r and r.ok: return "\n".join(f"{e.get('date','')}|{e.get('country','')}|{e.get('title','')}|{e.get('impact','')}" for e in r.json()[:40] if e.get("impact") in ["High","Medium"])
    except: pass
    return "Calendar unavailable"

SUBS=["wallstreetbets","investing","economics","geopolitics","IndiaInvestments","stocks","worldnews","MacroEconomics","finance","Commodities","CryptoMarkets","energy","GlobalPowers","CredibleDefense","IndiaStock","collapse"]
def fetch_reddit():
    posts=[]
    for sub in SUBS:
        try:
            r=requests.get(f"https://www.reddit.com/r/{sub}/hot.json?limit=4",headers={"User-Agent":"NeuralCortex/5.1"},timeout=10)
            if r.ok:
                for p in r.json()["data"]["children"]:posts.append(f"r/{sub}[{p['data']['score']}] {p['data']['title']}")
        except: pass
    return "\n".join(posts[:50])

def fetch_fear_greed():
    try:
        r=safe_get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",source_name="FG")
        if r and r.ok:
            d=r.json()["fear_and_greed"]
            return {"score":round(d["score"],1),"rating":d["rating"],"prev_week":round(d.get("previous_1_week",0),1),"prev_month":round(d.get("previous_1_month",0),1)}
    except: pass
    return {"score":"?","rating":"unavailable"}

PATTERN_LIBRARY=[{"name":"Mar/Apr Dump-Recover","trigger_months":[3,4],"record":"7/10 years","data":"VIX>25+HYG holds=manufactured","india":"Nifty -5-10% sympathy"},{"name":"JPY Carry Unwind","record":"2007,2015,2024","data":"USDJPY<142","india":"FII out, Nifty -8-15%"},{"name":"Credit Leads Equity","record":"HYG diverges 2-6wk before","data":"HYG vol surge no news=forced selling","india":"NBFC stress in FD/CP"},{"name":"Dollar Wrecking Ball","record":"DXY>105=EM stress 3-6mo","data":"DXY>104 sustained","india":"DXY>107=INR pressure"},{"name":"Fed Liquidity","record":"TGA+RRP drain=injection 2-6wk lag","data":"Both declining=pump","india":"Dollar liq->EM inflows->Nifty"},{"name":"Commodity-CPI Lag","record":"Oil->CPI 3-4mo","data":"Brent 3mo trend","india":"Crude>$90=RBI delayed"},{"name":"Options Tell","record":"Large PUT 2-4wk before events","data":"P/C>1.2, VVIX spike","india":"NSE FII F&O daily"},{"name":"Minsky Moment","record":"Low-vol+leverage->cascade","data":"VIX<14 90d, HY>500bps","india":"Global risk-off hits FII"},{"name":"AI Capex Cascade","record":"Hyperscaler $200B+","data":"TSMC, HBM, DC REITs","india":"IT services layer"},{"name":"De-Dollar","record":"USD reserve 73%->58%","data":"COFER, gold % reserves","india":"RBI gold buying"}]
def get_patterns(now):
    m=now.month;active=[];always=[]
    for p in PATTERN_LIBRARY:
        if "trigger_months" in p and m in p["trigger_months"]:active.append(p)
        else:always.append(p)
    return active+always


# ══════════════════════════════════════════════════════════════════
# 6-AGENT SYSTEM — ADVERSARIAL + STRUCTURED REASONING
# ══════════════════════════════════════════════════════════════════

def groq_call(messages, max_tokens=4000, temperature=0.7, retries=2):
    if not GROQ_KEY: return "ERROR: GROQ_API_KEY not set"
    for attempt in range(retries+1):
        try:
            r=requests.post(GROQ_URL,headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"},json={"model":GROQ_MODEL,"messages":messages,"max_tokens":max_tokens,"temperature":temperature},timeout=120)
            if r.ok: return r.json()["choices"][0]["message"]["content"]
            print(f"  [WARN] Groq #{attempt+1}: {r.status_code}")
            if attempt<retries:time.sleep(3)
        except Exception as e:
            print(f"  [WARN] Groq #{attempt+1}: {e}")
            if attempt<retries:time.sleep(3)
    return "LLM ERROR"

REASONING_FORMAT = """
USE THIS FORMAT for each analytical point:
OBSERVE: [raw data point you're reacting to]
LOGIC: [interpretation] -> [consequence] -> [implication]
COUNTER: [best argument against your conclusion]
CONCLUDE: [your call]
DEPENDS ON: [the single data point that would change your mind]
WRONG IF: [specific falsification condition]

This format is MANDATORY. Your reasoning will be logged and scored against reality."""

def run_agents(all_data, brain_output, recon_text, agent_scores):
    ts=all_data["ts"]; bs=brain_output["brain_summary"]; regime=bs["regime"]

    # Brain context for all agents
    pred_lines="\n".join(f"  {p['asset']}({p['timeframe']}): {p['call']} | {p['distribution']}" for p in bs.get("active_predictions",[])[:8])
    brain_ctx=f"BRAIN: Regime={regime} ({bs['regime_confidence']:.0%})\n{pred_lines}\nBrier: {bs.get('brier_overall','building')} | Patterns: {brain_output.get('pattern_memory_size',0)}"

    pats="\n".join(f"PATTERN: {p['name']}: {p['record']} | {p['data']}" for p in all_data["patterns"][:6])

    # ── Agent 1: Upstream ──
    print("    [1/6] Upstream...")
    a1_ctx=build_agent_context(agent_scores,"upstream",regime)
    a1=groq_call([{"role":"system","content":f"You are the UPSTREAM ANALYST. Tier 1 only. You may OVERRIDE the brain's regime if your data contradicts it. Challenge everything.\n{REASONING_FORMAT}\n{a1_ctx}"},
        {"role":"user","content":f"{brain_ctx}\n{recon_text}\n\n[WARN] {chr(10).join(all_data['warn'][:10])}\n[OFAC] {chr(10).join(all_data['ofac'][:8])}\n[CONTRACTS] {chr(10).join(all_data['contracts'][:8])}\n[PATENTS] {chr(10).join(all_data['patents'][:8])}\n[PBOC] {json.dumps(all_data['pboc'])}\n[SHIPPING] {json.dumps(all_data['shipping'])[:800]}\n[LEADERS] {chr(10).join(all_data['leaders'][:15])}\n[POLICY] {chr(10).join(all_data['policy'][:10])}"}],max_tokens=1800)

    # ── Agent 2: Flow ──
    print("    [2/6] Flow...")
    a2_ctx=build_agent_context(agent_scores,"flow",regime)
    a2=groq_call([{"role":"system","content":f"You are the FLOW ANALYST. Tier 2. You see Upstream's output - CHALLENGE it where flows contradict. If the brain's probabilities don't match what flows show, say so.\n{REASONING_FORMAT}\n{a2_ctx}"},
        {"role":"user","content":f"{brain_ctx}\n{recon_text}\nUPSTREAM SAID:\n{a1[:1500]}\n\n[FED LIQ] {json.dumps(all_data['fed_liq'],indent=1)[:1500]}\n[SOFR] {json.dumps(all_data['sofr'])[:800]}\n[ETF] {json.dumps(all_data['etf_flows'],indent=1)[:1200]}\n[OPTIONS] {json.dumps(all_data['options'])[:800]}\n[INSIDER] {chr(10).join(all_data['insider'][:10])}\n[FII/DII] {json.dumps(all_data['fii_dii'])[:600]}\n[LEGISLATIVE] {chr(10).join(all_data['legislative'][:8])}"}],max_tokens=1800)

    # ── Agent 3: Market ──
    print("    [3/6] Market...")
    a3_ctx=build_agent_context(agent_scores,"market",regime)
    mkt_sum=[f"{nm}:{d.get('price',0)} 1d:{d.get('chg_1d',0):+.2f}% 10d:{d.get('chg_10d',0):+.2f}% vol:{d.get('vol_ratio',1)}x" for nm,d in all_data['markets'].items() if isinstance(d,dict) and "error" not in d]
    corr_lines = []
    for _c in all_data['correlations']:
        _flag = ' ***BREAK***' if _c.get('flag') == 'DIVERGENCE' else ''
        corr_lines.append('{}: A={:+.2f}% B={:+.2f}% {} {}'.format(
            _c.get('pair',''), _c.get('a_chg',0), _c.get('b_chg',0), _c.get('interp',''), _flag).strip())
    corr_text = '\n'.join(corr_lines)
    a3=groq_call([{"role":"system","content":f"You are the MARKET ANALYST. Tier 3 + brain probabilities. You have VETO POWER on brain probabilities. CHALLENGE Upstream and Flow where price action disagrees.\n{REASONING_FORMAT}\n{a3_ctx}"},
        {"role":"user","content":f"{brain_ctx}\n{recon_text}\nUPSTREAM: {a1[:800]}\nFLOW: {a2[:800]}\n\nMARKETS:\n{chr(10).join(mkt_sum)}\n\nCORRELATIONS:\n{corr_text}\n\nANOMALIES:\n{chr(10).join(all_data['anomalies'])}\n\nF&G: {json.dumps(all_data['fg'])}"}],max_tokens=1800)

    # ── Agent 4: Narrative ──
    print("    [4/6] Narrative...")
    a4_ctx=build_agent_context(agent_scores,"narrative",regime)
    a4=groq_call([{"role":"system","content":f"You are the NARRATIVE ANALYST. Map what the CROWD believes (=priced). Flag where Agents 1-3 FOLLOW THE HERD. That's not edge, that's consensus.\n{REASONING_FORMAT}\n{a4_ctx}"},
        {"role":"user","content":f"{brain_ctx}\n{recon_text}\nAGENTS 1-3:\nUpstream:{a1[:600]}\nFlow:{a2[:600]}\nMarket:{a3[:600]}\n\nNEWS:\n{all_data['news'][:4000]}\nGDELT:\n{all_data['gdelt'][:1000]}\nCALENDAR:\n{all_data['macro_cal']}\nREDDIT:\n{all_data['reddit'][:1500]}"}],max_tokens=1800)

    # ── Agent 5: Synthesizer ──
    print("    [5/6] Synthesizer...")
    a5=groq_call([{"role":"system","content":f"""You are the SYNTHESIZER. You see ALL agents + brain + reasoning history.

1. RECONCILE contradictions between agents. Name which agent you side with and WHY.
2. Generate THESES using this format:
   THESIS: [title]
   CHAIN: [signal] -> [mechanism] -> [consequence] -> [implication]
   CONFIDENCE: [0-100]
   KILL IF: [specific condition]
   TIMEFRAME: [days]
   ASSET: [ticker]
3. INDIA INTELLIGENCE: INR, Nifty sectors, FII/DII read, RBI, risks, opportunities
4. THE UNCOMFORTABLE TRUTH (one paragraph)
5. WHAT TO WATCH: 5 specific "if X then Y" conditions

{REASONING_FORMAT}"""},
        {"role":"user","content":f"Timestamp: {ts}\n\n[UPSTREAM]\n{a1}\n\n[FLOW]\n{a2}\n\n[MARKET]\n{a3}\n\n[NARRATIVE]\n{a4}\n\n{brain_ctx}\n\n{recon_text}\n\nPATTERNS:\n{pats}"}],max_tokens=6000,temperature=0.72)

    # ── Agent 6: Adversary ──
    print("    [6/6] Adversary...")
    a6=groq_call([{"role":"system","content":f"You are the ADVERSARY. DESTROY the Synthesizer's output. Attack theses, regime, probabilities, signal weights. Name the WEAKEST THESIS and WHY. Name the BIGGEST BLIND SPOT.\n{REASONING_FORMAT}"},
        {"role":"user","content":f"Attack:\n{a5[:6000]}\n\nBrain:{brain_ctx}"}],max_tokens=2000)

    return {"upstream":a1,"flow":a2,"market":a3,"narrative":a4,"synthesis":a5,"adversary":a6}


# ══════════════════════════════════════════════════════════════════
# DASHBOARD RENDERER (compact but complete)
# ══════════════════════════════════════════════════════════════════

def md_to_html(t):
    if not t: return "<p>--</p>"
    t=re.sub(r"^## (.+)$",r'<h2>\1</h2>',t,flags=re.MULTILINE);t=re.sub(r"^### (.+)$",r'<h3>\1</h3>',t,flags=re.MULTILINE)
    t=re.sub(r"\*\*(.+?)\*\*",r"<strong>\1</strong>",t);t=re.sub(r"\*(.+?)\*",r"<em>\1</em>",t)
    t=re.sub(r"^[-] (.+)$",r"<li>\1</li>",t,flags=re.MULTILINE)
    t=re.sub(r"^(?:OBSERVE|LOGIC|COUNTER|CONCLUDE|DEPENDS|WRONG IF)[:\s]",r"<strong>\g<0></strong>",t,flags=re.MULTILINE)
    return f"<p>{t.replace(chr(10)+chr(10),'</p><p>').replace(chr(10),'<br>')}</p>"

def render_html(ts,mkts,fg,anoms,corrs,brain_out,agents,reasoning_out,patterns,hist_len):
    bs=brain_out["brain_summary"];regime=bs["regime"];rc=bs["regime_confidence"]
    brier=brain_out.get("brier_data",{});bv=brier.get("overall_brier");bstr=f"{bv:.4f}" if bv else "building..."
    bcol="#00e676" if bv and bv<0.2 else "#ffd740" if bv and bv<0.3 else "#ff4444" if bv else "var(--d)"
    rgcol={"CRISIS":"#ff4444","RISK_OFF":"#ff8888","TRANSITION":"#ffd740","RISK_ON":"#00e676","EUPHORIA":"#a78bfa"}.get(regime,"#ffd740")
    GROUPS=[("EQUITIES",["SP500","NASDAQ","DOW","NIFTY50","SENSEX","BANKNIFTY","DAX","FTSE","HANGSENG","NIKKEI"]),("VOL",["VIX","VIX9D","VVIX","INDIA_VIX"]),("BONDS",["US10Y","US2Y","US30Y","TLT","HYG","JNK","LQD","EMB"]),("METALS",["GOLD","SILVER","PLATINUM","COPPER"]),("ENERGY",["OIL_WTI","OIL_BRENT","NATGAS"]),("AGRICULTURE",["WHEAT","CORN","SOYBEAN","SUGAR","COFFEE"]),("STRATEGIC",["LITHIUM","URANIUM","RARE_EARTH","STEEL"]),("FX",["DXY","USDINR","EURUSD","USDJPY","USDCNH","GBPUSD"]),("CRYPTO",["BITCOIN","ETHEREUM","SOLANA"]),("SECTORS",["SEMIS","DEFENSE","ENERGY_ETF","FINANCIALS","HEALTHCARE","UTILITIES","MATERIALS","INDUSTRIALS"])]
    cards=""
    for gn,keys in GROUPS:
        gc=""
        for k in keys:
            d=mkts.get(k,{})
            if not isinstance(d,dict) or "error" in d or d.get("price") is None: continue
            pr,c1,vr=d["price"],d.get("chg_1d",0),d.get("vol_ratio",1)
            col="#00e676" if c1>=0 else "#ff4444";ps=f"{pr:,.4f}".rstrip("0").rstrip(".") if pr<100 else f"{pr:,.2f}"
            vf=f'<span style="font-size:7px;color:#ffd740">{vr}x</span>' if vr>1.8 else ""
            gc+=f'<div class="cd"><div class="cl">{k}</div><div class="cv">{ps}</div><div class="cc" style="color:{col}">{"&#9650;" if c1>=0 else "&#9660;"}{abs(c1):.2f}%{vf}</div></div>'
        if gc:cards+=f'<div class="mg"><div class="ml">{gn}</div><div class="gr">{gc}</div></div>'
    # Prob dist cards
    phtml=""
    for pred in brain_out.get("predictions_new",[])[:12]:
        dist=pred.get("distribution",{});call=pred.get("call",{})
        dc="#00e676" if call.get("direction")=="UP" else "#ff4444" if call.get("direction")=="DOWN" else "#ffd740"
        bars=""
        for bk in ["down_big","down","flat","up","up_big"]:
            pct=dist.get(bk,0);w=max(2,int(pct*100));bc={"down_big":"#ff4444","down":"#ff8888","flat":"#666","up":"#88ff88","up_big":"#00e676"}[bk];bl={"down_big":"--","down":"-","flat":"=","up":"+","up_big":"++"}[bk]
            bars+=f'<div style="display:flex;gap:4px;margin:1px 0"><span style="font-size:7px;width:16px;color:var(--d)">{bl}</span><div style="height:8px;width:{w}px;background:{bc};border-radius:1px"></div><span style="font-size:8px;color:var(--d)">{pct:.0%}</span></div>'
        phtml+=f'<div class="pc"><div style="display:flex;justify-content:space-between;margin-bottom:4px"><span style="font-size:12px;font-weight:600;color:var(--hi)">{pred["asset"]}</span><span style="font-size:9px;color:var(--d)">{pred["timeframe_days"]}d</span></div><div style="font-size:16px;font-weight:700;color:{dc};margin-bottom:4px">{call.get("direction","?")} <span style="font-size:10px;font-weight:400;color:var(--d)">{call.get("conviction","")}</span></div>{bars}</div>'
    # Agent scores
    aschtml=""
    for an in ["upstream","flow","market","narrative","synthesis"]:
        ad=reasoning_out.get("agent_scores",{}).get("agents",{}).get(an,{})
        t=ad.get("total",0);c=ad.get("correct",0);acc=round(c/t*100,1) if t>0 else None
        col="#00e676" if acc and acc>55 else "#ff4444" if acc and acc<45 else "#ffd740" if acc else "var(--d)"
        aschtml+=f'<span style="font-size:9px;margin-right:12px">{an}: <span style="color:{col}">{f"{acc}%" if acc else "..."}</span> ({t})</span>'
    # Regime transition
    trans=reasoning_out.get("regime_transition",{})
    trans_risk=trans.get("transition_risk",0)
    trans_html=""
    if trans_risk>0.3:
        tcol="#ff4444" if trans_risk>0.6 else "#ffd740"
        trans_html=f'<div style="background:#1a0a0a;border:1px solid {tcol};padding:10px;margin-bottom:12px;border-radius:2px;font-size:10px;color:{tcol}">REGIME TRANSITION RISK: {trans_risk:.0%} {trans.get("detail","")}</div>'
    # TF flags
    tf_flags=reasoning_out.get("timeframe_flags",[])
    tf_html=""
    if tf_flags:
        tf_html='<div style="background:var(--s1);border:1px solid var(--amber);padding:10px;margin-bottom:12px;border-radius:2px">'
        for f in tf_flags[:4]: tf_html+=f'<div style="font-size:10px;color:var(--amber)">{f.get("type","")}: {f.get("detail","")}</div>'
        tf_html+='</div>'
    # Meta failures
    meta=reasoning_out.get("meta_failures",[])
    meta_html=""
    if meta:
        meta_html='<div style="background:#0a0a1a;border:1px solid var(--violet);padding:10px;margin-bottom:12px;border-radius:2px"><div style="font-size:7px;font-weight:700;letter-spacing:.2em;color:var(--violet);margin-bottom:6px">META-LEARNING: SYSTEM FAILURE PATTERNS</div>'
        for m in meta[:5]: meta_html+=f'<div style="font-size:10px;color:var(--tx);margin-bottom:3px">{m.get("detail","")}</div>'
        meta_html+='</div>'
    ts_=len(SOURCE_HEALTH);hs_=sum(1 for s in SOURCE_HEALTH.values() if s["ok"]);hp_=round(hs_/ts_*100) if ts_ else 0
    hcol="#00e676" if hp_>75 else "#ffd740" if hp_>50 else "#ff4444"
    fgv=fg.get("score","?");fgc="#ff4444" if isinstance(fgv,(int,float)) and fgv<25 else "#00e676" if isinstance(fgv,(int,float)) and fgv>75 else "#ffd740"
    u10=mkts.get("US10Y",{}).get("price");u2=mkts.get("US2Y",{}).get("price");yc=round(u10-u2,3) if u10 and u2 else "--";ycc="#ff4444" if isinstance(yc,float) and yc<0 else "#00e676"
    anhtml="".join(f'<div style="font-size:10px;line-height:1.6;padding:2px 0;border-bottom:.5px solid var(--bd)">{a}</div>' for a in anoms)
    crhtml="".join(f'<div style="display:flex;gap:6px;font-size:9px;padding:3px 5px;border:.5px solid {"#ff4444" if c.get("flag")=="DIVERGENCE" else "#333"};margin-bottom:3px;border-radius:2px"><span style="font-size:8px;font-weight:600;color:var(--hi);min-width:75px">{c["pair"]}</span><span style="color:{"#00e676" if c["a_chg"]>=0 else "#ff4444"}">{c["a_chg"]:+.2f}%</span> vs <span style="color:{"#00e676" if c["b_chg"]>=0 else "#ff4444"}">{c["b_chg"]:+.2f}%</span>{"<span style=font-size:7px;color:var(--r);margin-left:auto>BREAK</span>" if c.get("flag")=="DIVERGENCE" else ""}</div>' for c in corrs)
    synthesis=agents.get("synthesis","");adversary=agents.get("adversary","")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><meta http-equiv="refresh" content="21600"><title>NEURAL CORTEX v5.1 -- {ts}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0}}:root{{--bg:#06060e;--s1:#0b0b1a;--s2:#101020;--bd:#16162a;--bd2:#20203a;--tx:#9090b0;--d:#44445a;--hi:#e8e8f8;--g:#00e676;--r:#ff4444;--amber:#ffd740;--blue:#40a9ff;--violet:#a78bfa;--teal:#2dd4bf}}body{{background:var(--bg);color:var(--tx);font-family:'IBM Plex Mono',monospace;font-size:12px}}header{{border-bottom:1px solid var(--bd);padding:14px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;background:var(--s1)}}.logo{{font-size:12px;font-weight:600;letter-spacing:.2em;color:var(--hi);display:flex;align-items:center;gap:8px}}.pulse{{width:7px;height:7px;border-radius:50%;background:var(--g);animation:gp 2s ease-in-out infinite}}@keyframes gp{{0%,100%{{opacity:1;box-shadow:0 0 8px var(--g)}}50%{{opacity:.15}}}}main{{max-width:1400px;margin:0 auto;padding:24px 18px}}.mg{{margin-bottom:10px}}.ml{{font-size:7px;font-weight:600;letter-spacing:.2em;color:var(--d);margin-bottom:4px}}.gr{{display:grid;grid-template-columns:repeat(auto-fill,minmax(105px,1fr));gap:4px}}.cd{{background:var(--s1);border:1px solid var(--bd);border-radius:2px;padding:8px 7px 6px}}.cd:hover{{background:var(--s2)}}.cl{{font-size:7px;letter-spacing:.12em;color:var(--d);margin-bottom:2px}}.cv{{font-size:12px;font-weight:600;color:var(--hi);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.cc{{font-size:9px;font-weight:600}}.rb{{background:linear-gradient(135deg,#0d0d20,#14142e);border:2px solid {rgcol};border-radius:2px;padding:20px;margin-bottom:14px;text-align:center}}.pg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-bottom:14px}}.pc{{background:var(--s1);border:1px solid var(--bd);border-radius:2px;padding:12px}}.r3{{display:grid;grid-template-columns:180px 1fr 240px;gap:12px;margin-bottom:14px;align-items:start}}@media(max-width:900px){{.r3{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.r3{{grid-template-columns:1fr}}}}.fg{{background:var(--s1);border:1px solid var(--bd);border-radius:2px;padding:14px;text-align:center}}.ab{{background:var(--s1);border:1px solid var(--bd);border-radius:2px;padding:12px;margin-bottom:8px}}.al{{font-size:7px;font-weight:700;letter-spacing:.2em;color:var(--d);margin-bottom:6px}}.sec{{background:var(--s1);border:1px solid var(--bd);border-radius:2px;padding:18px;margin-bottom:14px}}.sec h2{{font-size:8px;font-weight:700;letter-spacing:.2em;color:var(--violet);text-transform:uppercase;margin:20px 0 8px;border-bottom:.5px solid var(--bd);padding-bottom:3px}}.sec h2:first-child{{margin-top:0}}.sec h3{{font-family:'IBM Plex Sans',sans-serif;font-size:13px;font-weight:600;color:var(--hi);margin:10px 0 4px}}.sec p,.sec li{{font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;line-height:1.85;color:var(--tx)}}.sec strong{{color:var(--hi)}}.sec em{{color:var(--blue)}}.sec li{{margin-left:18px;list-style:disc}}.meta{{margin-top:18px;padding:12px;background:var(--s1);border:1px solid var(--bd);border-radius:2px;display:flex;flex-wrap:wrap;gap:14px}}.mi{{font-size:8px;color:var(--d)}}.mi span{{color:var(--tx)}}footer{{text-align:center;padding:20px;font-size:7px;letter-spacing:.14em;color:var(--d);border-top:1px solid var(--bd);margin-top:24px}}.hb{{height:4px;border-radius:2px;background:var(--bd);margin-top:3px;overflow:hidden}}.hf{{height:100%;border-radius:2px}}</style></head><body>
<header><div class="logo"><div class="pulse"></div>NEURAL CORTEX <span style="font-size:8px;color:var(--d);font-weight:300;margin-left:4px">v5.1 INTELLIGENCE BRAIN</span></div><div style="font-size:8px;color:var(--d);text-align:right">{ts} UTC | <span style="color:{hcol}">{hs_}/{ts_} ({hp_}%)</span><div class="hb"><div class="hf" style="width:{hp_}%;background:{hcol}"></div></div></div></header>
<main>
<div class="rb"><div style="font-size:8px;letter-spacing:.3em;color:var(--d);margin-bottom:6px">MECHANICAL REGIME</div><div style="font-size:28px;font-weight:700;color:{rgcol}">{regime}</div><div style="font-size:11px;color:var(--d);margin-top:4px">{rc:.0%} conf | Brier: <span style="color:{bcol}">{bstr}</span> | Scored: {brier.get("total_scored",0)} | Patterns: {brain_out.get("pattern_memory_size",0)} | Reasoning chains: {reasoning_out.get("reasoning_db_size",0)}</div><div style="font-size:9px;color:var(--d);margin-top:6px">AGENT SCORES: {aschtml}</div></div>
{trans_html}{tf_html}{meta_html}
{cards}
<div style="font-size:7px;font-weight:700;letter-spacing:.2em;color:var(--teal);margin:14px 0 8px">PROBABILITY DISTRIBUTIONS (mechanical brain)</div>
<div class="pg">{phtml}</div>
<div style="background:var(--s1);border:1px solid var(--bd);padding:12px;margin-bottom:12px;border-radius:2px;text-align:center"><div style="font-size:7px;letter-spacing:.2em;color:var(--d);margin-bottom:3px">YIELD CURVE (10Y-2Y)</div><div style="font-size:24px;font-weight:700;color:{ycc}">{yc}%</div></div>
<div class="r3"><div class="fg"><div style="font-size:7px;letter-spacing:.2em;color:var(--d);margin-bottom:4px">FEAR &amp; GREED</div><div style="font-size:38px;font-weight:700;color:{fgc}">{fgv}</div><div style="font-size:9px;letter-spacing:.12em;color:{fgc}">{str(fg.get("rating","")).upper()}</div></div><div><div class="ab"><div class="al">ANOMALIES</div>{anhtml}</div></div><div><div class="ab"><div class="al">CORRELATIONS</div>{crhtml}</div></div></div>
<div class="sec" style="border-color:var(--g)"><div style="font-size:7px;font-weight:700;letter-spacing:.24em;color:var(--g);margin-bottom:8px">INTELLIGENCE SYNTHESIS (6-agent adversarial + structured reasoning)</div>{md_to_html(synthesis)}</div>
<div class="sec" style="border-color:var(--r)"><div style="font-size:7px;font-weight:700;letter-spacing:.24em;color:var(--r);margin-bottom:8px">RED TEAM</div><div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;line-height:1.85">{md_to_html(adversary)}</div></div>
<div class="meta">
<div class="mi">INSTRUMENTS <span>{len([k for k,v in mkts.items() if isinstance(v,dict) and "error" not in v])}</span></div>
<div class="mi">REGIME <span style="color:{rgcol}">{regime} ({rc:.0%})</span></div>
<div class="mi">BRIER <span style="color:{bcol}">{bstr}</span></div>
<div class="mi">PREDICTIONS <span>{len(brain_out.get("predictions_active",[]))}</span></div>
<div class="mi">PATTERNS <span>{brain_out.get("pattern_memory_size",0)}</span></div>
<div class="mi">REASONING <span>{reasoning_out.get("reasoning_db_size",0)} chains</span></div>
<div class="mi">DISAGREEMENTS <span>{reasoning_out.get("disagreements_found",0)}</span></div>
<div class="mi">HISTORY <span>{hist_len}</span></div>
</div></main>
<footer>NEURAL CORTEX v5.1 -- MECHANICAL BRAIN + REASONING ENGINE + 6-AGENT INTELLIGENCE -- BRIER CALIBRATION -- PATTERN LEARNING -- NOT FINANCIAL ADVICE</footer></body></html>"""


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    now=datetime.datetime.utcnow();ts=now.strftime("%Y-%m-%d %H:%M")
    print("="*60);print(f"NEURAL CORTEX v5.1 PRODUCTION -- {ts} UTC");print("Brain thinks. Agents argue. Reasoning learns.");print("="*60)

    history=load_json(f"{DATA_DIR}/history.json",[])

    print("\n[1/8] Tier 1: Upstream...")
    warn=fetch_warn_act();ofac=fetch_ofac();contracts=fetch_contracts();patents=fetch_patents()
    pboc=fetch_pboc();shipping=fetch_shipping();leaders=fetch_leaders();policy=fetch_policy()

    print("\n[2/8] Tier 2: Flows...")
    fed_liq=fetch_fed_liquidity();sofr=fetch_sofr();legislative=fetch_legislative()
    insider=fetch_insider();fii_dii=fetch_fii_dii();etf_flows=fetch_etf_flows();options=fetch_options()

    print("\n[3/8] Tier 3: Markets...")
    markets=fetch_markets();corrs=compute_correlations(markets);fg=fetch_fear_greed()
    anomalies=detect_anomalies(markets,fg,corrs)
    live=len([k for k,v in markets.items() if isinstance(v,dict) and "error" not in v])
    print(f"  {live} instruments")

    print("\n[4/8] Tier 4+5: News+Social...")
    news=fetch_news();gdelt=fetch_gdelt();macro_cal=fetch_macro_cal();reddit=fetch_reddit()
    patterns=get_patterns(now)

    print("\n[5/8] Mechanical brain...")
    brain_out=run_brain(markets,fed_liq,etf_flows,options,fg,corrs,anomalies,fii_dii)

    print("\n[6/8] Reasoning reconciliation...")
    recon_text,agent_scores,reasoning_db=get_pre_agent_context(markets,{"fed_liq":fed_liq,"etf_flows":etf_flows})

    print("\n[7/8] 6-agent synthesis...")
    all_data={"ts":ts,"markets":markets,"correlations":corrs,"anomalies":anomalies,"fg":fg,
        "warn":warn,"ofac":ofac,"contracts":contracts,"patents":patents,"pboc":pboc,"shipping":shipping,
        "leaders":leaders,"policy":policy,"fed_liq":fed_liq,"sofr":sofr,"legislative":legislative,
        "insider":insider,"fii_dii":fii_dii,"etf_flows":etf_flows,"options":options,
        "news":news,"gdelt":gdelt,"macro_cal":macro_cal,"reddit":reddit,"patterns":patterns}
    agents=run_agents(all_data,brain_out,recon_text,agent_scores)

    print("\n[8/8] Reasoning extraction + logging...")
    reasoning_out=run_reasoning_pass(agents,brain_out,markets,all_data)
    print(f"  Chains:{reasoning_out['reasoning_chains']} Disagreements:{reasoning_out['disagreements_found']} Theses:{len(reasoning_out['theses_extracted'])}")

    # Log EVERYTHING
    raw_log={"timestamp":ts,"regime":brain_out.get("regime",{}),"tier1":{"warn":warn,"ofac":ofac,"contracts":contracts,"patents":patents,"pboc":pboc,"shipping":shipping,"leaders":leaders,"policy":policy},"tier2":{"fed_liq":fed_liq,"sofr":sofr,"legislative":legislative,"insider":insider,"fii_dii":fii_dii,"etf_flows":etf_flows,"options":options},"tier3":{"markets_live":live,"correlations":corrs,"anomalies":anomalies},"tier4":{"news_len":len(news),"gdelt_len":len(gdelt)},"tier5":{"reddit_posts":len(reddit.splitlines()),"fg":fg},"brain":{"predictions":len(brain_out.get("predictions_active",[])),"brier":brain_out.get("brier_data",{}).get("overall_brier"),"patterns":brain_out.get("pattern_memory_size",0)},"agents":{k:v[:2000] for k,v in agents.items()},"reasoning":{"chains":reasoning_out["reasoning_chains"],"disagreements":reasoning_out["disagreements_found"],"meta_failures":reasoning_out.get("meta_failures",[])},"source_health":SOURCE_HEALTH}
    save_raw_log(ts,raw_log)

    # Log theses
    for th in reasoning_out.get("theses_extracted",[]):
        save_json(f"{DATA_DIR}/thesis_log.json",load_json(f"{DATA_DIR}/thesis_log.json",[])[-500:]+[th])

    entry={"timestamp":ts,"markets":markets,"fear_greed":fg,"regime":brain_out.get("regime",{})}
    history.append(entry);hist_len=len(history)
    save_json(f"{DATA_DIR}/history.json",history[-150:])
    save_json(f"{DATA_DIR}/source_health.json",SOURCE_HEALTH)

    html=render_html(ts,markets,fg,anomalies,corrs,brain_out,agents,reasoning_out,patterns,hist_len)
    with open("index.html","w",encoding="utf-8") as f: f.write(html)

    ts_=len(SOURCE_HEALTH);hs_=sum(1 for s in SOURCE_HEALTH.values() if s["ok"])
    print(f"\n{'='*60}")
    print(f"DONE. {len(html):,} bytes | {hist_len} runs")
    print(f"  Regime: {brain_out.get('regime',{}).get('regime','?')} ({brain_out.get('regime',{}).get('confidence',0):.0%})")
    print(f"  Brier: {brain_out.get('brier_data',{}).get('overall_brier','building')}")
    print(f"  Predictions: {len(brain_out.get('predictions_active',[]))} | Patterns: {brain_out.get('pattern_memory_size',0)}")
    print(f"  Reasoning: {reasoning_out['reasoning_chains']} chains | {reasoning_out['disagreements_found']} disagreements")
    print(f"  Meta-failures: {len(reasoning_out.get('meta_failures',[]))}")
    print(f"  Sources: {hs_}/{ts_} | Raw log saved")
    print("="*60)

if __name__=="__main__":
    main()
