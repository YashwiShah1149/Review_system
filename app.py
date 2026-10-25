import html as htmllib
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ReviewPulse", page_icon="📊", layout="wide")

CATS = ["Refunds, cancellations & order problems", "App problems & general complaints",
        "Customer service & support", "Delivery", "Pricing & charges", "Food quality"]
# The data keeps the original cluster name; the app shows a name that matches what the reviews say.
DISPLAY = {CATS[1]: "General complaints & switching apps"}


def nice(c):
    return DISPLAY.get(c, c)


SHORT = {"Refunds & orders": CATS[0], "General & switching": CATS[1], "Support": CATS[2],
         "Delivery": CATS[3], "Pricing": CATS[4], "Food": CATS[5]}
CAT_INFO = {
    CATS[0]: "Refunds not received, cancelled orders, wrong or missing items.",
    CATS[1]: "Angry reviews that do not point at one specific problem (worst food, worst delivery, worst "
             "application) and people saying they will stop using Swiggy or switch to Zomato. A few mention the AI "
             "chat bot, bugs, login or payment. The topic model first called this group App problems, but its "
             "phrases show it is really general dissatisfaction and switching, so it was renamed here.",
    CATS[2]: "Complaints about customer care and support.",
    CATS[3]: "Late or slow delivery, delivery partner issues.",
    CATS[4]: "High prices, extra charges, offers and coupons not working.",
    CATS[5]: "Reviews that talk about food quality and taste. The full group also contains praise, so only its negative reviews are counted as complaints here.",
}
LANG_LABEL = {"english": "English", "hinglish": "Hinglish", "hindi_script": "Hindi"}

st.markdown("""<style>
.stApp {background:#f3f4f6;}
[data-testid="stHeader"] {background:#f3f4f6;}
[data-testid="stToolbar"] * {color:#374151 !important;}
[data-testid="stSidebar"] {background:#111827;}
[data-testid="stSidebar"] * {color:#e5e7eb !important;}
.card {background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px 18px 18px; margin:0 0 14px 0;}
.card h4 {margin:0 0 4px 0; font-size:15px; font-weight:700; color:#111827;}
div[class*="st-key-card"] {background:#ffffff; border-radius:14px; border:1px solid #e5e7eb; padding:14px 16px 20px 16px !important; margin-bottom:14px;}
.small {color:#6b7280; font-size:13px; line-height:1.45;}
.kpi-label {color:#6b7280; font-size:14px;}
.kpi-value {font-size:32px; font-weight:600; color:#111827; line-height:1.3; margin:2px 0 4px 0;}
.up-good {color:#0b7a4b; font-size:13px;}
.up-bad {color:#b42318; font-size:13px;}
.flat {color:#6b7280; font-size:13px;}
.row {display:flex; align-items:center; gap:10px; margin:12px 0;}
.barbg {background:#e5e7eb; border-radius:6px; height:10px; flex:1;}
.bar {height:10px; border-radius:6px;}
.pill {display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; white-space:nowrap;}
.pill-red {background:#fde8e4; color:#a32a12;}
.pill-yellow {background:#fdf0c9; color:#7a5a00;}
.pill-green {background:#dcf3e8; color:#0b6b46;}
.pill-grey {background:#eceef2; color:#374151;}
.item {display:flex; gap:12px; margin:14px 0; align-items:flex-start;}
.lang-table {width:100%; font-size:14px; border-collapse:collapse;}
.lang-table td, .lang-table th {padding:6px 4px; text-align:left;}
.lang-table th {color:#6b7280; font-weight:500; font-size:12px;}
.rcard {background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 18px; margin:10px 0;}
.rmeta {display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; font-size:13px; color:#6b7280;}
.rtext {font-size:16px; color:#111827; line-height:1.5;}
.card li {color:#111827; margin-bottom:8px; line-height:1.5;}
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 {color:#111827 !important;}
[data-testid="stMain"] label p {color:#111827 !important;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {color:#111827;}
[data-testid="stMain"] b, [data-testid="stMain"] strong {color:#111827;}
[data-testid="stMain"] li, [data-testid="stMain"] li p {color:#111827 !important;}
[data-testid="stMain"] .small {color:#6b7280 !important;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    d = pd.read_csv("data/dashboard_data_v2.csv", parse_dates=["date"])
    d["week_start"] = pd.to_datetime(d["week_start"])
    return d


def load_phrases():          # NOT cached, so a newly uploaded file is noticed
    """Prefer the distinctive-phrase file (v4); fall back to the old frequency file (v3)."""
    for version in ("v4", "v3"):
        path = f"data/category_phrases_{version}.csv"
        if os.path.exists(path):
            return _read_phrases(path, os.path.getmtime(path)), version
    return None, None


@st.cache_data
def _read_phrases(path, mtime):    # mtime makes the cache reset when the file changes
    p = pd.read_csv(path)
    p["week_start"] = pd.to_datetime(p["week_start"])
    return p


@st.cache_data
def load_sample():
    if not os.path.exists("data/review_sample_masked.csv"):
        return None
    s = pd.read_csv("data/review_sample_masked.csv", parse_dates=["date"])
    s["text"] = s["text"].fillna("").astype(str)
    return s


df = load()


def pct_change(c, p):
    return (c - p) / p * 100 if p else 0.0


def pos_neg(x):
    s = x[x["sentiment_pred"] != "not scored"]
    if len(s) == 0:
        return 0.0, 0.0
    return (s["sentiment_pred"] == "pos").mean() * 100, (s["sentiment_pred"] == "neg").mean() * 100


def complaints(d):
    """Reviews that are a real complaint: in one of the 6 categories AND predicted negative.
    Without the sentiment filter, praise inside a category (e.g. Food quality is ~57% positive)
    would be counted as a complaint."""
    return d[d["aspect"].isin(CATS) & (d["sentiment_pred"] == "neg")]


def delta(d, mag_text, higher_is_good=None, thresh=0.0):
    if abs(d) <= thresh:
        return "No real change vs previous 4 weeks", None
    arrow = "▲" if d > 0 else "▼"
    good = None if higher_is_good is None else ((d > 0) == higher_is_good)
    return f"{arrow} {mag_text} vs previous 4 weeks", good


def card_html(inner):
    return f'<div class="card">{inner}</div>'


def head(title, sub=None):
    t = f'<h4>{title}</h4>'
    if sub:
        t += f'<div class="small" style="margin-bottom:8px;">{sub}</div>'
    return t


def kpi(col, label, value, d_text, good):
    cls = {True: "up-good", False: "up-bad", None: "flat"}[good]
    col.markdown(card_html(f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                           f'<div class="{cls}">{d_text}</div>'), unsafe_allow_html=True)


def full_weeks():
    full = df[~df["partial_week"]]
    return full, sorted(full["week_start"].unique())


def style_fig(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=55, r=15, t=10, b=45),
                      xaxis_title="", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#374151"))
    fig.update_yaxes(rangemode="tozero", gridcolor="#eef0f3", automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


def change_cell(ch, old):
    """Text and colour for the 'vs previous 4 weeks' column. Only moves of 10%+ get a colour."""
    if old == 0:
        return "new", "#6b7280"
    if round(ch) == 0:
        return "no change", "#6b7280"
    colour = "#b42318" if ch >= 10 else ("#0b7a4b" if ch <= -10 else "#6b7280")
    arrow = "▲" if ch > 0 else "▼"
    return f"{arrow} {abs(ch):.0f}%", colour


# ---------------- sidebar ----------------
st.sidebar.markdown('<div style="font-size:22px;font-weight:700;">ReviewPulse</div>'
                    '<div style="font-size:13px;">Swiggy app reviews</div><br>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Overview", "Category detail", "Review explorer", "Live demo", "About this data"],
                        label_visibility="collapsed")
st.sidebar.markdown("<br><br><div style='font-size:12px;'>Reads English, Hinglish and Hindi reviews</div>",
                    unsafe_allow_html=True)

# ---------------- page 1: overview ----------------
def overview():
    full, weeks = full_weeks()
    if len(weeks) < 8:
        st.warning("Need at least 8 full weeks of data for this page.")
        return
    last4, prev4 = weeks[-4:], weeks[-8:-4]
    start = pd.Timestamp(last4[0])
    end = pd.Timestamp(last4[-1]) + pd.Timedelta(days=6)

    h_l, h_r = st.columns([3, 3])
    with h_l:
        st.markdown(f'<div class="small">Swiggy app · last 4 full weeks ({start:%d %b} to {end:%d %b %Y})</div>'
                    f'<h1 style="margin:0;padding:0;">Review insights</h1>', unsafe_allow_html=True)
    with h_r:
        choice = st.radio("Language", ["All languages", "English", "Hinglish", "Hindi"], horizontal=True,
                          label_visibility="collapsed", key="lang_choice")
    code = {"English": "english", "Hinglish": "hinglish", "Hindi": "hindi_script"}.get(choice)
    dl = full if code is None else full[full["language"] == code]

    cur = dl[dl["week_start"].isin(last4)]
    prv = dl[dl["week_start"].isin(prev4)]
    if len(cur) == 0 or len(prv) == 0:
        st.warning("Not enough reviews for this language.")
        return
    pos_c, neg_c = pos_neg(cur)
    pos_p, neg_p = pos_neg(prv)

    k = st.columns(4)
    dr = cur["rating"].mean() - prv["rating"].mean()
    t, g = delta(dr, f"{abs(dr):.2f}", True, 0.02)
    kpi(k[0], "Overall rating", f'{cur["rating"].mean():.2f}<span style="font-size:16px;color:#6b7280;"> / 5</span>', t, g)
    dn_ = pct_change(len(cur), len(prv))
    t, g = delta(dn_, f"{abs(dn_):.0f}%", None, 1)
    kpi(k[1], "New reviews", f"{len(cur):,}", t, g)
    t, g = delta(pos_c - pos_p, f"{abs(pos_c - pos_p):.1f} pts", True, 0.5)
    kpi(k[2], "Positive reviews", f"{pos_c:.1f}%", t, g)
    t, g = delta(neg_c - neg_p, f"{abs(neg_c - neg_p):.1f} pts", False, 0.5)
    kpi(k[3], "Negative reviews", f"{neg_c:.1f}%", t, g)

    # complaints = negative reviews with a topic (praise inside a category is NOT counted)
    comp_c, comp_p = complaints(cur), complaints(prv)
    cc, cp = comp_c["aspect"].value_counts(), comp_p["aspect"].value_counts()
    tot_c, tot_p = len(comp_c), len(comp_p)
    rows = []
    for c in CATS:
        cnt, old = int(cc.get(c, 0)), int(cp.get(c, 0))
        rows.append((c, cnt, old, pct_change(cnt, old)))
    rows.sort(key=lambda r: -r[1])

    mid_l, mid_r = st.columns(2)
    with mid_l:
        tot_txt, tot_col = change_cell(pct_change(tot_c, tot_p), tot_p)
        h = head("Complaints by category",
                 f"<b>{tot_c:,}</b> negative reviews with a clear topic in the last 4 weeks "
                 f'(<span style="color:{tot_col};">{tot_txt}</span> vs the previous 4 weeks)')
        # column headers
        h += ('<div class="row" style="margin:10px 0 2px 0;">'
              '<div style="width:38%;"></div>'
              '<div class="small" style="flex:1;">Share of complaints</div>'
              '<div class="small" style="width:44px;"></div>'
              '<div class="small" style="width:92px;text-align:right;">vs prev 4 weeks</div></div>')
        for name, cnt, old, ch in rows:
            share = cnt / tot_c * 100 if tot_c else 0
            ch_txt, ch_col = change_cell(ch, old)
            h += (f'<div class="row"><div style="width:38%;"><b style="color:#111827;">{nice(name).replace("&", "&amp;")}</b>'
                  f'<div class="small">{cnt:,} complaints</div></div>'
                  f'<div class="barbg"><div class="bar" style="width:{share:.1f}%;background:#2563eb;"></div></div>'
                  f'<div style="width:44px;text-align:right;font-size:14px;font-weight:600;color:#111827;">{share:.0f}%</div>'
                  f'<div style="width:92px;text-align:right;color:{ch_col};font-size:13px;">{ch_txt}</div></div>')
        h += ('<div class="small">Bars add up to 100% of complaints with a topic. In the last column, red = up 10% or more, '
              'green = down 10% or more, grey = a smaller move, which is often normal weekly noise. '
              'Only negative reviews are counted, so praise about food is not shown as a complaint.</div>')
        st.markdown(card_html(h), unsafe_allow_html=True)
    with mid_r:
        with st.container(border=True, key="card_trend"):
            st.markdown(head("Negative reviews by week",
                             "Share of scored reviews that are negative, all full weeks (not just the last 4)"),
                        unsafe_allow_html=True)
            s = dl[dl["sentiment_pred"] != "not scored"]
            wk = (s.groupby("week_start")
                  .agg(reviews=("review_id", "size"),
                       negative_pct=("sentiment_pred", lambda x: (x == "neg").mean() * 100))
                  .reset_index())
            fig = px.line(wk, x="week_start", y="negative_pct", markers=True, hover_data=["reviews"])
            fig.update_traces(line_color="#2563eb")
            fig = style_fig(fig, 360)
            fig.update_layout(yaxis_title="% negative")
            st.plotly_chart(fig, use_container_width=True, theme=None)

    items = []
    big = [r for r in rows if r[1] >= 80 and r[2] > 0]
    if big:
        w = max(big, key=lambda r: r[3])
        if w[3] >= 15:
            items.append(("Needs attention", "pill-red", f"{nice(w[0])} complaints are rising",
                          f"Up {w[3]:.0f}% ({w[2]:,} to {w[1]:,} complaints) vs the previous 4 weeks."))
        elif w[3] >= 5:
            items.append(("Watch", "pill-yellow", f"{nice(w[0])} complaints are creeping up",
                          f"Up {w[3]:.0f}% ({w[2]:,} to {w[1]:,} complaints) vs the previous 4 weeks."))
        b = min(big, key=lambda r: r[3])
        if b[3] <= -10:
            items.append(("Healthy", "pill-green", f"{nice(b[0])} complaints are falling",
                          f"Down {abs(b[3]):.0f}% ({b[2]:,} to {b[1]:,} complaints) vs the previous 4 weeks."))
    dneg = neg_c - neg_p
    rel = pct_change(neg_c, neg_p)
    if abs(rel) >= 10:          # same 10% rule as the category panel
        up = dneg > 0
        items.append(("Needs attention" if up else "Healthy", "pill-red" if up else "pill-green",
                      f"Negative reviews are {'up' if up else 'down'}",
                      f"{neg_p:.1f}% to {neg_c:.1f}% of scored reviews ({abs(dneg):.1f} pts)."))
    elif abs(dneg) >= 0.5:
        items.append(("Stable", "pill-grey", "Negative share is roughly stable",
                      f"{neg_p:.1f}% to {neg_c:.1f}% of scored reviews ({abs(dneg):.1f} pts), "
                      f"a {abs(rel):.0f}% move, within normal noise."))
    if not items:
        items.append(("Stable", "pill-grey", "Nothing unusual", "No big changes vs the previous 4 weeks."))

    bot_l, bot_r = st.columns(2)
    with bot_l:
        h = head("What changed")
        for tag, cls, title, text in items:
            h += (f'<div class="item"><span class="pill {cls}">{tag}</span>'
                  f'<div><b style="color:#111827;">{title.replace("&", "&amp;")}</b><div class="small">{text}</div></div></div>')
        st.markdown(card_html(h), unsafe_allow_html=True)
    with bot_r:
        cur_all = full[full["week_start"].isin(last4)]
        ca = cur_all[cur_all["language"].isin(LANG_LABEL.keys())]
        n_emoji = int((cur_all["language"] == "emoji_only").sum())
        colours = {"english": "#1d4ed8", "hinglish": "#6b8ff0", "hindi_script": "#bcd0fb"}
        bar, table = "", ""
        for lg, label in LANG_LABEL.items():
            sub = ca[ca["language"] == lg]
            share = len(sub) / len(ca) * 100 if len(ca) else 0
            share_txt = f"{share:.1f}%" if share < 10 else f"{share:.0f}%"
            p_, n_ = pos_neg(sub)
            bar += f'<div style="width:{share:.1f}%;background:{colours[lg]};height:12px;"></div>'
            table += (f'<tr><td style="color:#111827;"><span style="color:{colours[lg]};">■</span> {label}</td>'
                      f'<td style="color:#111827;">{len(sub):,}</td><td style="color:#111827;">{share_txt}</td>'
                      f'<td style="color:#0b7a4b;">{p_:.0f}%</td><td style="color:#b42318;">{n_:.0f}%</td></tr>')
        h = head("Reviews by language", "Last 4 full weeks, all languages")
        h += (f'<div style="display:flex;border-radius:6px;overflow:hidden;margin:10px 0;">{bar}</div>'
              f'<table class="lang-table"><tr><th>Language</th><th>Reviews</th><th>Share</th><th>Positive</th><th>Negative</th></tr>{table}</table>'
              f'<div class="small" style="margin-top:8px;">Plus {n_emoji:,} emoji-only reviews, which have no words to '
              'tag or score. Hinglish and Hindi reviews look more negative mostly because they are longer, '
              'and long reviews are mostly complaints. Rows with few reviews can swing a lot.</div>')
        st.markdown(card_html(h), unsafe_allow_html=True)

    st.markdown('<div class="small">Positive and negative come from a sentiment model (96.5% accurate on 283 labelled '
                'test reviews). See About this data for how the labels were made and the limits.</div>',
                unsafe_allow_html=True)


# ---------------- page 2: category detail ----------------
def category_detail():
    full, weeks = full_weeks()
    if len(weeks) < 8:
        st.warning("Need at least 8 full weeks of data for this page.")
        return
    last4, prev4 = weeks[-4:], weeks[-8:-4]

    st.markdown('<div class="small">Swiggy app · complaint categories</div>'
                '<h1 style="margin:0;padding:0;">Category detail</h1>', unsafe_allow_html=True)
    choice = st.radio("Category", list(SHORT.keys()), horizontal=True,
                      label_visibility="collapsed", key="cat_choice")
    cat = SHORT[choice]

    comp = complaints(full)
    cur_all = comp[comp["week_start"].isin(last4)]
    prv_all = comp[comp["week_start"].isin(prev4)]
    cur = cur_all[cur_all["aspect"] == cat]
    prv = prv_all[prv_all["aspect"] == cat]
    n_c, n_p = len(cur), len(prv)
    share_c = n_c / len(cur_all) * 100 if len(cur_all) else 0.0
    share_p = n_p / len(prv_all) * 100 if len(prv_all) else 0.0

    # all scored reviews in each period, used to turn counts into a rate
    scored = full[full["sentiment_pred"] != "not scored"]
    n_scored_c = int(scored["week_start"].isin(last4).sum())
    n_scored_p = int(scored["week_start"].isin(prev4).sum())
    rate_c = n_c / n_scored_c * 1000 if n_scored_c else 0.0
    rate_p = n_p / n_scored_p * 1000 if n_scored_p else 0.0

    k = st.columns(3)
    ch = pct_change(n_c, n_p)
    t, g = delta(ch, f"{abs(ch):.0f}%", False, 10)
    kpi(k[0], "Complaints, last 4 weeks", f"{n_c:,}", t, g)
    rch = pct_change(rate_c, rate_p)
    t, g = delta(rch, f"{abs(rch):.0f}%", False, 10)
    kpi(k[1], "Complaints per 1,000 reviews", f"{rate_c:.0f}", t, g)
    t, g = delta(share_c - share_p, f"{abs(share_c - share_p):.1f} pts", None, 1.0)
    kpi(k[2], "Share of all topic complaints", f"{share_c:.0f}%", t, g)

    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card_c_trend"):
            st.markdown(head(f"{choice}: complaints per week",
                             "Negative reviews in this category, full weeks only. "
                             "Dark bars = the last 4 weeks used in the numbers above."),
                        unsafe_allow_html=True)
            wk = comp[comp["aspect"] == cat].groupby("week_start").size().reset_index(name="complaints")
            last4_ts = set(pd.to_datetime(list(last4)))
            wk["period"] = ["Last 4 weeks" if w in last4_ts else "Earlier" for w in wk["week_start"]]
            fig = px.bar(wk, x="week_start", y="complaints", color="period",
                         color_discrete_map={"Last 4 weeks": "#2563eb", "Earlier": "#bfd0f5"})
            fig = style_fig(fig, 380)
            fig.update_layout(yaxis_title="Complaints", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, theme=None)

    with right:
        h = head("What people say")
        ph, version = load_phrases()
        if ph is None:
            h += '<div class="small">Phrase file not found. Put <b>category_phrases_v4.csv</b> in the data folder.</div>'
        else:
            sub = ph[(ph["aspect"] == cat) & (ph["week_start"].isin(pd.to_datetime(list(last4))))]
            period = "last 4 weeks"
            if sub["count"].sum() < 20:          # too thin for 4 weeks, fall back to all weeks
                sub = ph[ph["aspect"] == cat]
                period = "all weeks (too few in the last 4 to show)"
            top = sub.groupby("phrase")["count"].sum().nlargest(8)
            top = top[top >= 3]
            if version == "v4":
                h += (f'<div class="small">Phrases that set this category apart from the other complaints, '
                      f'with the number of complaints mentioning each, {period}.</div>')
            else:
                h += f'<div class="small">Most common two-word phrases, {period} (old phrase file).</div>'
            if top.empty:
                h += '<div class="small">Too few complaints in this category to show reliable phrases.</div>'
            else:
                mx = top.max()
                for phrase, cnt in top.items():
                    h += (f'<div class="row"><div style="width:45%;color:#111827;">{htmllib.escape(str(phrase))}</div>'
                          f'<div class="barbg"><div class="bar" style="width:{cnt / mx * 100:.0f}%;background:#b42318;"></div></div>'
                          f'<div style="width:50px;text-align:right;font-size:13px;color:#111827;">{int(cnt):,}</div></div>')
                h += ('<div class="small">Counts only, no review text is stored. Each phrase is counted once per review. '
                      'Hinglish is not translated.</div>')
        st.markdown(card_html(h), unsafe_allow_html=True)

    n_no_topic_c = int(((full["aspect"] == "No clear topic") & full["week_start"].isin(last4)).sum())
    st.markdown(card_html(head("What this category covers") +
                          f'<div class="small">{htmllib.escape(CAT_INFO[cat])}</div>'
                          '<div class="small" style="margin-top:6px;"><b>How to read the numbers.</b> '
                          'Complaints per 1,000 reviews adjusts for how many reviews came in, so it shows whether this '
                          'problem is really growing or the app just got more reviews. Share of all topic complaints '
                          f'leaves out {n_no_topic_c:,} short complaints in the last 4 weeks that name no topic '
                          '("worst", "bad", "waste"). Categories come from topic modelling on longer reviews plus a '
                          'classifier for short complaints, and were not fully validated.</div>'),
                unsafe_allow_html=True)


# ---------------- page 3: review explorer ----------------
def highlight(text, q):
    esc = htmllib.escape(text)
    if q:
        pat = re.compile(re.escape(htmllib.escape(q)), re.I)
        esc = pat.sub(lambda m: f'<mark style="background:#fde8a8;">{m.group(0)}</mark>', esc)
    return esc


def review_explorer():
    st.markdown('<div class="small">Swiggy app · a small masked sample</div>'
                '<h1 style="margin:0;padding:0;">Review explorer</h1>', unsafe_allow_html=True)
    s = load_sample()
    if s is None:
        st.warning("Sample file not found. Put review_sample_masked.csv in the data folder.")
        return

    with st.container(border=True, key="card_ex_filters"):
        q = st.text_input("Search a word", placeholder="Search a word in any language, e.g. refund, paise, late",
                          key="ex_q").strip()
        c1, c2 = st.columns(2)
        with c1:
            sent = st.radio("Sentiment", ["All", "Positive", "Negative"], horizontal=True, key="ex_sent")
        with c2:
            lang = st.radio("Language", ["All", "English", "Hinglish", "Hindi"], horizontal=True, key="ex_lang")
        c3, c4 = st.columns(2)
        with c3:
            present = s["aspect"].dropna().unique().tolist()
            complaint_cats = sorted([c for c in CATS if c in present], key=nice)      # the 6 complaint groups, A to Z
            other_cats = sorted([c for c in present if c not in CATS], key=nice)    # praise, unclear, etc. at the end
            cat_opts = ["All categories"] + complaint_cats + other_cats
            cat = st.selectbox("Category", cat_opts, key="ex_cat", format_func=nice)
        with c4:
            order = st.selectbox("Sort", ["Newest first", "Oldest first", "Lowest rating first", "Highest rating first"],
                                 key="ex_sort")

    r = s.copy()
    if q:
        r = r[r["text"].str.contains(q, case=False, regex=False)]
    if sent != "All":
        r = r[r["sentiment_pred"] == {"Positive": "pos", "Negative": "neg"}[sent]]
    if lang != "All":
        r = r[r["language"] == {"English": "english", "Hinglish": "hinglish", "Hindi": "hindi_script"}[lang]]
    if cat != "All categories":
        r = r[r["aspect"] == cat]
    if order == "Newest first":
        r = r.sort_values("date", ascending=False)
    elif order == "Oldest first":
        r = r.sort_values("date", ascending=True)
    elif order == "Lowest rating first":
        r = r.sort_values(["rating", "date"], ascending=[True, False])
    else:
        r = r.sort_values(["rating", "date"], ascending=[False, False])

    shown = r.head(30)
    st.markdown(f'<div class="small">Showing {len(shown)} of {len(r)} matching reviews '
                f'(sample of {len(s)} reviews, 4 to 60 words, personal details masked).</div>',
                unsafe_allow_html=True)
    if r.empty:
        st.info("No reviews match these filters.")
    for _, row in shown.iterrows():
        sent_pill = ('<span class="pill pill-green">Positive</span>' if row["sentiment_pred"] == "pos"
                     else '<span class="pill pill-red">Negative</span>')
        lang_name = LANG_LABEL.get(row["language"], str(row["language"]))
        date_txt = f"{row['date']:%d %b %Y}" if pd.notna(row["date"]) else ""
        st.markdown(
            f'<div class="rcard"><div class="rmeta"><b style="color:#111827;">{int(row["rating"])} of 5 stars</b>'
            f'<span class="pill pill-grey">{htmllib.escape(lang_name)}</span>{sent_pill}'
            f'<span class="pill pill-grey">{htmllib.escape(nice(str(row["aspect"])))}</span>'
            f'<span style="margin-left:auto;">{date_txt}</span></div>'
            f'<div class="rtext">{highlight(row["text"], q)}</div></div>',
            unsafe_allow_html=True)
    st.markdown('<div class="small">Sentiment and category tags are model predictions, so some are wrong. '
                'Reviews are shown as written (Hinglish is not translated).</div>', unsafe_allow_html=True)


# ---------------- page 4: about ----------------
ABOUT_CSS = """<style>
.flow {display:flex; align-items:stretch; gap:8px; flex-wrap:wrap; margin:10px 0;}
.node {flex:1; min-width:170px; background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px;}
.node .t {color:#111827; font-weight:700; font-size:14px; margin-bottom:3px;}
.node .d {color:#4b5563; font-size:13px; line-height:1.45;}
.arrow {align-self:center; color:#9ca3af; font-size:20px;}
.branches {display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; margin:10px 0;}
.branch {border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; background:#ffffff;}
.branch .t {color:#111827; font-weight:700; font-size:14px; margin-bottom:4px;}
.branch .d {color:#4b5563; font-size:13px; line-height:1.5;}
.b1 {border-top:4px solid #1d4ed8;} .b2 {border-top:4px solid #0b7a4b;} .b3 {border-top:4px solid #b45309;}
.hl td {background:#eaf1ff; font-weight:600;}
.lang-table tr {border-bottom:1px solid #eef0f3;}
.callout {background:#f3f6ff; border:1px solid #d6e2ff; border-radius:10px; padding:10px 14px; margin:12px 0 4px 0;
          color:#111827; font-size:14px; line-height:1.5;}
@media (max-width: 900px) {.branches {grid-template-columns:1fr;}}
</style>"""


def _tbl(headers, rows, hl=None):
    """Small HTML table. hl = index of the row to highlight (the chosen method)."""
    h = "<table class='lang-table'><tr>" + "".join(f"<th>{x}</th>" for x in headers) + "</tr>"
    for i, r in enumerate(rows):
        cls = ' class="hl"' if i == hl else ""
        h += f"<tr{cls}>" + "".join(f'<td style="color:#111827;">{c}</td>' for c in r) + "</tr>"
    return h + "</table>"


def _bullets(items):
    return "<ul>" + "".join(f"<li>{b}</li>" for b in items) + "</ul>"


def _callout(text):
    return f'<div class="callout">{text}</div>'


def about():
    st.markdown(ABOUT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="small">Swiggy app reviews</div>'
                '<h1 style="margin:0;padding:0;">About this data</h1>', unsafe_allow_html=True)

    t_arch, t_lang, t_sent, t_topic, t_limits = st.tabs(
        ["Architecture & cleaning", "Language detection", "Sentiment", "Topics", "Limitations"])

    # ======================= TAB 1: architecture + preprocessing =======================
    with t_arch:
        h = head("How the project fits together",
                 "Every review goes through the same cleaning, then three separate models add one column each.")
        h += ('<div class="flow">'
              '<div class="node"><div class="t">Collect</div><div class="d">google-play-scraper, English and Hindi feeds, '
              'newest first. 39,457 Swiggy reviews, Jul 12 to Oct 9, 2026.</div></div><div class="arrow">→</div>'
              '<div class="node"><div class="t">Clean</div><div class="d">Lowercase, remove links, shorten stretched '
              'letters and symbols. Numbers are never touched.</div></div><div class="arrow">→</div>'
              '<div class="node"><div class="t">Prepare</div><div class="d">Count words. Split short (1 to 3 words) '
              'from long (4+). Set emoji-only reviews aside.</div></div>'
              '</div>')
        h += ('<div class="branches">'
              '<div class="branch b1"><div class="t">1. Language detection</div><div class="d">MuRIL language-ID model '
              'labels each word as Hindi or English, then a rule decides English, Hinglish or Hindi. '
              'Devanagari script is caught by a regex.<br><b style="color:#111827;">Adds:</b> language</div></div>'
              '<div class="branch b2"><div class="t">2. Sentiment</div><div class="d">TF-IDF on character n-grams '
              'plus logistic regression, trained on star ratings.<br>'
              '<b style="color:#111827;">Adds:</b> positive or negative, and a negative probability</div></div>'
              '<div class="branch b3"><div class="t">3. Topics</div><div class="d">MiniLM sentence embeddings, '
              'KMeans clusters for long reviews, a small classifier for short complaints.<br>'
              '<b style="color:#111827;">Adds:</b> complaint category</div></div>'
              '</div>')
        h += ('<div class="flow">'
              '<div class="node"><div class="t">Combine</div><div class="d">One row per review in '
              '<b>dashboard_data_v2.csv</b>: date, rating, language, sentiment, category. Also weekly phrase counts '
              'and a masked sample of 300 reviews.</div></div><div class="arrow">→</div>'
              '<div class="node"><div class="t">Streamlit app</div><div class="d">Overview, Category detail and '
              'Review explorer read those files. No model runs when you open the app.</div></div>'
              '</div>')
        h += _callout("Links between the steps: the language tag decides how topics are grouped (Hinglish reviews are "
                      "clustered on their own), and emoji-only reviews skip sentiment and topics. A complaint on the "
                      "Overview and Category pages is a review with a category AND a negative sentiment prediction.")
        st.markdown(card_html(h), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            h = head("Cleaning steps")
            h += _bullets([
                "<b>Lowercase</b> everything and turn links into spaces.",
                "<b>Stretched letters:</b> 3 or more repeats become 2 (wowwww becomes woww).",
                "<b>Stretched symbols and emoji:</b> 3 or more repeats become 2.",
                "<b>Numbers are left alone.</b> The first version shortened 3000 to 300 and 10000x to 100x, "
                "so the rule was changed to letters only.",
                "1,819 of 39,457 reviews (4.6%) changed during cleaning.",
            ])
            st.markdown(card_html(h), unsafe_allow_html=True)
        with c2:
            h = head("Deliberately not done")
            h += _bullets([
                "<b>No stemming or lemmatizing.</b> Character n-grams already absorb spelling variants "
                "(bhot, bohot, bahut), and an English stemmer would damage Hinglish words.",
                "<b>No translation or transliteration.</b> The models read Hinglish as written.",
                "<b>No stop-word removal</b> for the models. It is used only when picking the phrases shown on "
                "Category detail, and there negations (no, not) are kept so no refund and not received survive.",
            ])
            st.markdown(card_html(h), unsafe_allow_html=True)

        h = head("Splitting and labelling")
        h += _bullets([
            "<b>Short vs long reviews:</b> reviews under 4 words are handled separately, because the language rule "
            "and the topic model were validated on 4+ word reviews.",
            "<b>Training labels for sentiment:</b> stars. 1 to 2 stars = negative, 4 to 5 = positive, 3 stars dropped. "
            "That gave 23,538 training reviews (12,022 negative, 11,516 positive).",
            "<b>No leaking:</b> the same text never sits in both training and validation (grouped split, 80/20), "
            "and none of the hand-labelled test reviews appear in training (0 overlaps found).",
            "<b>Hand-labelled checks:</b> 196 reviews for language, 300 for sentiment, 120 and 300 for short-review topics.",
            "<b>Privacy:</b> the Review explorer sample has emails, phone numbers, long ids and @handles masked.",
        ])
        st.markdown(card_html(h), unsafe_allow_html=True)

    # ======================= TAB 2: language detection =======================
    with t_lang:
        l, r = st.columns([3, 2])
        with l:
            h = head("Five ways to tag Hinglish, compared",
                     "Scored on 196 hand-labelled reviews of 4+ words (74 Hinglish). Precision = how many flagged "
                     "reviews were really Hinglish. Recall = how many real Hinglish reviews were found.")
            h += _tbl(["Method", "Flagged", "Precision", "Recall", "F1"], [
                ["Word list, strict", "56", "1.00", "0.76", "0.86"],
                ["Word list, loose", "68", "0.96", "0.88", "0.92"],
                ["MuRIL: 3+ Hindi words", "110", "0.64", "0.95", "0.76"],
                ["MuRIL: 3+ words and 20%+ share (chosen)", "73", "0.95", "0.93", "0.94"],
                ["MuRIL: 2+ words and 40%+ share", "67", "0.99", "0.89", "0.94"],
            ], hl=3)
            h += _callout("Counting words alone flagged many long English reviews that mention one or two Hindi words. "
                          "Adding the 20% share condition fixed that. Rule 5 tied on F1 but missed more real Hinglish.")
            st.markdown(card_html(h), unsafe_allow_html=True)
        with r:
            h = head("What was used")
            h += _bullets([
                "<b>Model:</b> PhysicsWallahAI/muril-hinglish-lid, a pretrained word-level language-ID model "
                "(labels each word HIN or ENG).",
                "<b>Rule:</b> Hinglish if it finds 3+ Hindi words and they make up 20%+ of the review. "
                "Emoji and symbols are ignored when counting.",
                "<b>Devanagari script</b> (Hindi letters) is tagged by a regex.",
                "<b>Emoji-only</b> reviews are tagged separately and skip sentiment and topics.",
            ])
            st.markdown(card_html(h), unsafe_allow_html=True)

        l, r = st.columns(2)
        with l:
            h = head("Short reviews (1 to 3 words)", "Harder: too few words to judge. 93 hand-labelled, 17 Hinglish.")
            h += _tbl(["Rule", "Precision", "Recall", "F1"], [
                ["1+ Hindi word", "0.33", "1.00", "0.49"],
                ["2+ Hindi words", "0.77", "0.59", "0.67"],
                ["Word list only", "1.00", "0.41", "0.58"],
                ["2+ words OR word list (chosen)", "0.80", "0.71", "0.75"],
            ], hl=3)
            st.markdown(card_html(h), unsafe_allow_html=True)
        with r:
            h = head("Final result", "All 39,457 reviews")
            h += _tbl(["Language", "Reviews", "Share"], [
                ["English", "36,396", "92.2%"],
                ["Hinglish", "1,784", "4.5%"],
                ["Emoji only", "1,131", "2.9%"],
                ["Hindi script", "146", "0.4%"],
            ])
            h += ('<div class="small" style="margin-top:8px;">Among reviews with 4+ words, about 9% are Hinglish. '
                  'Short-review tags are the least reliable (F1 0.75).</div>')
            st.markdown(card_html(h), unsafe_allow_html=True)

    # ======================= TAB 3: sentiment =======================
    with t_sent:
        l, r = st.columns([3, 2])
        with l:
            h = head("Three models, same test",
                     "283 hand-labelled reviews (positive or negative; 17 neutral ones left out).")
            h += _tbl(["Model", "Accuracy", "Precision", "Recall", "Macro F1"], [
                ["TF-IDF + logistic regression (chosen)", "0.965", "0.960", "0.968", "0.963"],
                ["DistilBERT multilingual, fine-tuned", "0.968", "0.964", "0.971", "0.967"],
                ["MuRIL sentiment, pretrained", "0.972", "0.968", "0.975", "0.971"],
            ], hl=0)
            h += _callout("The gap between best and chosen is about 2 reviews out of 283, which is noise on a test "
                          "set this small. The simplest model was kept: it trains in seconds, needs no GPU "
                          "and is easy to deploy.")
            st.markdown(card_html(h), unsafe_allow_html=True)
        with r:
            h = head("How each was set up")
            h += _bullets([
                "<b>TF-IDF + LogReg:</b> character n-grams (2 to 5 letters) so spelling variants match, "
                "balanced class weights. Five TF-IDF settings all scored 0.937 to 0.940 on validation.",
                "<b>DistilBERT:</b> fine-tuned 3 epochs on 18,885 reviews, best validation F1 0.949, "
                "about 3 minutes per epoch on a laptop GPU.",
                "<b>MuRIL:</b> used as is, no training. It also has a neutral class, "
                "but only 83% three-way accuracy on all 300 reviews.",
            ])
            st.markdown(card_html(h), unsafe_allow_html=True)

        l, r = st.columns(2)
        with l:
            h = head("Chosen model, by group", "Test reviews only")
            h += _tbl(["Group", "Reviews", "Accuracy"], [
                ["English", "117", "0.97"],
                ["Hinglish", "84", "0.96"],
                ["Hindi script", "8", "1.00"],
                ["Short (1 to 3 words)", "74", "0.95"],
            ])
            h += ('<div class="small" style="margin-top:8px;">On short reviews, accuracy looks high only because '
                  'most are positive. Macro F1 there is 0.79.</div>')
            st.markdown(card_html(h), unsafe_allow_html=True)
        with r:
            h = head("Where it goes wrong")
            h += _bullets([
                "Slang and typos (bhota leta, chutyagiri).",
                "Negation in Hinglish (aadha saman nahin bhejte).",
                "Comparisons that sound positive (zepto is faster and cheaper than instamart).",
                "Most errors are real complaints read as positive, so negatives are slightly undercounted.",
            ])
            h += ('<div class="small">Overall prediction on all scored reviews: about 67% positive, 33% negative.</div>')
            st.markdown(card_html(h), unsafe_allow_html=True)

    # ======================= TAB 4: topic modelling =======================
    with t_topic:
        h = head("Two stages: find the topics, then label the short reviews")
        h += ('<div class="flow">'
              '<div class="node"><div class="t">Embed</div><div class="d">paraphrase-multilingual-MiniLM-L12-v2 '
              'turns each review into 384 numbers. 14,397 reviews with 4+ words.</div></div><div class="arrow">→</div>'
              '<div class="node"><div class="t">Cluster</div><div class="d">KMeans with 12 clusters, read by eye '
              'and merged into 6 complaint categories, praise and other.</div></div><div class="arrow">→</div>'
              '<div class="node"><div class="t">Fix Hinglish</div><div class="d">Hinglish reviews clustered by '
              'language, not topic, so they are clustered separately with character n-grams.</div></div><div class="arrow">→</div>'
              '<div class="node"><div class="t">Label short ones</div><div class="d">Logistic regression on the same '
              'embeddings for 2,040 short negative reviews.</div></div>'
              '</div>')
        st.markdown(card_html(h), unsafe_allow_html=True)

        l, r = st.columns(2)
        with l:
            h = head("Short complaints: three attempts", "Checked on 120 hand-labelled short reviews")
            h += _tbl(["Version", "Result"], [
                ["Trained on long reviews only", "43%"],
                ["Added 300 hand-labelled short reviews, added a No clear topic class", "about 82% (chosen)"],
                ["Final model", "retrained on all 420 hand-labelled short reviews"],
            ], hl=1)
            h += ('<div class="small" style="margin-top:8px;">Version 1 scored 92% on long reviews, but that mostly '
                  'measures how well it copies the KMeans labels, so it was not trusted. Words like worst, poor and '
                  'waste have no topic, and version 1 forced them into Support. The extra class fixed most of that.</div>')
            st.markdown(card_html(h), unsafe_allow_html=True)
        with r:
            full, _ = full_weeks()
            comp = complaints(full)
            shares = comp["aspect"].value_counts(normalize=True).reindex(CATS).fillna(0) * 100
            shares = shares.sort_values(ascending=False)
            h = head("Result: share of complaints by category",
                     f"Full weeks, negative reviews with a clear topic ({len(comp):,} complaints)")
            h += _tbl(["Category", "Share"],
                      [[nice(c).replace("&", "&amp;"), f"{v:.1f}%"] for c, v in shares.items()])
            st.markdown(card_html(h), unsafe_allow_html=True)

        h = head("What the clusters looked like")
        h += _bullets([
            "<b>Clear:</b> Delivery delay, Pricing and charges, Cancellations and refunds, Food quality.",
            "<b>Overlapping:</b> Refund and support complaints (merged into one category), "
            "App problems and fraud.",
            "<b>Hinglish:</b> 871 of 1,337 Hinglish reviews fell into one cluster, so they were re-clustered. "
            "About 680 of them (roughly half) got a topic that could be trusted. The rest are marked Other / unclear.",
        ])
        st.markdown(card_html(h), unsafe_allow_html=True)

        h = head("How the phrases on Category detail are picked",
                 "Two-word phrases from negative reviews only, rebuilt in category_phrases_v4.csv")
        h += _bullets([
            "<b>Counted once per review.</b> A long review that repeats a phrase still counts once.",
            "<b>Ranked by how distinctive they are, not how common.</b> Each phrase's share of reviews in a "
            "category is compared with its share in the other five (a smoothed log-odds ratio as a z-score). "
            "Phrases every complaint uses, such as delivery partner or ordered food, score near zero and drop out.",
            "<b>One category per phrase.</b> A phrase is shown only where it is most distinctive, so customer "
            "support appears under Support and not also under Refunds.",
            "<b>Filler removed.</b> Phrases that only name the service (food delivery, order food) or only "
            "express a feeling (extremely disappointed, disappointing experience) are dropped. Negations such as "
            "no refund and not received are kept.",
            "<b>Limits:</b> phrases need at least 8 reviews over all weeks, so small categories such as Food quality "
            "show only a few. Hinglish phrases appear as written, without translation.",
        ])
        st.markdown(card_html(h), unsafe_allow_html=True)

    # ======================= TAB 5: limitations =======================
    with t_limits:
        h = head("What to keep in mind when reading the numbers")
        h += _bullets([
            "<b>Test labels:</b> drafted by an AI assistant and not independently verified, so every accuracy figure "
            "here is measured against labels that may contain mistakes.",
            "<b>Small test sets:</b> 283 reviews for sentiment, 196 for language, 120 for short-review topics. "
            "Differences of one or two points are not meaningful.",
            "<b>No neutral class:</b> 3-star reviews were not used for training, so every review is called "
            "positive or negative.",
            "<b>Short reviews dominate:</b> about 60% of reviews have 1 to 3 words and most are plain praise, "
            "so overall positive and negative shares partly reflect review length.",
            "<b>Star-based training labels:</b> stars and hand labels agreed on 95% of reviews, "
            "so some training labels are wrong.",
            "<b>Topics only cover part of the data:</b> long reviews get a topic from clustering and short reviews "
            "only if they are negative. Complaint counts use negative reviews only, because the categories "
            "themselves include some praise (Food quality is about 57% positive).",
            "<b>Complaint counts depend on the sentiment model:</b> it misses a few real complaints, "
            "so category counts are slightly low.",
            "<b>Short-review topic accuracy:</b> the final model was trained on the same 120 reviews used for the "
            "82% check, so there is no clean held-out score for it.",
            "<b>Hinglish topics:</b> only about half are reliable.",
            "<b>Time window:</b> about 3 months of one app on one store, so seasonal effects are not visible.",
        ])
        st.markdown(card_html(h), unsafe_allow_html=True)

# ---------------- page 5: live demo ----------------
DEMO_URL = "https://yashwishah-reviewpulse-demo.hf.space"


def live_demo():
    import streamlit.components.v1 as components
    st.markdown('<div class="small">Swiggy app · try it yourself</div>'
                '<h1 style="margin:0;padding:0;">Live demo</h1>', unsafe_allow_html=True)
    st.markdown('<div class="small">Type a review in English, Hinglish or Hindi and watch it go through the same '
                'cleaning, language, sentiment and topic steps. The first load can take a minute while the demo wakes up.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<a href="{DEMO_URL}" target="_blank" style="display:inline-block;padding:8px 16px;'
                'margin:8px 0 12px 0;background:#2563eb;color:#ffffff;border-radius:8px;text-decoration:none;'
                'font-weight:600;font-size:14px;">Open the demo in a new tab ↗</a>', unsafe_allow_html=True)
    components.iframe(DEMO_URL + "/?__theme=light", height=1000, scrolling=True)

# ---------------- routing ----------------
if page == "Overview":
    overview()
elif page == "Category detail":
    category_detail()
elif page == "Review explorer":
    review_explorer()
elif page == "Live demo":
    live_demo()
else:
    about()
