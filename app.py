import html as htmllib
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ReviewPulse", page_icon="📊", layout="wide")

CATS = ["Refunds, cancellations & order problems", "App problems & general complaints",
        "Customer service & support", "Delivery", "Pricing & charges", "Food quality"]
SHORT = {"Refunds & orders": CATS[0], "App problems": CATS[1], "Support": CATS[2],
         "Delivery": CATS[3], "Pricing": CATS[4], "Food": CATS[5]}
CAT_INFO = {
    CATS[0]: "Refunds not received, cancelled orders, wrong or missing items.",
    CATS[1]: "App crashes, bugs, payment or login trouble, general complaints, fraud mentions, and some people comparing with other apps.",
    CATS[2]: "Complaints about customer care and support.",
    CATS[3]: "Late or slow delivery, delivery partner issues.",
    CATS[4]: "High prices, extra charges, offers and coupons not working.",
    CATS[5]: "Reviews that talk about food quality and taste. This group includes some praise, so its average rating is higher than the others.",
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


@st.cache_data
def load_phrases():
    if not os.path.exists("data/category_phrases_v3.csv"):
        return None
    p = pd.read_csv("data/category_phrases_v3.csv")
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


# ---------------- sidebar ----------------
st.sidebar.markdown('<div style="font-size:22px;font-weight:700;">ReviewPulse</div>'
                    '<div style="font-size:13px;">Swiggy app reviews</div><br>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Overview", "Category detail", "Review explorer", "About this data"],
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
    kpi(k[2], "Positive reviews", f"{pos_c:.0f}%", t, g)
    t, g = delta(neg_c - neg_p, f"{abs(neg_c - neg_p):.1f} pts", False, 0.5)
    kpi(k[3], "Negative reviews", f"{neg_c:.0f}%", t, g)

    cc = cur[cur["aspect"].isin(CATS)]["aspect"].value_counts()
    cp = prv[prv["aspect"].isin(CATS)]["aspect"].value_counts()
    rows = []
    for c in CATS:
        cnt, old = int(cc.get(c, 0)), int(cp.get(c, 0))
        rows.append((c, cnt, old, pct_change(cnt, old)))
    rows.sort(key=lambda r: -r[1])
    mx = max(r[1] for r in rows) or 1

    mid_l, mid_r = st.columns(2)
    with mid_l:
        h = head("Complaints by category", "Reviews with a clear complaint topic, and the change vs the previous 4 weeks")
        for name, cnt, old, ch in rows:
            colour = "#b42318" if ch >= 10 else ("#0b7a4b" if ch <= -10 else "#6b7280")
            arrow = "▲" if ch > 0 else "▼"
            h += (f'<div class="row"><div style="width:42%;"><b style="color:#111827;">{name.replace("&", "&amp;")}</b>'
                  f'<div class="small">{cnt:,} reviews</div></div>'
                  f'<div class="barbg"><div class="bar" style="width:{cnt / mx * 100:.0f}%;background:{colour};"></div></div>'
                  f'<div style="width:70px;text-align:right;color:{colour};font-size:13px;">{arrow} {abs(ch):.0f}%</div></div>')
        h += '<div class="small">Red = rising 10%+, green = falling 10%+. Smaller moves are often normal weekly noise.</div>'
        st.markdown(card_html(h), unsafe_allow_html=True)
    with mid_r:
        with st.container(border=True, key="card_trend"):
            st.markdown(head("Negative reviews by week", "Share of scored reviews that are negative, full weeks only"),
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
    big = [r for r in rows if r[1] >= 80]
    if big:
        w = max(big, key=lambda r: r[3])
        if w[3] >= 15:
            items.append(("Needs attention", "pill-red", f"{w[0]} complaints are rising",
                          f"Up {w[3]:.0f}% ({w[2]:,} to {w[1]:,} reviews) vs the previous 4 weeks."))
        elif w[3] >= 5:
            items.append(("Watch", "pill-yellow", f"{w[0]} complaints are creeping up",
                          f"Up {w[3]:.0f}% ({w[2]:,} to {w[1]:,} reviews) vs the previous 4 weeks."))
        b = min(big, key=lambda r: r[3])
        if b[3] <= -10:
            items.append(("Healthy", "pill-green", f"{b[0]} complaints are falling",
                          f"Down {abs(b[3]):.0f}% ({b[2]:,} to {b[1]:,} reviews) vs the previous 4 weeks."))
    dneg = neg_c - neg_p
    if abs(dneg) >= 1:
        up = dneg > 0
        items.append(("Needs attention" if up else "Healthy", "pill-red" if up else "pill-green",
                      f"Negative reviews are {'up' if up else 'down'}",
                      f"{neg_p:.0f}% to {neg_c:.0f}% of scored reviews, a change of {abs(dneg):.1f} pts."))
    if not items:
        items.append(("Healthy", "pill-green", "Nothing unusual", "No big changes vs the previous 4 weeks."))

    bot_l, bot_r = st.columns(2)
    with bot_l:
        h = head("What changed")
        for tag, cls, title, text in items:
            h += (f'<div class="item"><span class="pill {cls}">{tag}</span>'
                  f'<div><b style="color:#111827;">{title.replace("&", "&amp;")}</b><div class="small">{text}</div></div></div>')
        st.markdown(card_html(h), unsafe_allow_html=True)
    with bot_r:
        ca = full[full["week_start"].isin(last4) & full["language"].isin(LANG_LABEL.keys())]
        colours = {"english": "#1d4ed8", "hinglish": "#6b8ff0", "hindi_script": "#bcd0fb"}
        bar, table = "", ""
        for lg, label in LANG_LABEL.items():
            sub = ca[ca["language"] == lg]
            share = len(sub) / len(ca) * 100 if len(ca) else 0
            p_, n_ = pos_neg(sub)
            bar += f'<div style="width:{share:.1f}%;background:{colours[lg]};height:12px;"></div>'
            table += (f'<tr><td style="color:#111827;"><span style="color:{colours[lg]};">■</span> {label}</td>'
                      f'<td style="color:#111827;">{len(sub):,}</td><td style="color:#111827;">{share:.0f}%</td>'
                      f'<td style="color:#0b7a4b;">{p_:.0f}%</td><td style="color:#b42318;">{n_:.0f}%</td></tr>')
        h = head("Reviews by language", "Last 4 full weeks, all languages")
        h += (f'<div style="display:flex;border-radius:6px;overflow:hidden;margin:10px 0;">{bar}</div>'
              f'<table class="lang-table"><tr><th>Language</th><th>Reviews</th><th>Share</th><th>Positive</th><th>Negative</th></tr>{table}</table>'
              '<div class="small" style="margin-top:8px;">Hinglish and Hindi reviews look more negative mostly because they are longer, '
              'and long reviews are mostly complaints. Rows with few reviews can swing a lot.</div>')
        st.markdown(card_html(h), unsafe_allow_html=True)

    st.markdown('<div class="small">Positive and negative come from a sentiment model (about 96% accurate on a 300-review test set; '
                'the test labels were drafted by an AI assistant, not independently verified).</div>',
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

    comp = full[full["aspect"].isin(CATS)]
    cur_all = comp[comp["week_start"].isin(last4)]
    prv_all = comp[comp["week_start"].isin(prev4)]
    cur = cur_all[cur_all["aspect"] == cat]
    prv = prv_all[prv_all["aspect"] == cat]
    n_c, n_p = len(cur), len(prv)
    share_c = n_c / len(cur_all) * 100 if len(cur_all) else 0.0
    share_p = n_p / len(prv_all) * 100 if len(prv_all) else 0.0

    k = st.columns(3)
    ch = pct_change(n_c, n_p)
    t, g = delta(ch, f"{abs(ch):.0f}%", False, 5)
    kpi(k[0], "Complaint reviews, last 4 weeks", f"{n_c:,}", t, g)
    t, g = delta(share_c - share_p, f"{abs(share_c - share_p):.1f} pts", None, 0.5)
    kpi(k[1], "Share of all complaints with a topic", f"{share_c:.0f}%", t, g)
    if n_c and n_p:
        dr = cur["rating"].mean() - prv["rating"].mean()
        t, g = delta(dr, f"{abs(dr):.2f}", True, 0.05)
        rating_txt = f'{cur["rating"].mean():.2f}<span style="font-size:16px;color:#6b7280;"> / 5</span>'
    else:
        t, g, rating_txt = "Not enough reviews to compare", None, "n/a"
    kpi(k[2], "Average rating of these reviews", rating_txt, t, g)

    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card_c_trend"):
            st.markdown(head(f"{choice}: complaint reviews per week", "Full weeks only"), unsafe_allow_html=True)
            wk = full[full["aspect"] == cat].groupby("week_start").size().reset_index(name="reviews")
            fig = px.bar(wk, x="week_start", y="reviews")
            fig.update_traces(marker_color="#2563eb")
            fig = style_fig(fig, 380)
            fig.update_layout(yaxis_title="Reviews")
            st.plotly_chart(fig, use_container_width=True, theme=None)

    with right:
        h = head("What people say")
        ph = load_phrases()
        if ph is None:
            h += '<div class="small">Phrase file not found. Put <b>category_phrases_v3.csv</b> in the data folder.</div>'
        else:
            sub = ph[(ph["aspect"] == cat) & (ph["week_start"].isin(pd.to_datetime(list(last4))))]
            label = "Most common two-word phrases, last 4 weeks"
            if sub.empty:
                sub = ph[ph["aspect"] == cat]
                label = "Most common two-word phrases, all weeks"
            top = sub.groupby("phrase")["count"].sum().nlargest(8)
            h += f'<div class="small">{label}</div>'
            if top.empty:
                h += '<div class="small">No phrases to show.</div>'
            else:
                mx = top.max()
                for phrase, cnt in top.items():
                    h += (f'<div class="row"><div style="width:45%;color:#111827;">{htmllib.escape(str(phrase))}</div>'
                          f'<div class="barbg"><div class="bar" style="width:{cnt / mx * 100:.0f}%;background:#b42318;"></div></div>'
                          f'<div style="width:50px;text-align:right;font-size:13px;color:#111827;">{int(cnt):,}</div></div>')
                h += ('<div class="small">Counts only, no review text is stored. Phrases come from negative reviews in this category. '
                      'They are most useful for Delivery, Pricing and Support. Refunds and App problems are mixed groups, '
                      'so their phrases are rougher. Hinglish is not translated.</div>')
        st.markdown(card_html(h), unsafe_allow_html=True)

    st.markdown(card_html(head("What this category covers") +
                          f'<div class="small">{htmllib.escape(CAT_INFO[cat])} Categories come from topic modelling on longer '
                          'reviews, plus a classifier for short complaints, and were not fully validated.</div>'),
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
            cat_opts = ["All categories"] + sorted(s["aspect"].dropna().unique().tolist())
            cat = st.selectbox("Category", cat_opts, key="ex_cat")
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
            f'<span class="pill pill-grey">{htmllib.escape(str(row["aspect"]))}</span>'
            f'<span style="margin-left:auto;">{date_txt}</span></div>'
            f'<div class="rtext">{highlight(row["text"], q)}</div></div>',
            unsafe_allow_html=True)
    st.markdown('<div class="small">Sentiment and category tags are model predictions, so some are wrong. '
                'Reviews are shown as written (Hinglish is not translated).</div>', unsafe_allow_html=True)


# ---------------- page 4: about ----------------
def about():
    st.markdown('<div class="small">Swiggy app reviews</div>'
                '<h1 style="margin:0;padding:0;">About this data</h1>', unsafe_allow_html=True)
    bullets = [
        "<b>Data:</b> 39,457 Google Play reviews of Swiggy, Jul 12 to Oct 9, 2026.",
        "<b>Language:</b> tagged English, Hinglish or Hindi script with a pretrained word-level language model.",
        "<b>Sentiment:</b> TF-IDF + logistic regression trained on star ratings, about 96% accurate on a 300-review test set.",
        "<b>Complaint topics:</b> topic modelling on longer reviews, then a classifier for short complaints (about 82% on a 120-review check).",
        "<b>Review explorer:</b> a random sample of 300 reviews with emails and phone numbers masked and names checked by hand. The rest of the app uses counts only, with no review text.",
        "<b>Labels:</b> the test labels were drafted by an AI assistant and not independently verified.",
    ]
    st.markdown(card_html(head("How it was built") + "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"),
                unsafe_allow_html=True)


# ---------------- routing ----------------
if page == "Overview":
    overview()
elif page == "Category detail":
    category_detail()
elif page == "Review explorer":
    review_explorer()
else:
    about()