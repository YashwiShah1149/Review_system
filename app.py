# ---------------- page 4: about (replace your old about() with this whole block) ----------------
# Uses helpers that already exist in app.py: card_html, head, htmllib, st

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
                      "clustered on their own), and emoji-only reviews skip sentiment and topics.")
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
                "<b>No stop-word removal</b> for the models. It only matters for the phrase counts shown on Category detail.",
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
            h = head("Result: share of complaints by category", "Full weeks, reviews with a clear topic")
            h += _tbl(["Category", "Share"], [
                ["Refunds, cancellations &amp; order problems", "29.8%"],
                ["App problems &amp; general complaints", "23.5%"],
                ["Customer service &amp; support", "15.5%"],
                ["Delivery", "12.7%"],
                ["Pricing &amp; charges", "11.0%"],
                ["Food quality", "7.4%"],
            ])
            st.markdown(card_html(h), unsafe_allow_html=True)

        h = head("What the clusters looked like")
        h += _bullets([
            "<b>Clear:</b> Delivery delay, Pricing and charges, Cancellations and refunds, Food quality.",
            "<b>Overlapping:</b> Refund and support complaints (merged into one category), "
            "App problems and fraud.",
            "<b>Hinglish:</b> 871 of 1,337 Hinglish reviews fell into one cluster, so they were re-clustered. "
            "About 680 of them (roughly half) got a topic that could be trusted. The rest are marked Other / unclear.",
            "<b>Phrases:</b> the Category detail page counts the most common two-word phrases per category and week.",
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
            "only if they are negative. Long-review categories can include some praise "
            "(Food quality is about 57% positive).",
            "<b>Short-review topic accuracy:</b> the final model was trained on the same 120 reviews used for the "
            "82% check, so there is no clean held-out score for it.",
            "<b>Hinglish topics:</b> only about half are reliable.",
            "<b>Time window:</b> about 3 months of one app on one store, so seasonal effects are not visible.",
        ])
        st.markdown(card_html(h), unsafe_allow_html=True)