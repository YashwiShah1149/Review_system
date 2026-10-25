
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


# ---------------- page navigation / app entry point ----------------
if page == "Overview":
    overview()
elif page == "Category detail":
    category_detail()
elif page == "Review explorer":
    review_explorer()
else:
    about()
