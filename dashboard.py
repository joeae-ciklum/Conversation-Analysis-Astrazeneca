import os, re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

from conv_parser import load_excel, parse_turns, detect_language

EXCEL_FILENAME = "other_domain_conversations_no_embedding_20260317_183612.xlsx"

def _find_excel():
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in [base, os.path.expanduser("~"),
                   os.path.join(os.path.expanduser("~"), "Downloads"),
                   os.path.join(os.path.expanduser("~"), "Desktop"),
                   os.path.join(base, "..")]:
        p = os.path.join(folder, EXCEL_FILENAME)
        if os.path.isfile(p): return os.path.abspath(p)
    return None

st.set_page_config(
    page_title="Conversation Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
.def-box{border-left:4px solid #185FA5;background:#EFF6FF;padding:.7rem 1rem;
          border-radius:0 8px 8px 0;font-size:.82rem;line-height:1.7;margin:.5rem 0 1rem;}
.def-title{font-weight:600;color:#0C447C;font-size:.85rem;margin-bottom:.3rem;}
.insight{border-left:4px solid;padding:.65rem 1rem;border-radius:0 8px 8px 0;
          font-size:.82rem;margin:.5rem 0 .9rem;line-height:1.65;}
.amber{border-color:#f59e0b;background:#fffbeb;color:#78350f;}
.red  {border-color:#ef4444;background:#fff1f0;color:#7f1d1d;}
.blue {border-color:#3b82f6;background:#eff6ff;color:#1e3a8a;}
.green{border-color:#22c55e;background:#f0fdf4;color:#14532d;}
.purple{border-color:#8b5cf6;background:#f5f3ff;color:#4c1d95;}
.health-green{color:#15803d;font-weight:600;}
.health-amber{color:#92400e;font-weight:600;}
.health-red  {color:#991b1b;font-weight:600;}
</style>""", unsafe_allow_html=True)

C = dict(blue="#185FA5",amber="#BA7517",red="#E24B4A",green="#1D9E75",
         purple="#7F77DD",gray="#888780",teal="#0F6E56",coral="#D85A30",pink="#D4537E")
OUTCOME_COLORS = {
    "Resolved":"#1D9E75","Informational Resolved":"#5DCAA5",
    "Geo-Blocked":"#7F77DD","Blocked / Policy":"#E24B4A",
    "Async / Pending":"#BA7517","Deflected to Human":"#0F6E56",
    "Incomplete - needs info":"#185FA5","Deflected / Unknown":"#888780",
}
TYPE_COLORS = {
    "Button Click / Notification Noise":"#888780","URL Share":"#B4B2A9",
    "Feedback / Negative Reaction":"#E24B4A","System / Meta":"#BA7517",
    "Science / R&D":"#7F77DD","HR Transactional":"#0F6E56",
    "IT / Access":"#185FA5","Knowledge Query":"#1D9E75","General / Unclear":"#D85A30",
}
QUALITY_COLORS = {"Direct":"#1D9E75","Partial":"#BA7517","Clarifying":"#185FA5",
                  "Deflected":"#888780","Blocked":"#E24B4A","Empty":"#B4B2A9"}
HEALTH_COLORS  = {"Green":"#1D9E75","Amber":"#BA7517","Red":"#E24B4A"}

def type_color(t):
    if str(t).startswith("Multilingual"): return "#D4537E"
    return TYPE_COLORS.get(t, C["gray"])

BASE = dict(font=dict(family="Inter,sans-serif",size=12,color="#374151"),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40,r=20,t=36,b=36),
            xaxis=dict(gridcolor="#f1f5f9",linecolor="#e2e8f0"),
            yaxis=dict(gridcolor="#f1f5f9",linecolor="#e2e8f0"),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
def T(fig,**kw): fig.update_layout(**BASE,**kw); return fig
def ins(text,color="amber"):
    st.markdown(f'<div class="insight {color}">💡 {text}</div>',unsafe_allow_html=True)
def insight(text,color="amber"):
    if show_ins: ins(text,color)
def defbox(title, body):
    st.markdown(f'<div class="def-box"><div class="def-title">📖 {title}</div>{body}</div>',
                unsafe_allow_html=True)

def show_records(df_sub, label, cols=None, key_suffix=""):
    """Expandable table showing the actual conversation records."""
    import hashlib
    n = len(df_sub)
    if n == 0: return
    default_cols = ["conversation_id","conv_type","language","outcome",
                    "bot_answer_quality","frustration_score","conv_health",
                    "noise_reason","first_user_msg"]
    show_cols = [c for c in (cols or default_cols) if c in df_sub.columns]
    unique_key = "drill_" + hashlib.md5(f"{label}_{key_suffix}".encode()).hexdigest()[:12]
    with st.expander(f"View {n:,} conversation records — {label}", expanded=False):
        st.caption(f"Showing up to 500 of {n:,} records. Use the search box to filter.")
        search = st.text_input("Filter by keyword", key=unique_key,
                               placeholder="Type to filter rows…")
        display = df_sub[show_cols].copy()
        if search:
            mask = display.apply(
                lambda col: col.astype(str).str.contains(search, case=False, na=False)
            ).any(axis=1)
            display = display[mask]
        st.dataframe(display.head(500), use_container_width=True, hide_index=True)

@st.cache_data(show_spinner="Loading and analysing conversations — ~60 seconds for large files…")
def load(path): return load_excel(path)


# SIDEBAR
with st.sidebar:
    st.title("Conversation Analysis")
    st.caption("Review on other bucket")
    st.divider()

    excel_path = _find_excel()
    if excel_path:
        st.success("File found")
        st.caption(f"`{EXCEL_FILENAME}`")
        df_full = load(excel_path)
        st.info(f"**{len(df_full):,}** conversations loaded")
    else:
        st.warning("Excel file not found. Upload below.")
        uploaded = st.file_uploader("Upload Excel", type=["xlsx","xls"])
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f:
                f.write(uploaded.read()); tmp = f.name
            df_full = load(tmp); os.unlink(tmp)
        else:
            st.stop()

    st.divider()
    st.caption("**Filters**")
    all_types    = sorted(df_full["conv_type"].unique().tolist())
    all_outcomes = sorted(df_full["outcome"].unique().tolist())
    all_months   = sorted(df_full["month"].dropna().unique().tolist())

    sel_types    = st.multiselect("Conversation type", all_types, default=all_types)
    sel_outcomes = st.multiselect("Outcome",           all_outcomes, default=all_outcomes)
    sel_health   = st.multiselect("Health", ["Green","Amber","Red"], default=["Green","Amber","Red"])
    sel_months   = st.multiselect("Month", all_months, default=all_months) if all_months else []

    min_t = int(df_full["n_total_turns"].min())
    max_t = int(df_full["n_total_turns"].max())
    turn_range = st.slider("Turn range", min_t, max(max_t,min_t+1), (min_t,max(max_t,min_t+1)))

    st.divider()
    show_ins = st.toggle("Show insights", value=True)
    page = st.radio("Navigate to", [
        "Summary",
        "Conversation Types",
        "Why Other",
        "Outcomes",
        "Bot Answer Quality",
        "Link Relevance",
        "Frustration & Sentiment",
        "Multilingual",
        "Timing Patterns",
        "Improvement Actions",
        "Conversation Explorer",
    ], label_visibility="collapsed")

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (df_full["conv_type"].isin(sel_types) &
        df_full["outcome"].isin(sel_outcomes) &
        df_full["conv_health"].isin(sel_health) &
        df_full["n_total_turns"].between(turn_range[0], turn_range[1]))
if sel_months: mask &= df_full["month"].isin(sel_months)
df = df_full[mask].copy()
N  = len(df)

if N == 0: st.warning("No conversations match the current filters."); st.stop()
st.caption(f"Showing **{N:,}** of {len(df_full):,} conversations")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Summary":
    st.title("Conversation Analysis")
    st.caption("All conversations shown here were classified into the 'Other' routing category.")

    noise_n   = (df["conv_type"]=="Button Click / Notification Noise").sum()
    url_n     = (df["conv_type"]=="URL Share").sum()
    real_n    = N - noise_n - url_n
    res_t     = (df["outcome"]=="Resolved").sum()
    res_i     = (df["outcome"]=="Informational Resolved").sum()
    total_res = res_t + res_i
    red_n     = (df["conv_health"]=="Red").sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total conversations",   f"{N:,}")
    c2.metric("Button click noise",    f"{noise_n:,}", f"{noise_n/N*100:.1f}% — not real queries")
    c3.metric("Real conversations",    f"{real_n:,}",  f"{real_n/N*100:.1f}% after removing noise")
    c4.metric("Transactional resolved",f"{res_t:,}",   f"{res_t/N*100:.1f}% — task completed")
    c5.metric("Informational resolved",f"{res_i:,}",   f"{res_i/N*100:.1f}% — direct answer given")
    c6.metric("Red health convos",     f"{red_n:,}",   f"{red_n/N*100:.1f}% — complete failures")

    st.divider()

    defbox("What counts as Resolved?",
           "There are two types of resolution:<br>"
           "<b>Transactional Resolved</b> — the bot completed a backend task (e.g. a Workday job title update). "
           "Detected when the bot reply contains phrases like <i>'successfully submitted'</i> or <i>'has been processed'</i>.<br>"
           "<b>Informational Resolved</b> — the bot gave a direct useful answer: a link, an ID number, an address, "
           "step-by-step instructions, or a factual definition. Detected when the bot reply contains a URL, "
           "<i>'your prid is'</i>, <i>'here is the link'</i>, etc. "
           "Button-click conversations are blocked from this category — a bot saying "
           "<i>'Glad you found this helpful!'</i> to a button click does <b>not</b> count as resolved.")

    res_df = pd.DataFrame({
        "Conversation example":[
            '"Change title" → bot: "successfully submitted"',
            '"Change weekly hours" → bot: "has been processed"',
            '"What is my PRID?" → bot gives the correct ID',
            '"Where is Degreed?" → bot gives the URL',
            '"Good to know!" → bot: "Glad you found this helpful!"',
            'Spanish query → bot replies with a direct answer',
        ],
        "Counted as Resolved?": [
            "✅  Yes — Transactional Resolved",
            "✅  Yes — Transactional Resolved",
            "✅  Yes — Informational Resolved",
            "✅  Yes — Informational Resolved",
            "❌  No — button click noise (not a real question)",
            "✅  Yes — Informational Resolved (if direct answer detected)",
        ],
    })
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Conversation type breakdown**")
        tc = df["conv_type"].value_counts().reset_index()
        tc.columns = ["type","count"]; tc["pct"] = (tc["count"]/N*100).round(1)
        fig = go.Figure(go.Bar(
            x=tc["count"], y=tc["type"], orientation="h",
            marker_color=[type_color(t) for t in tc["type"]],
            text=[f"{r['count']:,}  ({r['pct']}%)" for _,r in tc.iterrows()],
            textposition="outside",
        ))
        fig.update_layout(showlegend=False, xaxis_title="Conversations",
                          yaxis=dict(categoryorder="total ascending"),
                          height=max(320,len(tc)*38))
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Conversation health overview**")
        hc = df["conv_health"].value_counts().reset_index(); hc.columns = ["health","count"]
        fig2 = go.Figure(go.Pie(
            labels=hc["health"], values=hc["count"], hole=0.58,
            marker=dict(colors=[HEALTH_COLORS.get(h,C["gray"]) for h in hc["health"]],
                        line=dict(color="white",width=2)),
            textinfo="percent+label",
        ))
        T(fig2); st.plotly_chart(fig2, use_container_width=True)

    insight("The most important number is not the raw resolution rate — it is the split between Transactional and Informational resolved. Combined they represent every conversation that provided genuine value to the user.", "green")

    defbox("What does conversation health mean? (Green / Amber / Red)",
           "Every conversation is automatically rated on a three-level health scale based on its outcome, "
           "bot answer quality, and frustration score:<br><br>"
           "<b style='color:#166534;'>Green</b> — The conversation went well. The bot gave a "
           "resolved outcome (Transactional or Informational) and the user showed little or no "
           "frustration (score 0–1). Example: user asked for the Degreed link, bot gave the URL, "
           "conversation ended in one turn.<br><br>"
           "<b style='color:#92400e;'>Amber</b> — The conversation had issues but was not a "
           "complete failure. Triggers: outcome is Async/Pending (task started but not confirmed), "
           "bot gave a Partial or Clarifying response, or user frustration score is 2. "
           "Example: user asked for payslip, bot said 'process is ongoing, please allow 15 minutes' "
           "— something happened but the user left without a confirmed result.<br><br>"
           "<b style='color:#991b1b;'>Red</b> — The conversation was a complete failure. "
           "Triggers: outcome is Blocked/Policy or Geo-Blocked (hard wall, no alternative offered), "
           "frustration score is 3 or above, or the bot deflected with no useful answer and no "
           "escalation path. Example: user asked to change a job title, bot said 'I couldn't find "
           "that employee' four times while the user became increasingly frustrated.")

    st.divider()
    st.markdown("**Outcome distribution**")
    oc = df["outcome"].value_counts().reset_index(); oc.columns = ["outcome","count"]
    oc["pct"] = (oc["count"]/N*100).round(1)
    fig3 = go.Figure(go.Bar(
        x=oc["pct"], y=oc["outcome"], orientation="h",
        marker_color=[OUTCOME_COLORS.get(o,C["gray"]) for o in oc["outcome"]],
        text=[f"{r['pct']}%  ({r['count']:,})" for _,r in oc.iterrows()],
        textposition="outside",
    ))
    fig3.update_layout(showlegend=False, xaxis=dict(ticksuffix="%"),
                       yaxis=dict(categoryorder="total ascending"), height=320)
    T(fig3); st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONVERSATION TYPES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Conversation Types":
    st.title("Conversation Types")
    st.caption("Every conversation is classified into one of 10 types by reading the first user message.")

    DEFS = {
        "Button Click / Notification Noise": (
            "A record created when an employee clicks a button or reacts to a broadcast message in "
            "Microsoft Teams or Slack — for example clicking 'Good to know!' at the bottom of a bot "
            "announcement. No real question was asked. The click itself creates the conversation record "
            "automatically. These inflate the Other bucket and should be pre-filtered before routing."
        ),
        "URL Share": (
            "A record where the user's entire first message is a URL — for example pasting a ServiceNow "
            "link or clicking a Workvivo notification. No question was typed. The bot has nothing to "
            "respond to. These are notification-click artefacts."
        ),
        "Feedback / Negative Reaction": (
            "The user's first message is a rating or complaint about the bot's previous response: "
            "'Not helpful', 'This is wrong', 'Useless', etc. These are genuine signals that the bot "
            "failed the user in a prior conversation."
        ),
        "System / Meta": (
            "The user wants to change something about the chat session itself rather than ask a business "
            "question: 'Set my language to English', 'Clear history', 'Show my tickets'. These are "
            "configuration actions, not domain queries."
        ),
        "HR Transactional": (
            "The user wants the bot to perform an HR task in Workday — change a job title, update working "
            "hours, retrieve a payslip, request annual leave, update bank details, etc. The conversation "
            "type is detected when the first message contains known HR keywords."
        ),
        "IT / Access": (
            "The user has a technology access problem or IT support request — VPN issues, password resets, "
            "IDM access, ServiceNow tickets, Teams issues, etc."
        ),
        "Knowledge Query": (
            "The user is asking a genuine question wanting a factual answer: 'What is my PRID?', "
            "'Who is my HR business partner?', 'Where can I find the Degreed portal?', "
            "'What does VLP stand for?'. Detected when the first message contains question words or "
            "link-seeking phrases."
        ),
        "Science / R&D": (
            "A pharma, lab, or clinical query that is genuinely outside the scope of HR, IT, and Finance. "
            "Examples: dilution calculations, clinical trial terminology, molecule definitions. "
            "These correctly land in Other — there is no routing destination for them yet."
        ),
        "Multilingual": (
            "A real user question written in a non-English language — Spanish, Swedish, German, Polish, "
            "Italian, Turkish, Chinese, etc. The routing system was trained primarily on English, so "
            "these queries cannot be matched to HR/IT/Finance domains even when the question is clear."
        ),
        "General / Unclear": (
            "A real English message that contains no keyword matching any of the above categories. "
            "The user asked something genuine but in phrasing the system did not recognise."
        ),
    }

    st.divider()
    for ctype, grp in df.groupby("conv_type"):
        count   = len(grp)
        pct_val = count/N*100
        res     = grp["outcome"].isin(["Resolved","Informational Resolved"]).mean()*100
        avg_f   = grp["frustration_score"].mean()
        red_pct = (grp["conv_health"]=="Red").mean()*100

        with st.expander(
            f"**{ctype}** — {count:,} conversations ({pct_val:.1f}%)  "
            f"·  Resolution {res:.1f}%  ·  Avg frustration {avg_f:.1f}/5"
        ):
            if ctype in DEFS:
                defbox(f"What is '{ctype}'?", DEFS[ctype])
            elif ctype.startswith("Multilingual"):
                defbox(f"What is '{ctype}'?", DEFS["Multilingual"])

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Metrics**")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Conversations", f"{count:,}")
                m2.metric("Resolution %",  f"{res:.1f}%")
                m3.metric("Avg frustration",f"{avg_f:.1f}/5")
                m4.metric("Red health",    f"{red_pct:.0f}%")

                if ctype == "Button Click / Notification Noise" and "noise_reason" in grp.columns:
                    st.markdown("**Noise reason breakdown**")
                    nr = grp["noise_reason"].value_counts().reset_index()
                    nr.columns = ["reason","count"]
                    nr["pct"] = (nr["count"]/count*100).round(1)
                    st.dataframe(nr, use_container_width=True, hide_index=True)
                else:
                    st.markdown("**Top 10 most frequent first messages**")
                    top = Counter(grp["first_user_msg"].str.lower().str.strip()).most_common(10)
                    for msg, cnt in top:
                        st.caption(f"{cnt:,}×  `{str(msg)[:100]}`")

            with col_r:
                st.markdown("**Outcome breakdown**")
                oc = grp["outcome"].value_counts()
                for outcome, cnt in oc.items():
                    color = OUTCOME_COLORS.get(outcome, C["gray"])
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                        f'<div style="width:10px;height:10px;border-radius:2px;background:{color};flex-shrink:0;"></div>'
                        f'<span style="font-size:12px;">{outcome}: {cnt:,} ({cnt/count*100:.0f}%)</span></div>',
                        unsafe_allow_html=True)

            show_records(grp, ctype,
                         cols=["conversation_id","noise_reason","first_user_msg",
                                "outcome","bot_answer_quality","frustration_score","conv_health"],
                         key_suffix=f"type_{ctype[:20]}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHY OTHER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Why Other":
    st.title("Why Did These Conversations Land in 'Other'?")
    st.caption("Every conversation gets a plain-English root cause explaining why the routing system couldn't assign it to a named category.")

    defbox("How 'Why Other' is derived",
           "After all other features are computed, the system combines conversation type, language, "
           "first message word count, turn count, and bot reply patterns to generate a root cause string. "
           "Each root cause points to a specific, actionable fix.")

    reason_counts = df["why_other"].value_counts().reset_index()
    reason_counts.columns = ["reason","count"]
    reason_counts["pct"] = (reason_counts["count"]/N*100).round(1)

    col_l, col_r = st.columns([2,1])
    with col_l:
        fig = go.Figure(go.Bar(
            x=reason_counts["count"], y=reason_counts["reason"], orientation="h",
            marker_color=C["blue"],
            text=[f"{r['count']:,}  ({r['pct']}%)" for _,r in reason_counts.iterrows()],
            textposition="outside",
        ))
        fig.update_layout(showlegend=False, xaxis_title="Conversations",
                          yaxis=dict(categoryorder="total ascending"),
                          height=max(380, len(reason_counts)*38))
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        for _, row in reason_counts.iterrows():
            st.metric(f"{row['pct']}%", f"{row['count']:,}")
            st.caption(row["reason"][:70])
            st.divider()

    insight("Each root cause requires a different fix. Button-click noise → add a pre-filter. Non-English → add language routing. HR/IT vocabulary mismatch → add training phrases. Each has a different owner and a different effort level.", "blue")

    st.divider()
    st.markdown("**Drill down — see the actual records for each root cause**")
    for reason, grp in df.groupby("why_other"):
        show_records(grp, reason,
                     cols=["conversation_id","conv_type","language","noise_reason",
                            "first_user_msg","outcome","conv_health"],
                     key_suffix=f"why_{reason[:20]}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Outcomes":
    st.title("Outcome Analysis")
    st.caption("How each conversation ended — derived from reading the bot's reply text.")

    defbox("How outcomes are detected",
           "Outcomes are detected by scanning the bot's complete reply text for specific trigger phrases, "
           "checked in priority order. The first phrase that matches assigns the outcome and checking stops.<br><br>"
           "<b>Resolved</b> — Workday task-completion phrase found: 'successfully submitted', 'has been processed', etc.<br>"
           "<b>Informational Resolved</b> — Bot gave a direct useful answer (URL, ID, address, steps) — "
           "only for real conversations, not button-click noise.<br>"
           "<b>Geo-Blocked</b> — Bot mentioned an unsupported country: 'located in france', 'cannot be processed through this system'.<br>"
           "<b>Blocked / Policy</b> — Bot explicitly refused: 'cannot', 'unable to', 'couldn't find'.<br>"
           "<b>Async / Pending</b> — Task started but not confirmed: 'is still ongoing', 'will be notified'.<br>"
           "<b>Deflected to Human</b> — Bot passed to a person: 'connect you to', 'reach out to'.<br>"
           "<b>Incomplete - needs info</b> — Bot asked for more details: 'please provide', 'could you please'.<br>"
           "<b>Deflected / Unknown</b> — None of the above phrases found in the bot reply. Default catch-all — 77% of conversations.")

    res_t = (df["outcome"]=="Resolved").sum()
    res_i = (df["outcome"]=="Informational Resolved").sum()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Transactional resolved",   f"{res_t:,}",  f"{res_t/N*100:.1f}%")
    c2.metric("Informational resolved",   f"{res_i:,}",  f"{res_i/N*100:.1f}%")
    c3.metric("Blocked / Policy",         f"{(df['outcome']=='Blocked / Policy').sum():,}",
              f"{(df['outcome']=='Blocked / Policy').mean()*100:.1f}%")
    c4.metric("Deflected / Unknown",      f"{(df['outcome']=='Deflected / Unknown').sum():,}",
              f"{(df['outcome']=='Deflected / Unknown').mean()*100:.1f}%")
    c5.metric("Geo-Blocked",              f"{(df['outcome']=='Geo-Blocked').sum():,}",
              f"{(df['outcome']=='Geo-Blocked').mean()*100:.1f}%")

    st.divider()
    st.markdown("**Click any outcome to see the actual records**")
    for outcome, grp in df.groupby("outcome"):
        show_records(grp, f"{outcome} — {len(grp):,} conversations",
                     cols=["conversation_id","conv_type","language","noise_reason",
                            "first_user_msg","bot_answer_quality","frustration_score","conv_health"],
                     key_suffix=f"out_{outcome[:20]}")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Resolution funnel**")
        funnel = {
            "Conversation started":            N,
            "Real conversation (non-noise)":   int(N - (df["conv_type"]=="Button Click / Notification Noise").sum() - (df["conv_type"]=="URL Share").sum()),
            "Not geo-blocked":                 int(N*(1-df["geo_blocked"].mean())),
            "Not policy-blocked":              int(N*(1-df["bot_blocked"].mean())),
            "Informational resolved":          int(res_i),
            "Transactional resolved":          int(res_t),
        }
        fig = go.Figure(go.Funnel(
            y=list(funnel.keys()), x=list(funnel.values()),
            textinfo="value+percent initial",
            marker=dict(color=[C["blue"],C["teal"],C["purple"],C["amber"],"#5DCAA5",C["green"]]),
        ))
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Resolution rate by conversation type**")
        res = df.groupby("conv_type")["outcome"].apply(
            lambda x: x.isin(["Resolved","Informational Resolved"]).mean()*100).reset_index()
        res.columns = ["type","resolution_rate"]
        res = res.sort_values("resolution_rate", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=res["resolution_rate"], y=res["type"], orientation="h",
            marker_color=[C["green"] if v>10 else C["amber"] if v>5 else C["red"]
                          for v in res["resolution_rate"]],
            text=[f"{v:.1f}%" for v in res["resolution_rate"]], textposition="outside",
        ))
        fig2.update_layout(showlegend=False, xaxis=dict(ticksuffix="%"), height=400)
        T(fig2); st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BOT ANSWER QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bot Answer Quality":
    st.title("Bot Answer Quality")
    st.caption("Rates the quality of the bot's response — independent of whether a task was completed.")

    defbox("How bot answer quality is rated",
           "The bot's complete reply text is scanned for four signals:<br>"
           "<b>has_direct</b> — bot gave a concrete answer (URL, ID, address, steps, Workday completion).<br>"
           "<b>has_block</b> — bot explicitly refused ('cannot', 'unable to', 'couldn't find').<br>"
           "<b>has_deflect</b> — bot passed to human or gave a generic deflection.<br>"
           "<b>has_clarify</b> — bot asked the user for more information.<br><br>"
           "These combine into 6 quality ratings:<br>"
           "<b>Direct</b> — concrete answer, no deflection.<br>"
           "<b>Partial</b> — some info given AND also deflected or clarified.<br>"
           "<b>Clarifying</b> — bot asked for more info, nothing given yet.<br>"
           "<b>Deflected</b> — passed to human or gave nothing useful.<br>"
           "<b>Blocked</b> — explicitly refused to help.<br>"
           "<b>Empty</b> — bot reply was blank.")

    c1,c2,c3,c4,c5 = st.columns(5)
    for metric,col in [("Direct",c1),("Partial",c2),("Clarifying",c3),("Deflected",c4),("Blocked",c5)]:
        cnt = (df["bot_answer_quality"]==metric).sum()
        col.metric(metric, f"{cnt:,}", f"{cnt/N*100:.1f}%")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Quality distribution**")
        qc = df["bot_answer_quality"].value_counts().reset_index(); qc.columns = ["quality","count"]
        fig = go.Figure(go.Pie(
            labels=qc["quality"], values=qc["count"], hole=0.55,
            marker=dict(colors=[QUALITY_COLORS.get(q,C["gray"]) for q in qc["quality"]],
                        line=dict(color="white",width=2)), textinfo="percent+label",
        ))
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Quality by conversation type**")
        qa = df.groupby(["conv_type","bot_answer_quality"]).size().reset_index(name="count")
        qw = qa.pivot(index="conv_type", columns="bot_answer_quality", values="count").fillna(0)
        qp = qw.div(qw.sum(axis=1), axis=0)*100
        fig2 = go.Figure()
        for q in ["Direct","Partial","Clarifying","Deflected","Blocked","Empty"]:
            if q in qp.columns:
                fig2.add_trace(go.Bar(name=q, x=qp.index, y=qp[q].round(1),
                                      marker_color=QUALITY_COLORS.get(q,C["gray"])))
        fig2.update_layout(barmode="stack", yaxis=dict(ticksuffix="%"), xaxis=dict(tickangle=25))
        T(fig2); st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**Drill down by quality rating**")
    for quality, grp in df.groupby("bot_answer_quality"):
        show_records(grp, f"{quality} quality — {len(grp):,} conversations",
                     cols=["conversation_id","conv_type","first_user_msg","outcome","conv_health"],
                     key_suffix=f"q_{quality}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LINK RELEVANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Link Relevance":
    st.title("Link Relevance Analysis")
    st.caption("When a user asked for a specific resource and the bot provided a URL — was it the right one?")

    defbox("How link relevance is determined",
           "When the bot's reply contains a URL, the system cross-checks whether the URL domain matches "
           "what the user was asking for using 11 topic-to-domain rules.<br><br>"
           "<b>Relevant</b> — user asked for a resource and bot gave a matching URL "
           "(e.g. user: 'send me the Degreed link' → bot URL contains 'degreed').<br>"
           "<b>Irrelevant</b> — user asked for resource X but bot gave a URL for resource Y "
           "(e.g. user asked for Degreed, bot gave Workvivo — wrong link).<br>"
           "<b>No link</b> — user asked for a resource but bot gave no URL at all.<br>"
           "<b>Not asked</b> — user was not asking for a link; link relevance is not applicable.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Relevant link given",      f"{(df['link_relevance']=='Relevant').sum():,}",   f"{(df['link_relevance']=='Relevant').mean()*100:.1f}%")
    c2.metric("WRONG link given",         f"{(df['link_relevance']=='Irrelevant').sum():,}", f"{(df['link_relevance']=='Irrelevant').mean()*100:.1f}%")
    c3.metric("Asked but no link given",  f"{(df['link_relevance']=='No link').sum():,}",    f"{(df['link_relevance']=='No link').mean()*100:.1f}%")
    c4.metric("Link not requested",       f"{(df['link_relevance']=='Not asked').sum():,}",  f"{(df['link_relevance']=='Not asked').mean()*100:.1f}%")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        lc = df["link_relevance"].value_counts().reset_index(); lc.columns = ["relevance","count"]
        color_map = {"Relevant":C["green"],"Irrelevant":C["red"],"No link":C["amber"],"Not asked":C["gray"]}
        fig = go.Figure(go.Bar(
            x=lc["count"], y=lc["relevance"], orientation="h",
            marker_color=[color_map.get(r,C["gray"]) for r in lc["relevance"]],
            text=lc["count"], textposition="outside",
        ))
        fig.update_layout(showlegend=False, xaxis_title="Conversations",
                          yaxis=dict(categoryorder="total ascending"))
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Conversations where user got the wrong link**")
        irr = df[df["link_relevance"]=="Irrelevant"][
            ["conversation_id","first_user_msg","outcome","conv_type"]].head(30)
        if len(irr):
            st.dataframe(irr, use_container_width=True, hide_index=True)
        else:
            st.success("No irrelevant links in current filter.")

    st.divider()
    for relevance, grp in df.groupby("link_relevance"):
        if relevance == "Not asked": continue
        show_records(grp, f"{relevance} — {len(grp):,} conversations",
                     cols=["conversation_id","conv_type","first_user_msg","outcome"],
                     key_suffix=f"lr_{relevance}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — FRUSTRATION & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Frustration & Sentiment":
    st.title("Frustration & Sentiment Analysis")
    st.caption("Frustration score (0–5) derived from user behaviour patterns — no survey needed.")

    defbox("How the frustration score (0–5) is calculated",
           "Five independent signals, each worth 1 point — capped at 5:<br>"
           "<b>+1 Repeated message</b> — user sent the exact same message more than once "
           "(bot didn't understand, user tried again).<br>"
           "<b>+1 Urgency words</b> — user typed: 'urgent', 'asap', 'deadline', 'by today', 'eod', 'immediately'.<br>"
           "<b>+1 Frustration language</b> — user typed: 'not working', 'already told', 'again', "
           "'wrong', 'ridiculous', 'waste', 'useless'.<br>"
           "<b>+1 Message escalation</b> — last message is 1.8× longer than first message "
           "(user adds more and more context because the bot keeps failing them).<br>"
           "<b>+1 Long loop</b> — user sent 4 or more messages in the same conversation.<br><br>"
           "<b>Score 0–1 = Calm</b> · <b>Score 2 = Mild frustration</b> · <b>Score 3–5 = High frustration → Red health</b>")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg frustration score",      f"{df['frustration_score'].mean():.2f} / 5")
    c2.metric("High frustration (≥ 3)",     f"{(df['frustration_score']>=3).sum():,}",
              f"{(df['frustration_score']>=3).mean()*100:.1f}%")
    c3.metric("Negative sentiment",         f"{(df['sentiment']=='Negative').sum():,}",
              f"{(df['sentiment']=='Negative').mean()*100:.1f}%")
    c4.metric("Users who repeated themselves", f"{df['has_repeated_user'].sum():,}",
              f"{df['has_repeated_user'].mean()*100:.1f}%")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Frustration score distribution**")
        fc = df["frustration_score"].value_counts().sort_index().reset_index()
        fc.columns = ["score","count"]
        fig = go.Figure(go.Bar(
            x=fc["score"].astype(str), y=fc["count"],
            marker_color=[C["green"] if s<=1 else C["amber"] if s<=2 else C["red"] for s in fc["score"]],
            text=fc["count"], textposition="outside",
        ))
        fig.update_layout(xaxis_title="Frustration score", yaxis_title="Conversations", showlegend=False)
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Avg frustration by conversation type**")
        af = df.groupby("conv_type")["frustration_score"].mean().reset_index()
        af.columns = ["type","avg_frustration"]
        af = af.sort_values("avg_frustration", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=af["avg_frustration"], y=af["type"], orientation="h",
            marker_color=[C["red"] if v>=3 else C["amber"] if v>=2 else C["green"] for v in af["avg_frustration"]],
            text=[f"{v:.2f}" for v in af["avg_frustration"]], textposition="outside",
        ))
        fig2.update_layout(showlegend=False, xaxis_title="Avg frustration score", height=400)
        T(fig2); st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**Drill down by frustration score**")
    for score in [5,4,3,2,1,0]:
        grp = df[df["frustration_score"]==score]
        if len(grp)==0: continue
        show_records(grp, f"Score {score} — {len(grp):,} conversations",
                     cols=["conversation_id","conv_type","first_user_msg","outcome","conv_health"],
                     key_suffix=f"fs_{score}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MULTILINGUAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Multilingual":
    st.title("Multilingual Query Analysis")
    st.caption("Real user questions in non-English languages that the routing system couldn't classify.")

    defbox("Why multilingual queries land in 'Other'",
           "The routing system's classifier was trained primarily on English text. When an employee "
           "writes a perfectly clear question in Spanish, Swedish, German, Polish, or another language, "
           "the system cannot match it to an HR, IT, or Finance domain — even if the equivalent English "
           "question would be classified immediately.<br><br>"
           "Detection: the system counts non-ASCII characters and checks whether 2 or more words from "
           "each language's vocabulary list appear in the message. Languages detected: Spanish, Swedish, "
           "German, Polish, Italian, French, Portuguese, Turkish, Chinese.")

    ml_df = df[df["language"] != "English"]
    c1,c2,c3 = st.columns(3)
    c1.metric("Non-English conversations", f"{len(ml_df):,}", f"{len(ml_df)/N*100:.1f}% of total")
    c2.metric("Languages detected",        df["language"].nunique()-1)
    c3.metric("Resolution rate",
              f"{ml_df['outcome'].isin(['Resolved','Informational Resolved']).mean()*100:.1f}%")

    col_l, col_r = st.columns(2)
    with col_l:
        lc = df[df["language"]!="English"]["language"].value_counts().reset_index()
        lc.columns = ["language","count"]
        fig = go.Figure(go.Bar(
            x=lc["count"], y=lc["language"], orientation="h",
            marker_color=C["pink"], text=lc["count"], textposition="outside",
        ))
        fig.update_layout(showlegend=False, xaxis_title="Conversations",
                          yaxis=dict(categoryorder="total ascending"), height=400)
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        ml_oc = ml_df["outcome"].value_counts().reset_index(); ml_oc.columns = ["outcome","count"]
        fig2 = go.Figure(go.Pie(
            labels=ml_oc["outcome"], values=ml_oc["count"], hole=0.55,
            marker=dict(colors=[OUTCOME_COLORS.get(o,C["gray"]) for o in ml_oc["outcome"]],
                        line=dict(color="white",width=2)), textinfo="percent+label",
        ))
        T(fig2); st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**Drill down by language**")
    for lang, grp in ml_df.groupby("language"):
        show_records(grp, f"{lang} — {len(grp):,} conversations",
                     cols=["conversation_id","first_user_msg","outcome","bot_answer_quality","conv_health"],
                     key_suffix=f"lang_{lang[:15]}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — TIMING PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Timing Patterns":
    st.title("Timing Patterns")
    st.caption("When do conversations happen, and what does timing tell us about what users need?")

    df_t = df[df["hour"].notna()].copy()
    if df_t.empty:
        st.info("No datetime information available."); st.stop()

    col_l, col_r = st.columns(2)
    with col_l:
        hourly = df_t.groupby("hour").size().reset_index(name="count")
        peak_h = int(hourly.loc[hourly["count"].idxmax(),"hour"])
        fig = go.Figure(go.Bar(
            x=hourly["hour"], y=hourly["count"],
            marker_color=[C["red"] if h==peak_h else C["blue"] for h in hourly["hour"]],
            text=hourly["count"], textposition="outside",
        ))
        fig.update_layout(xaxis=dict(title="Hour of day",dtick=1),
                          yaxis_title="Conversations", showlegend=False)
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with col_r:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily = df_t.groupby("day_of_week").size().reset_index(name="count")
        daily["day_of_week"] = pd.Categorical(daily["day_of_week"],categories=day_order,ordered=True)
        daily = daily.sort_values("day_of_week")
        peak_d = str(daily.loc[daily["count"].idxmax(),"day_of_week"])
        fig2 = go.Figure(go.Bar(
            x=daily["day_of_week"].astype(str), y=daily["count"],
            marker_color=[C["red"] if str(d)==peak_d else C["blue"] for d in daily["day_of_week"].astype(str)],
            text=daily["count"], textposition="outside",
        ))
        T(fig2); st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**Conversation volume by time slot**")
    if df_t["time_slot"].notna().sum() > 0:
        ts = df_t.groupby("time_slot").agg(
            total=("conversation_id","count"),
            resolved=("outcome", lambda x: x.isin(["Resolved","Informational Resolved"]).sum()),
        ).reset_index()
        ts["res_rate"] = (ts["resolved"]/ts["total"]*100).round(1)
        ts_order = ["Morning (6–12)","Afternoon (12–17)","Evening (17–22)","Night (22–6)"]
        ts["time_slot"] = pd.Categorical(ts["time_slot"], categories=ts_order, ordered=True)
        ts = ts.sort_values("time_slot")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(
            x=ts["time_slot"].astype(str), y=ts["total"],
            name="Volume", marker_color="#B5D4F4",
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=ts["time_slot"].astype(str), y=ts["res_rate"],
            name="Resolution %", mode="lines+markers",
            line=dict(color=C["green"], width=2.5), marker=dict(size=8),
        ), secondary_y=True)
        fig3.update_yaxes(title_text="Conversations", secondary_y=False, gridcolor="#f1f5f9")
        fig3.update_yaxes(title_text="Resolution %", ticksuffix="%", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        T(fig3); st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — IMPROVEMENT ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Improvement Actions":
    st.title("Improvement Actions")
    st.caption("Prioritised, quantified actions to reduce the Other bucket and improve conversation quality.")

    total     = len(df_full)
    noise_n   = (df_full["conv_type"]=="Button Click / Notification Noise").sum()
    url_n     = (df_full["conv_type"]=="URL Share").sum()
    ml_n      = (df_full["language"]!="English").sum()
    hr_it_n   = df_full["conv_type"].isin(["HR Transactional","IT / Access"]).sum()
    irr_n     = (df_full["link_relevance"]=="Irrelevant").sum()
    nolink_n  = (df_full["link_relevance"]=="No link").sum()
    hi_frust  = (df_full["frustration_score"]>=3).sum()
    sci_n     = (df_full["conv_type"]=="Science / R&D").sum()

    defbox("Priority levels",
           "<b>P0 — Zero effort</b>: one pre-filter rule, no model change, immediate impact.<br>"
           "<b>P1 — Low effort</b>: model or routing configuration change.<br>"
           "<b>P2 — Medium effort</b>: content, knowledge base, or vocabulary change.<br>"
           "<b>P3 — Low effort</b>: logic or flow improvement in the conversation design.")

    actions = [
        ("P0","red",  "Pre-filter: discard button-click noise records before routing",
         noise_n, "These are Teams/Slack button clicks, not real user questions. One regex rule removes them.",
         "None — one regex check"),
        ("P0","red",  "Pre-filter: discard bare URL paste messages",
         url_n, "Notification link clicks create empty conversation records. Zero real intent.",
         "None — one regex check"),
        ("P1","amber","Add language detection before routing for non-English queries",
         ml_n, "Route by detected language first, then to the correct domain. Spanish HR questions belong in HR.",
         "Low — add one pre-routing detection step"),
        ("P1","amber","Add misrouted HR/IT phrases to routing vocabulary",
         hr_it_n, "Conversations with clear HR/IT intent that used phrasing the routing system doesn't recognise.",
         "Medium — labelling exercise using the top message list below"),
        ("P2","blue", "Fix wrong link responses in knowledge base",
         irr_n, "Users asked for specific resources and received URLs for different resources.",
         "Medium — audit knowledge base link mappings"),
        ("P2","blue", "Add missing links for conversations where user asked but got nothing",
         nolink_n, "User asked for a resource, bot had no URL to give. Missing KB content.",
         "Medium — add KB articles for top unanswered resource requests"),
        ("P2","blue", "Create Science/R&D routing destination",
         sci_n, "Pharma/lab queries have no routing home. They need a dedicated destination.",
         "Medium — new routing rule + KB content"),
        ("P3","green","Add frustration-triggered human escalation (score ≥ 3)",
         hi_frust, "When a user's frustration score reaches 3, automatically offer a human agent.",
         "Low — add frustration check to conversation flow logic"),
    ]

    for pri, color, action, n, rationale, effort in actions:
        pct = round(n/total*100, 1)
        with st.expander(f"**{pri}** — {action}  ·  {n:,} conversations ({pct}%)"):
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(f"**Rationale:** {rationale}")
                st.markdown(f"**Effort:** {effort}")
            with c2:
                st.metric("Conversations affected", f"{n:,}")
                st.metric("% of Other bucket", f"{pct}%")

    st.divider()
    st.markdown("### Top messages to add to routing vocabulary")
    st.caption("The most repeated first messages — label these and add to the routing vocabulary.")
    real_df  = df_full[~df_full["conv_type"].isin(["Button Click / Notification Noise","URL Share"])]
    top_msgs = Counter(real_df["first_user_msg"].str.lower().str.strip()).most_common(80)
    top_df   = pd.DataFrame(top_msgs, columns=["first_message","count"])
    top_df   = top_df[top_df["first_message"].str.len() > 3]
    top_df["pct_of_other"] = (top_df["count"]/total*100).round(3)
    top_df["suggested_label"] = top_df["first_message"].apply(
        lambda m: ("HR Transactional" if any(s in m for s in ["title","hours","leave","payslip","absence","workday"])
              else "IT / Access"       if any(s in m for s in ["ticket","idm","vpn","password","teams","access","login"])
              else "Knowledge Query"   if any(s in m for s in ["what","who","where","how","can you","find","link","send"])
              else "System / Meta"     if any(s in m for s in ["clear","language","history","reset"])
              else "Review manually"))
    st.dataframe(top_df.head(60), use_container_width=True, hide_index=True)
    st.download_button("Download training candidates CSV",
                       data=top_df.to_csv(index=False),
                       file_name="training_candidates.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — CONVERSATION EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Conversation Explorer":
    st.title("Conversation Explorer")
    st.caption("Read any conversation in full with automatic quality annotations. Search across all conversations by keyword.")

    col_s, col_f = st.columns([2,2])
    with col_s:
        type_filter   = st.selectbox("Filter by type",   ["All"] + all_types)
        health_filter = st.selectbox("Filter by health",  ["All","Green","Amber","Red"])
        filtered = df.copy()
        if type_filter   != "All": filtered = filtered[filtered["conv_type"]==type_filter]
        if health_filter != "All": filtered = filtered[filtered["conv_health"]==health_filter]
        selected_id = st.selectbox("Select conversation",
                                   filtered["conversation_id"].tolist()
                                   if len(filtered) else df["conversation_id"].tolist())
    with col_f:
        search_term = st.text_input("Search all transcripts", placeholder="keyword…")

    if search_term:
        mask2   = df["raw_transcript"].str.contains(search_term,case=False,na=False)
        matches = df[mask2]
        st.info(f"**'{search_term}'** found in **{len(matches):,}** conversations")
        if not matches.empty:
            st.dataframe(
                matches[["conversation_id","conv_type","language","noise_reason",
                          "outcome","bot_answer_quality","link_relevance",
                          "frustration_score","conv_health","n_total_turns"]].head(50),
                use_container_width=True, hide_index=True)
        st.divider()

    row = df[df["conversation_id"]==selected_id]
    if row.empty: st.warning("Conversation not found."); st.stop()
    row = row.iloc[0]

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Type",           str(row["conv_type"])[:20])
    c2.metric("Language",       row["language"])
    c3.metric("Outcome",        row["outcome"])
    c4.metric("Answer quality", row["bot_answer_quality"])
    c5.metric("Link relevance", row["link_relevance"])
    c6.metric("Frustration",    f"{row['frustration_score']}/5")
    health_val   = row["conv_health"]
    health_color = {"Green":"green","Amber":"amber","Red":"red"}.get(health_val,"amber")
    c7.markdown(f"**Health**<br><span class='health-{health_color}'>{health_val}</span>",
                unsafe_allow_html=True)

    if row.get("noise_reason",""):
        st.info(f"**Noise reason:** {row['noise_reason']}")

    st.markdown(f"**Why it landed in Other:** _{row['why_other']}_")
    st.divider()

    turns = parse_turns(row["raw_transcript"])
    for i, (speaker, text) in enumerate(turns):
        if speaker == "User":
            with st.chat_message("user"):
                st.markdown(text)
                st.caption(f"Turn {i+1}  ·  {len(text.split())} words")
        else:
            with st.chat_message("assistant"):
                st.markdown(text[:3000] + ("…" if len(text)>3000 else ""))
                badges = []
                fc = text.lower().count("feel free to")
                if fc >= 2:  badges.append(f"⚠️ {fc}× filler phrase")
                if any(p in text.lower() for p in ["still ongoing","currently ongoing"]): badges.append("⏳ Async")
                if any(p in text.lower() for p in ["successfully submitted","has been submitted"]): badges.append("✅ Success")
                if any(p in text.lower() for p in ["can't","cannot","couldn't find"]): badges.append("🚫 Blocked")
                if any(p in text.lower() for p in ["located in","cannot be processed through"]): badges.append("🌍 Geo-block")
                if "http" in text.lower(): badges.append("🔗 Link provided")
                if badges: st.caption("  ·  ".join(badges))

    st.divider()
    st.markdown("**Anomaly flags**")
    flags = {
        "User repeated a message":      row["has_repeated_user"],
        "Urgency language detected":    row["has_urgency"],
        "Bot blocked by policy":        row["bot_blocked"],
        "Geo-blocked":                  row["geo_blocked"],
        "Async / pending state":        row["bot_async"],
        "Multi-response stitching bug": row["multi_bot_resp"],
        "Contradictory status":         row["contradictory"],
    }
    c1, c2 = st.columns(2)
    items = list(flags.items())
    for i, (flag, val) in enumerate(items):
        with (c1 if i < len(items)//2+1 else c2):
            st.markdown(f"{'🔴' if val else '🟢'}  {flag}")