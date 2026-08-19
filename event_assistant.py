"""Drafts replies to event and booking enquiries.

Answers only from the offerings text below, which is editable in the UI. Anything
it does not cover is flagged for a person instead of guessed, so the draft never
invents a price or a capacity.
"""

import streamlit as st
from openai import OpenAI

import config

# Values marked [CONFIRM] are left blank on purpose so the model flags them
# instead of inventing a number.
DEFAULT_OFFERINGS = """REID'S DISTILLERY, TORONTO
Craft distillery. Signature Gin voted Best Classic Gin in Canada at the 2023
World Gin Awards.

SPIRITS (750ml unless noted)
- Reid's Signature Gin, $37
- Reid's Citrus Gin, $37
- Reid's Spiced Gin, $37
- Reid's Navy Strength Gin, $59
- Reid's Vodka, $30
- Reid's Blood Orange Aperitivo Bitter Liqueur, $23
- Citronino, $30
- 50ml minis, $6 each, or a Mini 3-pack for $14
- Case of 6 (Signature, Citrus or Spiced), $221
- Personalised 750ml bottle, $48

READY TO SERVE AND KITS
- Negroni, Spiced Maple Manhattan, $23 each
- Strawberry Lime Gimlet, $25
- Gin and Tonic six-pack, $14
- French 75 Social Pack, $92; personalised bottle cocktail kit, $100
- Celebration Pack, $349
- Gin and tonic gift sets, from $41

EXPERIENCES
1. Signature Distillery Tour and Tasting, $51 per person (gift voucher price)
   - Group experience, approximately 90 minutes
   - Includes a finishing cocktail to enjoy afterward
   - Group size limits: [CONFIRM]

2. Reid's Cocktail Class, $51 per person (gift voucher price)
   - Open to all levels, no bartending experience needed
   - Guests make 3 cocktails and enjoy them in the lounge afterward
   - Group size limits: [CONFIRM]

3. Reid's Sunday High Tea, $51 per person (gift voucher price)

4. Free 30-minute virtual tasting session, no cost, bookable online

5. Weddings and private events
   - Venue space rentals and custom private experiences
   - Starts with an enquiry so the team can advise on date, headcount and format
   - Pricing, minimum spend, catering and capacity: [CONFIRM]

6. Concerts and special events, live music and ticketed events on site

NON-ALCOHOLIC OPTIONS
- Seedlip distilled non-alcoholic spirits, $34 (Garden 108, Grove 42, Spice 94)
- Full mixer range (Fever Tree, 1642), syrups and garnishes, so non-drinking
  guests can be served the same cocktails without the spirit
- Grove 42 and Spice 94 are currently out of stock

RETAIL SHOP
Around 90 items: cocktail kits and social packs, ready-to-drink cocktails,
mixers, bitters, syrups, dehydrated citrus garnishes, glassware, bar tools,
branded merchandise and gift sets.

BOOKING NOTES
- Tours, classes, High Tea and gift vouchers are bought through the online store
- Online gift cards available from $18.37
- Private events and weddings start with an enquiry form
- Accessibility and dietary accommodations: [CONFIRM]
- Voucher expiry and date-change policy: [CONFIRM]
"""

SAMPLE_INQUIRIES = {
    "Corporate team booking": (
        "Hi there,\n\nI'm organizing a team offsite for 14 people from my company "
        "in early October and we'd love to do something hands-on. A few of the team "
        "don't drink alcohol. Could you let me know what you offer, roughly what it "
        "costs per person, and whether a Thursday evening would work?\n\nThanks,\nPriya"
    ),
    "Wedding enquiry": (
        "Hello,\n\nMy fiance and I are getting married next summer and are looking "
        "for somewhere different for our reception, around 60 guests. Is your space "
        "available for weddings, and can you cater? Would love to come see it."
        "\n\nBest,\nDaniel"
    ),
    "Gift voucher question": (
        "Hi! I want to buy the cocktail class as a birthday gift for my sister but "
        "she may not be able to use it for a few months. Do the vouchers expire, "
        "and can she pick her own date? Also is it suitable for a complete "
        "beginner?\n\nThanks, Meera"
    ),
}


def get_client():
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def draft_reply(inquiry: str, offerings: str, tone: str) -> str:
    response = get_client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You draft email replies for the events team at Reid's "
                    "Distillery, a craft distillery in Toronto.\n\n"
                    "Rules:\n"
                    "1. Use only the offerings information provided. Never invent "
                    "prices, capacities, dates, availability or policies.\n"
                    "2. If the customer asks something the information marks "
                    "[CONFIRM] or does not cover, do not guess. Write the reply "
                    "without that detail, then list it under a heading "
                    "'NEEDS YOUR INPUT BEFORE SENDING'.\n"
                    "3. Recommend the experience that fits what they described, and "
                    "say why in one line.\n"
                    "4. If the enquiry mentions guests who do not drink alcohol, "
                    "mention the non-alcoholic options without being asked.\n"
                    "5. Keep it warm and concise. This is a small business, not a "
                    "call centre. No corporate filler.\n"
                    "6. End with a clear next step.\n\n"
                    f"Tone: {tone}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OFFERINGS INFORMATION:\n{offerings}\n\n"
                    f"CUSTOMER ENQUIRY:\n{inquiry}\n\n"
                    "Draft the reply."
                ),
            },
        ],
        temperature=0.4,
        max_tokens=900,
    )
    return response.choices[0].message.content


def render():
    st.header("Event and Booking Enquiries")
    st.write(
        "Turns an incoming tour, class, wedding or private event enquiry into a "
        "ready-to-send draft, and flags anything it should not answer on its own."
    )

    sample = st.selectbox(
        "Load a sample enquiry, or paste your own below",
        ["None"] + list(SAMPLE_INQUIRIES.keys()),
    )
    inquiry = st.text_area(
        "Customer enquiry",
        value=SAMPLE_INQUIRIES.get(sample, ""),
        height=180,
    )

    tone = st.radio(
        "Tone",
        ["Warm and friendly", "Polished and professional", "Short and direct"],
        horizontal=True,
    )

    with st.expander("Offerings reference (edit to match current pricing)"):
        offerings = st.text_area(
            "The assistant answers only from this text",
            value=DEFAULT_OFFERINGS,
            height=320,
        )

    if not st.button("Draft reply"):
        return

    if not config.DEEPSEEK_API_KEY:
        st.error("OpenRouter API key not set. Add it to your .env file.")
        return
    if not inquiry.strip():
        st.warning("Paste an enquiry first, or load one of the samples.")
        return

    with st.spinner("Drafting..."):
        reply = draft_reply(inquiry, offerings, tone)

    st.subheader("Draft")
    st.markdown(reply)
    st.download_button("Download draft", reply, file_name="reply_draft.txt")
