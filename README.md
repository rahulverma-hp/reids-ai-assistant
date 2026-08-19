# Reid's Distillery Operations Assistant

A prototype built for the Reid's Distillery Applied AI Solutions Development co-op
application, Fall 2026.

Reid's makes Signature, Citrus, Spiced and Navy Strength Gin plus Vodka, runs a
retail shop of around 90 items, and hosts tours, cocktail classes, Sunday High
Tea, weddings and concerts. That mix creates a lot of small repetitive admin jobs
between the still and the customer. This app takes on three of them.

A shorter summary written for the business rather than for developers is in
[BRIEF.md](BRIEF.md).

## Features

### Event and booking enquiries

Paste a tour, class, wedding or corporate enquiry and get a drafted reply that
recommends the experience matching what the customer described. If the enquiry
mentions guests who do not drink, it raises the non-alcoholic options without
being asked.

It answers only from an editable offerings reference. Anything that reference does
not cover, such as minimum spend, room capacity or voucher expiry, is listed under
"needs your input before sending" rather than guessed.

### Inventory

Tracks the botanicals and bottling supplies behind the spirits separately from
retail shop stock, because the two shortages cost different things:

- Production: juniper, angelica root, orris root, spirit, bottles, labels.
  Running short stalls a bottling run.
- Retail and classes: Fever Tree and 1642 mixers, Seedlip, glassware, dehydrated
  citrus garnishes. Running short loses a sale and can leave a cocktail class
  short.

Items are flagged out of stock, critical or low, production shortages sort first,
and the tool estimates what current stockouts are costing in missed shop revenue.

That estimate has one assumed input, units sold per week, exposed as a slider
because it is the one number the data cannot supply. Prices and dates come from
the data. Days are counted from the last restock, which is an upper bound on the
stockout window, and the UI says so.

Sample data mirrors Reid's real product lineup and listed retail prices. The
thirteen items at zero stock are the ones the store listed as sold out when this
was written.

### Document Q&A

Ask questions across your own documents: AGCO and excise paperwork, botanical
recipes and batch specs, tasting room scripts, venue policies. Answers cite the
source document and chunk, and the passages behind each answer are viewable.

A sample tasting room reference loads automatically so the tab works before you
upload anything. If nothing clears a relevance threshold the tool says so instead
of answering from the closest weak match.

## Stack

- Streamlit for the UI
- scikit-learn TF-IDF over word and bigram features for search
- LangChain text splitters for chunking
- DeepSeek via OpenRouter for generation
- Pandas for the inventory work

### Why not a vector database

The first version used ChromaDB with its default embedding model, which downloads
about 80MB of ONNX weights on first use. That download hung on a cold start and
left the page blank, which would have happened to anyone opening the app for the
first time.

For a few dozen SOPs and policy documents TF-IDF is accurate enough, starts
instantly and needs no network. Search sits behind two functions in
`retrieval.py`, so moving to embeddings later is a contained change.

## Setup

```bash
git clone https://github.com/rahulverma-hp/reids-ai-assistant.git
cd reids-ai-assistant

python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env         # then add your OpenRouter API key

streamlit run app.py
```

## Tests

28 tests, none of which call the model, so they run in seconds and give the same
result every time.

```bash
python test_inventory.py     # alerting rules and stockout cost arithmetic
python test_retrieval.py     # chunking, ranking and refusal
```

The retrieval tests cover refusal as well as recall, since a search layer that
always returns its least bad match is how a grounded tool ends up confidently
wrong. They include a regression test for a real bug: the word "does" in "how long
does the tour take" was matching a document that read "does not expire" and
outranking the one that actually described the tour.

## Layout

```
app.py                 Streamlit entry point, three tabs
event_assistant.py     Enquiry drafting against an editable offerings reference
inventory_helper.py    Stock alerting and stockout cost estimation
document_qa.py         Document Q&A tab
retrieval.py           Chunking and search, no Streamlit state so it is testable
config.py              API and chunking settings
test_inventory.py      Tests for the alerting and cost logic
test_retrieval.py      Tests for chunking, ranking and refusal
sample_docs/           Sample document preloaded into the Q&A tab
.streamlit/config.toml Theme and server defaults
BRIEF.md               Short summary written for the business
```

## Author

Rahul Verma, George Brown College, Applied A.I. Solutions Development.
[Portfolio](https://rahulverma-hp.github.io) |
[GitHub](https://github.com/rahulverma-hp)
