# Reid's Distillery Operations Assistant

Rahul Verma, Fall 2026 co-op application

I built a working prototype rather than describing one. It takes on three of the
repetitive jobs that sit between the still and the customer. All three run now,
against your real product lineup and prices.

## What it does

**Drafts replies to event enquiries.** Paste a tour, class, wedding or corporate
enquiry and get a reply that recommends the right experience and explains why. If
the enquiry mentions guests who do not drink, it brings up the Seedlip and mixer
options without being asked. It answers only from an offerings sheet you can edit,
and anything that sheet does not cover, such as minimum spend or room capacity,
comes back flagged for you rather than guessed.

**Shows what stockouts are costing.** It separates the botanicals and bottling
supplies behind the gin from retail shop stock, because those two shortages cost
different things. Short on angelica root and a bottling run stalls. Short on tonic
and you lose sales today. It then prices the current retail stockouts in missed
revenue.

**Answers questions from your own documents.** Upload AGCO paperwork, batch specs,
tasting room scripts or venue policies and ask questions across them. Every answer
cites the document it came from, so it can be checked.

## One thing the data showed

Thirteen retail items on the store were listed as sold out at once, including all
four dehydrated citrus garnishes, three of four glassware styles, and two of three
Seedlip products. At a modest assumed sales rate that is a four-figure figure in
missed shop revenue, and the Seedlip gap means a non-drinking guest at a cocktail
class currently has fewer options.

The tool makes that visible on one screen instead of one product page at a time.

## How it handles what it does not know

Each tool refuses rather than guesses. Unknown prices come back flagged. A
question with no good match in the documents gets told there is no good match. The
revenue figures are labelled as estimates, and the single assumed input is a slider
you can set yourself.

That choice comes from a year of QA and support work at Solitaire Infosys, where
the expensive tickets were never the errors that announced themselves. They were
the ones that returned something plausible and wrong. A tool that quotes a price
nobody approved is worse than one that leaves a blank.

## Where I would take it in a first month

1. Replace the sample inventory with a live export and set real reorder points and
   sales rates, which turns the estimates into measurements.
2. Load the documents staff ask about most and check the answers against people
   who know them.
3. Track how many drafted replies go out with light edits, since that number is
   the honest measure of whether this saves time.

## Stack

Python, Streamlit, scikit-learn, Pandas, DeepSeek via OpenRouter. 28 tests covering
the alerting rules, the cost arithmetic, and the search behaviour including
refusal. Runs on a laptop or deploys to a link.

- Code: github.com/rahulverma-hp/reids-ai-assistant
- Portfolio: rahulverma-hp.github.io
