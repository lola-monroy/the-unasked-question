# THE UNASKED QUESTION — The 90-second Prior Labs pitch

Predictive models answer questions about the data we already have.

I wanted to turn that around: **can TabPFN tell us which information is actually worth acquiring?**

Imagine a venture investor with 20 minutes to interrogate a founder. There are dozens of sensible questions. Asking the one that reduces uncertainty the most might still be a waste — because it may not change the investment decision.

So I built **The Unasked Question**, a decision-aware interviewer. It forms plausible complete versions of the company, predicts the outcome with TabPFN, and estimates the *expected decision value* of asking about churn, runway, margins or other missing information. It chooses the next question **after accounting for the cost of obtaining that answer**, and recommends stopping when asking more wouldn't help.

The live demo lets you watch the investment decision update after each answer. More importantly, it puts three strategies head to head on held-out cases: maximize expected decision value, maximize entropy reduction, or follow a conventional diligence checklist.

**The question I'm exploring is not whether TabPFN can predict startup success. It's whether a foundation model for tabular prediction can become an information-acquisition agent.**

I'm not claiming the value-of-information mathematics is new. The interesting unknown is whether TabPFN's data efficiency makes this approach meaningfully better than conventional predictive models when historical examples are scarce and answers are expensive.

## What I'd ask the creators

1. How should the **joint conditional answer distribution** be modeled? Would a PFN-style generative imputer outperform the donor sampler?
2. How stable is EVSI ranking under realistic **calibration errors** and small in-context datasets?
3. Could we **amortize** inference across hypothetical answer scenarios or expose a query-planning interface that makes this real-time?
4. Under a matched question-cost budget, when does decision-aware acquisition beat entropy-based acquisition and an LLM interviewer?

## The 10-second demo moment

> "Our model is unsure. But the AI does not ask for more data indiscriminately. It tells you which answer would actually be worth paying for — and knows when to stop."

**Experimental honesty:** first demo uses *synthetic* labels and optionally a **non-TabPFN** instant baseline; the app can run official TabPFN-3.5/Fast after model access is configured. Real-world VC validation is not claimed.
