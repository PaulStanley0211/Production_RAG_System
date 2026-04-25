"""Grading and decision-making prompts.

Different from templates.py — these prompts produce structured signals
(labels, JSON, single rewrites), not prose for the user.

Convention: every grading prompt outputs inside <output>...</output> tags
so the pipeline can parse reliably even if the model adds extra commentary.
"""

# ============================================================
# Query Router — classifies user intent
# ============================================================
# Used by app/services/query_router.py to decide which pipeline path to take.

QUERY_ROUTER_V1 = """Classify the user's query into ONE of these categories:

- factual         The user wants a specific fact or piece of information from the documents.
                  Example: "What is the company's refund policy?"
- comparative     The user wants to compare or contrast multiple things.
                  Example: "How does Plan A differ from Plan B?"
- chitchat        Greetings, small talk, or meta-questions about the assistant.
                  Example: "Hi", "Who are you?", "Thanks"
- no_retrieval    Generic knowledge question that doesn't need document retrieval.
                  Example: "What is 2+2?", "Translate hello to French"

Output ONLY the category name inside <output> tags. No explanation.

# User query
{query}

# Your classification
<output>"""


# ============================================================
# Query Decomposer — splits complex queries into sub-queries
# ============================================================
# Used by app/services/query_decomposer.py for comparative or multi-part queries.

QUERY_DECOMPOSER_V1 = """Break the user's query into 2-4 focused sub-queries that, together, fully answer it.

# Rules
1. Each sub-query should be answerable independently from the documents.
2. Cover all aspects of the original question — don't skip any.
3. Don't decompose simple queries — return a single-item list if no decomposition is needed.
4. Output a JSON array of strings inside <output> tags.

# Examples
Query: "Compare Plan A and Plan B on price and features"
<output>["What is the price and what features does Plan A include?", "What is the price and what features does Plan B include?"]</output>

Query: "What is the refund policy?"
<output>["What is the refund policy?"]</output>

# User query
{query}

# Your decomposition
<output>"""


# ============================================================
# Document Grader — per-chunk relevance scoring
# ============================================================
# Used by app/services/document_grader.py after retrieval, before generation.

DOCUMENT_GRADER_V1 = """Decide if the document chunk is relevant to the user's query.

# Categories
- relevant            The chunk directly answers the query or is essential context.
- partially_relevant  The chunk is related but doesn't fully answer; useful as supporting context.
- irrelevant          The chunk has no connection to the query, even if it shares keywords.

Output ONLY the category name inside <output> tags. No explanation.

# User query
{query}

# Document chunk
{chunk}

# Your grade
<output>"""


# ============================================================
# Memory Rewriter — turns follow-up queries into self-contained ones
# ============================================================
# Used by app/services/conversation.py before retrieval. Critical for follow-ups.

MEMORY_REWRITER_V1 = """The user is having a multi-turn conversation. Their latest query may rely on context from earlier turns.

Your job: rewrite the latest query so it's fully self-contained — readable without seeing the conversation history.

# Rules
1. Resolve all pronouns ("it", "that", "they") using the conversation history.
2. Resolve implicit references ("the second one", "what about cost?").
3. If the query is already self-contained, return it unchanged.
4. Output ONLY the rewritten query inside <output> tags.

# Examples
History:
User: What is Plan A's price?
Assistant: Plan A costs $10/month.
Latest query: What about its features?
<output>What features does Plan A include?</output>

History:
User: Tell me about Python
Assistant: Python is a high-level programming language...
Latest query: How do I install it?
<output>How do I install Python?</output>

# Conversation history
{history}

# Latest query
{query}

# Rewritten query
<output>"""


# ============================================================
# Public aliases — current default versions
# ============================================================

QUERY_ROUTER = QUERY_ROUTER_V1
QUERY_DECOMPOSER = QUERY_DECOMPOSER_V1
DOCUMENT_GRADER = DOCUMENT_GRADER_V1
MEMORY_REWRITER = MEMORY_REWRITER_V1