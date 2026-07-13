target_declaration = "Target is '{target}'"

guesser_prologue = """
Let us play the game of 20 questions. I am impersonating the thing, X.

Your goal is to identify X within at most {n} turns.

Rules:
1. Ask exactly one question only.
2. Each question should start with "Is X" whenever possible.
3. Do not ask more than one question in a turn.
4. Do not add explanation, justification, or extra text.
5. If this is the final turn, do not ask a question. Instead, make exactly one final guess.

Final-turn output rule:
- On the final turn, output only:
  X is "<your best guess>".

Let us begin. Ask me the first question.
"""

urge_prompt = ""

inform_prompt = "The X may be one of items below:\n{item_list_str}"

examiner_prologue = '''Let us play the game of 20 questions. You are the answerer and I am the guesser. X is "{item}".

You must answer each question truthfully based on being X, but you must follow this output rule strictly:

Allowed responses:
- "Yes"
- "No"
- "Partially"
- "Not applicable"
- "You guessed it. X is "{item}"."

Important rules:
1. Output exactly one of the allowed responses above.
2. Do not add any explanation, justification, hint, clarification, punctuation, or extra words.
3. Do not reveal any additional information about X beyond the allowed response.
4. If the guess is exactly correct, output: "You guessed it. X is "{item}"."
5. For all other questions, output only one of: "Yes", "No", "Partially", or "Not applicable".

Examples of forbidden outputs:
- "Yes, because it is a mammal."
- "No. It usually lives in Africa."
- "Partially, since that depends on context."

Examples of valid outputs:
- "Yes"
- "No"
- "Partially"
- "Not applicable"

Let us begin. Here is my first question.
'''

final_guess_prompt = """
This is the final turn.
You must stop asking questions and make exactly one final guess now.

Output format:
X is "<your best guess>".

Do not ask a question.
Do not add any explanation.
Do not output anything else.
"""

final_guess_prompt_inform = """
This is the final turn.
You must stop asking questions and make exactly one final guess now.

Possible candidates:
{item_list_str}

Output format:
X is "<your best guess>".

Do not ask a question.
Do not add any explanation.
Do not output anything else.
"""

extract_q_prompt = """{rsp}

Extract the question it wants to ask.
"""

extract_guess_prompt = """
Rewrite the following response into exactly one final guess in this format:

X is "<guess>".

Rules:
1. Output exactly one line.
2. Do not output a question.
3. Do not add explanation or extra words.
4. Keep only the guessed entity.

Response:
{rsp}
"""
