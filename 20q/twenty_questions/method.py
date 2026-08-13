import copy
import os
import re
from typing import Any, Tuple

from twenty_questions.models import get_response_method, unpack_model_response
from twenty_questions.twenty_question_utils import (
    extract_guess_entity,
    looks_like_guess,
    normalize_guess_entity,
)


def _split_thinking_content(text: str) -> Tuple[str, str]:
    """
    Backward-compatible parser for legacy wrappers that only return text.
    """
    if not isinstance(text, str):
        text = str(text)

    original_text = text

    if "</think>" in text:
        left, right = text.rsplit("</think>", 1)
        thinking_text = re.sub(r"</?think>", "", left, flags=re.IGNORECASE).strip()
        visible_text = right.strip()
        return visible_text, thinking_text

    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_blocks:
        thinking_text = "\n".join([b.strip() for b in think_blocks if b.strip()]).strip()
        visible_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return visible_text, thinking_text

    return original_text.strip(), ""


def _normalize_response_output(resp: Any) -> Tuple[Any, str, str, int]:
    """
    Normalize different model wrapper return formats into:
        (raw_response, visible_text, thinking_text, think_tokens)
    """
    raw, answer, cot, think_tokens = unpack_model_response(resp)

    if cot:
        return raw, str(answer), str(cot), int(think_tokens or 0)

    if answer:
        visible_text, thinking_text = _split_thinking_content(str(answer))
        if thinking_text and (think_tokens is None or int(think_tokens or 0) == 0):
            think_tokens = len(thinking_text.split())
        return raw, visible_text, thinking_text, int(think_tokens or 0)

    return raw, "", "", int(think_tokens or 0)


def _call_model(response_fn, messages, model: str, **kwargs) -> Tuple[Any, str, str, int]:
    """
    Unified model call wrapper.
    Always returns (raw_response, visible_text, thinking_text, think_tokens).
    """
    resp = response_fn(messages, model=model, **kwargs)
    return _normalize_response_output(resp)


def _compact_history_for_final_guess(history):
    if not history:
        return []

    system_content = history[0]["content"] if history[0].get("role") == "system" else ""
    transcript = []
    for msg in history[1:]:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            transcript.append(f"Guesser: {content}")
        elif role == "user":
            transcript.append(f"Answerer: {content}")

    return [{
        "role": "system",
        "content": system_content,
    }, {
        "role": "user",
        "content": "Conversation so far:\n" + "\n".join(transcript),
    }]


def is_winning_response(text: str) -> bool:
    if not isinstance(text, str):
        return False

    t = text.strip().lower()
    if "not correct" in t:
        return False
    if re.match(r"^(no|nope|incorrect|false)\b", t):
        return False

    winning_phrases = [
        "you guessed it",
        "you are right",
        "you're right",
        "that is correct",
        "yes, that's right",
        "yes, you are right",
        "yes, that's correct",
    ]
    return any(p in t for p in winning_phrases)


def looks_like_question(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    return t.endswith("?")


# Per-turn guesser progress lines are printed by default; set TWENTYQ_DEBUG=0 to silence
_GUESSER_TURN_DEBUG = os.environ.get("TWENTYQ_DEBUG", "1").lower() not in ("0", "false", "no")


def _print_guesser_turn_debug(turn_stat: dict):
    if not _GUESSER_TURN_DEBUG:
        return
    if not isinstance(turn_stat, dict):
        return

    action_type = turn_stat.get("action_type", "unknown")

    if action_type == "guess":
        print(
            f"[guesser] turn={turn_stat.get('turn')} "
            f"type=guess "
            f"guess={turn_stat.get('normalized_guess')} "
            f"repeated_guess={turn_stat.get('repeated_guess')} "
            f"cum_guess_count={turn_stat.get('cumulative_guess_action_count')} "
            f"cum_repeated_guess_count={turn_stat.get('cumulative_repeated_guess_count')} "
            f"cum_repeated_guess_rate={turn_stat.get('cumulative_repeated_guess_rate', 0.0):.4f}"
        )
    elif action_type == "question":
        print(
            f"[guesser] turn={turn_stat.get('turn')} "
            f"type=question "
            f"repeated_question={turn_stat.get('repeated_question')}"
        )
    else:
        print(
            f"[guesser] turn={turn_stat.get('turn')} "
            f"type={action_type}"
        )


def get_examiner_response(task, history):
    response_fn = get_response_method(task.examiner_model)

    if len(history) > 12:
        msg = [history[0]] + history[-11:]
    else:
        msg = history

    raw_response, visible_text, thinking_text, think_tokens = _call_model(
        response_fn,
        msg,
        model=task.examiner_model,
    )
    return raw_response, visible_text, thinking_text, think_tokens


def get_guesser_naive_response(task, history, ques_id):
    response_fn = get_response_method(task.guesser_model)

    is_final_turn = ques_id >= task.max_turn

    msg = (
        _compact_history_for_final_guess(history)
        if is_final_turn
        else copy.deepcopy(history)
    )
    prompt = ""

    if is_final_turn:
        if task.inform:
            prompt += (
                "This is your final turn. "
                "You must now make exactly one final guess only. "
                + task.prompts.final_guess_prompt_inform.format(
                    item_list_str=', '.join(task.set)
                )
            )
        else:
            prompt += (
                "This is your final turn. "
                "You must now make exactly one final guess only. "
                + task.prompts.final_guess_prompt
            )
    else:
        if ques_id > int(task.max_turn * 0.7):
            prompt += task.prompts.urge_prompt

        prompt += (
            "\nReply with exactly one action only.\n"
            "Allowed actions:\n"
            "1. one yes/no question, or\n"
            "2. one direct guess of X.\n"
            "Do not output multiple questions.\n"
            "Do not output both a question and a guess.\n"
            "Keep the reply short."
        )

    if len(msg) == 0:
        msg = [{"role": "system", "content": prompt.strip()}]
    else:
        msg[-1]["content"] += " " + prompt

    _, rsp_text, thinking_text, think_tokens = _call_model(
        response_fn,
        msg,
        model=task.guesser_model,
    )

    if not rsp_text.strip():
        raise RuntimeError(
            "The 20Q guesser returned no visible response. Its generation may "
            "have ended during reasoning; check the model/server generation budget."
        )

    def extract_ques(text: str) -> Tuple[str, str, int]:
        examiner_fn = get_response_method(task.examiner_model)
        message = [{
            "role": "user",
            "content": task.prompts.extract_q_prompt.format(rsp=text)
        }]
        _, extracted_text, extract_thinking, extract_think_tokens = _call_model(
            examiner_fn,
            message,
            model=task.examiner_model,
            max_tokens=task.expected_target_tokens + 32,
        )
        return extracted_text, extract_thinking, extract_think_tokens

    def extract_guess(text: str) -> Tuple[str, str, int]:
        examiner_fn = get_response_method(task.examiner_model)
        message = [{
            "role": "user",
            "content": task.prompts.extract_guess_prompt.format(rsp=text)
        }]
        _, extracted_text, extract_thinking, extract_think_tokens = _call_model(
            examiner_fn,
            message,
            model=task.examiner_model,
            max_tokens=task.expected_target_tokens + 32,
        )
        return extracted_text, extract_thinking, extract_think_tokens

    merged_thinking = thinking_text
    merged_think_tokens = int(think_tokens or 0)

    if is_final_turn:
        if len(rsp_text.split()) > task.expected_action_tokens or looks_like_question(rsp_text):
            extracted_text, extract_thinking, extract_think_tokens = extract_guess(rsp_text)
            merged_thinking = "\n".join(x for x in [merged_thinking, extract_thinking] if x).strip()
            merged_think_tokens += int(extract_think_tokens or 0)
            return extracted_text, merged_thinking, merged_think_tokens

        return rsp_text, merged_thinking, merged_think_tokens

    if len(rsp_text.split()) > task.expected_action_tokens:
        if looks_like_question(rsp_text):
            extracted_text, extract_thinking, extract_think_tokens = extract_ques(rsp_text)
        else:
            extracted_text, extract_thinking, extract_think_tokens = extract_guess(rsp_text)

        merged_thinking = "\n".join(x for x in [merged_thinking, extract_thinking] if x).strip()
        merged_think_tokens += int(extract_think_tokens or 0)

        if not looks_like_question(extracted_text) and not looks_like_guess(extracted_text):
            extracted_text = "Is it a living thing?"

        return extracted_text, merged_thinking, merged_think_tokens

    if looks_like_question(rsp_text) or looks_like_guess(rsp_text):
        return rsp_text, merged_thinking, merged_think_tokens

    extracted_q, q_thinking, q_tokens = extract_ques(rsp_text)
    if looks_like_question(extracted_q):
        merged_thinking = "\n".join(x for x in [merged_thinking, q_thinking] if x).strip()
        merged_think_tokens += int(q_tokens or 0)
        return extracted_q, merged_thinking, merged_think_tokens

    extracted_g, g_thinking, g_tokens = extract_guess(rsp_text)
    if looks_like_guess(extracted_g):
        merged_thinking = "\n".join(x for x in [merged_thinking, g_thinking] if x).strip()
        merged_think_tokens += int(g_tokens or 0)
        return extracted_g, merged_thinking, merged_think_tokens

    return "Is it a living thing?", merged_thinking, merged_think_tokens


def naive_converse(task, i):
    item = task.data[i]["target"]
    inform_prompt = ""

    if task.inform:
        inform_prompt = "\n\n" + task.prompts.inform_prompt.format(
            item_list_str=', '.join(task.set)
        )

    target_decl = task.prompts.target_declaration.format(target=item)
    print(target_decl)

    thinking_g = []
    thinking_e = []
    thinking_tokens_g = 0
    thinking_tokens_e = 0

    guesser_turn_stats = []
    seen_guess_entities = set()
    seen_questions = set()
    repeated_guess_count = 0
    guess_action_count = 0

    history_g = [{
        'role': 'system',
        'content': task.prompts.guesser_prologue.format(n=task.max_turn) + inform_prompt
    }]
    history_e = [{
        'role': 'system',
        'content': task.prompts.examiner_prologue.format(item=item)
    }]

    def _record_guesser_action(turn_idx: int, action_text: str):
        nonlocal repeated_guess_count, guess_action_count

        action_type = "other"
        normalized_guess = None
        repeated_guess = False
        repeated_question = False

        if looks_like_question(action_text):
            action_type = "question"
            q_norm = re.sub(r"\s+", " ", action_text.strip().lower())
            repeated_question = q_norm in seen_questions
            seen_questions.add(q_norm)

        elif looks_like_guess(action_text):
            action_type = "guess"
            guess_action_count += 1
            guess_raw = extract_guess_entity(action_text)
            normalized_guess = normalize_guess_entity(guess_raw)
            repeated_guess = normalized_guess in seen_guess_entities if normalized_guess else False
            if repeated_guess:
                repeated_guess_count += 1
            if normalized_guess:
                seen_guess_entities.add(normalized_guess)

        repeated_guess_rate = (
            repeated_guess_count / guess_action_count if guess_action_count > 0 else 0.0
        )

        guesser_turn_stats.append({
            "turn": turn_idx,
            "action_text": action_text,
            "action_type": action_type,
            "normalized_guess": normalized_guess,
            "repeated_guess": repeated_guess,
            "repeated_question": repeated_question,
            "cumulative_guess_action_count": guess_action_count,
            "cumulative_repeated_guess_count": repeated_guess_count,
            "cumulative_repeated_guess_rate": repeated_guess_rate,
        })

    print("------ DIALOGUE START ------")
    count = 0

    bot1_response_text, bot1_thinking_text, bot1_think_tokens = get_guesser_naive_response(task, history_g, count + 1)
    print("Bot 2:", bot1_response_text)
    _record_guesser_action(count + 1, bot1_response_text)
    _print_guesser_turn_debug(guesser_turn_stats[-1])

    if bot1_thinking_text:
        thinking_g.append({
            'turn': count + 1,
            'content': bot1_thinking_text,
            'tokens': int(bot1_think_tokens or 0),
        })
        thinking_tokens_g += int(bot1_think_tokens or 0)

    history_g.append({'role': 'assistant', 'content': bot1_response_text})
    history_e.append({'role': 'user', 'content': bot1_response_text})

    while True:
        _, bot2_response_text, bot2_thinking_text, bot2_think_tokens = get_examiner_response(task, history_e)

        if bot2_thinking_text:
            thinking_e.append({
                'turn': count + 1,
                'content': bot2_thinking_text,
                'tokens': int(bot2_think_tokens or 0),
            })
            thinking_tokens_e += int(bot2_think_tokens or 0)

        history_e.append({'role': 'assistant', 'content': bot2_response_text})
        history_g.append({'role': 'user', 'content': bot2_response_text})
        print("Bot 1:", bot2_response_text)

        if is_winning_response(bot2_response_text):
            state = 1
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("Bot 1: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        bot1_response_text, bot1_thinking_text, bot1_think_tokens = get_guesser_naive_response(task, history_g, count + 1)
        print("Bot 2:", bot1_response_text)
        _record_guesser_action(count + 1, bot1_response_text)
        _print_guesser_turn_debug(guesser_turn_stats[-1])

        if bot1_thinking_text:
            thinking_g.append({
                'turn': count + 1,
                'content': bot1_thinking_text,
                'tokens': int(bot1_think_tokens or 0),
            })
            thinking_tokens_g += int(bot1_think_tokens or 0)

        history_g.append({'role': 'assistant', 'content': bot1_response_text})
        history_e.append({'role': 'user', 'content': bot1_response_text})

    final_repeated_guess_rate = (
        repeated_guess_count / guess_action_count if guess_action_count > 0 else 0.0
    )

    print(
        f"[EPISODE_SUMMARY] "
        f"guess_actions={guess_action_count} "
        f"repeated_guesses={repeated_guess_count} "
        f"repeated_guess_rate={final_repeated_guess_rate:.4f}"
    )

    return {
        'index': i,
        'turn': count,
        'history_g': history_g,
        'history_e': history_e,
        'thinking_g': thinking_g,
        'thinking_e': thinking_e,
        'thinking_tokens_g': thinking_tokens_g,
        'thinking_tokens_e': thinking_tokens_e,
        'state': state,
        'item': task.data[i]["target"],

        'guesser_turn_stats': guesser_turn_stats,
        'guesser_guess_action_count': guess_action_count,
        'guesser_repeated_guess_count': repeated_guess_count,
        'guesser_repeated_guess_rate': final_repeated_guess_rate,
    }
