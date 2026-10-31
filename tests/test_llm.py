import numpy as np
import pytest

from irsys.llm import INSTRUCTION, format_pair, pack_ids, yes_no_log_odds


def test_prompt_puts_instruction_claim_and_abstract_in_the_user_turn():
    prefix, body, suffix = format_pair("Vitamin D prevents fractures.", "A trial of vitamin D in the elderly.")
    assert prefix.startswith("<|im_start|>system\n") and '"yes" or "no"' in prefix
    assert prefix.endswith("<|im_start|>user\n")
    assert body == (f"<Instruct>: {INSTRUCTION}\n<Query>: Vitamin D prevents fractures.\n"
                    "<Document>: A trial of vitamin D in the elderly.")
    # Non-thinking assistant turn: the very next token is the answer.
    assert suffix == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_pack_ids_keeps_prefix_and_suffix_and_truncates_the_body():
    assert pack_ids([1, 2], [10, 11, 12], [99], max_length=10) == [1, 2, 10, 11, 12, 99]
    # Too long: the end of the body (the end of the abstract) is cut, never the answer slot.
    assert pack_ids([1, 2], [10, 11, 12, 13, 14], [99], max_length=5) == [1, 2, 10, 11, 99]


def test_pack_ids_rejects_a_length_with_no_room_for_the_body():
    with pytest.raises(ValueError):
        pack_ids([1, 2], [10], [99], max_length=3)


def test_score_is_log_odds_of_yes_over_no():
    logits = np.array([[0.0, 2.0, 1.0], [0.0, -1.0, 1.0]])  # columns: other, yes, no
    scores = yes_no_log_odds(logits, yes_id=1, no_id=2)
    assert scores == pytest.approx([1.0, -2.0])
    # Monotone in P(yes) = sigmoid(score), so ranking by it = ranking by P(yes).
    p_yes = np.exp(logits[:, 1]) / (np.exp(logits[:, 1]) + np.exp(logits[:, 2]))
    assert p_yes == pytest.approx(1 / (1 + np.exp(-scores)))
