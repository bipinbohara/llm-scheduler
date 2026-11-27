from __future__ import annotations

from typing import Dict, List

from las_mbpp.telemetry import observe


CODING_SYSTEM_PROMPT = (
    "You are a precise Python coding assistant.\n"
    "Return exactly one Python code block with the complete function implementation.\n"
    "Use only Python standard library.\n"
    "Do not include explanations, markdown outside the code block, or test code."
)

CONSENSUS_SYSTEM_PROMPT = (
    "You are a senior code reviewer selecting the strongest candidate implementation.\n"
    "Return strict JSON only."
)


def format_tests_for_prompt(tests: List[str], limit: int = 5) -> str:
    if not tests:
        return "# No tests provided."
    subset = tests if limit <= 0 else tests[:limit]
    lines = [f"{idx + 1}. {case}" for idx, case in enumerate(subset)]
    if limit > 0 and len(tests) > limit:
        lines.append(f"... ({len(tests) - limit} more tests omitted)")
    return "\n".join(lines)


@observe("prompt.debate_initial", "processing")
def build_debate_initial_prompt(
    *,
    task_description: str,
    function_stub: str,
    function_name_from_tests: str,
    tests: List[str],
    agent_name: str,
    tests_preview_limit: int = 5,
    include_official_test_examples: bool = True,
    benchmark_name: str = "MBPP",
) -> str:
    tests_block = ""
    if include_official_test_examples:
        tests_text = format_tests_for_prompt(tests, limit=tests_preview_limit)
        tests_block = "Official test examples:\n" f"{tests_text}\n\n"
    return (
        f"You are Debate Agent {agent_name}.\n"
        f"Write a correct implementation for the {benchmark_name} task.\n"
        f"Required function name from official tests: {function_name_from_tests}\n\n"
        f"Problem:\n{task_description}\n\n"
        f"Function stub:\n{function_stub}\n\n"
        f"{tests_block}"
        "Return only one Python code block."
    )


@observe("prompt.debate_revision", "processing")
def build_debate_revision_prompt(
    *,
    task_description: str,
    function_stub: str,
    function_name_from_tests: str,
    tests: List[str],
    agent_name: str,
    own_code: str,
    peer_codes: Dict[str, str],
    tests_preview_limit: int = 5,
    include_official_test_examples: bool = True,
) -> str:
    tests_block = ""
    if include_official_test_examples:
        tests_text = format_tests_for_prompt(tests, limit=tests_preview_limit)
        tests_block = "Official test examples:\n" f"{tests_text}\n\n"
    peer_chunks = []
    for peer_name in sorted(peer_codes):
        peer_chunks.append(f"Agent {peer_name} code:\n{peer_codes[peer_name]}")
    peers_text = "\n\n".join(peer_chunks) if peer_chunks else "No peer code provided."
    return (
        f"You are Debate Agent {agent_name} in the final debate round.\n"
        "Revise your implementation after reading peer implementations.\n"
        f"Required function name from official tests: {function_name_from_tests}\n\n"
        f"Problem:\n{task_description}\n\n"
        f"Function stub:\n{function_stub}\n\n"
        "Your previous code:\n"
        f"{own_code}\n\n"
        "Peer implementations:\n"
        f"{peers_text}\n\n"
        f"{tests_block}"
        "Return only one Python code block."
    )


@observe("prompt.consensus", "processing")
def build_consensus_prompt(
    *,
    task_description: str,
    function_stub: str,
    function_name_from_tests: str,
    tests: List[str],
    candidate_codes: Dict[str, str],
    tests_preview_limit: int = 5,
    include_official_test_examples: bool = True,
) -> str:
    tests_block = ""
    if include_official_test_examples:
        tests_text = format_tests_for_prompt(tests, limit=tests_preview_limit)
        tests_block = "Official test examples:\n" f"{tests_text}\n\n"
    candidates_text = []
    for name in sorted(candidate_codes):
        candidates_text.append(f"Candidate {name}:\n{candidate_codes[name]}")
    joined_candidates = "\n\n".join(candidates_text)
    return (
        "Select the single best implementation from candidates A, B, and C.\n"
        f"Required function name from official tests: {function_name_from_tests}\n\n"
        f"Problem:\n{task_description}\n\n"
        f"Function stub:\n{function_stub}\n\n"
        f"{tests_block}"
        "Candidate implementations:\n"
        f"{joined_candidates}\n\n"
        "Return strict JSON with this schema only:\n"
        '{"selected_agent": "A|B|C", "reason": "short reason"}'
    )


@observe("prompt.reflection", "processing")
def build_reflection_prompt(
    *,
    task_description: str,
    function_stub: str,
    function_name_from_tests: str,
    selected_code: str,
) -> str:
    return (
        "Reflect on the selected code and improve it if needed.\n"
        f"Required function name from official tests: {function_name_from_tests}\n\n"
        f"Problem:\n{task_description}\n\n"
        f"Function stub:\n{function_stub}\n\n"
        "Selected code:\n"
        f"{selected_code}\n\n"
        "Return only one Python code block."
    )


@observe("prompt.verification", "processing")
def build_verification_prompt(
    *,
    task_description: str,
    function_stub: str,
    function_name_from_tests: str,
    reflected_code: str,
    tests: List[str],
    tests_preview_limit: int = 5,
    include_official_test_examples: bool = True,
) -> str:
    tests_block = ""
    if include_official_test_examples:
        tests_text = format_tests_for_prompt(tests, limit=tests_preview_limit)
        tests_block = "Official test examples:\n" f"{tests_text}\n\n"
    return (
        "Verify this implementation against the task and likely edge cases.\n"
        "If you find issues, fix them.\n"
        f"Required function name from official tests: {function_name_from_tests}\n\n"
        f"Problem:\n{task_description}\n\n"
        f"Function stub:\n{function_stub}\n\n"
        "Current code:\n"
        f"{reflected_code}\n\n"
        f"{tests_block}"
        "Return only one Python code block."
    )


@observe("prompt.error_correction", "processing")
def build_error_correction_prompt(
    *,
    task_description: str,
    function_stub: str,
    function_name_from_tests: str,
    verified_code: str,
    test_feedback: str,
    tests: List[str],
    tests_preview_limit: int = 5,
) -> str:
    tests_text = format_tests_for_prompt(tests, limit=tests_preview_limit)
    return (
        "The verified code was tested. And might have passed or failed.\n"
        "If it passed all test cases, you don't need to do any modifications and just return the previous code.\n"
        "If it failed on some cases, fix it based on the test feedback. Focus on the specific test cases that failed.\n"
        f"Required function name from official tests: {function_name_from_tests}\n\n"
        f"Problem:\n{task_description}\n\n"
        f"Function stub:\n{function_stub}\n\n"
        "Code to fix:\n"
        f"{verified_code}\n\n"
        "Official test examples:\n"
        f"{tests_text}\n\n"
        "Test feedback:\n"
        f"{test_feedback}\n\n"
        "Return only one Python code block."
    )
