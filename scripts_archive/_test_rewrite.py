#!/usr/bin/env python3
"""Quick test of NVIDIA API model for rewriting solution processes."""
import os, time
from openai import OpenAI

api_key = os.environ.get('NVIDIA_API_KEY') or os.environ.get('NVIDIA_NIM_API_KEY')
client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=api_key, timeout=30)

TESTS = [
    {
        "type": "gravity",
        "prompt": "A secret gravitational constant g is hidden. Given examples of distance d and time t where d=0.5*g*t^2, find d for t=4.41.\nExamples: t=1.37->d=14.92, t=4.27->d=144.96, t=3.28->d=85.54",
        "solution": "d=0.5*g*t^2. g=2*14.92/1.37^2=15.8986; g=2*144.96/4.27^2=15.9009; g=2*85.54/3.28^2=15.9020. g_avg=15.9008. d=0.5*15.9008*4.41^2=154.62",
        "answer": "154.62"
    },
    {
        "type": "cipher",
        "prompt": "Decrypt: 'brg wzrswvog hffk'",
        "solution": "Substitution cipher. Mapping: b->t, f->o, g->s, h->b, k->k, o->e, r->a, s->g, t->c, v->n, w->i, z->m. Result: cat imagines book",
        "answer": "cat imagines book"
    },
    {
        "type": "numeral",
        "prompt": "Convert 38 to Alice's number system. Examples: 14->XIV, 7->VII",
        "solution": "Arabic->Roman. 38 = 10x3=XXX, 5x1=V, 1x3=III -> XXXVIII",
        "answer": "XXXVIII"
    },
    {
        "type": "symbol",
        "prompt": "In Alice's Wonderland: \\(*[#=\\([#, '(*#[=']#[. Find: \\(*[#=?",
        "solution": "Symbol op '*' = concat. \\( * [# = \\([#",
        "answer": "\\([#"
    },
    {
        "type": "bit_ops",
        "prompt": "Transform 8-bit binary. Examples: 10110100->01001100, 11001010->00110110",
        "solution": "Per-bit rules: b0=NOT in[0]; b1=NOT in[1]; b2=NOT in[2]; b3=NOT in[3]; b4=NOT in[4]; b5=NOT in[5]; b6=NOT in[6]; b7=NOT in[7]. Result: 10010111",
        "answer": "10010111"
    },
    {
        "type": "unit_conv",
        "prompt": "Convert 25.09 from unit A to unit B. Examples: 10.08->6.69, 17.83->11.83, 35.85->23.79",
        "solution": "Linear conversion. 10.08->6.69(f=0.663690); 17.83->11.83(f=0.663489); 35.85->23.79(f=0.663598). avg_f=0.663584. 25.09*0.663584=16.65",
        "answer": "16.65"
    },
]

SYSTEM = """You are rewriting a compact machine-generated solution into a clear, natural step-by-step reasoning trace.

Rules:
1. The rewritten solution must lead to EXACTLY the same final answer.
2. Write as if you are thinking through the problem step by step.
3. Be concise but clear - explain the key reasoning, not fluff.
4. Use natural language mixed with math notation.
5. End with: Therefore, the answer is {answer}.
6. Output ONLY the rewritten solution. No preamble, no meta-commentary."""

for test in TESTS[:2]:  # Test 2 first
    user_msg = f"""Problem type: {test['type']}
Problem: {test['prompt']}
Machine solution: {test['solution']}
Final answer: {test['answer']}

Rewrite the machine solution into a clear step-by-step reasoning trace that arrives at the same answer."""

    t0 = time.time()
    resp = client.chat.completions.create(
        model='meta/llama-3.3-70b-instruct',
        messages=[
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_msg}
        ],
        temperature=0.3,
        max_tokens=400,
    )
    t1 = time.time()
    
    print(f"\n{'='*60}")
    print(f"TYPE: {test['type']} | Time: {t1-t0:.1f}s | Tokens: {resp.usage.completion_tokens}")
    print(f"ORIGINAL: {test['solution']}")
    print(f"REWRITTEN:\n{resp.choices[0].message.content}")
    
    time.sleep(1.6)  # rate limit
