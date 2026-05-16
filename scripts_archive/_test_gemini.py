"""Quick test for Gemini API key."""
import os
from google import genai

client = genai.Client(api_key="AIzaSyAIHB3tfmMz7S5a1-8vx3MBFP4QFwOIYG0")

# 列出可用模型
print("=== 可用 Gemini 模型 ===")
try:
    for m in client.models.list():
        if "gemini" in m.name.lower():
            print(f"  {m.name}")
except Exception as e:
    print(f"列出模型失败: {e}")

# 测试生成
print("\n=== 测试 Gemini 生成 ===")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence.",
    )
    print(f"模型: gemini-2.5-flash")
    print(f"回复: {response.text}")
    print("✅ API Key 可用!")
except Exception as e:
    print(f"❌ gemini-2.5-flash 错误: {e}")
    # fallback 试试其他型号
    for fallback in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
        try:
            response = client.models.generate_content(
                model=fallback,
                contents="Say hello in one sentence.",
            )
            print(f"\n✅ {fallback} 可用!")
            print(f"回复: {response.text}")
            break
        except Exception as e2:
            print(f"❌ {fallback}: {e2}")
