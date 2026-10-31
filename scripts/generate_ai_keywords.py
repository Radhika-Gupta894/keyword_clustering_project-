"""Generate a SYNTHETIC AI Search Query Dataset grounded in Generative AI brand taxonomy.

FOR PIPELINE TESTING ONLY - NOT REAL SEARCH LOGS.

Derived from the cleanor-search-index AI tool taxonomy:
Brands: ChatGPT, Gemini, Claude, Copilot, Grok, Perplexity, DeepSeek, Meta AI,
        Qwen, Mistral, Doubao, GitHub Copilot, Midjourney, Claude Code.

Creates:
- data/synthetic/ai_search_keywords.csv (synthetic input, ~700 rows)
- data/synthetic/ai_gold_topics.csv (synthetic ground truth topics for testing)
"""
import random
from pathlib import Path
import pandas as pd

SEED = 42

AI_TOPICS = {
    "coding_assistants": {
        "items": ["claude code", "github copilot", "deepseek coder", "chatgpt coding", "qwen coder", "copilot cli", "cursor ai alternative"],
        "facets": ["for python", "for react", "in vscode", "debugging", "for beginners", "terminal commands", "free tier", "code refactoring"],
        "verbs": ["install", "configure", "use", "debug with", "setup"]
    },
    "image_generation": {
        "items": ["midjourney", "meta ai image generator", "grok image generator", "chatgpt image generation", "flux ai"],
        "facets": ["prompts for beginners", "photorealistic styles", "architecture design", "character consistency", "commercial license", "anime style", "negative prompts"],
        "verbs": ["create", "generate", "prompt", "upscale", "edit with"]
    },
    "pricing_and_plans": {
        "items": ["chatgpt plus", "claude pro", "perplexity pro", "copilot pro", "gemini advanced", "midjourney subscription"],
        "facets": ["monthly price", "student discount", "annual plan", "free trial", "cancellation policy", "is it worth it", "pricing in india", "team pricing"],
        "verbs": ["buy", "cancel", "compare", "upgrade to", "renew"]
    },
    "web_research": {
        "items": ["perplexity ai", "gemini deep research", "grok search", "chatgpt web search", "perplexity research engine"],
        "facets": ["for academic papers", "source citations", "fact checking", "market research", "literature review", "real time news", "for students"],
        "verbs": ["search with", "cite sources using", "summarize with", "explore", "use"]
    },
    "open_source_local": {
        "items": ["deepseek r1", "mistral large", "qwen 2.5", "mistral nemo", "deepseek v3", "open source llm"],
        "facets": ["local install ollama", "system requirements", "quantization 4bit", "weights download", "on mac m3", "offline use", "fine tuning"],
        "verbs": ["run", "install", "download", "benchmark", "host"]
    },
    "model_comparisons": {
        "items": ["claude vs chatgpt", "gemini vs perplexity", "deepseek vs claude", "copilot vs gemini", "grok vs chatgpt", "mistral vs qwen"],
        "facets": ["for coding", "for academic writing", "reasoning benchmark", "math problem solving", "hallucination rate", "speed comparison", "2026 review"],
        "verbs": ["compare", "test", "evaluate", "benchmark", "choose between"]
    },
    "api_and_developer": {
        "items": ["anthropic claude api", "openai api", "google gemini api", "mistral api", "deepseek api", "qwen api"],
        "facets": ["rate limits", "pricing per token", "python sdk tutorial", "function calling", "structured output", "api key setup", "latency benchmarks"],
        "verbs": ["call", "integrate", "authenticate", "stream with", "set up"]
    },
    "mobile_and_desktop": {
        "items": ["chatgpt mobile app", "gemini assistant", "doubao app", "meta ai whatsapp", "claude desktop app", "copilot windows app"],
        "facets": ["voice mode", "for android apk", "for iphone", "offline shortcuts", "widget setup", "login issues", "background listening"],
        "verbs": ["download", "activate", "troubleshoot", "install", "connect"]
    }
}

TEMPLATES = [
    "best {item} {facet}",
    "how to {verb} {item}",
    "{item} {facet}",
    "{item} review",
    "free {item} {facet}",
    "what is the best {item} {facet}",
    "{item} tips for {facet}",
    "how to use {item} {facet}",
    "{item} vs alternative for {facet}",
    "cheap {item} {facet}"
]

AMBIGUOUS = ["gemini", "copilot", "python", "apple", "claude", "mercury", "amazon", "mouse", "deep", "terminal"]


def typo(s, rng):
    words = s.split()
    cand = [i for i, w in enumerate(words) if len(w) > 4]
    if not cand:
        return s
    i = rng.choice(cand)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def main():
    rng = random.Random(SEED)
    rows = []
    
    # Generate balanced queries per topic
    for topic, d in AI_TOPICS.items():
        seen = set()
        while len(seen) < 65:
            t = rng.choice(TEMPLATES)
            item = rng.choice(d["items"])
            facet = rng.choice(d["facets"])
            verb = rng.choice(d["verbs"])
            query = t.format(item=item, facet=facet, verb=verb)
            seen.add(query)
        rows += [(k, topic) for k in sorted(seen)]
        
    base = rows[:]
    # Injected real-world dirty data
    for k, t in rng.sample(base, 35): rows.append((k.upper(), t))                           # Case variants
    for k, t in rng.sample(base, 25): rows.append(("  " + k.replace(" ", "  ", 1) + " ", t)) # Whitespace variants
    for k, t in rng.sample(base, 30): rows.append((typo(k, rng), t))                         # Realistic typos
    for k, t in rng.sample(base, 35): rows.append((k, t))                                    # Exact duplicates
    rows += [(k, "ambiguous") for k in AMBIGUOUS]
    
    rng.shuffle(rows)
    
    data_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pipeline input: keywords only (SYNTHETIC)
    pd.DataFrame({"keyword": [r[0] for r in rows]}).to_csv(data_dir / "ai_search_keywords.csv", index=False)
    
    # 2. Benchmark evaluation ground truth (SYNTHETIC)
    pd.DataFrame(rows, columns=["keyword", "topic"]).to_csv(data_dir / "ai_gold_topics.csv", index=False)
    
    print(f"Generated {len(rows)} SYNTHETIC AI search queries across {len(AI_TOPICS)} topics.")
    print(f"Saved: {data_dir / 'ai_search_keywords.csv'}")
    print(f"Saved: {data_dir / 'ai_gold_topics.csv'}")


if __name__ == "__main__":
    main()

