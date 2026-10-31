"""Generate the SYNTHETIC datasets (pipeline testing only - NOT real search data).

Writes data/sample_keywords.csv (keyword only), data/synthetic_gold_topics.csv (keyword + topic,
for an optional pairwise precision/recall check) and data/smoke_keywords.csv (100 rows).
"""
import random
from pathlib import Path
import pandas as pd

SEED = 42
TOPICS = {
 "running_shoes": dict(items=["running shoes", "trail running shoes", "marathon shoes", "jogging sneakers", "racing flats", "stability running shoes", "carbon plate shoes", "minimalist running shoes"],
    facets=["beginners", "flat feet", "women", "men", "wide feet", "long distance", "rainy weather", "plantar fasciitis"], verbs=["lace", "clean", "choose", "break in", "size"]),
 "crm_software": dict(items=["crm software", "crm for small agency", "sales crm", "crm tool", "customer database software", "lead tracking crm", "free crm", "cloud crm"],
    facets=["small business", "agencies", "startups", "real estate", "email marketing", "freelancers", "teams", "pipeline management"], verbs=["set up", "migrate to", "choose", "integrate", "compare"]),
 "vegan_recipes": dict(items=["vegan recipes", "vegan dinner", "plant based meals", "vegan dessert", "vegan breakfast", "tofu recipes", "vegan meal prep", "vegan protein snacks"],
    facets=["beginners", "quick", "high protein", "kids", "budget", "one pot", "gluten free", "weeknight"], verbs=["cook", "prepare", "make", "bake", "season"]),
 "car_insurance": dict(items=["car insurance", "auto insurance quotes", "cheap car insurance", "full coverage insurance", "vehicle insurance policy", "liability insurance for cars", "car insurance renewal", "comprehensive car insurance"],
    facets=["young drivers", "seniors", "new cars", "high risk drivers", "students", "low mileage", "families", "classic cars"], verbs=["compare", "cancel", "renew", "buy", "switch"]),
 "laptops": dict(items=["laptop", "gaming laptop", "ultrabook", "student laptop", "business laptop", "budget notebook computer", "lightweight laptop", "laptop with long battery life"],
    facets=["video editing", "programming", "college", "travel", "under 50000 rupees", "coding", "office work", "streaming"], verbs=["buy", "upgrade", "clean", "choose", "repair"]),
 "home_workouts": dict(items=["home workout", "bodyweight exercises", "dumbbell workout", "no equipment workout", "hiit routine", "resistance band exercises", "home gym setup", "yoga at home"],
    facets=["beginners", "weight loss", "abs", "busy people", "women", "back pain", "30 minutes", "small space"], verbs=["start", "plan", "build", "track", "warm up for"]),
 "bali_travel": dict(items=["bali travel guide", "bali itinerary", "ubud hotels", "bali beaches", "bali honeymoon packages", "things to do in bali", "bali villas", "bali visa"],
    facets=["7 days", "budget", "couples", "families", "december", "first timers", "surfing", "monsoon season"], verbs=["book", "plan", "reach", "explore", "visit"]),
 "seo_marketing": dict(items=["seo tools", "keyword research", "backlink building", "on page seo", "content marketing strategy", "seo audit", "technical seo checklist", "local seo"],
    facets=["small business", "beginners", "ecommerce", "agencies", "blogs", "startups", "free", "2026"], verbs=["improve", "audit", "learn", "automate", "measure"]),
 "dog_training": dict(items=["dog training", "puppy training tips", "house training a puppy", "dog obedience classes", "leash training", "stop dog barking", "crate training", "positive reinforcement dog training"],
    facets=["beginners", "puppies", "rescue dogs", "aggressive dogs", "small breeds", "at home", "adult dogs", "labradors"], verbs=["teach", "train", "start", "correct", "reward"]),
 "credit_cards": dict(items=["credit card", "travel credit card", "cashback credit card", "best credit card", "student credit card", "business credit card", "credit card rewards", "low interest credit card"],
    facets=["beginners", "bad credit", "airline miles", "students", "no annual fee", "online shopping", "fuel", "lounge access"], verbs=["apply for", "compare", "cancel", "pay off", "choose"]),
}
TEMPLATES = ["best {item} for {facet}", "{item} for {facet}", "cheap {item}", "how to {verb} {item}", "{item} review",
             "{facet} {item}", "buy {item} online", "top {item} {facet}", "{item} tips", "what is the best {item}"]
AMBIGUOUS = ["apple", "jaguar", "mercury", "python", "java", "amazon", "mouse", "running", "insurance", "orange", "bank", "crane"]


def typo(s, rng):
    words = s.split()
    cand = [i for i, w in enumerate(words) if len(w) > 4]
    if not cand:
        return s
    i = rng.choice(cand); w = words[i]; j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def main():
    rng = random.Random(SEED)
    rows = []
    for topic, d in TOPICS.items():
        seen = set()
        while len(seen) < 56:
            t = rng.choice(TEMPLATES)
            seen.add(t.format(item=rng.choice(d["items"]), facet=rng.choice(d["facets"]), verb=rng.choice(d["verbs"])))
        rows += [(k, topic) for k in sorted(seen)]
    base = rows[:]
    for k, t in rng.sample(base, 30): rows.append((k.upper(), t))                    # case variants
    for k, t in rng.sample(base, 20): rows.append(("  " + k.replace(" ", "  ", 1) + " ", t))  # whitespace variants
    for k, t in rng.sample(base, 25): rows.append((typo(k, rng), t))                  # spelling variants
    for k, t in rng.sample(base, 30): rows.append((k, t))                             # exact duplicates
    rows += [(k, "ambiguous") for k in AMBIGUOUS]
    rng.shuffle(rows)
    out = Path(__file__).resolve().parents[1] / "data"
    pd.DataFrame({"keyword": [r[0] for r in rows]}).to_csv(out / "sample_keywords.csv", index=False)
    pd.DataFrame(rows, columns=["keyword", "topic"]).to_csv(out / "synthetic_gold_topics.csv", index=False)
    smoke = []
    for topic in TOPICS:
        smoke += [r[0] for r in rows if r[1] == topic][:10]
    pd.DataFrame({"keyword": smoke}).to_csv(out / "smoke_keywords.csv", index=False)
    print(f"sample_keywords.csv: {len(rows)} rows | smoke_keywords.csv: {len(smoke)} rows | SYNTHETIC DATA")


if __name__ == "__main__":
    main()
