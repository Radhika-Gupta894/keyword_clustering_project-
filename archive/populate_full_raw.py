import json
import os
from pathlib import Path

appdata = os.getenv("USERPROFILE", "")
full_transcript = Path(appdata) / ".gemini" / "antigravity" / "brain" / "d5070367-19a4-4777-815e-c90ae5563ca1" / ".system_generated" / "logs" / "transcript_full.jsonl"
for i, line in enumerate(reversed(full_transcript.read_text(encoding="utf-8").splitlines())):
    data = json.loads(line)
    if data.get("type") == "USER_INPUT":
        print(f"User step {i}: len={len(data.get('content', ''))}, preview={data.get('content', '')[:100]}")
    if i > 15:
        break
