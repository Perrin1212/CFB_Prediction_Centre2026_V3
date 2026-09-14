from __future__ import annotations
import json, os, random, time
from pathlib import Path
from typing import Any, Iterable
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.collegefootballdata.com"

class CFBDClient:
    """Small, cache-first CFBD client owned by V3.

    Every response can be persisted before transformation. Re-running acquisition
    therefore costs zero API calls for already-cached payloads unless force=True.
    """
    def __init__(self, cache_root: Path, timeout: int = 60):
        load_dotenv()
        key = os.getenv("CFBD_API_KEY", "").strip()
        if not key:
            raise RuntimeError("CFBD_API_KEY missing. Copy .env.example to .env and add your key.")
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
        self.new_calls = 0

    def _path(self, namespace: str, name: str) -> Path:
        p = self.cache_root / namespace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get(self, endpoint: str, params: dict[str, Any], namespace: str, name: str,
            force: bool = False, retries: int = 6) -> list[dict[str, Any]] | dict[str, Any]:
        path = self._path(namespace, name)
        # An empty JSON list is still a valid cached response. Treating its
        # two-byte file as missing caused repeated historical/future API calls.
        if path.exists() and path.stat().st_size > 0 and not force:
            return json.loads(path.read_text(encoding="utf-8"))
        url = BASE_URL + endpoint
        last = None
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(30, 2 ** attempt + random.random()))
                    last = RuntimeError(f"CFBD {r.status_code}: {r.text[:250]}")
                    continue
                r.raise_for_status()
                payload = r.json()
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
                self.new_calls += 1
                return payload
            except Exception as exc:
                last = exc
                time.sleep(min(20, 1.5 ** attempt))
        raise RuntimeError(f"CFBD request failed {endpoint} {params}: {last}")

    def get_games(self, year: int, force: bool = False):
        return self.get("/games", {"year": year, "seasonType": "both", "classification": "fbs"},
                        "games", f"{year}.json", force)

    def get_plays(self, year: int, week: int, force: bool = False):
        return self.get("/plays", {"year": year, "week": week, "seasonType": "both", "classification": "fbs"},
                        f"plays/{year}", f"week_{week:02d}.json", force)

    def get_team_game_stats(self, year: int, week: int, force: bool = False):
        return self.get("/games/teams", {"year": year, "week": week, "seasonType": "both", "classification": "fbs"},
                        f"team_game_stats/{year}", f"week_{week:02d}.json", force)

    def get_advanced_box(self, game_id: int, year: int, force: bool = False):
        return self.get("/game/box/advanced", {"id": int(game_id)}, f"advanced_box/{year}", f"{int(game_id)}.json", force)
