from __future__ import annotations
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from config.v3_settings import SETTINGS
from engine.v3_matchup_engine import V3_MATCHUP_FEATURES

def main():
    py=list(ROOT.rglob("*.py"));syntax=[]
    for p in py:
        if any(x in p.parts for x in (".venv","venv")):continue
        try:ast.parse(p.read_text(encoding="utf-8",errors="replace"))
        except Exception as e:syntax.append({"file":str(p.relative_to(ROOT)),"error":str(e)})
    forbidden=[f for f in V3_MATCHUP_FEATURES if any(t in f.lower() for t in SETTINGS.market_forbidden_tokens)]
    result={"python_files":len(py),"syntax_errors":syntax,"predictive_features":len(V3_MATCHUP_FEATURES),"market_feature_violations":forbidden,"app_lines":sum(1 for _ in open(ROOT/"app"/"app.py",encoding="utf-8",errors="ignore")),"status":"PASS" if not syntax and not forbidden else "FAIL"}
    print(json.dumps(result,indent=2));(SETTINGS.audit_root/"project_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if result["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
