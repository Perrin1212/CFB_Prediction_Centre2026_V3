from __future__ import annotations
"""Cached CFBD ingestion for V3. Historical weeks are fetched once and reused."""
from pathlib import Path
import json, os, time, requests
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parent.parent; load_dotenv(ROOT/'.env')
KEY=os.getenv('CFBD_API_KEY') or os.getenv('COLLEGE_FOOTBALL_DATA_API_KEY') or os.getenv('CFBD_API_TOKEN')
BASE='https://api.collegefootballdata.com'; RAW=ROOT/'data'/'raw'/'v3_cfbd'; RAW.mkdir(parents=True,exist_ok=True)
YEARS=range(2021,2027); WEEKS=range(0,17)

def get(path,params,retries=4):
    if not KEY: raise RuntimeError('CFBD_API_KEY not found in .env')
    for i in range(retries):
        r=requests.get(BASE+path,params=params,headers={'Authorization':f'Bearer {KEY}'},timeout=90)
        if r.status_code==200:return r.json()
        if r.status_code in (429,500,502,503,504): time.sleep(2**i); continue
        raise RuntimeError(f'CFBD {r.status_code}: {r.text[:500]}')
    raise RuntimeError(f'CFBD request failed after retries: {path} {params}')

def save_json(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj),encoding='utf-8')

def main():
    calls=0
    for year in YEARS:
        for week in WEEKS:
            p=RAW/'plays'/str(year)/f'week_{week:02d}.json'
            if p.exists() and p.stat().st_size>2: continue
            data=get('/plays',{'year':year,'week':week,'seasonType':'both','classification':'fbs'}); calls+=1
            save_json(p,data); print(f'{year} W{week}: {len(data):,} plays')
            time.sleep(.08)
    print(f'\nDone. New API calls: {calls}. Cached weeks were not refetched.')
    print('Next: python -m jobs.build_v3_drive_dataset')
if __name__=='__main__': main()
