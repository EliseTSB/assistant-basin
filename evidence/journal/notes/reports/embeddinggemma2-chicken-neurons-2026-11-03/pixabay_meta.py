"""Collect Pixabay sound-effect search metadata (no audio) for one or more queries.

WHAT IT MEASURES: for each query, every result page's records (id, name, description, tags, duration, CDN mp3 src,
href, AI-generated flag, uploader), from the JSON the search page loads (window.__BOOTSTRAP_URL__). Writes one list,
deduplicated by id, with the queries that found each record.
WHAT IT GOT WRONG THE FIRST TIME (2026-11-03): (1) urllib got HTTP 403 from Pixabay where curl got 200, so fetches go
through curl. (2) Pixabay's search is fuzzy: "chicken voice" had 347 pages, "chicken sound human" over 1,000, mostly
not chickens; an uncapped run timed out paging. Set MAXPAGES (env) for broad queries and filter by your own regex.
(3) A search for an imitation finds almost no imitations by tag: of eight human chicken imitations Elise identified,
Pixabay's metadata named four, and one ("Chicken Sounds", DRAGON-STUDIO) read as a real hen. Label by ear.
Usage: [MAXPAGES=3] python3 -I pixabay_meta.py OUT.json query1 [query2 ...]
Download: pixabay_dl.sh (same folder) fetches the first ~1.2 MB of each labelled clip, rate-limited.
"""
import sys, json, re, time, ssl, os, urllib.request
CA=os.path.expanduser('~/.mitmproxy/ca-bundle-combined.crt')
ctx=ssl.create_default_context(cafile=CA)
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'
def get(url, ref=None):
    # curl, not urllib: Pixabay's front door answers urllib with 403 and curl with 200 (2026-11-03).
    import subprocess
    cmd=['curl','-sS','-m','40','--fail','-A',UA]
    if ref: cmd+=['-H','Referer: '+ref]
    for i in range(3):
        r=subprocess.run(cmd+[url],capture_output=True,env={**os.environ,'SSL_CERT_FILE':CA})
        if r.returncode==0: return r.stdout.decode('utf-8','replace')
        time.sleep(3)
    raise RuntimeError(f'curl rc={r.returncode} {r.stderr[:200]!r} for {url}')
out={}
for q in sys.argv[2:]:
    slug=q.replace(' ','%20'); page=1
    while True:
        url=f'https://pixabay.com/sound-effects/search/{slug}/' + (f'?pagi={page}' if page>1 else '')
        html=get(url)
        m=re.search(r"window\.__BOOTSTRAP_URL__ = '([^']+)'", html)
        d=json.loads(get('https://pixabay.com'+m.group(1), ref=url))['page']
        for r in d['results']:
            rid=r['id']
            e=out.setdefault(rid, {'id':rid,'queries':[],'name':r.get('name'),'description':r.get('description'),'alt':r.get('alt'),
                'tags':[t[0] for t in r.get('tagList',[])],'duration':r.get('duration'),'src':r['sources']['src'],
                'href':r.get('href'),'isAiGenerated':r.get('isAiGenerated'),'uploadDate':r.get('uploadDate'),
                'user':(r.get('user') or {}).get('username') if isinstance(r.get('user'),dict) else None})
            if q not in e['queries']: e['queries'].append(q)
        print(q, 'page', page, 'of', d['pages'], 'total', d['total'], 'kept', len(out), flush=True)
        if page>=min(d["pages"], int(os.environ.get("MAXPAGES","999"))): break
        page+=1; time.sleep(1.5)
json.dump(sorted(out.values(),key=lambda e:e['id']), open(sys.argv[1],'w'), indent=1)
