"""Public static shell; all usage data requires the shim's API key."""


def usage_html():
    return r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codex Shim · Usage</title>
<style>
body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1200px;margin:40px auto;padding:0 24px}h1{margin-bottom:8px}p{color:#a9b5c9;line-height:1.5}input,button{font:inherit;padding:10px;border-radius:6px;border:1px solid #4b5563;background:#1f2937;color:inherit}button{cursor:pointer}input{width:300px}#cards{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.card{background:#1f2937;padding:18px;border-radius:8px;min-width:180px}.card strong{display:block;font-size:28px}small{color:#a9b5c9}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #374151}.scroll{overflow:auto}#notice{color:#fbbf24}
</style>
<h1>Request usage</h1><p>Local, persistent token counters — not a billing statement. Unknown means the provider did not report a counter; it does not mean zero.</p>
<form id="auth"><input id="key" type="password" autocomplete="off" placeholder="Shim API key" aria-label="Shim API key"> <button>Load / refresh</button> <button id="lock" type="button">Clear key &amp; data</button></form>
<p id="notice" role="status">Enter the same key used by your API client. It stays in this page's memory only.</p>
<section id="cards"></section><h2>Recent requests</h2><div class="scroll"><table><thead><tr><th>Started</th><th>Model / endpoint</th><th>Status</th><th>Duration</th><th>Input</th><th>Output</th><th>Cached</th><th>Cache write</th><th>Request ID</th></tr></thead><tbody id="rows"></tbody></table></div>
<p>Only primary calls are counted. Routing classifiers and other auxiliary calls are excluded. Ensemble / IO requests have unknown counters, not a made-up sum. Partial or disconnected requests may have consumed unreported tokens; provider billing is authoritative. Cached tokens are a subset of normalized input. No prices or dollar estimates are inferred.</p>
<script>
const $=id=>document.getElementById(id), fmt=v=>v==null?'Unknown':Number(v).toLocaleString();
$('lock').onclick=()=>{$('key').value='';$('cards').replaceChildren();$('rows').replaceChildren();$('notice').textContent='Locked.'};
$('auth').onsubmit=async e=>{e.preventDefault();$('notice').textContent='Loading…';try{
 const r=await fetch('/v1/usage',{headers:{Authorization:'Bearer '+$('key').value},cache:'no-store'});
 if(!r.ok)throw new Error(r.status===401?'API key required or incorrect.':'Usage unavailable ('+r.status+').');
 const d=await r.json();$('cards').replaceChildren();$('rows').replaceChildren();
 for(const [name,s] of Object.entries(d.totals)){const box=document.createElement('div');box.className='card';const label=document.createElement('span');label.textContent=name.replaceAll('_',' ');const n=document.createElement('strong');n.textContent=fmt(s.known_total);const detail=document.createElement('small');detail.textContent=s.known_requests+' known / '+s.unknown_requests+' unknown requests';box.append(label,n,detail);$('cards').append(box)}
 for(const row of d.recent){const tr=document.createElement('tr');for(const value of [new Date(row.started_at*1000).toLocaleString(),row.model+' · '+row.endpoint,row.status+' / '+(row.http_status??'—')+' · '+row.coverage,fmt(row.duration_ms)+' ms',...['input_tokens','output_tokens','cached_tokens','cache_write_tokens'].map(k=>fmt(row[k])),row.request_id]){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('rows').append(tr)}
 $('notice').textContent=d.requests+' retained requests · '+d.retention_days+' days / '+d.max_rows+' rows maximum · latest 200 shown.';
}catch(err){$('notice').textContent=err.message}};
</script></html>'''
