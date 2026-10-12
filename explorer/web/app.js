const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const fmt = (value) => new Intl.NumberFormat('en-US').format(value ?? 0);
const bytes = (value) => value == null ? '—' : value >= 1e9 ? `${(value / 1e9).toFixed(1)} GB` : value >= 1e6 ? `${(value / 1e6).toFixed(1)} MB` : value >= 1e3 ? `${(value / 1e3).toFixed(1)} KB` : `${value} B`;
const shortName = (name) => String(name).split('.').slice(-2).join('.');
const api = async (path) => { const response = await fetch(path); if (!response.ok) { let detail = `Request failed (${response.status})`; try { detail = (await response.json()).detail || detail; } catch {} throw new Error(detail); } return response.json(); };
const endpoint = (task, route, params = {}) => `/api/tasks/${encodeURIComponent(task)}/${route}${Object.keys(params).length ? `?${new URLSearchParams(params)}` : ''}`;
const svgEl = (tag, attrs = {}) => { const el = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value); return el; };

let catalog, task, focus, selectedNode, page = 'explore', mode = 'graph', repoFilter = 'all', selectionVersion = 0, graphVersion = 0, symbolTimer, snapshotTimer, toastTimer;
let graphView = {x:-550,y:-340,w:1100,h:680};
let graphDefaultView = {...graphView};

function issuePreview(statement, title) {
  const clean = String(statement || '').replace(/<!--[^]*?-->/g, ' ').replace(/<details>[^]*?<\/details>/gi, ' ').replace(/\r/g, '');
  const description = clean.match(/^##\s+Description\s*\n([^]*?)(?=^##\s|(?![^]*))/im)?.[1];
  const paragraphs = (description || clean).split(/\n\s*\n/).map(value => value.replace(/^#+\s*/, '').replace(/\s+/g, ' ').trim()).filter(value => value && value !== title && !/^(Pull Request|Discussion:|Checklist|AI Disclaimer)$/i.test(value));
  const preview = paragraphs.find(value => value.length > 45) || paragraphs[0] || 'No issue description supplied.';
  return preview.slice(0, 230) + (preview.length > 230 ? '…' : '');
}

function toast(message) { const node = $('#toast'); node.textContent = message; node.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove('show'), 3400); }
function showError(error) { console.error(error); toast(error.message || String(error)); }

function showPage(next) {
  page = next;
  $$('.nav-button').forEach(button => button.classList.toggle('active', button.dataset.view === next));
  $$('.view').forEach(view => view.classList.toggle('hidden', view.id !== `${next}View`));
  $('#sidebar').classList.remove('open');
  if (next === 'files') renderFiles();
}

function renderSidebar() {
  if (!catalog) return;
  const repos = Object.entries(catalog.summary.repos).sort((a,b) => b[1]-a[1]);
  $('#repoFilters').innerHTML = `<button class="repo-filter ${repoFilter === 'all' ? 'active' : ''}" data-repo="all">All</button>` + repos.map(([name]) => `<button class="repo-filter ${repoFilter === name ? 'active' : ''}" data-repo="${escapeHtml(name)}">${escapeHtml(name)}</button>`).join('');
  $$('.repo-filter').forEach(button => button.addEventListener('click', () => { repoFilter = button.dataset.repo; renderSidebar(); }));
  const query = $('#taskSearch').value.trim().toLowerCase();
  const list = catalog.tasks.filter(row => (repoFilter === 'all' || row.repo_short === repoFilter) && (!query || `${row.title} ${row.id} ${row.changed_files.join(' ')}`.toLowerCase().includes(query)));
  $('#caseCount').textContent = `${list.length} / ${catalog.tasks.length}`;
  $('#taskList').innerHTML = list.length ? list.map(row => `<button class="task-item ${task?.id === row.id ? 'active' : ''}" data-task="${escapeHtml(row.id)}" role="option" aria-selected="${task?.id === row.id}"><div class="task-item-top"><b>${escapeHtml(row.repo_short)}</b><span>${escapeHtml(row.id)}</span></div><div class="task-item-title">${escapeHtml(row.title)}</div><div class="task-item-foot"><span>${row.changed_files.length} changed files</span><span>${row.snapshot_bytes ? '● snapshot' : '○ no snapshot'}</span></div></button>`).join('') : '<div class="loading-state">No cases match this search.</div>';
  $$('.task-item').forEach(button => button.addEventListener('click', () => selectTask(button.dataset.task)));
}

function renderOverview() {
  const summary = catalog.summary;
  const stats = [['Tasks',summary.task_count,'public development issues'],['Base commits',summary.artifact_count,'unique repository states'],['Call graph nodes',summary.total_nodes,'symbols across all graphs'],['Vector records',summary.total_vectors,'256D symbol embeddings']];
  $('#overviewStats').innerHTML = stats.map(([label,value,note]) => `<div class="stat-card"><span class="eyebrow">${escapeHtml(label)}</span><strong>${fmt(value)}</strong><small>${escapeHtml(note)}</small></div>`).join('');
  const repos = Object.entries(summary.repos).sort((a,b) => b[1]-a[1]);
  const max = Math.max(...repos.map(([,n]) => n));
  $('#repoOverview').innerHTML = repos.map(([name,count]) => `<div class="repo-row"><button data-repo="${escapeHtml(name)}">${escapeHtml(name)}</button><div class="bar-track"><div class="bar-fill" style="width:${count/max*100}%"></div></div><span>${count}</span></div>`).join('');
  $$('#repoOverview button').forEach(button => button.addEventListener('click', () => { repoFilter = button.dataset.repo; renderSidebar(); showPage('explore'); }));
  const counts = {}; const sizes = {};
  for (const file of catalog.files) { counts[file.kind] = (counts[file.kind] || 0) + 1; sizes[file.kind] = (sizes[file.kind] || 0) + file.bytes; }
  const order = ['snapshots','graphs','embeddings','wheels','docker','sandbox','sample_submission','guide','tasks','archive'];
  $('#assetOverview').innerHTML = order.filter(kind => counts[kind]).map(kind => `<div class="asset-tile"><strong>${escapeHtml(kind.replace('_',' '))}</strong><span>${fmt(counts[kind])} files · ${bytes(sizes[kind])}</span></div>`).join('');
}

async function selectTask(id) {
  const version = ++selectionVersion;
  showPage('explore');
  $('#exploreTitle').textContent = 'Loading case…';
  $('#graphLoading').classList.remove('hidden');
  $('#graphLoading').textContent = 'Loading code graph…';
  $('#graphSvg').replaceChildren();
  try {
    const [detail, focusData] = await Promise.all([api(`/api/tasks/${encodeURIComponent(id)}`), api(endpoint(id,'focus'))]);
    if (version !== selectionVersion) return;
    task = detail; focus = focusData; selectedNode = focus.default_node;
    history.replaceState(null, '', `?task=${encodeURIComponent(id)}`);
    renderSidebar(); renderTask();
    $('#symbolSearch').value = '';
    $('#snapshotSearch').value = (task.changed_files[0] || '').split('/').slice(0,-1).join('/');
    $('#symbolResults').classList.add('hidden');
    await showMode(mode);
  } catch (error) { if (version === selectionVersion) showError(error); }
}

function renderTask() {
  $('#taskRepo').textContent = task.repo.toUpperCase();
  $('#exploreTitle').textContent = task.title;
  $('#taskIdBadge').textContent = task.id;
  $('#taskMeta').innerHTML = `<span>Base ${escapeHtml(task.base_commit.slice(0,12))}</span><span>${escapeHtml(task.created_at?.slice(0,10) || 'Date unavailable')}</span><span>${task.changed_files.length} changed files</span>`;
  $('#taskPreview').textContent = issuePreview(task.problem_statement, task.title);
  $('#changedFiles').innerHTML = focus.groups.length ? focus.groups.map(group => `<button class="changed-file ${group.count ? 'linked' : ''}" data-path="${escapeHtml(group.path)}" title="${group.count} matching graph symbols">${escapeHtml(group.path)}<small>${group.count ? `${group.count} symbols` : 'no graph symbol'}</small></button>`).join('') : '<span class="muted">No changed paths found in patch.</span>';
  $$('.changed-file.linked').forEach(button => button.addEventListener('click', () => { const group = focus.groups.find(item => item.path === button.dataset.path); if (group?.symbols[0]) chooseNode(group.symbols[0]); }));
  const a = task.artifact_summary;
  $('#artifactFacts').textContent = `${fmt(a.nodes)} symbols · ${fmt(a.edges)} calls · ${fmt(a.vectors)} vectors`;
  $('#dialogTitle').textContent = task.title;
}

async function showMode(next) {
  mode = next;
  $$('.mode-button').forEach(button => button.classList.toggle('active', button.dataset.mode === next));
  $('#graphPanel').classList.toggle('hidden', next !== 'graph');
  $('#embeddingPanel').classList.toggle('hidden', next !== 'embedding');
  $('#snapshotPanel').classList.toggle('hidden', next !== 'snapshot');
  $('#graphControls').classList.toggle('hidden', next !== 'graph');
  $('#symbolResults').classList.add('hidden');
  $('#workspaceTitle').textContent = next === 'graph' ? 'Call graph' : next === 'embedding' ? 'Embedding space' : 'Repository snapshot';
  if (!task) return;
  try { if (next === 'graph') await renderGraph(); else if (next === 'embedding') await renderEmbedding(); else await renderSnapshot(); }
  catch (error) { showError(error); }
}

async function chooseNode(name) {
  if (!task || !name) return;
  selectedNode = name;
  $('#symbolResults').classList.add('hidden');
  $('#symbolSearch').value = '';
  await Promise.allSettled([renderInspectorNode(), mode === 'graph' ? renderGraph() : mode === 'embedding' ? renderEmbedding() : Promise.resolve()]);
}

function layoutGraph(nodes, edges) {
  const groups = {caller:[],callee:[],both:[],second:[],center:[]};
  nodes.forEach(node => (groups[node.role] || groups.second).push(node));
  const pos = new Map();
  const seed = (name) => [...name].reduce((n,c) => (n * 33 + c.charCodeAt(0)) >>> 0, 5381);
  const place = (items, start, end, radius) => items.forEach((node,i) => { const angle = start + (end-start) * (i+.5)/Math.max(items.length,1); const jitter = ((seed(node.id)%100)/100-.5)*35; pos.set(node.id,{x:Math.cos(angle)*(radius+jitter),y:Math.sin(angle)*(radius+jitter)}); });
  place(groups.caller,Math.PI*.58,Math.PI*1.42,245);
  place(groups.callee,-Math.PI*.42,Math.PI*.42,245);
  place(groups.both,-Math.PI*.72,-Math.PI*.28,205);
  place(groups.second,Math.PI*.22,Math.PI*.78,285);
  pos.set(groups.center[0]?.id,{x:0,y:0});
  for (let tick=0;tick<75;tick++) {
    const force = new Map(nodes.map(node => [node.id,{x:0,y:0}]));
    for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {
      const a=pos.get(nodes[i].id), b=pos.get(nodes[j].id); if (!a || !b) continue;
      let dx=b.x-a.x, dy=b.y-a.y, dist=Math.max(1,Math.hypot(dx,dy));
      if (dist<72) { const strength=(72-dist)*.035; force.get(nodes[i].id).x-=dx/dist*strength;force.get(nodes[i].id).y-=dy/dist*strength;force.get(nodes[j].id).x+=dx/dist*strength;force.get(nodes[j].id).y+=dy/dist*strength; }
    }
    for (const edge of edges) { const a=pos.get(edge.source),b=pos.get(edge.target); if (!a||!b) continue; const dx=b.x-a.x,dy=b.y-a.y,dist=Math.max(1,Math.hypot(dx,dy)); const strength=(dist-155)*.0015; force.get(edge.source).x+=dx/dist*strength;force.get(edge.source).y+=dy/dist*strength;force.get(edge.target).x-=dx/dist*strength;force.get(edge.target).y-=dy/dist*strength; }
    for (const node of nodes) { if(node.role==='center') continue; const p=pos.get(node.id),f=force.get(node.id);p.x=Math.max(-490,Math.min(490,p.x+f.x));p.y=Math.max(-285,Math.min(285,p.y+f.y)); }
  }
  return pos;
}

function applyGraphView() { const v=graphView; $('#graphSvg').setAttribute('viewBox',`${v.x} ${v.y} ${v.w} ${v.h}`); }
async function renderGraph() {
  if (!selectedNode || !task) return;
  const version = ++graphVersion;
  $('#graphLoading').classList.remove('hidden');
  $('#graphLoading').textContent = 'Tracing the selected symbol…';
  const depth = Number($('#depthSelect').value);
  const data = await api(endpoint(task.id,'graph/neighborhood',{node:selectedNode,depth,limit:75}));
  if (version!==graphVersion || data.center!==selectedNode) return;
  const svg=$('#graphSvg'); svg.replaceChildren();
  const defs=svgEl('defs'), marker=svgEl('marker',{id:'arrow',markerWidth:8,markerHeight:8,refX:14,refY:3,orient:'auto',markerUnits:'userSpaceOnUse'});marker.append(svgEl('path',{d:'M0,0 L7,3 L0,6 Z',fill:'#5b819b'}));defs.append(marker);svg.append(defs);
  const position=layoutGraph(data.nodes,data.edges);
  for(const edge of data.edges){const a=position.get(edge.source),b=position.get(edge.target);if(!a||!b)continue;svg.append(svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:`graph-link ${edge.source===selectedNode||edge.target===selectedNode?'highlight':''}`,'marker-end':'url(#arrow)'}));}
  const colors={center:'#bef09c',caller:'#ba9af1',callee:'#6fe0d2',both:'#e9bc78',second:'#7e91ad'};
  const labels=new Set(data.nodes.filter(node=>node.role==='center').map(node=>node.id));
  data.nodes.filter(node=>node.role!=='center').sort((a,b)=>(b.in_degree+b.out_degree)-(a.in_degree+a.out_degree)).slice(0,9).forEach(node=>labels.add(node.id));
  for(const node of data.nodes){const p=position.get(node.id);if(!p)continue;const group=svgEl('g',{class:'graph-node',transform:`translate(${p.x},${p.y})`});const circle=svgEl('circle',{r:node.role==='center'?13:Math.min(9,5+Math.log1p(node.in_degree+node.out_degree)*1.2),fill:colors[node.role]||colors.second});group.append(circle);if(labels.has(node.id)){const label=svgEl('text',{x:node.role==='caller'?-15:15,y:4,'text-anchor':node.role==='caller'?'end':'start'});label.textContent=shortName(node.id);group.append(label);}const title=svgEl('title');title.textContent=`${node.id} · ${node.in_degree} callers · ${node.out_degree} callees`;group.append(title);group.addEventListener('click',()=>chooseNode(node.id));svg.append(group);}
  $('#graphLoading').classList.add('hidden');
  $('#graphCaption').textContent=`Showing ${fmt(data.nodes.length)} of ${fmt(task.artifact_summary.nodes)} symbols around ${shortName(selectedNode)} · ${data.direct_callers} direct callers · ${data.direct_callees} direct callees${data.limited?' · limit reached':''}. Click a node to recenter.`;
  const points=[...position.values()];
  const x0=Math.min(...points.map(point=>point.x)),x1=Math.max(...points.map(point=>point.x));
  const y0=Math.min(...points.map(point=>point.y)),y1=Math.max(...points.map(point=>point.y));
  const width=Math.max(600,x1-x0+260,(y1-y0+210)*1100/680);
  const height=width*680/1100;
  graphDefaultView={x:(x0+x1-width)/2,y:(y0+y1-height)/2,w:width,h:height};
  graphView={...graphDefaultView};applyGraphView();
  renderInspectorNode().catch(showError);
}

async function renderInspectorNode(){
  if(!selectedNode||!task)return;
  const id=task.id,name=selectedNode;
  const data=await api(endpoint(id,'graph/node',{node:name}));
  if(task?.id!==id||selectedNode!==name)return;
  $('#inspectorTitle').textContent=shortName(name);
  $('#inspectorSubtitle').textContent=name;
  const content=$('#inspectorContent');
  content.innerHTML=`<div class="inspector-section"><h3>Connections</h3><div class="inspector-fact"><b>Callers</b><span>${fmt(data.callers.length)}</span></div><div class="inspector-fact"><b>Callees</b><span>${fmt(data.callees.length)}</span></div></div><div class="inspector-section"><h3>Source symbol</h3><pre id="symbolCode"></pre></div><div class="inspector-section"><h3>Calls this symbol</h3><div class="connection-list" id="callersList"></div></div><div class="inspector-section"><h3>Called by this symbol</h3><div class="connection-list" id="calleesList"></div></div><button id="semanticJump" class="small-button">Explore embedding neighbors →</button>`;
  $('#symbolCode').textContent=data.text||'No source text supplied for this node.';
  for(const [selector,names] of [['#callersList',data.callers],['#calleesList',data.callees]]){
    const holder=$(selector);holder.innerHTML=names.length?names.slice(0,24).map(value=>`<button title="${escapeHtml(value)}" data-node="${escapeHtml(value)}">${escapeHtml(shortName(value))}</button>`).join(''):'<span class="inspector-note">None in this graph.</span>';
    holder.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>chooseNode(button.dataset.node)));
  }
  $('#semanticJump').addEventListener('click',()=>showMode('embedding'));
}

async function renderEmbedding(){
  if(!selectedNode||!task)return;
  const id=task.id,name=selectedNode;
  $('#embeddingTitle').textContent=`Neighbors of ${shortName(name)}`;
  $('#embeddingSvg').replaceChildren();
  $('#embeddingList').innerHTML='<div class="loading-state">Comparing symbol vectors…</div>';
  const data=await api(endpoint(id,'embeddings/nearest',{node:name,k:29}));
  if(task?.id!==id||selectedNode!==name||mode!=='embedding')return;
  const svg=$('#embeddingSvg');svg.replaceChildren();
  svg.append(svgEl('line',{class:'embedding-axis',x1:390,y1:35,x2:390,y2:395}));svg.append(svgEl('line',{class:'embedding-axis',x1:45,y1:215,x2:735,y2:215}));
  data.points.forEach((point,index)=>{const x=390+point.x*325,y=215-point.y*165;const circle=svgEl('circle',{class:'embedding-point',cx:x,cy:y,r:index===0?11:index<6?7:5,fill:index===0?'#bef09c':index<6?'#ba9af1':'#78aaf4',opacity:index===0?1:.62+Math.max(0,point.similarity)*.35});const title=svgEl('title');title.textContent=`${point.id} · cosine ${point.similarity.toFixed(5)}`;circle.append(title);circle.addEventListener('click',()=>chooseNode(point.id));svg.append(circle);if(index<5){const label=svgEl('text',{class:'embedding-label',x:x+12,y:y-8});label.textContent=shortName(point.id);svg.append(label);}});
  $('#embeddingList').innerHTML=data.points.slice(1).map((point,index)=>`<div class="nearest-row"><span>${String(index+1).padStart(2,'0')}</span><button title="${escapeHtml(point.id)}" data-node="${escapeHtml(point.id)}">${escapeHtml(point.id)}</button><div class="similarity-track"><i style="width:${Math.max(0,Math.min(100,(point.similarity+1)*50))}%"></i><em>${point.similarity.toFixed(5)}</em></div></div>`).join('');
  $$('#embeddingList button').forEach(button=>button.addEventListener('click',()=>chooseNode(button.dataset.node)));
  $('#inspectorTitle').textContent=shortName(name);
  $('#inspectorSubtitle').textContent=`${data.dimensions}-dimensional vector · ${fmt(data.archive_vectors)} symbols in this archive`;
  $('#inspectorContent').innerHTML=`<div class="inspector-section"><h3>Vector sample</h3><pre>${escapeHtml(data.vector_preview.join(', '))}</pre></div><div class="inspector-section"><h3>Reading this map</h3><p class="inspector-note">Ranking uses cosine similarity across the original 256 values. The plot is a PCA projection of this local group, so its 2D distances are only a guide.</p></div><button id="graphJump" class="small-button">See calls around this symbol →</button>`;
  $('#graphJump').addEventListener('click',()=>showMode('graph'));
}

async function renderSnapshot(){
  if(!task)return;
  const id=task.id,q=$('#snapshotSearch').value.trim();
  $('#snapshotResults').innerHTML='<div class="loading-state">Scanning snapshot paths…</div>';
  const data=await api(endpoint(id,'snapshot',{q,limit:300}));
  if(task?.id!==id||mode!=='snapshot'||$('#snapshotSearch').value.trim()!==q)return;
  $('#snapshotResults').innerHTML=`<div class="muted" style="font-size:11px;margin-bottom:9px">${fmt(data.total)} matching files${data.total>data.files.length?` · first ${data.files.length} shown`:''}</div>`+data.files.map(file=>`<button class="snapshot-row" data-path="${escapeHtml(file.path)}"><span>${escapeHtml(file.path)}</span><span>${bytes(file.bytes)}</span></button>`).join('');
  $$('#snapshotResults .snapshot-row').forEach(button=>button.addEventListener('click',()=>previewSnapshotFile(button.dataset.path)));
  $('#inspectorTitle').textContent='Repository snapshot';$('#inspectorSubtitle').textContent=`${task.id}.tgz · ${bytes(task.snapshot_bytes)}`;
  $('#inspectorContent').innerHTML='<p class="inspector-note">Select a source file to read its contents at the task’s base commit. Use the path filter to find a module.</p>';
}

async function previewSnapshotFile(path){
  if(!task)return;const id=task.id;
  try{const data=await api(endpoint(id,'snapshot/file',{path}));if(task?.id!==id)return;$('#inspectorTitle').textContent=path.split('/').pop();$('#inspectorSubtitle').textContent=path;const holder=$('#inspectorContent');holder.innerHTML=`<div class="inspector-section"><h3>Base snapshot · ${bytes(data.bytes)}</h3>${data.text==null?`<p class="inspector-note">${escapeHtml(data.reason||'No preview available')}</p>`:'<pre id="snapshotText"></pre>'}</div>`;if(data.text!=null)$('#snapshotText').textContent=data.text;}catch(error){showError(error);}
}

function renderFiles(){
  if(!catalog)return;
  const select=$('#fileKind');if(select.options.length===1){const kinds=[...new Set(catalog.files.map(file=>file.kind))].sort();for(const kind of kinds){const option=document.createElement('option');option.value=kind;option.textContent=kind.replace('_',' ');select.append(option);}}
  const q=$('#fileSearch').value.trim().toLowerCase(),kind=select.value;
  const files=catalog.files.filter(file=>(kind==='all'||file.kind===kind)&&(!q||file.path.toLowerCase().includes(q)));
  $('#fileCount').textContent=`${fmt(files.length)} files`;
  $('#fileList').innerHTML=files.length?files.map(file=>`<button class="file-row" data-path="${escapeHtml(file.path)}"><span class="file-row-kind">${escapeHtml(file.kind)}</span><span class="file-row-path" title="${escapeHtml(file.path)}">${escapeHtml(file.path)}</span><span class="file-row-size">${bytes(file.bytes)}</span></button>`).join(''):'<div class="loading-state">No files match this filter.</div>';
  $$('#fileList .file-row').forEach(button=>button.addEventListener('click',()=>previewFile(button.dataset.path)));
}

async function previewFile(path){
  $$('#fileList .file-row').forEach(row=>row.classList.toggle('active',row.dataset.path===path));
  $('#filePreviewTitle').textContent=path;
  $('#filePreviewBody').textContent='Loading preview…';
  try{
    const data=await api(`/api/files/preview?${new URLSearchParams({path})}`);
    const holder=$('#filePreviewBody');holder.replaceChildren();
    const summary=data.summary;
    if(summary){const card=document.createElement('div');card.className='file-summary';card.innerHTML=`<div><b>${fmt(summary.nodes)}</b><span>graph symbols</span></div><div><b>${fmt(summary.edges)}</b><span>call edges</span></div><div><b>${fmt(summary.vectors)}</b><span>embedding vectors</span></div><div><b>${fmt(summary.dimensions)}D</b><span>vector size</span></div>`;holder.append(card);const p=document.createElement('p');p.textContent=`Base commit ${summary.base_commit.slice(0,12)} · ${summary.tasks.length} related task${summary.tasks.length===1?'':'s'}`;holder.append(p);for(const id of summary.tasks){const button=document.createElement('button');button.className='small-button';button.style.marginRight='7px';button.textContent=`Explore ${id} →`;button.addEventListener('click',()=>selectTask(id));holder.append(button);}}
    if(data.reason){const p=document.createElement('p');p.textContent=data.reason;holder.append(p);}
    if(data.entries!=null){const p=document.createElement('p');p.textContent=`${fmt(data.entries)} files in wheel`;holder.append(p);}
    if(data.text!=null){const pre=document.createElement('pre');pre.textContent=data.text;holder.append(pre);}
    if(data.sample_entries?.length){const p=document.createElement('p');p.textContent='First archive entries';holder.append(p);const pre=document.createElement('pre');pre.textContent=data.sample_entries.join('\n');holder.append(pre);}
  }catch(error){showError(error);$('#filePreviewBody').textContent='Preview unavailable.';}
}

function openTaskDialog(detail='issue'){
  if(!task)return;
  const key=detail==='patch'?'patch':detail==='test'?'test_patch':'problem_statement';
  $$('.dialog-tab').forEach(button=>button.classList.toggle('active',button.dataset.detail===detail));
  $('#dialogBody').textContent=task[key]||'No content supplied.';
  if(!$('#taskDialog').open)$('#taskDialog').showModal();
}

async function searchSymbols(){
  if(!task)return;
  const q=$('#symbolSearch').value.trim();
  if(!q){$('#symbolResults').classList.add('hidden');return;}
  const id=task.id;
  try{const data=await api(endpoint(id,'graph/search',{q,limit:40}));if(task?.id!==id||$('#symbolSearch').value.trim()!==q)return;const holder=$('#symbolResults');holder.classList.remove('hidden');holder.innerHTML=data.results.length?data.results.map(item=>`<button class="symbol-result" data-node="${escapeHtml(item.id)}"><span>${escapeHtml(item.id)}</span><span>${item.degree} calls</span></button>`).join(''):'<div class="loading-state">No matching symbols.</div>';holder.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>chooseNode(button.dataset.node)));}catch(error){showError(error);}
}

function bindEvents(){
  $$('.nav-button').forEach(button=>button.addEventListener('click',()=>showPage(button.dataset.view)));
  $('#mobileMenu').addEventListener('click',()=>$('#sidebar').classList.toggle('open'));
  $('#taskSearch').addEventListener('input',renderSidebar);
  $$('.mode-button').forEach(button=>button.addEventListener('click',()=>showMode(button.dataset.mode).catch(showError)));
  $('#depthSelect').addEventListener('change',()=>renderGraph().catch(showError));
  $('#resetGraph').addEventListener('click',()=>{graphView={...graphDefaultView};applyGraphView();});
  $('#symbolSearch').addEventListener('input',()=>{clearTimeout(symbolTimer);symbolTimer=setTimeout(searchSymbols,170);});
  $('#snapshotSearch').addEventListener('input',()=>{clearTimeout(snapshotTimer);snapshotTimer=setTimeout(()=>renderSnapshot().catch(showError),200);});
  $('#fileSearch').addEventListener('input',renderFiles);$('#fileKind').addEventListener('change',renderFiles);
  $('#readIssue').addEventListener('click',()=>openTaskDialog('issue'));
  $('#closeDialog').addEventListener('click',()=>$('#taskDialog').close());
  $$('.dialog-tab').forEach(button=>button.addEventListener('click',()=>openTaskDialog(button.dataset.detail)));
  $('#taskDialog').addEventListener('click',event=>{if(event.target===$('#taskDialog'))$('#taskDialog').close();});
  $('#graphSvg').addEventListener('wheel',event=>{event.preventDefault();const factor=event.deltaY>0?1.12:.89;const cx=graphView.x+graphView.w/2,cy=graphView.y+graphView.h/2;graphView.w=Math.max(350,Math.min(2200,graphView.w*factor));graphView.h=graphView.w*680/1100;graphView.x=cx-graphView.w/2;graphView.y=cy-graphView.h/2;applyGraphView();},{passive:false});
}

async function initialize(){
  bindEvents();
  try{catalog=await api('/api/catalog');renderSidebar();renderOverview();renderFiles();const params=new URLSearchParams(location.search);const requested=params.get('task');const first=catalog.tasks.find(row=>row.id===requested)?.id||catalog.tasks.find(row=>row.id==='fastapi_15589')?.id||catalog.tasks[0]?.id;if(first)await selectTask(first);}catch(error){showError(error);$('#taskList').innerHTML='<div class="loading-state">Could not load the local dataset. Check the server terminal.</div>';}
}
initialize();
