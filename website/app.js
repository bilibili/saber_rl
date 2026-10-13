(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  let lang = 'en';
  try { lang = localStorage.getItem('saber-language') === 'zh' ? 'zh' : 'en'; } catch (_) {}
  const state = { mode: 'FastThink', caseMode: 'FastThink', size: '1.5B', benchmark: 'MATH', result: 'FastThink' };
  const I = (en, zh) => lang === 'zh' ? zh : en;
  const fmt = n => n.toLocaleString('en-US');
  const modeNames = ['NoThink', 'FastThink', 'CoreThink', 'DeepThink'];
  const order = ['Base', 'DeepThink', 'CoreThink', 'FastThink', 'NoThink', 'Base-NoThink'];
  const colors = { Base: '#3e3d35', DeepThink: '#7585a8', CoreThink: '#b28a4c', FastThink: '#47745c', NoThink: '#8a7771', 'Base-NoThink': '#aaa396' };
  const descriptions = {
    NoThink: ['A direct answer, with no explicit thinking trace. For the moments when you just need the answer.', '不生成显式思考过程，直接给出答案，适合需要迅速响应的场景。'],
    FastThink: ['Just the essential steps. A shorter route to an answer, with a smaller reasoning budget.', '聚焦核心求解步骤，以更少的思考预算走向答案。'],
    CoreThink: ['Room for intermediate steps and reflection. A balance of reasoning depth and token cost.', '为中间步骤与反思保留空间，在推理深度与 token 成本之间取得平衡。'],
    DeepThink: ['Room to explore, reflect, and self-check. No explicit target budget on the thinking process.', '充分探索、反思与自检，不为思考过程设置显式目标预算。']
  };
  function renderMode() {
    $$('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
    $$('[data-route]').forEach(g => g.classList.toggle('active', g.dataset.route === state.mode));
    $('#mode-description').textContent = I(...descriptions[state.mode]);
    $('.mode-index').textContent = `0${modeNames.indexOf(state.mode) + 1} / 04`;
    $('.mode-index').style.color = colors[state.mode];
  }
  function renderCase() {
    $$('[data-case-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.caseMode === state.caseMode)));
    const traces = {
      NoThink: ['<p>The number of positive whole-number divisors of 196 is <mark>9</mark>.</p>', '<p>196 共有 <mark>9 个</mark>正整数约数。</p>'],
      FastThink: ['<p><mark>196 = 2² × 7².</mark></p><p>Add one to each exponent, then multiply:<br><mark>(2 + 1) × (2 + 1) = 9.</mark></p>', '<p><mark>196 = 2² × 7²。</mark></p><p>将各质因子的指数加一，再相乘：<br><mark>(2 + 1) × (2 + 1) = 9。</mark></p>'],
      CoreThink: ['<p><mark class="reflection">Start with the prime factorization.</mark> Divide 196 by 2 twice to get 49, which is 7². So <mark>196 = 2² × 7².</mark></p><p><mark class="reflection">Each exponent tells us how many choices we have.</mark> The divisor count is <mark>(2 + 1) × (2 + 1) = 9.</mark></p>', '<p><mark class="reflection">先从质因数分解入手。</mark>196 连续除以两次 2，得到 49，也就是 7²。因此 <mark>196 = 2² × 7²。</mark></p><p><mark class="reflection">每个指数决定了相应质因子的可选次数。</mark>约数个数为 <mark>(2 + 1) × (2 + 1) = 9。</mark></p>'],
      DeepThink: ['<p><mark>196 = 2² × 7²</mark>, giving <mark>(2 + 1)(2 + 1) = 9</mark> divisors. <mark class="reflection">Check the factorization and apply the divisor formula.</mark></p><p><mark class="self-check">Verify by listing them: 1, 2, 4, 7, 14, 28, 49, 98, 196.</mark> Nine in total—the answer is consistent.</p>', '<p>由 <mark>196 = 2² × 7²</mark> 得到 <mark>(2 + 1)(2 + 1) = 9</mark> 个约数。<mark class="reflection">检查分解过程，再应用约数个数公式。</mark></p><p><mark class="self-check">再逐一列举验证：1、2、4、7、14、28、49、98、196。</mark>共九个，与公式结果一致。</p>']
    };
    $('#case-text').innerHTML = `<div class="case-mode-label">${state.caseMode.toUpperCase()} / ${I('ILLUSTRATED SUMMARY', '精简示意')}</div>${I(...traces[state.caseMode])}`;
  }
  function renderChart() {
    const ds = window.SABER_DATA[state.size];
    const rows = order.map(mode => ({ mode, ...ds[mode][state.benchmark] }));
    const mobile = window.matchMedia('(max-width: 600px)').matches;
    const width = mobile ? 360 : 740, left = mobile ? 34 : 70, right = mobile ? 313 : 667, top = mobile ? 35 : 42, bottom = mobile ? 270 : 365;
    $('#results-chart').setAttribute('viewBox', `0 0 ${width} ${mobile ? 333 : 440}`);
    const maxTokens = Math.max(...rows.map(r => r.thinkTokens));
    const step = maxTokens > 6000 ? 2000 : 1000;
    const xMax = Math.ceil(maxTokens / step) * step;
    const yMin = Math.max(0, Math.floor((Math.min(...rows.map(r => r.accuracy)) - 3) / 5) * 5);
    const yMax = Math.min(100, Math.ceil((Math.max(...rows.map(r => r.accuracy)) + 3) / 5) * 5);
    const x = n => left + n / xMax * (right - left);
    const y = n => bottom - (n - yMin) / (yMax - yMin) * (bottom - top);
    let svg = `<title id="chart-title">${I('Accuracy versus average thinking tokens', '准确率与平均思考 token 数')}</title><desc id="chart-desc">${state.size} ${state.benchmark}. ${rows.map(r => `${r.mode}: ${r.accuracy}% accuracy, ${r.thinkTokens} tokens`).join('; ')}</desc>`;
    svg += `<text x="${left}" y="17" class="axis-label">${I('Accuracy (%) ↑', '准确率（%）↑')}</text>`;
    const yStep = yMax - yMin > 25 ? 10 : 5;
    for (let tick = yMin; tick <= yMax; tick += yStep) svg += `<line x1="${left}" x2="${right + (mobile ? 4 : 18)}" y1="${y(tick)}" y2="${y(tick)}" class="chart-grid"/><text x="${left - 14}" y="${y(tick) + 4}" text-anchor="end" class="axis-text">${tick}</text>`;
    for (let tick = 0; tick <= xMax; tick += step) svg += `<text x="${x(tick)}" y="${bottom + 25}" text-anchor="middle" class="axis-text">${tick === 0 ? '0' : `${tick / 1000}k`}</text>`;
    svg += `<line x1="${left}" x2="${right + (mobile ? 4 : 18)}" y1="${bottom}" y2="${bottom}" stroke="#c9c0ac"/><text x="${(right + left) / 2}" y="${bottom + (mobile ? 53 : 63)}" text-anchor="middle" class="axis-label">${I('Average thinking tokens →', '平均思考 token 数 →')}</text>`;
    const curve = ['NoThink', 'FastThink', 'CoreThink', 'DeepThink'].map(m => rows.find(r => r.mode === m));
    svg += `<polyline points="${curve.map(r => `${x(r.thinkTokens)},${y(r.accuracy)}`).join(' ')}" fill="none" stroke="#a7b498" stroke-width="1.4" stroke-dasharray="4 5"/>`;
    const labels = [];
    rows.forEach(r => {
      const px=x(r.thinkTokens), py=y(r.accuracy), active=r.mode===state.result, color=colors[r.mode];
      let label = r.mode === 'Base-NoThink' ? I('Base · NoThink', '基线 · NoThink') : r.mode === 'Base' ? I('Base model', '基座模型') : r.mode;
      let anchor = r.mode === 'Base' ? 'end' : 'start';
      let lx = anchor === 'end' ? px - 12 : px + 12, ly = py - 12;
      const fontSize = mobile ? 10 : 12;
      const labelWidth = [...label].reduce((w, ch) => w + (/[^\x00-\x7f]/.test(ch) ? fontSize : fontSize * .57), 0);
      const candidates = [-12, 23, -30, 40].flatMap(offset => ['start', 'end'].map(align => ({x: px + (align === 'start' ? 12 : -12), y: py + offset, align})));
      if(r.mode === 'Base') candidates.reverse();
      for(const c of candidates) {
        const rect = {left: c.align === 'start' ? c.x : c.x - labelWidth, right: c.align === 'start' ? c.x + labelWidth : c.x, top: c.y - fontSize, bottom: c.y + 3};
        if(rect.left < 4 || rect.right > width - 5 || rect.top < 20 || rect.bottom > bottom - 2) continue;
        const intersects = labels.some(p => rect.left < p.right + 4 && rect.right > p.left - 4 && rect.top < p.bottom + 3 && rect.bottom > p.top - 3);
        const coversPoint = rows.some(p => Math.abs(x(p.thinkTokens) - px) + Math.abs(y(p.accuracy) - py) > 1 && x(p.thinkTokens) > rect.left - 5 && x(p.thinkTokens) < rect.right + 5 && y(p.accuracy) > rect.top - 5 && y(p.accuracy) < rect.bottom + 5);
        if(!intersects && !coversPoint) { lx=c.x; ly=c.y; anchor=c.align; labels.push(rect); break; }
      }
      svg += `<g class="chart-point" role="button" tabindex="0" data-result="${r.mode}" aria-label="${label}: ${r.accuracy}%, ${r.thinkTokens} ${I('thinking tokens','思考 token')}" aria-pressed="${active}">${active?`<circle cx="${px}" cy="${py}" r="15" fill="${color}" opacity=".10"/>`:''}<circle cx="${px}" cy="${py}" r="${active ? 6.5 : 5}" fill="${r.mode.startsWith('Base')?'#f4f0e5':color}" stroke="${color}" stroke-width="1.8"/><text x="${lx}" y="${ly}" fill="${color}" text-anchor="${anchor}" class="chart-label">${label}</text></g>`;
    });
    $('#results-chart').innerHTML=svg;
    $('.chart-legend').innerHTML=rows.map(r=>`<button type="button" data-result="${r.mode}" style="--mode-color:${colors[r.mode]}" aria-pressed="${r.mode===state.result}"><i></i>${r.mode === 'Base-NoThink' ? I('Base · NoThink','基线 · NoThink') : r.mode === 'Base' ? I('Base model','基座模型') : r.mode}</button>`).join('');
    $$('[data-result]').forEach(b=>{
      b.addEventListener('click',()=>{state.result=b.dataset.result;renderChart();});
      if(b.tagName.toLowerCase()==='g')b.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();state.result=b.dataset.result;renderChart();const restored=$(`g[data-result="${state.result}"]`);restored?.focus();}});
    });
    const r=rows.find(r=>r.mode===state.result);
    const baseline=rows.find(candidate=>candidate.mode===(r.mode==='NoThink'?'Base-NoThink':'Base'));
    $('#result-context').textContent=`${state.size} / ${state.benchmark}`;
    $('#result-mode').textContent=r.mode === 'Base-NoThink'?I('Base · NoThink','基线 · NoThink'):r.mode === 'Base'?I('Base model','基座模型'):r.mode;
    $('#result-mode').style.color=colors[r.mode];
    $('#result-accuracy').textContent=`${r.accuracy.toFixed(1)}%`;
    $('#result-tokens').textContent=fmt(r.thinkTokens);
    const delta=r.accuracy-baseline.accuracy;
    if(r.mode.startsWith('Base')) $('#result-comparison').innerHTML=I('The reference point for this model size and benchmark.','该模型规模与测试集下的对照基线。');
    else if(r.mode==='NoThink') $('#result-comparison').innerHTML=I(`<strong>+${delta.toFixed(1)} percentage points</strong> in accuracy, with no explicit thinking trace.`,`不生成显式思考过程，准确率提升 <strong>${delta.toFixed(1)} 个百分点</strong>。`);
    else {
      const saved=((baseline.thinkTokens-r.thinkTokens)/baseline.thinkTokens*100).toFixed(1);
      $('#result-comparison').innerHTML=I(`<strong>${saved}% fewer thinking tokens</strong> and <strong>${delta>=0?'+':''}${delta.toFixed(1)} pp</strong> accuracy versus the base model.`,`相对基座，思考 token <strong>减少 ${saved}%</strong>，准确率<strong>${delta>=0?'提升':'下降'} ${Math.abs(delta).toFixed(1)} 个百分点</strong>。`);
    }
    $('#table-caption').textContent=I(`Reported results · ${state.size} · ${state.benchmark}`,`论文实验结果 · ${state.size} · ${state.benchmark}`);
    $('#results-table').innerHTML=rows.map(r=>`<tr class="${r.mode===state.result?'selected':''}"><td>${r.mode==='Base'?'DeepSeek-R1-Distill-Qwen-'+state.size:r.mode==='Base-NoThink'?'Base · NoThink':'SABER · '+r.mode}</td><td>${r.accuracy.toFixed(1)}</td><td>${fmt(r.thinkTokens)}</td></tr>`).join('');
  }
  function setSize(size) {
    state.size=size;
    $$('[data-size]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.size===size)));
    $('#benchmark option[value="LiveBench-R"]').disabled=size!=='7B';
    if(size==='1.5B'&&state.benchmark==='LiveBench-R'){state.benchmark='MATH';$('#benchmark').value='MATH';}
    renderChart();
  }
  function translate() {
    document.documentElement.lang=lang==='zh'?'zh-CN':'en';
    document.body.classList.toggle('lang-zh',lang==='zh');
    $$('[data-en][data-zh]').forEach(el=>el.innerHTML=el.dataset[lang]);
    $('#language').innerHTML=lang==='zh'?'EN <span>↗</span>':'中文 <span>↗</span>';
    $('#language').setAttribute('aria-label',lang==='zh'?'Switch to English':'切换到中文');
    document.title=I('SABER — Give every question just enough thought.','SABER — 让每个问题得到恰到好处的思考');
    $('#copy-status').textContent='';
    renderMode();renderCase();renderChart();
  }
  $$('[data-mode]').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;renderMode();}));
  $$('[data-case-mode]').forEach(b=>b.addEventListener('click',()=>{state.caseMode=b.dataset.caseMode;renderCase();}));
  $$('[data-size]').forEach(b=>b.addEventListener('click',()=>setSize(b.dataset.size)));
  $('#benchmark').addEventListener('change',e=>{state.benchmark=e.target.value;renderChart();});
  $('#language').addEventListener('click',()=>{lang=lang==='en'?'zh':'en';try{localStorage.setItem('saber-language',lang);}catch(_){}translate();});
  $('#show-generalization').addEventListener('click',()=>{state.size='7B';state.benchmark='LiveBench-R';state.result='DeepThink';$('#benchmark').value='LiveBench-R';setSize('7B');$('#results').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});});
  $('#copy-citation').addEventListener('click',async()=>{
    const citation=$('#citation').textContent;
    let copied=false;
    try { if(navigator.clipboard){await navigator.clipboard.writeText(citation);copied=true;} } catch(_) {}
    if(!copied){
      const textarea=document.createElement('textarea');textarea.value=citation;textarea.style.position='fixed';textarea.style.opacity='0';document.body.append(textarea);textarea.select();
      try{copied=document.execCommand('copy');}catch(_){} textarea.remove();$('#copy-citation').focus();
    }
    $('#copy-status').textContent=copied?I('Citation copied.','引用已复制。'):I('Select and copy the citation above.','请选中上方引用文本复制。');
  });
  let resizeFrame;
  window.addEventListener('resize',()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(renderChart);});
  translate();
})();
