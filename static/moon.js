'use strict';

const byId = id => document.getElementById(id);
const map = byId('map');
const ctx = map.getContext('2d');
const current = {
  panel: null, settings: null, config: null, mode: 'illumination', view: 'marked',
  tool: 'point', first: null, pin: null, saved: null, image: null, proposal: null,
  drawing: 0, computing: 0, reading: 0, opening: 0,
};
const columns = {
  disk_px: '月盘像素', light_px: '亮区像素', shadow_px: '暗区像素', light_pct: '亮区 %',
  light_arcsec2: '亮区角秒²', y: '源 Y', x: '源 X', jump: '亮度落差', candidates: '候选数',
  distance_arcsec: '距离（角秒）', luma: '亮度', in_disk: '月盘', bright: '亮度归类',
  reference: '参照', pixels: '像素数', p10: '第 10 百分位', median: '中位数', p90: '第 90 百分位',
  lower: '下界', upper: '上界', sky_px: '背景像素',
  channel: '颜色通道', samples: '像素数', floor_px: '接近黑位', ceiling_px: '接近白位',
};
const directions = {
  point: '点击照片读取源像素。', inspect: '依次点选检查窗的两个角。',
  disk: '点选月盘外接框的两个角。', bright: '框选一块受光区作为参照。',
  dark: '框选一块暗面作为参照。', line: '依次点选亮度剖线的两端。',
};
const inputKeys = ['threshold', 'arcsec_per_pixel', 'sun_side'];

function status(message, error = false) {
  const element = byId('status');
  element.textContent = message;
  element.toggleAttribute('data-error', error);
}

function safe(action) {
  return async event => {
    try {
      await action(event);
    } catch (error) {
      status(error.message, true);
    }
  };
}

async function response(path, method = 'GET', data) {
  const options = {method};
  if (data instanceof FormData) {
    options.body = data;
  } else if (data !== undefined) {
    options.body = JSON.stringify(data);
    options.headers = {'Content-Type': 'application/json'};
  }
  const reply = await fetch(path, options);
  if (!reply.ok) {
    const problem = await reply.json().catch(() => ({}));
    throw Error(problem.error || '读取检查服务失败。');
  }
  return reply;
}

async function json(path, method, data) {
  return (await response(path, method, data)).json();
}

function endpoint(suffix = '') {
  return '/moon/panels/' + current.panel.id + suffix;
}

function sourceUrl(layer, download = false) {
  const query = new URLSearchParams();
  if (current.saved) query.set('report', current.saved.id);
  if (download) query.set('download', '1');
  return endpoint('/source/' + layer + '.png') + (query.size ? '?' + query.toString() : '');
}

function reportUrl(suffix = '') {
  return endpoint('/reports/' + current.saved.id + suffix);
}

function snapshot() {
  return {settings: current.settings, mode: current.mode, revision: current.panel.revision};
}

function selected(button, active) {
  button.classList.toggle('active', active);
  button.setAttribute('aria-pressed', String(active));
}

function announceMap(message) {
  byId('map-note').textContent = message;
}

function drawMap() {
  if (!current.image) return;
  ctx.drawImage(current.image, 0, 0, map.width, map.height);
  ctx.strokeStyle = '#a6eddf';
  ctx.lineWidth = Math.max(1, map.width / 800);
  ctx.setLineDash([6, 4]);
  const options = current.saved?.settings || current.settings;
  if (current.view !== 'marked') {
    const [x0, y0, x1, y1] = options.inspect;
    ctx.strokeRect(x0 + 1, y0 + 1, x1 - x0 - 2, y1 - y0 - 2);
  }
  ctx.setLineDash([]);
  const proposal = current.proposal;
  if (proposal) {
    const {cx, cy, radius} = proposal.circle;
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = '#f7cea0';
    ctx.lineWidth = Math.max(1.5, map.width / 500);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const [x, y, side] of proposal.samples) {
      ctx.fillStyle = side === 'bright' ? '#f8c471' : '#7fb5f8';
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
  }
  const point = current.first || current.pin;
  if (!point) return;
  const [x, y] = point;
  const arm = Math.max(7, map.width / 105);
  ctx.strokeStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(x - arm, y);
  ctx.lineTo(x + arm, y);
  ctx.moveTo(x, y - arm);
  ctx.lineTo(x, y + arm);
  ctx.stroke();
  ctx.strokeStyle = '#121b1b';
  ctx.strokeRect(x - arm, y - arm, arm * 2, arm * 2);
}

async function paint() {
  const sequence = ++current.drawing;
  const settings = current.saved?.settings || current.settings;
  let source = sourceUrl(settings.light);
  let marker;
  if (current.view === 'marked') {
    if (current.saved) {
      source = reportUrl('/overlay.png');
    } else {
      const reply = await response(endpoint('/preview.png'), 'POST', snapshot());
      marker = URL.createObjectURL(await reply.blob());
      source = marker;
    }
  }
  const picture = new Image();
  picture.src = source;
  try {
    await picture.decode();
  } finally {
    if (marker) URL.revokeObjectURL(marker);
  }
  if (sequence !== current.drawing) return;
  current.image = picture;
  drawMap();
}

function renderArchive() {
  const holder = byId('archive');
  holder.replaceChildren();
  if (!current.panel.reports.length) {
    const empty = document.createElement('p');
    empty.textContent = '目前没有保存的检查。';
    holder.append(empty);
  }
  for (const report of current.panel.reports) {
    const button = document.createElement('button');
    button.className = 'archive-row';
    const date = document.createElement('small');
    date.textContent = new Date(report.created).toLocaleString('zh-CN', {hour12: false});
    button.append(report.heading, date);
    selected(button, report.id === current.saved?.id);
    button.onclick = safe(() => openReport(report.id));
    holder.append(button);
  }
}

function renderHeader() {
  const panel = current.panel;
  const settings = current.saved?.settings || current.settings;
  const sources = current.saved?.sources || panel.sources;
  byId('title').textContent = panel.title;
  const details = [panel.width + ' × ' + panel.height + ' px'];
  if (panel.synthetic) details.push('合成图像');
  if (sources.b) details.push(sources.b.prepared ? '对照图来自照片准备' : '两张对照照片');
  details.push(current.config.lights[settings.light]);
  byId('metadata').textContent = details.join(' · ');
  byId('prepare').href = '/prepare?panel=' + panel.id;
  byId('source').href = sourceUrl(settings.light, true);
  byId('desk').hidden = false;
  byId('empty').hidden = true;
  byId('result').hidden = false;
  for (const button of document.querySelectorAll('[data-light]')) {
    selected(button, button.dataset.light === settings.light);
    button.disabled = Boolean(current.saved) || !sources[button.dataset.light];
  }
  for (const button of document.querySelectorAll('[data-view]')) selected(button, button.dataset.view === current.view);
  for (const button of document.querySelectorAll('[data-tool]')) {
    selected(button, button.dataset.tool === current.tool);
    button.disabled = Boolean(current.saved) && button.dataset.tool !== 'point';
  }
  byId('fit-limb').disabled = Boolean(current.saved) || !sources.b;
  for (const key of inputKeys) {
    byId(key).value = settings[key];
    byId(key).disabled = Boolean(current.saved);
  }
  for (const button of document.querySelectorAll('[data-mode]')) {
    selected(button, button.dataset.mode === current.mode);
    button.disabled = Boolean(current.saved);
  }
  byId('retain-settings').disabled = Boolean(current.saved);
  byId('save').hidden = Boolean(current.saved);
  byId('return-current').hidden = !current.saved;
  byId('exports').hidden = !current.saved;
  if (current.saved) {
    byId('csv').href = reportUrl('/report.csv');
    byId('overlay').href = reportUrl('/overlay.png?download=1');
  }
  renderArchive();
}

function display(value, key) {
  if (value === null) return '—';
  if (key === 'in_disk') return value ? '盘内' : '盘外';
  if (key === 'bright') return value ? '高于门槛' : '低于门槛';
  return String(value);
}

function renderResult(report) {
  byId('result-heading').textContent = report.heading;
  const [x0, y0, x1, y1] = report.settings.inspect;
  const note = [current.config.lights[report.settings.light], '窗 ' + (x1 - x0) + ' × ' + (y1 - y0) + ' px'];
  if (current.saved) note.push('已存档');
  if (report.mode === 'profile' && report.summary) {
    const summary = report.summary;
    if (summary.span_arcsec === null) {
      note.push('剖线未穿过月盘');
    } else {
      let span = '盘内 ' + summary.span_arcsec + ' 角秒';
      if (summary.entry) span += ' · 进盘 (' + summary.entry.join(', ') + ')';
      if (summary.exit) span += ' · 出盘 (' + summary.exit.join(', ') + ')';
      note.push(span);
    }
  }
  byId('result-note').textContent = note.join(' · ');
  const keys = Object.keys(report.rows[0] || {}).filter(key => key !== 'box');
  const header = document.createElement('tr');
  for (const key of keys) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = columns[key] || key;
    header.append(cell);
  }
  byId('columns').replaceChildren(header);
  const body = document.createDocumentFragment();
  for (const row of report.rows) {
    const entry = document.createElement('tr');
    for (const key of keys) {
      const cell = document.createElement('td');
      cell.textContent = display(row[key], key);
      entry.append(cell);
    }
    if (row.x !== undefined && row.y !== undefined) {
      entry.dataset.sample = 'yes';
      entry.tabIndex = 0;
      entry.onclick = safe(() => inspectPoint(Math.floor(row.x), row.y));
      entry.onkeydown = safe(event => {
        if (event.key === 'Enter') return inspectPoint(Math.floor(row.x), row.y);
      });
    }
    body.append(entry);
  }
  byId('rows').replaceChildren(body);
  byId('empty-rows').hidden = Boolean(report.rows.length);
}

async function refresh() {
  const sequence = ++current.computing;
  renderHeader();
  const compute = current.saved || json(endpoint('/preview'), 'POST', snapshot());
  const [report] = await Promise.all([compute, paint()]);
  if (sequence === current.computing) renderResult(report);
}

async function refreshLibrary() {
  const records = await json('/moon/panels');
  const holder = byId('library');
  holder.replaceChildren();
  for (const panel of records) {
    const button = document.createElement('button');
    button.textContent = panel.title;
    selected(button, panel.id === current.panel?.id);
    button.onclick = safe(async () => {
      await openPanel(panel.id);
      status('已打开观测。');
    });
    holder.append(button);
  }
  return records;
}

async function activate(panel) {
  current.opening++;
  current.reading++;
  current.panel = panel;
  current.settings = structuredClone(panel.settings);
  current.mode = 'illumination';
  current.view = 'marked';
  current.first = null;
  current.pin = null;
  current.saved = null;
  current.image = null;
  current.proposal = null;
  current.tool = 'point';
  map.width = panel.width;
  map.height = panel.height;
  localStorage.setItem('moondesk-panel', panel.id);
  announceMap(directions.point);
  await Promise.all([refreshLibrary(), refresh()]);
}

async function openPanel(id) {
  const sequence = ++current.opening;
  const panel = await json('/moon/panels/' + id);
  if (sequence === current.opening) await activate(panel);
}

async function openReport(id) {
  const sequence = ++current.opening;
  const report = await json(endpoint('/reports/' + id));
  if (sequence !== current.opening) return;
  current.saved = report;
  current.mode = report.mode;
  current.view = 'marked';
  current.pin = null;
  current.first = null;
  current.tool = 'point';
  current.reading++;
  announceMap('点图面可核对存档使用的源像素。');
  await refresh();
  status('已打开保存的检查。');
}

async function inspectPoint(x, y) {
  const sequence = ++current.reading;
  current.pin = [x, y];
  drawMap();
  const chosen = current.saved ? {report: current.saved.id} : {settings: current.settings};
  const sample = await json(endpoint('/point'), 'POST', {...chosen, point: [x, y]});
  if (sequence !== current.reading) return;
  const description = ['源像素 (' + sample.x + ', ' + sample.y + ')', current.config.lights[sample.light],
    'RGB ' + sample.rgb.join(' / '), '亮度 ' + sample.luma, sample.in_disk ? '月盘内' : '月盘外'];
  announceMap(description.join(' · '));
}

function pointOnMap(event) {
  const bounds = map.getBoundingClientRect();
  const x = (event.clientX - bounds.left - map.clientLeft) * map.width / map.clientWidth;
  const y = (event.clientY - bounds.top - map.clientTop) * map.height / map.clientHeight;
  return [Math.max(0, Math.min(map.width - 1, Math.floor(x))), Math.max(0, Math.min(map.height - 1, Math.floor(y)))];
}

map.onclick = safe(async event => {
  const point = pointOnMap(event);
  if (current.tool === 'point') return inspectPoint(...point);
  if (!current.first) {
    current.first = point;
    drawMap();
    announceMap('已选择第一点。请在图面选择第二点。');
    return;
  }
  const [x0, y0] = current.first;
  const [x1, y1] = point;
  const tool = current.tool;
  const rect = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1) + 1, Math.max(y0, y1) + 1];
  current.settings[tool] = tool === 'line' ? [x0, y0, x1, y1] : rect;
  current.first = null;
  current.pin = null;
  current.tool = 'point';
  announceMap('已更新' + current.config.tools[tool] + '。' + directions.point);
  await refresh();
});

for (const button of document.querySelectorAll('[data-tool]')) {
  button.onclick = safe(() => {
    current.first = null;
    current.pin = null;
    current.tool = button.dataset.tool;
    current.reading++;
    announceMap(directions[current.tool]);
    renderHeader();
    drawMap();
  });
}

for (const button of document.querySelectorAll('[data-light]')) {
  button.onclick = safe(async () => {
    current.settings.light = button.dataset.light;
    current.reading++;
    announceMap(directions[current.tool]);
    await refresh();
  });
}

for (const button of document.querySelectorAll('[data-view]')) {
  button.onclick = safe(async () => {
    current.view = button.dataset.view;
    renderHeader();
    await paint();
  });
}

for (const key of inputKeys) {
  byId(key).onchange = safe(async () => {
    const field = byId(key);
    if (!field.value || !field.checkValidity()) throw Error('请检查设置的数值范围。');
    current.settings[key] = key === 'sun_side' ? field.value : Number(field.value);
    current.reading++;
    announceMap(directions[current.tool]);
    await refresh();
  });
}

byId('demo').onclick = safe(async () => {
  await activate(await json('/moon/demo', 'POST'));
  status('已打开合成练习。');
});

byId('retain-settings').onclick = safe(async () => {
  const panel = await json(endpoint('/settings'), 'PUT', snapshot());
  if (current.panel.id !== panel.id) return;
  current.panel = panel;
  await refresh();
  status('已保留检查设置。');
});

byId('save').onclick = safe(async () => {
  const {panel, report} = await json(endpoint('/reports'), 'POST', snapshot());
  if (current.panel.id !== panel.id) return;
  current.panel = panel;
  await openReport(report.id);
  status('已保存检查。');
});

byId('return-current').onclick = safe(async () => {
  current.saved = null;
  current.pin = null;
  current.reading++;
  announceMap(directions.point);
  await refresh();
  status('已返回当前检查。');
});

const limbDialog = byId('limb-dialog');

function limbArcs(arcs) {
  return arcs.length ? arcs.map(pair => pair[0] + '° ~ ' + pair[1] + '°').join('，') : '—';
}

function renderLimbDialog(proposal) {
  const detail = byId('limb-detail');
  detail.replaceChildren();
  const circle = proposal.circle;
  const lead = document.createElement('p');
  lead.textContent = '建议圆：圆心 (' + circle.cx + ', ' + circle.cy + ') · 半径 ' + circle.radius +
    ' px · 整体偏差 RMS ' + proposal.rms_px + ' px';
  const table = document.createElement('table');
  const head = document.createElement('tr');
  for (const label of ['月缘', '取自', '用到的弧段', '点数', '偏差 RMS', '最大偏差']) {
    const cell = document.createElement('th');
    cell.textContent = label;
    head.append(cell);
  }
  table.append(head);
  for (const [side, label] of [['bright', '亮侧（短曝）'], ['dark', '暗侧（长曝）']]) {
    const report = proposal.sides[side];
    const row = document.createElement('tr');
    const values = [label, current.config.lights[report.light], limbArcs(report.arcs), report.count,
      report.rms_px === null ? '—' : report.rms_px + ' px', report.max_px === null ? '—' : report.max_px + ' px'];
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    table.append(row);
  }
  const comparison = document.createElement('p');
  comparison.className = 'limb-note';
  comparison.textContent = proposal.conflict
    ? '亮暗两侧各自拟合的出入：圆心相差 ' + proposal.conflict.center_shift_px +
      ' px，半径相差 ' + proposal.conflict.radius_gap_px + ' px；建议圆取两侧折中。'
    : '某一侧可用月缘不足，建议圆主要由单侧边缘决定。';
  const convention = document.createElement('p');
  convention.className = 'limb-note';
  convention.textContent = '弧段角度自画面右侧起顺时针计（图像坐标）；图上橙点为亮侧取样、蓝点为暗侧取样，虚线为建议圆。';
  detail.append(lead, table, comparison, convention);
  for (const text of proposal.notes) {
    const extra = document.createElement('p');
    extra.className = 'limb-note';
    extra.textContent = text;
    detail.append(extra);
  }
}

byId('fit-limb').onclick = safe(async () => {
  const proposal = await json(endpoint('/limb'), 'POST', {settings: current.settings});
  current.proposal = proposal;
  renderLimbDialog(proposal);
  byId('limb-error').textContent = '';
  drawMap();
  limbDialog.showModal();
});

byId('close-limb').onclick = () => limbDialog.close();

limbDialog.addEventListener('close', () => {
  current.proposal = null;
  drawMap();
});

byId('adopt-limb').onclick = async () => {
  if (!current.proposal) return;
  const settings = {...current.settings, disk: current.proposal.disk};
  let panel;
  try {
    panel = await json(endpoint('/settings'), 'PUT', {settings, revision: current.panel.revision});
  } catch (error) {
    byId('limb-error').textContent = error.message;
    return;
  }
  if (current.panel.id !== panel.id) return;
  current.panel = panel;
  current.settings = structuredClone(panel.settings);
  current.proposal = null;
  limbDialog.close();
  await refresh();
  status('已采用圆形月缘。可切换到亮度剖线查看进盘和出盘。');
};

const dialog = byId('import-dialog');
byId('open-import').onclick = () => { byId('import-error').textContent = ''; dialog.showModal(); };
byId('close-import').onclick = () => dialog.close();
byId('import-form').onsubmit = async event => {
  event.preventDefault();
  const form = event.currentTarget;
  event.submitter.disabled = true;
  try {
    const panel = await json('/moon/panels', 'POST', new FormData(form));
    dialog.close();
    form.reset();
    await activate(panel);
    status('已导入观测。');
  } catch (error) {
    byId('import-error').textContent = error.message;
  } finally {
    event.submitter.disabled = false;
  }
};

safe(async () => {
  current.config = await json('/moon/config');
  for (const [mode, label] of Object.entries(current.config.modes)) {
    const modeButton = document.createElement('button');
    modeButton.textContent = label;
    modeButton.dataset.mode = mode;
    modeButton.onclick = safe(async () => {
      current.mode = mode;
      await refresh();
    });
    byId('modes').append(modeButton);
  }
  const items = await refreshLibrary();
  const remembered = localStorage.getItem('moondesk-panel');
  if (items.length) await activate(items.find(item => item.id === remembered) || items[0]);
})();
