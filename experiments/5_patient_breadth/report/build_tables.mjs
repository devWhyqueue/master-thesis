// Run from this directory: node build_tables.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const read = p => JSON.parse(readFileSync(p, 'utf8').replace(/^\uFEFF/, ''));
for (const [file, hash] of Object.entries(read('assets/provenance.json').sha256)) {
  assert.equal(createHash('sha256').update(readFileSync(`assets/${file}`)).digest('hex'), hash);
}
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = xs => Math.sqrt(mean(xs.map(x => (x - mean(xs)) ** 2)));
const f = x => x.toFixed(2);
const estimate = x => `${f(x.point)} [${f(x.ci_2_5)}, ${f(x.ci_97_5)}]`;
const table = (spec, header, rows) => `\\begin{tabular}{@{}${spec}@{}}\n\\toprule\n${header} \\\\\n\\midrule\n${rows.join(' \\\\\n')} \\\\\n\\bottomrule\n\\end{tabular}\n`;
const cohorts = ['bracs', 'tcga_ut'].map(id => {
  const base = `assets/${id}`;
  const bytes = readFileSync(`${base}/preflight.json`);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), readFileSync(`${base}/preflight.json.sha256`, 'utf8').trim());
  const preflight = read(`${base}/preflight.json`);
  const analysis = read(`${base}/analysis.json`);
  const runs = read(`${base}/run_summary.json`);
  assert.equal(preflight.status, 'pass');
  assert.equal(runs.length, 135);
  assert.equal(new Set(runs.map(r => [r.split, r.g, r.m, r.draw].join(':'))).size, 135);
  assert(runs.every(r => r.converged));
  assert(runs.every(r => [0,1,2].includes(r.split) && [0,1,2,3,4].includes(r.draw)));
  for (const classes of Object.values(preflight.counts_audit)) {
    for (const depths of Object.values(classes)) {
      assert.equal(new Set(Object.values(depths)).size, 1);
      assert(depths['32'] >= 20);
    }
  }
  for (const g of [5, 10, 20]) for (const m of [8, 16, 32]) {
    const cell = runs.filter(r => r.g === g && r.m === m);
    assert.equal(cell.length, 15);
    assert(Math.abs(mean(cell.map(r => r.accuracy)) - analysis.cell_accuracies[`G${g}_m${m}`].point) < 1e-9);
  }
  const point = (g, m) => analysis.cell_accuracies[`G${g}_m${m}`].point;
  assert(Math.abs(point(20, 8) - point(5, 32) - analysis.contrasts.equal_budget_advantage_X.point) < 1e-9);
  return { id, name: id === 'bracs' ? 'BRACS' : 'TCGA-UT', preflight, analysis, runs };
});
for (const c of cohorts) {
  const a = c.analysis;
  writeFileSync(`tables/${c.id}/grid_accuracies.tex`, table('lrrr', '$G$ & $m=8$ & $m=16$ & $m=32$', [5, 10, 20].map(g => `${g} & ` + [8, 16, 32].map(m => estimate(a.cell_accuracies[`G${g}_m${m}`])).join(' & '))));
  writeFileSync(`tables/${c.id}/contrasts.tex`, table('lr', 'Contrast & Estimate [95\\% interval]', [
    ...[5, 10, 20].map(g => `$\\Delta_m(${g})$ & ${estimate(a.contrasts.delta_m[g])}`),
    ...[8, 16, 32].map(m => `$\\Delta_G(${m})$ & ${estimate(a.contrasts.delta_g[m])}`),
    `$X$ & ${estimate(a.contrasts.equal_budget_advantage_X)}`,
  ]));
  const p = a.surface_parameters;
  writeFileSync(`tables/${c.id}/surface_fits.tex`, table('lrr', 'Predictor & Slope [95\\% interval] & Residual SD', [
    `$\\log n$ & ${estimate(p.beta_n)} & ${f(p.res_std_n.point)}`,
    `$\\log G$ & ${estimate(p.beta_g)} & ${f(p.res_std_g.point)}`,
    `$\\log\\Neff$ & ${estimate(p.beta_neff)} & ${f(p.res_std_neff.point)}`,
  ]));
  const names = {N:'Normal',PB:'Pathological benign',UDH:'Usual ductal hyperplasia',FEA:'Flat epithelial atypia',ADH:'Atypical ductal hyperplasia',DCIS:'Ductal carcinoma in situ',IC:'Invasive carcinoma'};
  writeFileSync(`tables/${c.id}/class_iccs.tex`, table('p{9cm}rrr', 'Class & $\\ICC_c$ & $\\DE_c(8)$ & $\\DE_c(32)$', Object.entries(c.preflight.cohort_iccs).map(([name,v]) => `${names[name] ?? name.replaceAll('_',' ')} & ${v.toFixed(3)} & ${f(1+7*v)} & ${f(1+31*v)}`)));
  const s = c.analysis.secondary;
  const secondaryKeys = ['macro_nll', 'expected_calibration_error', 'patch_micro_balanced_accuracy'];
  for (const k of secondaryKeys) assert(Math.abs(s.cells['G20_m8'][k].point - s.cells['G5_m32'][k].point - s.equal_budget_X[k].point) < 1e-9);
  writeFileSync(`tables/${c.id}/secondary.tex`, table('lrrr', '$(G, m)$ & Macro NLL (nats) & ECE (\\%) & Patch-micro BA (\\%)', [
    ...[5, 10, 20].flatMap(g => [8, 16, 32].map(m => `(${g}, ${m}) & ` + secondaryKeys.map(k => estimate(s.cells[`G${g}_m${m}`][k])).join(' & '))),
    `$X$ & ` + secondaryKeys.map(k => estimate(s.equal_budget_X[k])).join(' & '),
  ]));
  writeFileSync(`tables/${c.id}/class_recalls.tex`, table('p{6cm}rrr', 'Class & $(5, 32)$ & $(20, 8)$ & $X$', Object.entries(s.class_recalls).map(([name, e]) => `${names[name] ?? name.replaceAll('_',' ')} & ${estimate(e.G5_m32)} & ${estimate(e.G20_m8)} & ${estimate(e.equal_budget_X)}`)));
}
writeFileSync('tables/residual_breadth.tex', table('llrr', 'Dataset & Adjustment & $\\gamma$ [95\\% interval] & Residual SD', cohorts.flatMap(c => ['n','e'].map(k => `${c.name} & ${k === 'n' ? 'Nominal' : 'Effective'} & ${estimate(c.analysis.surface_parameters[`gamma_${k}`])} & ${f(c.analysis.surface_parameters[k === 'n' ? 'res_std_aug_nom' : 'res_std_aug_eff'].point)}`))));
writeFileSync('tables/split_results.tex', table('lrrrr', 'Dataset / split & $X$ & $\\Delta_m(20)$ & $\\Delta_G(8)$ & Draw SD range', cohorts.flatMap(c => [0,1,2].map(s => {
  const points = (g,m) => c.runs.filter(r => r.split === s && r.g === g && r.m === m).map(r => r.accuracy);
  const dispersions = [5,10,20].flatMap(g => [8,16,32].map(m => sd(points(g,m))));
  return `${c.name} / ${s} & ${f(mean(points(20,8))-mean(points(5,32)))} & ${f(mean(points(20,32))-mean(points(20,8)))} & ${f(mean(points(20,8))-mean(points(5,8)))} & ${f(Math.min(...dispersions))}--${f(Math.max(...dispersions))}`;
}))));
console.log('Verified signed preflights, 270 converged records, and all 18 pooled cell estimates. Tables written.');
for (const c of cohorts) console.log(c.name, 'pool range', Math.min(...Object.values(c.preflight.counts_audit).flatMap(x=>Object.values(x).map(d=>d['32']))), Math.max(...Object.values(c.preflight.counts_audit).flatMap(x=>Object.values(x).map(d=>d['32']))), 'mean ICC', mean(Object.values(c.preflight.cohort_iccs)));
