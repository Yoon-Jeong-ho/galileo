#!/usr/bin/env node
/**
 * Audit LaTeX-style citation keys (\cite{...}, \citet{...}, \citep{...}, etc.)
 * against a BibTeX file.
 *
 * Usage:
 *   node scripts/audit_citations.js \
 *     --bib references.bib \
 *     --paths docs/paper
 *
 * Exit code:
 *   0 if no missing keys; 2 if missing keys; 1 on other errors.
 */

const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const args = {
    bib: 'references.bib',
    paths: ['docs/paper'],
    exclude: [
      // Vendor templates / style bundles often contain placeholder citations.
      'docs/paper/emnlp_template',
      'docs/paper/acl_style_files',
      'docs/paper/latex_skeleton',
      'docs/paper/latex_skeleton_emnlp2023',
    ],
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--bib') args.bib = argv[++i];
    else if (a === '--paths') {
      args.paths = [];
      while (i + 1 < argv.length && !argv[i + 1].startsWith('--')) args.paths.push(argv[++i]);
      if (args.paths.length === 0) throw new Error('Expected at least one path after --paths');
    } else if (a === '--exclude') {
      args.exclude = [];
      while (i + 1 < argv.length && !argv[i + 1].startsWith('--')) args.exclude.push(argv[++i]);
      if (args.exclude.length === 0) throw new Error('Expected at least one path after --exclude');
    } else if (a === '--help' || a === '-h') {
      console.log('Usage: node scripts/audit_citations.js [--bib references.bib] [--paths docs/paper ...] [--exclude <dir> ...]');
      process.exit(0);
    } else {
      throw new Error(`Unknown arg: ${a}`);
    }
  }
  return args;
}

function walk(p) {
  const out = [];
  const st = fs.statSync(p);
  if (st.isFile()) return [p];
  if (!st.isDirectory()) return [];
  const entries = fs.readdirSync(p, { withFileTypes: true });
  for (const e of entries) {
    const child = path.join(p, e.name);
    if (e.isDirectory()) out.push(...walk(child));
    else if (e.isFile()) out.push(child);
  }
  return out;
}

function stripMarkdownCode(md) {
  // Remove fenced code blocks ```...``` (including optional language tag).
  md = md.replace(/```[\s\S]*?```/g, '');
  // Remove inline code spans `...`.
  md = md.replace(/`[^`]*`/g, '');
  return md;
}

function extractCiteKeys(text) {
  // Covers: \cite{..}, \citep{..}, \citet{..}, \citealp{..}, \citeyearpar{..}, \citeposs{..}, etc.
  // Also supports optional natbib [] pre/post notes.
  const re = /\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)?(?:\[[^\]]*\]\s*)?\{([^}]+)\}/g;
  const keys = new Set();
  let m;
  while ((m = re.exec(text))) {
    m[1]
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      // Ignore common placeholder snippets in drafts/docs like \cite{...}
      .filter((k) => k !== '...')
      // Be conservative: keep only plausible BibTeX keys.
      .filter((k) => /^[A-Za-z0-9:._-]+$/.test(k))
      .forEach((k) => keys.add(k));
  }
  return keys;
}

function extractBibKeys(bibText) {
  const re = /@\w+\s*\{\s*([^,\s]+)\s*,/g;
  const keys = new Set();
  let m;
  while ((m = re.exec(bibText))) keys.add(m[1].trim());
  return keys;
}

function main() {
  const args = parseArgs(process.argv);

  const bibPath = args.bib;
  if (!fs.existsSync(bibPath)) throw new Error(`BibTeX file not found: ${bibPath}`);
  const bibKeys = extractBibKeys(fs.readFileSync(bibPath, 'utf8'));

  const isExcluded = (f) => {
    const norm = f.split(path.sep).join('/');
    return args.exclude.some((ex) => norm === ex || norm.startsWith(ex + '/'));
  };

  const files = args.paths
    .flatMap((p) => walk(p))
    .filter((f) => (f.endsWith('.tex') || f.endsWith('.md')) && !isExcluded(f));
  const cited = new Set();
  const perFile = [];

  for (const f of files) {
    let text = fs.readFileSync(f, 'utf8');
    if (f.endsWith('.md')) text = stripMarkdownCode(text);
    const keys = extractCiteKeys(text);
    for (const k of keys) cited.add(k);
    if (keys.size) perFile.push({ file: f, keys: [...keys].sort() });
  }

  const missing = [...cited].filter((k) => !bibKeys.has(k)).sort();

  console.log(`# Citation audit`);
  console.log(`BibTeX: ${bibPath}`);
  console.log(`Scanned: ${files.length} files under: ${args.paths.join(', ')}`);
  console.log(`Excluded: ${args.exclude.join(', ')}`);
  console.log(`Total cited keys: ${cited.size}`);
  console.log(`BibTeX keys: ${bibKeys.size}`);
  console.log('');

  if (missing.length === 0) {
    console.log('OK: no missing citation keys.');
    process.exit(0);
  }

  console.log(`MISSING (${missing.length}):`);
  for (const k of missing) console.log(`  - ${k}`);
  console.log('');
  console.log('Tip: add missing entries to references.bib or fix typos in \\cite{...}.');

  process.exit(2);
}

try {
  main();
} catch (err) {
  console.error(`ERROR: ${err.message}`);
  process.exit(1);
}
