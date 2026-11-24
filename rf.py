"""Logika Random Forest (Entropy, Gain, Split Info, Gain Ratio) untuk simulasi manual."""
import math
import random
from collections import Counter
from html import escape

DEF = """Usia,Pendapatan,Diskon,Frekuensi,Beli Produk
Muda,Rendah,Ya,Sering,Ya
Muda,Rendah,Tidak,Jarang,Tidak
Muda,Sedang,Ya,Jarang,Ya
Muda,Tinggi,Tidak,Sering,Tidak
Dewasa,Rendah,Tidak,Sering,Ya
Dewasa,Sedang,Tidak,Jarang,Ya
Dewasa,Tinggi,Ya,Jarang,Ya
Dewasa,Tinggi,Tidak,Sering,Ya
Senior,Sedang,Tidak,Jarang,Tidak
Senior,Rendah,Ya,Jarang,Ya
Senior,Sedang,Ya,Sering,Ya
Senior,Tinggi,Tidak,Jarang,Tidak
Muda,Sedang,Tidak,Sering,Tidak
Dewasa,Sedang,Ya,Sering,Ya"""

# bootstrap dan pemilihan kolom acak yang dipakai pada laporan
BOOT = [[4, 10, 9, 3, 6, 10, 8, 11, 10, 2, 10, 1, 14, 8],
        [12, 8, 9, 14, 9, 8, 7, 11, 14, 3, 4, 11, 3, 14],
        [3, 13, 10, 1, 5, 13, 1, 14, 14, 5, 8, 10, 12, 7]]
PRE = {(0, ''): [2, 3], (0, 'Diskon=Tidak'): [0, 3],
       (1, ''): [3, 2], (1, 'Diskon=Tidak'): [0, 3],
       (2, ''): [3, 1], (2, 'Pendapatan=Sedang'): [3, 2], (2, 'Pendapatan=Tinggi'): [0, 2]}


def parse(text):
    lines = [[c.strip() for c in l.split(',')] for l in text.strip().splitlines() if l.strip()]
    head = lines[0]
    rows = [{'no': i + 1, 'v': r[:-1], 'y': r[-1]} for i, r in enumerate(lines[1:]) if len(r) == len(head)]
    return head[:-1], head[-1], rows


def cnt(rs):
    return Counter(r['y'] for r in rs)


def ent(rs):
    n = len(rs)
    return -sum(k / n * math.log2(k / n) for k in cnt(rs).values())


def eval_attr(rs, a):
    n, g, si, det = len(rs), ent(rs), 0.0, []
    for v in sorted({r['v'][a] for r in rs}):
        s = [r for r in rs if r['v'][a] == v]
        p = len(s) / n
        g -= p * ent(s)
        si -= p * math.log2(p)
        det.append({'v': v, 'n': len(s), 'c': cnt(s), 'e': ent(s), 'sp': -p * math.log2(p)})
    return {'a': a, 'g': g, 'si': si, 'gr': g / si if si > 0 else 0.0, 'd': det}


def build(rs, feats, cols, path, t, rng, m, log, rep):
    c = cnt(rs)
    if len(c) == 1 or not feats:
        return {'leaf': True, 'label': c.most_common(1)[0][0], 'hl': False}
    pick = PRE.get((t, path)) if rep else None
    pick = pick or rng.sample(feats, min(m, len(feats)))
    ev = [eval_attr(rs, a) for a in pick]
    best = max(ev, key=lambda x: x['gr'])  # max() memilih yang pertama saat seri
    log.append({'path': path, 'n': len(rs), 'c': c, 'e': ent(rs), 'ev': ev, 'best': best['a']})
    ch = []
    for d in best['d']:
        sub = [r for r in rs if r['v'][best['a']] == d['v']]
        p = (path + ', ' if path else '') + f"{cols[best['a']]}={d['v']}"
        ch.append((d['v'], build(sub, [x for x in feats if x != best['a']], cols, p, t, rng, m, log, rep)))
    return {'leaf': False, 'label': cols[best['a']], 'ch': ch, 'hl': False}


def forest(text, n_trees, m, seed, mode):
    cols, tgt, rows = parse(text)
    classes = sorted({r['y'] for r in rows})
    rng = random.Random(seed)
    rep = mode == 'lap' and text.strip().replace('\r', '') == DEF
    out = []
    for t in range(n_trees):
        use = rep and t < len(BOOT)
        idx = BOOT[t] if use else [rng.randint(1, len(rows)) for _ in rows]
        rs = [rows[i - 1] for i in idx]
        log = []
        root = build(rs, list(range(len(cols))), cols, '', t, rng, m, log, use)
        out.append({'idx': idx, 'rows': rs, 'log': log, 'root': root})
    return cols, tgt, rows, classes, out


def predict(root, cols, x):
    """Telusuri pohon, tandai jalur (hl), kembalikan (jalur, hasil)."""
    def clear(n):
        n['hl'] = False
        for _, k in n.get('ch', []):
            clear(k)
    clear(root)
    n, path = root, []
    while not n['leaf']:
        a = cols.index(n['label'])
        nxt = next((k for e, k in n['ch'] if e == x[a]), None)
        if nxt is None:
            return path, None
        path.append(f"{n['label']} = {x[a]}")
        nxt['hl'] = True
        n = nxt
    return path, n['label']


def svg(root):
    c, d = [0], [0]

    def lay(n, k):
        n['y'] = k
        d[0] = max(d[0], k)
        if n['leaf']:
            n['x'] = c[0]
            c[0] += 1
            return
        for _, x in n['ch']:
            lay(x, k + 1)
        n['x'] = sum(x['x'] for _, x in n['ch']) / len(n['ch'])
    lay(root, 0)
    sx, sy = 130, 85
    W, H = max(c[0], 1) * sx, (d[0] + 1) * sy
    ln, nd = [], []

    def rec(n):
        cx, ty = n['x'] * sx + sx / 2, n['y'] * sy
        lab = escape(n['label'])
        if n['leaf']:
            nd.append(f'<rect x="{cx-34}" y="{ty+6}" width="68" height="28" rx="8" fill="#92D050"/>'
                      f'<text x="{cx}" y="{ty+25}" text-anchor="middle" font-size="13" font-weight="700" fill="#10300a">{lab}</text>')
            return
        nd.append(f'<ellipse cx="{cx}" cy="{ty+20}" rx="52" ry="19" fill="#92D050"/>'
                  f'<text x="{cx}" y="{ty+25}" text-anchor="middle" font-size="13" font-weight="700" fill="#10300a">{lab}</text>')
        for e, x in n['ch']:
            x2, y2 = x['x'] * sx + sx / 2, x['y'] * sy + (6 if x['leaf'] else 1)
            col, w = ('#f57c00', 4) if x['hl'] else ('#7cb342', 2)
            ln.append(f'<line x1="{cx}" y1="{ty+39}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}"/>'
                      f'<text x="{(cx+x2)/2}" y="{(ty+39+y2)/2+4}" text-anchor="middle" font-size="12" class="el">{escape(e)}</text>')
            rec(x)
    rec(root)
    return f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="Pohon keputusan">{"".join(ln)}{"".join(nd)}</svg>'
