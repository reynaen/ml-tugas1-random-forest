from flask import Flask, render_template, request
import rf

app = Flask(__name__)
app.jinja_env.filters['koma'] = lambda x: f'{x:.6f}'.replace('.', ',')


@app.route('/', methods=['GET', 'POST'])
def index():
    f = request.form
    ctx = {'csv': f.get('csv', rf.DEF).replace('\r', ''), 'nt': f.get('nt', 3, type=int),
           'm': f.get('m', 2, type=int), 'seed': f.get('seed', 3, type=int), 'mode': f.get('mode', 'lap'),
           'hasil': None, 'error': None}
    try:
        cols, tgt, rows, classes, trees = rf.forest(ctx['csv'], max(1, min(ctx['nt'], 10)), max(1, ctx['m']), ctx['seed'], ctx['mode'])
        if not rows:
            raise ValueError('data kosong')
    except Exception as e:  # input csv tidak valid
        ctx['error'] = f'Format data tidak valid: {e}'
        return render_template('index.html', **ctx)
    opsi = [sorted({r['v'][i] for r in rows}) for i in range(len(cols))]
    default = ['Muda', 'Sedang', 'Ya', 'Sering'] if ctx['csv'] == rf.DEF else [o[0] for o in opsi]
    x = [f.get(f'x{i}', default[i]) for i in range(len(cols))]
    vote, jalur = {}, []
    if f.get('aksi') == 'prediksi':
        for t in trees:
            p, y = rf.predict(t['root'], cols, x)
            jalur.append((p, y))
            if y:
                vote[y] = vote.get(y, 0) + 1
    for t in trees:
        t['svg'] = rf.svg(t['root'])
    ctx['hasil'] = dict(cols=cols, tgt=tgt, rows=rows, classes=classes, trees=trees, opsi=opsi, x=x,
                        jalur=jalur, vote=sorted(vote.items(), key=lambda kv: -kv[1]), prediksi=f.get('aksi') == 'prediksi')
    return render_template('index.html', **ctx)


if __name__ == '__main__':
    app.run(debug=True)
