import pandas as pd
import streamlit as st
from src.financial_calculator import DP_OPTIONS, TENORS
from src.prediction import load_assets, ROOT
from src.workflow import Request, simulate, result_checks, recommend_for, export_result

st.set_page_config(page_title='AutoFinance Lab | Simulasi Mobil', page_icon='🚘', layout='wide')

@st.cache_resource
def assets():
    try:
        models, meta, catalog = load_assets()
        return models, meta, catalog, None
    except Exception as exc:
        catalog = pd.read_csv(ROOT / 'data/katalog_mobil.csv')
        return {}, {}, catalog, f'{type(exc).__name__}: {exc}'

@st.cache_data
def report(name):
    return pd.read_csv(ROOT / 'reports' / name)


def rp(n, decimals=0):
    return 'Rp ' + f'{n:,.{decimals}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def pct(n):
    return f'{n:.2%}'.replace('.', ',')


def money_col(label):
    return st.column_config.NumberColumn(label, format='Rp %.0f')


def table(df, config=None, height=None):
    kwargs = {'hide_index': True, 'use_container_width': True, 'column_config': config or {}}
    if height is not None:
        kwargs['height'] = height
    st.dataframe(df, **kwargs)


def section(num, title, caption):
    st.markdown(f'#### {num} · {title}')
    st.caption(caption)


def snapshot(req):
    st.caption(f'{req.company} {req.model}  |  {rp(req.price)}  |  DP {req.dp}%  |  {req.tenure} bulan  |  ' + ('Tanpa bunga' if req.mode == 'dataset' else f'Bunga flat {req.rate:g}%/tahun'))
    st.caption(f'Penghasilan {rp(req.income)} · Cicilan lain {rp(req.other)} · Anggaran DP {rp(req.budget)}')


def reset():
    for key in list(st.session_state):
        if key.startswith(('form_', 'rec_')) or key in ('simulation', 'sim_error', 'run_count'):
            del st.session_state[key]


def draw_figures(options, key):
    label = st.selectbox('Pilih grafik', list(options), key=key)
    path = ROOT / 'reports/figures' / options[label]
    if path.is_file():
        st.image(str(path), use_container_width=True)
    else:
        st.warning('Grafik tidak tersedia di repository.')


try:
    models, metadata, catalog, asset_error = assets()
except Exception as exc:
    st.error('Katalog tidak dapat dimuat. Periksa data/katalog_mobil.csv.')
    with st.expander('Detail teknis'):
        st.code(f'{type(exc).__name__}: {exc}')
    st.stop()

catalog = catalog.sort_values(['company', 'model']).reset_index(drop=True)
car_ids = list(catalog.index)
example = catalog.index[(catalog.company == 'Daihatsu') & (catalog.model == 'Ayla')].tolist()
initial = example[0] if example else 0

st.markdown('''<style>
.block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: clamp(1.05rem, 2.3vw, 1.65rem);}
[data-testid="stMetricLabel"] {white-space: normal;}
[data-testid="stSidebar"] {border-right: 1px solid #e2ebe8;}
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### AutoFinance Lab')
    st.caption('DATA SCIENCE PROJECT / V4')
    st.markdown('**Cara menggunakan**')
    st.markdown('1. Isi kondisi finansial dan pilihan kendaraan.\n2. Tekan **Jalankan simulasi**.\n3. Baca hasil dan periksa proses algoritma.\n4. Cari alternatif mobil sesuai anggaran.')
    st.divider()
    st.markdown('**Status sistem**')
    if asset_error:
        st.warning('Model ML tidak tersedia. Kalkulator tetap dapat digunakan.')
        with st.expander('Detail model'):
            st.code(asset_error)
    else:
        st.success('Model dan katalog siap')
    st.caption(f'{len(catalog)} kendaraan dalam katalog penelitian')
    st.button('Reset seluruh simulasi', on_click=reset, use_container_width=True)
    st.divider()
    st.caption('Simulasi edukatif, bukan persetujuan kredit atau harga pasar terkini. Tidak menyimpan input ke database.')

st.caption('AUTOFINANCE LAB  /  SIMULASI PEMBIAYAAN')
st.title('Rencanakan pembelian mobilmu')
st.write('Hitung cicilan, pahami kemampuan bayar, dan telusuri bagaimana model menghasilkan prediksinya.')

sim_tab, rec_tab, trace_tab, study_tab = st.tabs(['01 Simulasi', '02 Rekomendasi', '03 Proses & diagnostik', '04 Hasil penelitian'])

with sim_tab:
    left, right = st.columns([1, 1.15], gap='large')
    with left:
        with st.container(border=True):
            st.subheader('Pengaturan simulasi')
            with st.form('simulation_form', clear_on_submit=False):
                section('1', 'Keuangan', 'Gunakan penghasilan bulanan dan kewajiban cicilan aktual.')
                income = st.number_input('Penghasilan per bulan (Rp)', 1_000_000, 500_000_000, 15_000_000, 500_000, key='form_income')
                other = st.number_input('Cicilan lain per bulan (Rp)', 0, 500_000_000, 0, 500_000, key='form_other')
                budget = st.number_input('Anggaran DP tersedia (Rp)', 0, 970_000_000, 100_000_000, 5_000_000, key='form_budget')
                st.divider()
                section('2', 'Kendaraan', 'Harga dan tipe kendaraan berasal dari dataset penelitian.')
                choice = st.selectbox('Pilih mobil', car_ids, index=initial, format_func=lambda i: f'{catalog.loc[i, "company"]} {catalog.loc[i, "model"]} | {rp(catalog.loc[i, "price_idr"])}', key='form_car')
                st.divider()
                section('3', 'Pembiayaan', 'Pilih DP, tenor, serta jenis perhitungan.')
                c1, c2 = st.columns(2)
                dp = c1.selectbox('DP (%)', DP_OPTIONS, index=1, key='form_dp')
                tenure = c2.selectbox('Tenor (bulan)', TENORS, index=3, key='form_tenure')
                mode = st.radio('Mode cicilan', ['dataset', 'flat'], format_func=lambda x: 'Tanpa bunga (dataset)' if x == 'dataset' else 'Bunga flat (asumsi)', key='form_mode')
                rate = st.number_input('Bunga flat per tahun (%)', 0.0, 30.0, 5.0, 0.5, key='form_rate', help='Diabaikan jika memilih mode tanpa bunga.')
                st.caption('Input baru diterapkan hanya saat tombol simulasi ditekan.')
                submitted = st.form_submit_button('Jalankan simulasi', type='primary', use_container_width=True)
            if submitted:
                st.session_state.pop('simulation', None)
                st.session_state.pop('rec_result', None)
                st.session_state.pop('rec_error', None)
                st.session_state.pop('sim_error', None)
                car = catalog.loc[choice]
                req = Request(str(car.company), str(car.model), str(car.body_style), int(car.price_idr), int(income), int(other), int(budget), int(dp), int(tenure), mode, float(rate) if mode == 'flat' else 0.0)
                try:
                    with st.spinner('Memvalidasi input, menghitung cicilan dan menjalankan model...'):
                        st.session_state.simulation = simulate(req, models, metadata, asset_error)
                    st.session_state.run_count = st.session_state.get('run_count', 0) + 1
                except Exception as exc:
                    st.session_state.sim_error = f'{type(exc).__name__}: {exc}'
    with right:
        st.subheader('Ringkasan hasil')
        result = st.session_state.get('simulation')
        if st.session_state.get('sim_error'):
            st.error('Simulasi gagal. Periksa input dan jalankan kembali.')
            st.code(st.session_state.sim_error)
        if result is None:
            with st.container(border=True):
                st.markdown('### Belum ada simulasi')
                st.write('Lengkapi formulir di sebelah kiri, lalu tekan **Jalankan simulasi**. Hasil dan proses algoritma akan muncul di sini.')
                st.caption('Batas penilaian penelitian: rasio seluruh cicilan maksimal 35% penghasilan dan uang muka tidak melebihi anggaran DP.')
        else:
            req, f = result['request'], result['finance']
            st.caption(f"SIMULASI #{st.session_state.run_count} · Hasil terakhir yang dikirim")
            st.markdown(f'### {req.company} {req.model}')
            snapshot(req)
            e = result['eligibility']
            with st.container(border=True):
                st.caption('KEPUTUSAN KEMAMPUAN BAYAR')
                if e['eligible']:
                    st.success('✅ LAYAK DALAM SIMULASI')
                    st.write('Kedua syarat terpenuhi: seluruh cicilan tidak melebihi 35% penghasilan dan uang muka masih dalam anggaran.')
                else:
                    st.error('❌ TIDAK LAYAK DALAM SIMULASI')
                    failed = []
                    if not e['debt_ok']:
                        failed.append(f'total cicilan melampaui batas sebesar {rp(-e["debt_gap"])} per bulan')
                    if not e['dp_ok']:
                        failed.append(f'anggaran DP kurang {rp(-e["dp_gap"])}')
                    st.write('Belum memenuhi syarat karena ' + ' dan '.join(failed) + '.')
                st.markdown('#### Mengapa hasilnya demikian?')
                a, b = st.columns(2)
                a.metric('Rasio seluruh cicilan', pct(f.ratio), help='Cicilan mobil dan cicilan lain dibagi penghasilan.')
                b.metric('Batas rasio penelitian', '35%')
                if e['debt_ok']:
                    st.success(f'✓ Rasio memenuhi batas. Total cicilan {rp(e["total_debt"])} dari batas {rp(e["debt_limit"])} per bulan. Sisa ruang cicilan {rp(e["debt_gap"])}.')
                else:
                    st.error(f'✗ Rasio melebihi batas. Total cicilan {rp(e["total_debt"])} dari batas {rp(e["debt_limit"])} per bulan. Kelebihan {rp(-e["debt_gap"])}.')
                if e['dp_ok']:
                    st.success(f'✓ DP sesuai anggaran. Dibutuhkan {rp(f.down_payment)} dari {rp(req.budget)} yang tersedia. Sisa anggaran {rp(e["dp_gap"])}.')
                else:
                    st.error(f'✗ DP melebihi anggaran. Dibutuhkan {rp(f.down_payment)} dari {rp(req.budget)} yang tersedia. Kekurangan {rp(-e["dp_gap"])}.')
                if abs(f.ratio - .35) < .00005:
                    st.caption(f'Rasio sangat dekat batas (nilai sebelum pembulatan: {f.ratio*100:.6f}%). Keputusan memakai nilai penuh.')
                st.caption('Kedua syarat wajib terpenuhi. Ini penilaian berbasis asumsi penelitian, bukan persetujuan kredit resmi.')
            with st.container(border=True):
                st.caption('ESTIMASI CICILAN MOBIL / BULAN')
                st.markdown(f'## {rp(f.installment)}')
                a, b = st.columns(2)
                a.metric('Total cicilan per bulan', rp(e['total_debt']))
                b.metric('Batas cicilan mobil', rp(f.installment_limit))
                a, b = st.columns(2)
                a.metric('Uang muka', rp(f.down_payment))
                b.metric('Sisa penghasilan', rp(f.remaining_income))
            with st.container(border=True):
                st.markdown('#### Rincian pembiayaan')
                breakdown = pd.DataFrame({'Komponen': ['Harga katalog', 'Uang muka', 'Pokok pembiayaan', 'Estimasi bunga total', 'Total pembayaran mobil', 'Cicilan lain per bulan'], 'Nominal (Rp)': [f.price, f.down_payment, f.principal, f.total_payment-f.price, f.total_payment, req.other]})
                table(breakdown, {'Nominal (Rp)': money_col('Nominal')})
                if f.remaining_income >= 0:
                    chart = pd.DataFrame({'Alokasi': ['Cicilan mobil', 'Cicilan lain', 'Sisa penghasilan'], 'Rp per bulan': [f.installment, req.other, f.remaining_income]})
                    st.bar_chart(chart, x='Alokasi', y='Rp per bulan', horizontal=True)
                else:
                    st.warning(f'Kewajiban melebihi pendapatan sebesar {rp(-f.remaining_income)} per bulan.')
            with st.container(border=True):
                st.markdown('#### Langkah selanjutnya')
                if e['eligible']:
                    st.write('Skema ini memenuhi dua syarat simulasi. Sebelum membeli, periksa biaya hidup, dana darurat, asuransi, dan penawaran pembiayaan aktual.')
                else:
                    if not e['debt_ok']:
                        st.write(f'**Batas cicilan mobil:** {rp(f.installment_limit)} per bulan. Untuk mempertahankan skema ini, penghasilan minimal menurut aturan 35% adalah {rp(e["min_income"])} per bulan (selisih {rp(e["income_shortfall"])}).')
                        if e['other_exhausts_limit']:
                            st.warning('Cicilan lain sudah menghabiskan atau melebihi batas 35%; saat ini tidak ada ruang cicilan mobil yang positif dalam batas tersebut.')
                        else:
                            st.write(f'Kurangi total cicilan sebesar setidaknya {rp(-e["debt_gap"])} per bulan, misalnya dengan memilih skema atau mobil yang lebih terjangkau.')
                    if not e['dp_ok']:
                        st.write(f'**Kekurangan anggaran DP:** {rp(-e["dp_gap"])}. Pertimbangkan kendaraan lebih murah atau DP yang sesuai dana tersedia, selama rasio cicilan juga memenuhi batas.')
                    st.markdown('**Alternatif skema untuk mobil yang sama**')
                    if e['alternatives']:
                        st.caption(f'{len(e["alternatives"])} skema lain memenuhi kedua syarat. Tiga skema terdekat dari pilihan awal ditampilkan; bunga, pendapatan dan anggaran tetap sama.')
                        options = pd.DataFrame(e['alternatives'][:3])
                        options['ratio'] *= 100
                        table(options, {'dp_percent': st.column_config.NumberColumn('DP', format='%d%%'), 'tenure': 'Tenor (bln)', 'down_payment': money_col('Uang muka'), 'installment': money_col('Cicilan/bln'), 'ratio': st.column_config.NumberColumn('Rasio', format='%.2f%%'), 'total_payment': money_col('Total bayar')})
                        st.caption('Tenor lebih panjang dapat meningkatkan total biaya dalam mode bunga. Gunakan alternatif hanya jika sesuai kebutuhan.')
                    else:
                        st.info('Tidak ada kombinasi DP 20–40% dan tenor 12–60 bulan untuk mobil ini yang memenuhi kedua syarat pada kondisi saat ini.')
                    st.write('Buka tab **02 Rekomendasi**, lalu tekan **Cari rekomendasi** untuk mencari mobil alternatif.')
            with st.container(border=True):
                st.markdown('#### Prediksi ML (mode dataset)')
                a,b = st.columns(2)
                reg = result['models'].get('reg')
                clf = result['models'].get('clf')
                a.metric('Linear Regression', rp(reg['prediction'], 2) if reg else 'Tidak tersedia')
                b.metric('Decision Tree', ('Sesuai' if clf['prediction'] else 'Melebihi') if clf else 'Tidak tersedia')
                st.caption('Prediksi ML mengikuti definisi label dataset: tanpa bunga, cicilan lain, atau anggaran DP. Hasil ML tidak menentukan status akhir simulasi.')
                if clf and bool(clf['prediction']) != e['eligible']:
                    st.info('Status ML dan keputusan simulasi berbeda. Periksa definisi label dataset dan syarat finansial pada tab Proses & diagnostik; status akhir tetap memakai kedua syarat simulasi.')
            if result['errors'] or result['trace_errors']:
                st.warning('Sebagian proses ML/interpretasi gagal. Kalkulasi finansial tetap tersedia. Lihat tab Proses & diagnostik.')
            elif result['warnings']:
                st.warning('Sebagian input di luar rentang data latih. Lihat tab Proses & diagnostik.')
            st.download_button('Unduh ringkasan simulasi (CSV)', export_result(result), 'hasil_simulasi.csv', 'text/csv', use_container_width=True)
            st.caption('Hasil belum termasuk biaya hidup, administrasi, asuransi, pajak, dan perawatan. Mode dataset membulatkan cicilan ke bawah; sisa pembulatan pokok diselesaikan pada pembayaran terakhir. Untuk bunga flat, total mengikuti cicilan bulanan yang dibulatkan.')

with rec_tab:
    st.subheader('Temukan alternatif mobil')
    result = st.session_state.get('simulation')
    if result is None:
        st.info('Jalankan simulasi pertama untuk menggunakan rekomendasi.')
    else:
        req = result['request']
        snapshot(req)
        st.caption('Rekomendasi menggunakan kondisi finansial dari hasil simulasi terakhir, bukan perubahan formulir yang belum dikirim.')
        with st.form('recommendation_form'):
            a, b = st.columns(2)
            ranking = a.selectbox('Prioritas', ['Cicilan terendah', 'DP terendah', 'Tenor terpendek', 'Total pembayaran terendah', 'Harga mobil tertinggi'], key='rec_ranking')
            body = b.selectbox('Tipe bodi', ['Semua', *sorted(catalog.body_style.unique())], key='rec_body')
            explore = st.checkbox('Cari kombinasi DP dan tenor alternatif', value=True, key='rec_explore')
            search = st.form_submit_button('Cari rekomendasi', type='primary', use_container_width=True)
        if search:
            st.session_state.pop('rec_result', None)
            st.session_state.pop('rec_error', None)
            try:
                with st.spinner('Menyaring katalog dan mengurutkan alternatif...'):
                    rows, stats = recommend_for(req, catalog, explore, ranking, body)
                st.session_state.rec_result = (rows, stats, ranking, body, explore)
            except Exception as exc:
                st.session_state.rec_error = f'{type(exc).__name__}: {exc}'
        if st.session_state.get('rec_error'):
            st.error('Pencarian rekomendasi gagal.')
            st.code(st.session_state.rec_error)
        if 'rec_result' in st.session_state:
            rows, stats, applied_rank, applied_body, applied_explore = st.session_state.rec_result
            st.caption(f'Hasil pencarian terakhir: {applied_rank} | {applied_body} | ' + ('DP/tenor alternatif' if applied_explore else 'Skema simulasi'))
            a,b,c,d = st.columns(4)
            a.metric('Kombinasi diuji', stats['combinations'])
            b.metric('Lolos kedua batas', stats['both_pass'])
            c.metric('Mobil unik lolos', stats['unique_cars'])
            d.metric('Ditampilkan', stats['displayed'])
            if rows.empty:
                st.warning('Tidak ada alternatif yang memenuhi kedua batas. Periksa anggaran DP, rasio, atau filter tipe bodi. Jangan menaikkan anggaran jika tidak sesuai kemampuan.')
            else:
                st.markdown('#### Pilihan terbaik berdasarkan prioritas')
                best = rows.iloc[0]
                with st.container(border=True):
                    st.caption('PERINGKAT 1 · ' + applied_rank.upper())
                    st.markdown(f'### {best.company} {best.model}')
                    a,b,c = st.columns(3)
                    a.metric('Cicilan per bulan', rp(best.installment))
                    b.metric('Uang muka', rp(best.down_payment))
                    c.metric('Tenor', f'{best.tenure} bulan')
                view = rows.copy()
                view.insert(0, 'rank', range(1, len(view)+1))
                view['ratio_pct'] = view.pop('ratio') * 100
                table(view, {'rank':'#', 'company':'Merek', 'model':'Model', 'body_style':'Tipe', 'price':money_col('Harga'), 'dp_percent':st.column_config.NumberColumn('DP', format='%d%%'), 'down_payment':money_col('Uang muka'), 'tenure':'Tenor (bln)', 'installment':money_col('Cicilan/bln'), 'ratio_pct':st.column_config.NumberColumn('Rasio', format='%.2f%%'), 'total_payment':money_col('Total bayar')})
                chart = rows.assign(Mobil=rows.company+' '+rows.model).rename(columns={'installment':'Cicilan/bln (Rp)'})
                st.bar_chart(chart, x='Mobil', y='Cicilan/bln (Rp)', horizontal=True)
                st.download_button('Unduh rekomendasi (CSV)', rows.to_csv(index=False).encode('utf-8-sig'), 'rekomendasi.csv', 'text/csv')
                st.caption('Maksimal 5 mobil unik. CSV menggunakan rasio desimal. Mobil yang sedang disimulasikan dikecualikan.')
            with st.expander('Jejak filter rekomendasi', expanded=rows.empty):
                st.caption('Lolos rasio dan lolos DP dihitung secara independen; keduanya dapat tumpang tindih.')
                table(pd.DataFrame({'Tahap':['Mobil setelah filter tipe dan pengecualian','Kombinasi DP/tenor','Lolos rasio 35%','Lolos anggaran DP','Lolos kedua batas','Mobil unik setelah ranking','Top 5'], 'Jumlah':[stats[k] for k in ('cars_considered','combinations','ratio_pass','dp_pass','both_pass','unique_cars','displayed')]}))
        else:
            st.info('Tekan Cari rekomendasi untuk mulai memeriksa katalog.')

with trace_tab:
    st.subheader('Bagaimana hasil dihitung?')
    st.caption('Jejak proses aktual pada simulasi terakhir, bukan training model ulang.')
    st.markdown('**Input → Validasi → Feature engineering → Inferensi ML → Pemeriksaan hasil → Keputusan finansial**')
    result = st.session_state.get('simulation')
    if result is None:
        st.info('Belum ada jejak proses. Jalankan simulasi terlebih dahulu.')
        if st.session_state.get('sim_error'):
            st.code(st.session_state.sim_error)
    else:
        req, f = result['request'], result['finance']
        snapshot(req)
        st.markdown('#### 1. Pemeriksaan setiap tahap')
        checks = result_checks(result)
        table(checks.style.map(lambda s: 'color: #147362; font-weight: 600' if s == 'Lulus' else 'color: #a12a26; font-weight: 600' if s == 'Gagal' else 'color: #9a5a15; font-weight: 600', subset=['Status']))
        st.caption('Melebihi batas adalah hasil penilaian finansial, bukan error program. Ambang selisih Rp 1 hanya diagnostik model pada dataset ini.')
        with st.expander('2. Input dan perhitungan finansial', expanded=True):
            table(result['raw'])
            table(pd.DataFrame({'Langkah': ['Harga × DP%', 'Harga − DP', 'Pokok ÷ tenor (floor)', 'Cicilan akhir sesuai mode', 'Cicilan mobil + cicilan lain', 'Total cicilan ÷ penghasilan', '35% penghasilan − cicilan lain (min. 0)'], 'Hasil': [rp(f.down_payment), rp(f.principal), rp(f.base_installment), rp(f.installment), rp(f.installment+req.other), pct(f.ratio), rp(f.installment_limit)]}))
            st.caption('Model ML menggunakan cicilan dasar tanpa bunga. Mode flat dan cicilan lain hanya memengaruhi kalkulator finansial.')
        for task, heading in [('reg','3. Regresi: prediksi cicilan'), ('clf','4. Klasifikasi: prediksi label')]:
            with st.expander(heading, expanded=True):
                if task in result['errors']:
                    st.error('Inferensi gagal. Periksa versi library, model dan skema fitur.')
                    st.code(result['errors'][task])
                    continue
                data = result['models'][task]
                pred = data['prediction']
                a,b,c = st.columns(3)
                if task == 'reg':
                    a.metric(data['name'], rp(pred, 2))
                    b.metric('Rumus dataset', rp(f.base_installment))
                    c.metric('Selisih absolut', rp(abs(pred-f.base_installment), 4))
                else:
                    a.metric(data['name'], 'Sesuai' if pred else 'Melebihi')
                    b.metric('Label aturan dataset', 'Sesuai' if result['rule_label'] else 'Melebihi')
                    c.metric('Konsistensi', 'Sama' if pred == result['rule_label'] else 'Berbeda')
                st.caption('Nilai fitur aktual yang dikirim ke pipeline model.')
                trace = data.get('trace')
                if trace:
                    table(pd.DataFrame(trace['steps']))
                    table(trace['features'], {'Nilai input':st.column_config.NumberColumn(format='%.8g'), 'Nilai ke estimator':st.column_config.NumberColumn(format='%.8g')})
                    if 'contributions' in trace:
                        st.markdown('**Linear Regression: kontribusi aktual tiap fitur**')
                        table(trace['contributions'], {'Nilai transformasi':st.column_config.NumberColumn(format='%.8g'), 'Koefisien':st.column_config.NumberColumn(format='%.8g'), 'Kontribusi (Rp)':st.column_config.NumberColumn(format='%.6f')})
                        st.write(f"Jumlah kontribusi + intercept {rp(trace['intercept'], 4)} = **{rp(trace['reconstructed'], 4)}**")
                        st.caption('RobustScaler mentransformasi fitur sebelum perkalian koefisien. Kontribusi adalah dekomposisi matematis prediksi, bukan efek kausal.')
                    if 'path' in trace:
                        st.markdown('**Decision Tree: jalur node yang benar-benar dilewati**')
                        table(trace['path'], {'Nilai':st.column_config.NumberColumn(format='%.9g'), 'Ambang':st.column_config.NumberColumn(format='%.9g')})
                        st.caption(f"Kedalaman tree: {trace['depth']}. Kelas 1 = sesuai rasio dataset, kelas 0 = melebihi. Perbandingan node mengikuti presisi float32 internal tree.")
                else:
                    table(data['features'].T.reset_index().rename(columns={'index':'Fitur', 0:'Nilai'}))
                    st.warning('Detail interpretasi tidak tersedia: ' + result['trace_errors'].get(task,'tidak diketahui'))
        with st.expander('5. Rentang data latih dan batas keputusan', expanded=True):
            if result['warnings']:
                st.warning('Input di luar rentang training terdeteksi; prediksi ML bersifat eksploratif.')
                table(pd.DataFrame(result['warnings']))
            else:
                st.write('Tidak ada peringatan rentang pada model yang berhasil dijalankan.')
            st.write('Keputusan akhir memakai rasio seluruh cicilan dan anggaran DP. Perbedaan dengan classifier dapat terjadi karena bunga, cicilan lain, atau anggaran DP bukan bagian dari target ML.')
            st.caption('Rentang min/maks bukan jaminan semua kombinasi pernah muncul dalam training.')
        st.download_button('Unduh diagnostik (CSV)', checks.to_csv(index=False).encode('utf-8-sig'), 'diagnostik_simulasi.csv', 'text/csv')

with study_tab:
    st.subheader('Hasil penelitian dan pemilihan algoritma')
    st.caption('Data evaluasi tersimpan dari notebook, bukan hasil training langsung saat simulasi.')
    if metadata:
        r,c = metadata['models']['reg'],metadata['models']['clf']
        a,b,d = st.columns(3)
        a.metric('Data kredit valid', f"{metadata['n_credit_valid']:,}".replace(',','.'))
        b.metric('MAE · Linear Regression', rp(r['test_metrics']['MAE'], 4))
        d.metric('Macro-F1 · Decision Tree', pct(c['test_metrics']['Macro_F1']))
        st.caption(f"{metadata['cv_folds']}-fold CV · {metadata['n_cv_sample']:,} sampel tuning · {metadata['n_test']:,} final test. Target deterministik, bukan prediksi persetujuan kredit aktual.")
    st.markdown('**Alur penelitian:** Profil data → Cleaning → EDA & outlier → Feature engineering → Split → CV & tuning → Stress test → Final test → Ekspor model')
    view = st.radio('Pilih bagian penelitian', ['Perbandingan model','Evaluasi final & generalisasi','EDA & kategorikal','Metadata'], horizontal=True)
    try:
        if view == 'Perbandingan model':
            cv = report('cv_model_comparison.csv')
            a,b = st.columns(2)
            task = a.selectbox('Jenis model', ['reg','clf'], format_func=lambda x: 'Regresi (MAE, lebih kecil lebih baik)' if x=='reg' else 'Klasifikasi (macro-F1, lebih besar lebih baik)')
            scenarios = {'1_raw_baseline':'Fitur asli / baseline', '2_raw_tuned':'Fitur asli / tuning', '3_financial_tuned':'Fitur finansial / tuning'}
            scenario = b.selectbox('Skenario', list(scenarios), index=2, format_func=scenarios.get)
            selected = cv[(cv.task == task) & (cv.scenario == scenario)].sort_values('cv_primary', ascending=task=='reg')
            metric_label = 'MAE (Rp)' if task=='reg' else 'Macro-F1'
            st.bar_chart(selected.rename(columns={'cv_primary':metric_label}), x='model', y=metric_label, horizontal=True)
            cols=['model','train_primary','cv_primary','cv_std','generalization_gap']
            table(selected[cols], {'model':'Algoritma','train_primary':st.column_config.NumberColumn('Train', format='%.6g'),'cv_primary':st.column_config.NumberColumn('CV '+metric_label,format='%.6g'),'cv_std':st.column_config.NumberColumn('SD CV',format='%.6g'),'generalization_gap':st.column_config.NumberColumn('Gap',format='%.6g')})
            st.caption('Random Forest dan XGBoost tetap termasuk kandidat. Gap adalah selisih train dan validasi sesuai definisi notebook. Skor serupa dapat mengutamakan model sederhana.')
            with st.expander('Parameter model yang diuji'):
                table(selected[['model','params']])
            st.download_button('Unduh seluruh hasil CV', cv.to_csv(index=False).encode('utf-8-sig'), 'cv_model_comparison.csv', 'text/csv')
        elif view == 'Evaluasi final & generalisasi':
            for name, label in [('regression_metrics.csv','Regresi / final test'),('classification_metrics.csv','Klasifikasi / final test'),('generalization_stress.csv','Stress test generalisasi'),('near_threshold.csv','Kasus dekat ambang 35%'),('outlier_sensitivity.csv','Sensitivitas outlier')]:
                with st.expander(label, expanded=name=='regression_metrics.csv'):
                    table(report(name))
            draw_figures({'Learning curve regresi':'09_learning_curve_reg.png','Learning curve klasifikasi':'09_learning_curve_clf.png','Residual & prediksi regresi':'10_regression_diagnostics.png','Confusion matrix, ROC & PR':'12_classification_diagnostics.png','Feature importance regresi':'13_importance_reg.png','Feature importance klasifikasi':'13_importance_clf.png'}, 'study_fig')
        elif view == 'EDA & kategorikal':
            st.write('Analisis kategorikal menggunakan data development. Hubungan kategori bersifat deskriptif, bukan sebab-akibat atau fitur tambahan model.')
            for label,name in [('Profil kategorikal','categorical_profile.csv'),('Asosiasi kategori','categorical_association.csv')]:
                if (ROOT/'reports'/name).exists():
                    with st.expander(label): table(report(name))
            grouped = report('categorical_financial_summary.csv')
            assoc = report('categorical_association.csv')
            labels = dict(zip(assoc.variable, assoc.label))
            field = st.selectbox('Variabel kategorikal', list(labels), format_func=labels.get)
            group = grouped[grouped.variable == field].copy()
            cat_metric = st.radio('Metrik per kategori', ['transactions','median_installment_million','eligible_pct'], format_func=lambda x: {'transactions':'Jumlah transaksi','median_installment_million':'Median cicilan (juta Rp)','eligible_pct':'Sesuai batas (%)'}[x], horizontal=True)
            st.bar_chart(group, x='category', y=cat_metric, horizontal=True)
            table(group[['category','transactions','median_price_million','median_installment_million','median_dti_pct','eligible_pct']], {'category':'Kategori','transactions':'Jumlah','median_price_million':st.column_config.NumberColumn('Harga median (juta Rp)',format='%.2f'),'median_installment_million':st.column_config.NumberColumn('Cicilan median (juta Rp)',format='%.2f'),'median_dti_pct':st.column_config.NumberColumn('Median rasio',format='%.2f%%'),'eligible_pct':st.column_config.NumberColumn('Sesuai batas',format='%.2f%%')})
            st.download_button('Unduh ringkasan kategorikal', grouped.to_csv(index=False).encode('utf-8-sig'), 'categorical_financial_summary.csv', 'text/csv')
            draw_figures({'Distribusi kategori utama':'categorical_16_1.png','Distribusi kategori tambahan':'categorical_16_2.png','Cicilan per kategori':'categorical_18_22.png','Distribusi cicilan & status':'categorical_20_0.png','Bivariat kategori (1)':'categorical_20_1.png','Bivariat kategori (2)':'categorical_20_2.png','Crosstab merek/tipe dan mesin/transmisi':'categorical_22_0.png','Status menurut merek dan tenor':'categorical_22_1.png','Cramér’s V dan eta squared':'categorical_24_1.png','Outlier IQR':'03_outlier_counts.png'}, 'eda_fig')
        else:
            if metadata: st.json(metadata, expanded=False)
            else: st.warning('Metadata model tidak dapat dimuat.')
            st.caption('Notebook lengkap tersedia di folder notebooks/ pada repository.')
    except Exception as exc:
        st.error('Laporan penelitian tidak dapat ditampilkan.')
        with st.expander('Detail teknis'): st.code(f'{type(exc).__name__}: {exc}')
