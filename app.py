import pandas as pd
import streamlit as st
from src.financial_calculator import calculate, DP_OPTIONS, TENORS
from src.prediction import load_assets, predict, ROOT
from src.recommendation import recommend

st.set_page_config(page_title='Simulasi Cicilan Mobil', page_icon='🚗', layout='wide')

@st.cache_resource
def assets():
    return load_assets()


def rp(value):
    return 'Rp ' + f'{value:,.0f}'.replace(',', '.')


def pct(value):
    return f'{value * 100:.2f}%'.replace('.', ',')


try:
    models, metadata, catalog = assets()
except Exception as exc:
    st.error(f'Gagal memuat aset penelitian: {exc}')
    st.stop()

st.title('Simulasi Cicilan & Kemampuan Finansial Mobil')
st.caption('Machine learning · Simulasi finansial · Rekomendasi kendaraan')
st.info('Simulasi edukatif, bukan keputusan persetujuan kredit. Harga berasal dari katalog dataset, bukan penawaran terkini.')

with st.sidebar:
    st.header('1. Kondisi finansial')
    income = st.number_input('Penghasilan bulanan (Rp)', 1_000_000, 500_000_000, 15_000_000, 500_000)
    other = st.number_input('Cicilan lain per bulan (Rp)', 0, 500_000_000, 0, 500_000)
    budget = st.number_input('Anggaran uang muka (Rp)', 0, 970_000_000, 100_000_000, 5_000_000)
    st.header('2. Kendaraan')
    company = st.selectbox('Merek mobil', sorted(catalog.company.unique()))
    cars = catalog[catalog.company == company].sort_values('model')
    name = st.selectbox('Model mobil', cars.model.tolist())
    car = cars[cars.model == name].iloc[0]
    price = int(car.price_idr)
    st.header('3. Skema pembayaran')
    dp = st.select_slider('DP (%)', options=DP_OPTIONS, value=25)
    tenure = st.selectbox('Tenor (bulan)', TENORS, index=3)
    mode_label = st.radio('Mode simulasi', ['Dataset (tanpa bunga)', 'Bunga flat (asumsi)'])
    mode = 'dataset' if mode_label.startswith('Dataset') else 'flat'
    rate = st.number_input('Bunga flat per tahun (%)', 0.0, 30.0, 5.0, 0.5) if mode == 'flat' else 0.0
    st.caption('Hasil diperbarui otomatis saat input berubah.')

f = calculate(price, dp, tenure, income, other, rate, mode)
ml_amount, ml_status, outside = predict(models, metadata, price, f.down_payment, tenure, income)
valid_budget = f.down_payment <= budget

sim, rec, study = st.tabs(['Simulasi & hasil', 'Rekomendasi mobil', 'Evaluasi model'])
with sim:
    st.subheader(f'{company} {name}')
    st.caption(f'{car.body_style} · {tenure} bulan · DP {dp}%')
    c = st.columns(3)
    c[0].metric('Harga katalog', rp(price))
    c[1].metric('Uang muka', rp(f.down_payment))
    c[2].metric('Pokok pembiayaan', rp(f.principal))
    st.divider()
    c = st.columns(3)
    c[0].metric('Cicilan mobil / bulan', rp(f.installment))
    c[1].metric('Rasio seluruh cicilan', pct(f.ratio))
    c[2].metric('Batas cicilan mobil / bulan', rp(f.installment_limit))
    st.progress(min(1.0, f.ratio), text=f'Rasio cicilan: {pct(f.ratio)} · Batas simulasi 35%')
    if f.affordable and valid_budget:
        st.success('LAYAK DALAM SIMULASI — Rasio cicilan dan uang muka sesuai batas input.')
    else:
        reasons = []
        if not f.affordable:
            reasons.append('rasio seluruh cicilan melebihi 35%')
        if not valid_budget:
            reasons.append('uang muka melebihi anggaran')
        st.error('BELUM SESUAI BATAS SIMULASI — ' + '; '.join(reasons) + '.')
        st.write('Buka tab Rekomendasi mobil untuk mencari alternatif.')
    c = st.columns(2)
    c[0].metric('Total pembayaran mobil', rp(f.total_payment))
    c[1].metric('Sisa penghasilan setelah cicilan', rp(f.remaining_income))
    st.caption('Belum termasuk biaya hidup, administrasi, asuransi, pajak, atau perawatan. Mode dataset membulatkan cicilan ke bawah; sisa pembulatan pokok dianggap diselesaikan pada pembayaran terakhir. Bunga flat adalah asumsi, bukan tarif lembaga pembiayaan.')
    st.subheader('Hasil prediksi machine learning')
    c = st.columns(3)
    c[0].metric(metadata['models']['reg']['name'], rp(ml_amount), help='Prediksi cicilan tanpa bunga.')
    c[1].metric(metadata['models']['clf']['name'], 'Layak' if ml_status else 'Tidak layak', help='Label rasio cicilan mobil tanpa bunga ≤35%.')
    c[2].metric('Selisih absolut ML vs rumus', f'Rp {abs(ml_amount-f.base_installment):.2f}'.replace('.', ','))
    st.caption('Prediksi ML mengikuti definisi dataset (tanpa bunga dan tanpa utang lain). Keputusan simulasi memakai satu aturan finansial, termasuk kewajiban lain dan batas DP. Perbedaan status bukan otomatis kesalahan model.')
    if outside:
        st.warning('Sebagian input berada di luar rentang training: ' + ', '.join(outside) + '. Prediksi ML bersifat eksploratif.')
    if ml_status != int(f.base_installment * 100 <= income * 35):
        st.warning('Classifier berbeda dari aturan label dataset pada input ini. Keputusan simulasi tetap memakai perhitungan finansial.')

with rec:
    st.subheader('Alternatif kendaraan sesuai anggaran')
    explore = st.checkbox('Cari kombinasi DP dan tenor lain', value=True)
    ranking = st.selectbox('Prioritas rekomendasi', ['Cicilan terendah', 'DP terendah', 'Tenor terpendek', 'Total pembayaran terendah', 'Harga mobil tertinggi'])
    body = st.selectbox('Preferensi tipe bodi', ['Semua', *sorted(catalog.body_style.unique())])
    result = recommend(catalog, income, budget, dp, tenure, other, rate, mode, explore, ranking, body, (company, name))
    if result.empty:
        st.warning('Tidak ada alternatif yang memenuhi batas rasio dan anggaran DP. Ubah kriteria jika sesuai kemampuan finansial.')
    else:
        result = result.rename(columns={'company':'Merek','model':'Model','body_style':'Tipe','price':'Harga','dp_percent':'DP (%)','down_payment':'Uang muka','tenure':'Tenor (bulan)','installment':'Cicilan/bulan','ratio':'Rasio','total_payment':'Total bayar'})
        raw_csv = result.to_csv(index=False).encode('utf-8-sig')
        for col in ['Harga','Uang muka','Cicilan/bulan','Total bayar']:
            result[col] = result[col].map(rp)
        result['Rasio'] = result['Rasio'].map(pct)
        st.dataframe(result, hide_index=True, use_container_width=True)
        st.caption('Maksimal lima mobil berbeda, satu skema terbaik per mobil menurut prioritas. Mobil yang sedang dipilih tidak ditampilkan kembali. CSV menggunakan rasio desimal.')
        st.download_button('Unduh rekomendasi CSV', raw_csv, 'rekomendasi_mobil.csv', 'text/csv')

with study:
    st.subheader('Hasil penelitian dari notebook')
    r = metadata['models']['reg']; c = metadata['models']['clf']
    cols = st.columns(3)
    cols[0].metric('Data kredit valid', f"{metadata['n_credit_valid']:,}".replace(',', '.'))
    cols[1].metric('MAE regresi (final test)', f"Rp {r['test_metrics']['MAE']:.2f}")
    cols[2].metric('Macro-F1 klasifikasi (final test)', f"{c['test_metrics']['Macro_F1']:.2%}")
    st.write(f"Regresi terpilih: **{r['name']}** · Klasifikasi terpilih: **{c['name']}**")
    st.caption(f"{metadata['cv_folds']}-fold CV · {metadata['n_cv_sample']:,} sampel tuning · {metadata['n_test']:,} final test. Target deterministik; skor bukan bukti prediksi persetujuan kredit aktual.")
    for label, file in [('Regresi','regression_metrics.csv'), ('Klasifikasi','classification_metrics.csv'), ('Cross-validation seluruh kandidat','cv_model_comparison.csv'), ('Uji generalisasi','generalization_stress.csv')]:
        with st.expander(label):
            st.dataframe(pd.read_csv(ROOT / 'reports' / file), hide_index=True, use_container_width=True)
    with st.expander('Grafik evaluasi'):
        for path in sorted((ROOT / 'reports/figures').glob('*.png')):
            if path.name.startswith(('07','08','09','10','11','12','13')):
                st.image(str(path), caption=path.stem, use_container_width=True)
    with st.expander('Metadata model dan keterbatasan'):
        st.json(metadata, expanded=False)
