# Dashboard hasil penelitian: streamlit run dashboard.py
import contextlib
import io
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui

BASE = Path(__file__).parent
# Run final: forecast 5 Nov (identik arsip/5_nov/) dan portofolio window 63/126/252 dalam satu folder
FORECAST_DIR = PORTO_DIR = BASE / "8_nov"

# Warna tetap per seri supaya mudah dibedakan di semua grafik
WARNA = {
    "NSGA-III": "#3683c8", "GA": "#30954f",
    "NSGA-III · N-BEATS": "#3683c8", "NSGA-III · LSTM": "#70b8fe", "NSGA-III · N-BEATS + LSTM": "#01528d",
    "GA · N-BEATS": "#30954f", "GA · LSTM": "#6acb81", "GA · N-BEATS + LSTM": "#026129",
    "Equal Weight (rebalance)": "#475569", "Equal Weight (buy & hold)": "#94a3b8",
    "NBEATS": "#4f46e5", "LSTM": "#f97316", "Actual": "#0f172a",
    "IBIT": "#8b5cf6", "GLD": "#f59e0b", "SPY": "#0ea5e9",
    "E(R) - TC": "#4f46e5", "Volatilitas": "#f97316", "Skewness": "#10b981", "Kurtosis": "#e11d48",
    "Pareto": "#94a3b8", "TOPSIS solution": "#e11d48", "Final run weights": "#0f172a",
}
OBJEKTIF = ["E(R) - TC", "Volatilitas", "Skewness", "Kurtosis"]

LABEL_BIAYA = {"dengan_biaya": "0.3%", "tanpa_biaya": "0% (tanpa biaya)"}
LABEL_NAMA = {"nbeats": "N-BEATS", "lstm": "LSTM", "kombinasi": "N-BEATS + LSTM", "semua": "Semua model (berdampingan)",
              "historis": "Rata-rata historis", "dengan": "dengan biaya"}

st.set_page_config(page_title="Portofolio IBIT-GLD-SPY", page_icon="📈", layout="wide")


@st.cache_data
def baca(folder, nama, **kwargs):
    return pd.read_csv(folder / nama, **kwargs)


def saring(df, **syarat):
    for kolom, nilai in syarat.items():
        df = df[df[kolom] == nilai]
    return df


def grafik_garis(df, y_label, format_tooltip=",.2f", sumbu_x="Tanggal", tipe_x="T", tinggi=380):
    # Altair agar sumbu-y mengikuti rentang data, bukan dari 0
    data = df.rename_axis(sumbu_x).reset_index().melt(sumbu_x, var_name="Seri", value_name=y_label)
    chart = alt.Chart(data).mark_line(strokeWidth=2.2, point=tipe_x == "O").encode(
        x=alt.X(f"{sumbu_x}:{tipe_x}", title=None if tipe_x == "T" else sumbu_x, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y_label}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Seri:N", sort=list(df.columns), title=None, legend=alt.Legend(orient="bottom", labelLimit=0),
                        scale=alt.Scale(domain=list(df.columns), range=[WARNA[c] for c in df.columns])),
        tooltip=[f"{sumbu_x}:{tipe_x}", "Seri:N", alt.Tooltip(f"{y_label}:Q", format=format_tooltip)],
    ).properties(height=tinggi)
    st.altair_chart(chart, use_container_width=True)


def grafik_bobot(w, freq):
    # Batang bertumpuk per periode tanpa zoom; tooltip menampilkan bobot ketiga aset periode itu
    data = w.reset_index()[["Rebalance Date"] + [f"w_{a}" for a in ASSETS]]
    data["Periode"] = data["Rebalance Date"].dt.strftime("%Y-%m-%d")
    long = data.melt(["Rebalance Date", "Periode"], var_name="Asset", value_name="Bobot")
    long = long.merge(data.drop(columns="Rebalance Date"), on="Periode")
    long["Asset"] = long["Asset"].str[2:]
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("Periode:O", title=f"Rebalance date ({freq})", axis=alt.Axis(labelAngle=-45, labelOverlap=True)),
        y=alt.Y("Bobot:Q", stack="zero", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Asset:N", sort=ASSETS, title=None, legend=alt.Legend(orient="bottom", labelLimit=0),
                        scale=alt.Scale(domain=ASSETS, range=[WARNA[a] for a in ASSETS])),
        order=alt.Order("Asset:N", sort="ascending"),
        tooltip=[alt.Tooltip("Periode:N", title="Rebalance date")]
                + [alt.Tooltip(f"w_{a}:Q", title=a, format=".1%") for a in ASSETS],
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


def pct(x, desimal=2):
    return "-" if pd.isna(x) else f"{x:.{desimal}f}%"


# Format tampilan tiap metrik kinerja; Max Drawdown tersimpan sebagai pecahan
FORMAT_METRIK = {
    "Total Return (%)": pct,
    "Final Value": lambda x: f"{'-' if x < 0 else ''}${abs(x):,.2f}",
    "Sharpe": lambda x: "-" if pd.isna(x) else f"{x:.3f}",
    "Sortino": lambda x: "-" if pd.isna(x) else f"{x:.3f}",
    "Max Drawdown": lambda x: pct(x * 100),
    "Turnover": lambda x: f"{x:.3f}",
}
NAMA_METRIK = {"Total Return (%)": "Total Return"}


def tabel(df, key, **kwargs):
    # Tabel statis shadcn, tanpa menu format kolom bawaan st.dataframe
    ui.table(df.fillna("-").astype(str), key=key, **kwargs)


@st.cache_resource(show_spinner="Memuat definisi optimasi dari portofolio_mvsk.ipynb...")
def muat_optimasi():
    """Fungsi dan data portofolio_mvsk.ipynb untuk merekonstruksi Pareto front."""
    sumber = json.loads((PORTO_DIR / "portofolio_mvsk.ipynb").read_text())
    penanda = ["# @core", "def load_forecast", "from pymoo", "class SimplexRepair", "def make_nsga3_solver"]
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in sumber["cells"]:
            src = "".join(cell["source"])
            # Sel diagnostik hypervolume ikut mengimpor pymoo tetapi butuh variabel contoh, jadi dilewati
            if cell["cell_type"] != "code" or not any(p in src for p in penanda) or "pymoo.indicators" in src:
                continue
            src = (src.replace('"data_IBIT_GLD_SPY.csv"', repr(str(PORTO_DIR / "data_IBIT_GLD_SPY.csv")))
                      .replace('"forecast_', f'"{PORTO_DIR}/forecast_')
                      .replace('Path("gambar_portofolio")', f'Path({str(PORTO_DIR / "gambar_portofolio")!r})'))
            exec(src, ns)
    return ns


@st.cache_data(show_spinner=False)
def front_pareto(model, freq, mode, biaya, window, t):
    """Front NSGA-III periode t, berangkat dari bobot run final periode sebelumnya."""
    ns = muat_optimasi()
    data = ns["SKEMA"][freq]
    rate = config["transaction_cost"][biaya]
    skn = dict(Model=model, Frekuensi=freq, Mode=mode, Biaya=biaya, Window=window)
    w_final = saring(baca(PORTO_DIR, "portfolio_weights.csv"), Strategi="NSGA-III", **skn)[
        [f"w_{a}" for a in ASSETS]].values
    seed = int(saring(baca(PORTO_DIR, "portfolio_equity.csv"), Strategi="NSGA-III", **skn)["Seed"].iloc[0])

    w_minus = np.full(len(ASSETS), 1 / len(ASSETS)) if t == 0 else ns["drift_weights"](w_final[t - 1], data["realized_test"][t - 1])
    sigma, m3, m4 = data["moments_test"][window][t]
    r_hat = data["r_hat_test"][(model, mode)][t]

    _, extra = ns["make_nsga3_solver"](rate=rate)(r_hat, w_minus, sigma, m3, m4, seed)
    index, lam, closeness = ns["entropy_topsis"](extra["pareto_F"])
    f_final = ns["evaluate_objectives"](w_final[t], r_hat, w_minus, sigma, m3, m4, rate)[0]

    # Tanda minimisasi dikembalikan ke bentuk asli: return dan skewness positif = baik
    alami = np.array([-1, 1, -1, 1])
    return {"X": extra["pareto_X"], "F": extra["pareto_F"] * alami, "closeness": closeness, "index": index,
            "lam": lam, "w_final": w_final[t], "f_final": f_final * alami, "seed": seed}


config = json.loads((PORTO_DIR / "portfolio_config.json").read_text())
ASSETS = config["assets"]

HALAMAN = ["Forecast", "Bobot portofolio", "Backtest", "Evaluasi mingguan"]

with st.sidebar:
    st.subheader("Menu")
    halaman = ui.tabs(HALAMAN, value=HALAMAN[0], orientation="vertical", key="halaman")
    st.divider()

    if halaman == "Forecast":
        st.subheader("Pilihan forecast")
        asset = ui.select("Asset", ASSETS, key="asset")
        split = ui.select("Data", ["test", "val"], format_func={"test": "Test", "val": "Validation"}.get, key="split")
        satuan = ui.select("Horizon", ["harian", "mingguan"], format_func=str.capitalize, key="satuan")
    else:
        st.subheader("Skenario portofolio")
        # Expected return hanya dari forecast (rata-rata historis disembunyikan); "semua" = ketiga model berdampingan
        pilihan_model = config["forecast_models"] + [config["combined_model"]]
        model = ui.select("Model expected return", pilihan_model + ["semua"], format_func=LABEL_NAMA.get, key="model")
        models = pilihan_model if model == "semua" else [model]
        if config["combined_model"] in models:
            st.caption("N-BEATS + LSTM: tiap aset per periode memakai model dengan error forecast periode sebelumnya terkecil.")
        freq = ui.select("Rebalancing frequency", config["frequencies"], format_func=str.capitalize, key="freq")
        # Satu expected return per frekuensi: akhir minggu (mingguan), t+1 (harian)
        mode = dict(config["scenarios"])[freq]
        biaya = ui.select("Biaya transaksi", list(config["transaction_cost"]), format_func=LABEL_BIAYA.get,
                          key="biaya")
        window = ui.select("Rolling window", config["windows"], format_func=lambda w: f"{w} trading days", key="window")
        dasar = dict(Frekuensi=freq, Mode=mode, Biaya=biaya, Window=window)


def skenario(m):
    """Filter satu skenario portofolio untuk model m."""
    return dict(Model=m, **dasar)


def nama_seri(algo, m):
    """Nama seri grafik; model ikut disebut pada opsi semua model."""
    return algo if len(models) == 1 else f"{algo} · {LABEL_NAMA[m]}"



st.title("Optimasi Portofolio IBIT, GLD, dan SPY")
if halaman != "Forecast":
    st.caption(f"Forecast N-BEATS vs LSTM · optimasi MVSK NSGA-III vs GA · initial capital "
               f"${config['initial_capital']:,.0f} · {config['n_seeds']} seed")

if halaman == "Forecast":
    if satuan == "harian":
        data = saring(baca(FORECAST_DIR, f"forecast_daily_{split}.csv", parse_dates=["Date"]), Asset=asset)
        tanggal, ramalan, aktual = "Date", "Forecast 1-step", "Actual"
        sumbu, skala, keterangan = "Price (USD)", 1, "Forecast 1-step: harga t+1 dari harga aktual kemarin"
    else:
        data = saring(baca(FORECAST_DIR, f"forecast_weekly_{split}.csv", parse_dates=["Target Date"]),
                      Asset=asset)
        tanggal, ramalan, aktual = "Target Date", "Forecast Price", "Actual Price"
        sumbu, skala, keterangan = "Price (USD)", 1, "Recursive forecast harga hari perdagangan terakhir minggu"

    chart = data.pivot_table(index=tanggal, columns="Model", values=ramalan)
    chart.columns = [m.upper() for m in chart.columns]
    chart["Actual"] = data.groupby(tanggal)[aktual].first()

    st.subheader(f"{asset}: actual vs forecast")
    st.caption(keterangan)
    grafik_garis(chart[["Actual", "NBEATS", "LSTM"]] * skala, sumbu)

    # Akurasi dihitung dari CSV forecast sesuai horizon dan data yang dipilih di sidebar
    st.subheader(f"Forecast accuracy {satuan}")
    if satuan == "harian":
        f_harga, a_harga, acuan = data["Forecast 1-step"], data["Actual"], data["Previous Actual"]
    else:
        f_harga, a_harga, acuan = data["Forecast Price"], data["Actual Price"], data["Anchor Price"]

    baris = []
    for nama, idx in data.groupby("Model").groups.items():
        f, a, prev = f_harga[idx], a_harga[idx], acuan[idx]
        rf, ra = f / prev - 1, a / prev - 1
        arah = ra != 0  # return aktual nol dikecualikan dari DA
        baris.append({
            "Model": LABEL_NAMA[nama], "n": len(idx),
            "RMSE Return": f"{np.sqrt(((rf - ra) ** 2).mean()):.4f}",
            "MAE Return": f"{(rf - ra).abs().mean():.4f}",
            "RMSE Price": f"{np.sqrt(((f - a) ** 2).mean()):.4f}",
            "MAE Price": f"{(f - a).abs().mean():.4f}",
            "MAPE Price": pct(((f - a).abs() / a).mean() * 100),
            "DA": pct((np.sign(rf[arah]) == np.sign(ra[arah])).mean() * 100),
        })
    tabel(pd.DataFrame(baris), key="akurasi")

elif halaman == "Bobot portofolio":
    weights = baca(PORTO_DIR, "portfolio_weights.csv", parse_dates=["Rebalance Date"])
    kolom_bobot = [f"w_{a}" for a in ASSETS]

    for m in models:
        if len(models) > 1:
            st.header(LABEL_NAMA[m])

        st.subheader("Komposisi bobot portofolio")
        st.caption("Bobot tiap periode rebalancing, seed dengan final value median. Arahkan kursor ke batang untuk melihat bobotnya.")

        for col, algo in zip(st.columns(2), ["NSGA-III", "GA"]):
            w = saring(weights, Strategi=algo, **skenario(m)).set_index("Rebalance Date")
            with col:
                st.markdown(f"**{algo}**")
                grafik_bobot(w, freq)
                rata = w[kolom_bobot].mean()
                ui.badges([ui.BadgeItem(f"{c[2:]} rata-rata {rata[c]:.0%}", "outline") for c in kolom_bobot]
                          + [ui.BadgeItem(f"Turnover {w['Turnover'].sum():.2f}", "secondary")],
                          key=f"badge_{algo}_{m}")

        st.subheader("Entropy weights (Entropy Weight-TOPSIS) NSGA-III per periode")
        # Bobot entropi hasil run (seed yang sama dengan grafik bobot), urutan kolom = OBJEKTIF
        lam = saring(weights, Strategi="NSGA-III", **skenario(m)).set_index("Rebalance Date")
        lam = lam[[c for c in weights.columns if c.startswith("Entropy ")]].set_axis(OBJEKTIF, axis=1)
        grafik_garis(lam, "Entropy weight", ".3f")
        st.caption("Makin besar entropy weight, makin besar peran objektif itu saat TOPSIS memilih satu solusi dari Pareto front.")

        st.subheader("Rincian bobot per periode rebalancing")
        algo = ui.select("Algorithm", ["NSGA-III", "GA"], key=f"algo_bobot_{m}")
        rinci = saring(weights, Strategi=algo, **skenario(m)).drop(columns=["Strategi"] + list(skenario(m)))
        rinci["Rebalance Date"] = rinci["Rebalance Date"].dt.date
        for c in kolom_bobot:
            rinci[c] = (rinci[c] * 100).map(lambda x: pct(x, 1))
        for c in ["Cost", "Gross Return", "Net Return"]:
            rinci[c] = (rinci[c] * 100).map(lambda x: pct(x, 3))
        rinci["Turnover"] = rinci["Turnover"].round(4)
        rinci["Value"] = rinci["Value"].map(FORMAT_METRIK["Final Value"])
        tabel(rinci.rename(columns=lambda c: c[2:] if c in kolom_bobot else c), key=f"rinci_{m}", max_height=420)

elif halaman == "Backtest":
    # Kartu ringkasan: median 30 seed skenario terpilih
    all_seeds = baca(PORTO_DIR, "portfolio_metrics_all_seeds.csv")
    all_seeds = all_seeds[all_seeds["Model"].isin(models)]
    median = saring(all_seeds, **dasar).groupby(["Model", "Algoritma"])[list(FORMAT_METRIK)].median()
    seri = [(m, algo) for m in models for algo in ["NSGA-III", "GA"]]
    kartu = [("Total Return", "Total Return (%)"), ("Final Value", "Final Value"), ("Sharpe Ratio", "Sharpe"),
             ("Sortino Ratio", "Sortino"), ("Max Drawdown", "Max Drawdown"), ("Turnover", "Turnover")]
    # Kartu HTML agar NSGA-III dan GA tampil setara berdampingan
    isi = "".join(
        f'<div class="kartu"><div class="judul">{judul}</div><div class="nilai">'
        + "".join(f'<div><span style="color:{WARNA[nama_seri(algo, m)]}">{nama_seri(algo, m)}</span>'
                  f'<b>{FORMAT_METRIK[kolom](median.loc[(m, algo), kolom])}</b></div>' for m, algo in seri)
        + "</div></div>"
        for judul, kolom in kartu)
    st.html("""<style>
      .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-family:Inter,sans-serif}
      .kartu{border:1px solid #e2e8f0;border-radius:12px;background:#fff;padding:14px 18px}
      .judul{font-size:.85rem;color:#64748b;margin-bottom:8px}
      .nilai{display:grid;grid-template-columns:1fr 1fr;gap:8px}
      .nilai span{display:block;font-size:.8rem;font-weight:600}
      .nilai b{font-size:1.45rem;font-weight:600;color:#0f172a}
    </style><div class="grid">""" + isi + "</div>")

    teks_lam = []
    for m in models:
        lam = config["best_lambda"][f"{m}|{freq}|{mode}|{biaya}|{window}"]
        teks_lam.append((f"{LABEL_NAMA[m]}: " if len(models) > 1 else "")
                        + " · ".join(f"{n} {x:.2f}" for n, x in zip(["return", "volatilitas", "skewness", "kurtosis"], lam)))
    st.caption(f"Median {config['n_seeds']} seed. λ GA terpilih di data validasi (grid Das-Dennis p = 8, "
               f"{config['n_ref_dirs']} kombinasi): " + "; ".join(teks_lam))

    equity = baca(PORTO_DIR, "portfolio_equity.csv", parse_dates=["Date"])
    algo_eq = saring(equity[equity["Model"].isin(models)], **dasar).copy()
    algo_eq["Strategi"] = [nama_seri(a, m) for a, m in zip(algo_eq["Strategi"], algo_eq["Model"])]
    bench_eq = saring(equity[equity["Window"].isna()], Frekuensi=freq, Biaya=biaya)
    urutan = [nama_seri(algo, m) for m, algo in seri] + ["Equal Weight (rebalance)", "Equal Weight (buy & hold)"]
    chart = pd.concat([algo_eq, bench_eq]).pivot_table(index="Date", columns="Strategi", values="Value")[urutan]

    # Return kumulatif, supaya sumbu-y tidak tertekan oleh baseline 0
    st.subheader("Cumulative return")
    st.caption("Grafik memakai satu seed per algoritma (urutan ke-16 dari 30 menurut final value), "
               "sedangkan nilai di atas memakai median 30 seed, jadi nilai akhirnya bisa berbeda sedikit.")
    grafik_garis((chart / config["initial_capital"] - 1) * 100, "Cumulative return (%)")

    st.subheader("Performance metrics per window")
    st.caption("Median [Q1; Q3] dari 30 seed")
    baris = []
    sub = saring(all_seeds, Frekuensi=freq, Mode=mode, Biaya=biaya)
    for (m, algo, win), g in sub.groupby(["Model", "Algoritma", "Window"]):
        teks = {}
        for kolom, fmt in FORMAT_METRIK.items():
            q1, med, q3 = g[kolom].quantile([0.25, 0.5, 0.75])
            teks[NAMA_METRIK.get(kolom, kolom)] = f"{fmt(med)} [{fmt(q1)}; {fmt(q3)}]"
        baris.append({**({"Model": LABEL_NAMA[m]} if len(models) > 1 else {}), "Algorithm": algo, "Window": win, **teks})
    tabel(pd.DataFrame(baris), key="summary")

    st.subheader("Benchmark")
    bench = saring(baca(PORTO_DIR, "portfolio_benchmark.csv"), Frekuensi=freq, Biaya=biaya)
    if not bench["Strategi"].str.startswith("100%").any():
        # Run lama belum mengekspor benchmark satu aset: dihitung dari harga harian periode test, tanpa biaya awal
        harga = baca(PORTO_DIR, "data_IBIT_GLD_SPY.csv", parse_dates=["Date"], index_col="Date")[ASSETS]
        tanggal = bench_eq["Date"]
        nilai = harga.loc[tanggal.min():tanggal.max()]
        nilai = config["initial_capital"] * nilai / nilai.iloc[0]
        satu_aset = []
        for a in ASSETS:
            r = nilai[a].pct_change().dropna().values
            satu_aset.append({"Strategi": f"100% {a} (buy & hold)",
                              "Sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252),
                              "Sortino": r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252),
                              "Max Drawdown": (nilai[a] / nilai[a].cummax() - 1).min(), "Turnover": 0.0,
                              "Final Value": nilai[a].iloc[-1],
                              "Total Return (%)": (nilai[a].iloc[-1] / config["initial_capital"] - 1) * 100})
        bench = pd.concat([bench, pd.DataFrame(satu_aset)], ignore_index=True)
    for kolom, fmt in FORMAT_METRIK.items():
        bench[kolom] = bench[kolom].map(fmt)
    tabel(bench.drop(columns=["Frekuensi", "Biaya"]).rename(columns={"Strategi": "Strategy", **NAMA_METRIK}),
          key="bench")

    def bagian_pareto(m):
        """Pareto front periode terpilih untuk model m, direkonstruksi ulang."""
        jadwal = saring(baca(PORTO_DIR, "portfolio_weights.csv", parse_dates=["Rebalance Date"]),
                        Strategi="NSGA-III", **skenario(m))["Rebalance Date"].dt.date.tolist()
        t = ui.select("Periode rebalancing", list(range(len(jadwal))), format_func=lambda i: f"{i + 1} · {jadwal[i]}",
                      key=f"periode_{freq}_{m}")
        with st.spinner("Merekonstruksi Pareto front..."):
            hasil = front_pareto(m, freq, mode, biaya, window, t)

        front = pd.DataFrame(hasil["F"], columns=OBJEKTIF)
        front["Solusi"] = "Pareto"
        front.loc[hasil["index"], "Solusi"] = "TOPSIS solution"
        front["Closeness"] = hasil["closeness"]
        for i, a in enumerate(ASSETS):
            front[a] = hasil["X"][:, i]
        final = pd.DataFrame([dict(zip(OBJEKTIF, hasil["f_final"]), Solusi="Final run weights",
                                   **dict(zip(ASSETS, hasil["w_final"])))])
        titik = pd.concat([front, final], ignore_index=True)

        solusi = ["Pareto", "TOPSIS solution", "Final run weights"]
        cols = st.columns(2)
        for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 3), (2, 3)]):
            plot = alt.Chart(titik).mark_point(filled=True, opacity=0.9, size=140).encode(
                x=alt.X(f"{OBJEKTIF[i]}:Q", scale=alt.Scale(zero=False)),
                y=alt.Y(f"{OBJEKTIF[j]}:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Solusi:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=0),
                                scale=alt.Scale(domain=solusi, range=[WARNA[x] for x in solusi])),
                shape=alt.Shape("Solusi:N", scale=alt.Scale(domain=solusi, range=["circle", "circle", "diamond"]),
                                legend=None),
                tooltip=[alt.Tooltip(f"{a}:Q", format=".1%") for a in ASSETS]
                        + [alt.Tooltip(f"{o}:Q", format=".4f") for o in OBJEKTIF],
            ).properties(height=300)
            cols[k % 2].altair_chart(plot, use_container_width=True)

        ui.badges([ui.BadgeItem(f"Entropy weight {o} {x:.3f}", "outline") for o, x in zip(OBJEKTIF, hasil["lam"])],
                  key=f"badge_entropi_{m}")

        tampil = titik.sort_values("Closeness", ascending=False, na_position="last")
        for a in ASSETS:
            tampil[a] = (tampil[a] * 100).map(lambda x: pct(x, 1))
        tabel(tampil[["Solusi"] + ASSETS + OBJEKTIF + ["Closeness"]].round(4), key=f"pareto_{m}")


    st.subheader("Pareto front NSGA-III dan solusi Entropy Weight-TOPSIS")
    if len(models) == 1:
        bagian_pareto(models[0])
    else:
        for tab, m in zip(st.tabs([LABEL_NAMA[m] for m in models]), models):
            with tab:
                bagian_pareto(m)

    st.subheader("Pareto front convexity")
    st.caption("Unsupported point = titik Pareto di bagian non-convex front, yang tidak bisa dicapai GA "
               "dengan linear scalarization berbobot λ.")
    konveks = saring(baca(PORTO_DIR, "pareto_konveksitas.csv"), **dasar)
    konveks = konveks[konveks["Model"].isin(models)].copy()
    konveks["Model"] = konveks["Model"].map(LABEL_NAMA)
    for c in [c for c in konveks.columns if c.endswith("(%)")]:
        konveks[c] = konveks[c].map(lambda x: pct(x, 1))
    tabel(konveks.drop(columns=list(dasar) + ([] if len(models) > 1 else ["Model"])).round(2).rename(columns={
        "Periode": "Periods", "Ukuran front": "Mean front size",
        "Tak terdukung (%)": "Unsupported points (mean)", "Tak terdukung maks (%)": "Unsupported points (max)",
        "Periode non-konveks (%)": "Non-convex front periods", "TOPSIS tak terdukung (%)": "Unsupported TOPSIS solution",
    }), key="konveks")

else:
    # Evaluasi per minggu kalender (median antar seed); halaman Backtest tetap ringkasan keseluruhan
    if not (PORTO_DIR / "portfolio_mingguan_median.csv").exists():
        ui.alert(title="Evaluasi mingguan belum tersedia",
                 description="Run ini belum mengekspor portfolio_mingguan_median.csv; jalankan portofolio_mvsk.ipynb versi terbaru.",
                 key="mingguan_kosong")
        st.stop()

    mingguan = baca(PORTO_DIR, "portfolio_mingguan_median.csv", parse_dates=["Tanggal"])
    algo_m = saring(mingguan[mingguan["Model"].isin(models)], **dasar).copy()
    algo_m["Strategi"] = [nama_seri(a, m) for a, m in zip(algo_m["Algoritma"], algo_m["Model"])]
    benchmark = ["Equal Weight (rebalance)", "Equal Weight (buy & hold)"]
    bench_m = saring(mingguan, Frekuensi=freq, Biaya=biaya)
    bench_m = bench_m[bench_m["Algoritma"].isin(benchmark)]
    data_m = pd.concat([algo_m, bench_m.assign(Strategi=bench_m["Algoritma"])])
    data_m["Max Drawdown"] *= 100
    urutan = [nama_seri(algo, m) for m in models for algo in ["NSGA-III", "GA"]] + benchmark

    st.subheader("Evaluasi kinerja per minggu")

    grafik = [("Return per minggu (%)", "Return Minggu (%)"), ("Cumulative return (%)", "Total Return (%)"),
              ("Sharpe ratio kumulatif", "Sharpe"), ("Max drawdown kumulatif (%)", "Max Drawdown")]
    for baris in [grafik[:2], grafik[2:]]:
        for col, (judul, kolom) in zip(st.columns(2), baris):
            with col:
                st.markdown(f"**{judul}**")
                grafik_garis(data_m.pivot_table(index="Minggu", columns="Strategi", values=kolom)[urutan],
                             judul, ".3f" if kolom == "Sharpe" else ".2f", sumbu_x="Minggu ke-", tipe_x="O", tinggi=300)

    st.subheader("Rincian per minggu")
    strategi = ui.select("Strategy", urutan, key="strategi_mingguan")
    rinci = data_m[data_m["Strategi"] == strategi].sort_values("Minggu")
    tabel(pd.DataFrame({
        "Minggu": rinci["Minggu"].astype(int),
        "Akhir minggu": rinci["Tanggal"].dt.date,
        "Return minggu": rinci["Return Minggu (%)"].map(pct),
        "Cumulative return": rinci["Total Return (%)"].map(pct),
        "Sharpe": rinci["Sharpe"].map(FORMAT_METRIK["Sharpe"]),
        "Sortino": rinci["Sortino"].map(FORMAT_METRIK["Sortino"]),
        "Max Drawdown": rinci["Max Drawdown"].map(pct),
        "Turnover": rinci["Turnover"].map(FORMAT_METRIK["Turnover"]),
        "Value": rinci["Final Value"].map(FORMAT_METRIK["Final Value"]),
    }), key="rinci_mingguan", max_height=520)
