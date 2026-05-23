from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)
app.jinja_env.auto_reload = True

df = pd.read_csv("dataset_credito_vivienda_colombia_final.csv", encoding="utf-8")
CIUDADES  = sorted(df["Ciudad"].unique().tolist())
CONTRATOS = sorted(df["Tipo_Contrato"].unique().tolist())

def apply_filters(data):
    p = request.args
    if p.get("ciudad"):   data = data[data["Ciudad"]           == p["ciudad"]]
    if p.get("estado"):   data = data[data["Estado_Solicitud"] == p["estado"]]
    if p.get("riesgo"):   data = data[data["Nivel_Riesgo"]     == p["riesgo"]]
    if p.get("contrato"): data = data[data["Tipo_Contrato"]    == p["contrato"]]
    if p.get("vivienda"): data = data[data["Tipo_Vivienda"]    == p["vivienda"]]
    if p.get("mora"):     data = data[data["Mora_Previa"]      == p["mora"]]
    return data

def pct_breakdown(data, group_col):
    grp    = data.groupby([group_col, "Estado_Solicitud"]).size().reset_index(name="n")
    totals = data.groupby(group_col).size().reset_index(name="total")
    grp    = grp.merge(totals, on=group_col)
    grp["pct"] = (grp["n"] / grp["total"] * 100).round(1)
    return grp.to_dict(orient="records")


@app.route("/")
def index():
    return render_template("index.html", ciudades=CIUDADES, contratos=CONTRATOS)

@app.route("/api/data")
def api_data():
    d = apply_filters(df)
    n = len(d)

    if n == 0:
        return jsonify({"n": 0, "kpis": {}, "charts": {}})

    apr = int((d["Estado_Solicitud"] == "Aprobado").sum())
    rec = int((d["Estado_Solicitud"] == "Rechazado").sum())
    est = int((d["Estado_Solicitud"] == "En estudio").sum())
    sub = int((d["Aplica_Subsidio"] == "Sí").sum())

    bins   = [0, 2, 4, 6, 8, 10, 15, 20, 9999]
    labels = ["<2M","2-4M","4-6M","6-8M","8-10M","10-15M","15-20M",">20M"]
    dc = d.copy()
    dc["ing_bin"] = pd.cut(dc["Ingresos_Mensuales_MCOP"], bins=bins, labels=labels, right=False)
    ing_hist = dc.groupby("ing_bin", observed=True).size().reset_index(name="n")

    def _rate(col, val, target):
        g = d[d[col] == val]
        if not len(g): return None
        return round(float((g["Estado_Solicitud"] == target).mean() * 100), 1)

    insights = {
        "hist_bueno_apr":   _rate("Historial_Crediticio", "Bueno",      "Aprobado"),
        "hist_malo_rec":    _rate("Historial_Crediticio", "Malo",       "Rechazado"),
        "mora_si_rec":      _rate("Mora_Previa",          "Sí",         "Rechazado"),
        "contrato_ind_apr": _rate("Tipo_Contrato",        "Indefinido", "Aprobado"),
        "contrato_inf_rec": _rate("Tipo_Contrato",        "Informal",   "Rechazado"),
    }

    return jsonify({
        "n": n,
        "kpis": {
            "total":          n,
            "aprobados_pct":  round(apr / n * 100, 1),
            "rechazados_pct": round(rec / n * 100, 1),
            "enestudio_pct":  round(est / n * 100, 1),
            "subsidio_pct":   round(sub / n * 100, 1),
            "ing_prom":       round(float(d["Ingresos_Mensuales_MCOP"].mean()), 1),
            "apr_abs": apr, "rec_abs": rec, "est_abs": est, "sub_abs": sub,
        },
        "insights": insights,
        "charts": {
            "estado":           d["Estado_Solicitud"].value_counts().to_dict(),
            "historial_estado": pct_breakdown(d, "Historial_Crediticio"),
            "contrato_estado":  pct_breakdown(d, "Tipo_Contrato"),
            "ciudad":           d["Ciudad"].value_counts().head(10).to_dict(),
            "segmento":         d["Segmento_Cliente"].value_counts().to_dict(),
            "ing_hist":         ing_hist.rename(columns={"ing_bin":"label"})[["label","n"]].to_dict(orient="records"),
        }
    })

@app.route("/intro")
def intro():
    return render_template("intro.html")

@app.route("/conclusiones")
def conclusiones():
    return render_template("conclusiones.html")

@app.route("/api/table")
def api_table():
    d     = apply_filters(df)
    page  = max(1, int(request.args.get("page", 1)))
    size  = min(50, max(5, int(request.args.get("size", 15))))
    total = len(d)
    start = (page - 1) * size
    cols  = ["ID_Solicitud", "Ciudad", "Edad", "Ingresos_Mensuales_MCOP",
             "Tipo_Contrato", "Historial_Crediticio", "Nivel_Riesgo",
             "Tipo_Vivienda", "Mora_Previa", "Estado_Solicitud"]
    rows  = d[cols].iloc[start:start + size].to_dict(orient="records")
    return jsonify({"total": total, "page": page, "size": size, "rows": rows})

if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, port=5000)
