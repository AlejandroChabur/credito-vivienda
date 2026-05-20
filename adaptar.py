"""
Script de adaptación: Credit Risk Dataset (Kaggle) → Crédito Vivienda Colombia
Proyecto: Perfilación de Clientes en Créditos de Vivienda - Universidad de Cundinamarca

INSTRUCCIONES:
1. Descarguen el dataset desde: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
2. Pongan el archivo 'credit_risk_dataset.csv' en la misma carpeta que este script
3. Ejecuten: python adaptar_dataset_kaggle.py
4. El resultado será: dataset_credito_vivienda_colombia_final.csv
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ─── 1. CARGAR DATASET ORIGINAL ───────────────────────────────────────────────
print("📂 Cargando dataset original de Kaggle...")
df = pd.read_csv("credit_risk_dataset.csv")
print(f"   Filas originales: {len(df)}")
print(f"   Columnas: {list(df.columns)}\n")

# ─── 2. LIMPIEZA BÁSICA (igual que hacen los notebooks de Kaggle) ─────────────
df = df[df["person_age"] < 80]           # eliminar edades atípicas
df = df[df["person_emp_length"] <= 40]   # empleo >40 años es error
df = df[df["person_income"] < 2_000_000] # ingresos atípicos
df = df.dropna()
print(f"   Filas tras limpieza: {len(df)}\n")

# ─── 3. CONVERSIÓN DE COLUMNAS EXISTENTES ─────────────────────────────────────
TRM = 4_200  # Tasa de cambio USD/COP aproximada 2025

# Edad
df["Edad"] = df["person_age"].astype(int)

# Ingresos: anual USD → mensual millones COP
df["Ingresos_Mensuales_MCOP"] = (df["person_income"] * TRM / 12 / 1_000_000).round(2)

# Antigüedad laboral
df["Antiguedad_Laboral_Anos"] = df["person_emp_length"].fillna(0).astype(int)

# Historial crediticio desde loan_grade
grado_a_historial = {"A": "Bueno", "B": "Bueno", "C": "Regular",
                     "D": "Regular", "E": "Malo", "F": "Malo", "G": "Inexistente"}
df["Historial_Crediticio"] = df["loan_grade"].map(grado_a_historial)

# Monto solicitado: USD → millones COP (como crédito de vivienda, escalado)
# El dataset original es crédito general, lo escalamos a rangos hipotecarios
df["Monto_Solicitado_MCOP"] = (df["loan_amnt"] * TRM / 1_000_000 * 8).round(1)
df["Monto_Solicitado_MCOP"] = df["Monto_Solicitado_MCOP"].clip(30, 900)

# Tasa de interés ajustada a Colombia (rangos hipotecarios 11-16% EA)
df["Tasa_Interes_EA_Pct"] = (df["loan_int_rate"].clip(11, 20)).round(1)

# Relación cuota/ingreso
df["Relacion_Cuota_Ingreso_Pct"] = (df["loan_percent_income"] * 100).round(1)

# Mora previa
df["Mora_Previa"] = df["cb_person_default_on_file"].map({"Y": "Sí", "N": "No"})

# Años de historial crediticio
df["Anos_Historial_Crediticio"] = df["cb_person_cred_hist_length"].astype(int)

# Vivienda actual
ownership_map = {"RENT": "Arrendatario", "OWN": "Propietario",
                 "MORTGAGE": "Hipotecado", "OTHER": "Otro"}
df["Vivienda_Actual"] = df["person_home_ownership"].map(ownership_map)

# Estado solicitud desde loan_status (0=pagó bien → Aprobado base)
def estado_desde_status(row):
    if row["loan_status"] == 0:
        if row["Historial_Crediticio"] == "Bueno" and row["Relacion_Cuota_Ingreso_Pct"] < 35:
            return np.random.choice(["Aprobado", "En estudio"], p=[0.92, 0.08])
        elif row["Historial_Crediticio"] == "Regular":
            return np.random.choice(["Aprobado", "En estudio", "Rechazado"], p=[0.60, 0.25, 0.15])
        else:
            return np.random.choice(["Aprobado", "En estudio", "Rechazado"], p=[0.50, 0.30, 0.20])
    else:  # loan_status == 1 (incumplimiento)
        return np.random.choice(["Rechazado", "En estudio"], p=[0.80, 0.20])

df["Estado_Solicitud"] = df.apply(estado_desde_status, axis=1)

# ─── 4. COLUMNAS NUEVAS — CONTEXTO COLOMBIANO ─────────────────────────────────

# Ciudad
ciudades = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga",
            "Pereira", "Manizales", "Ibagué", "Cartagena", "Cúcuta",
            "Villavicencio", "Santa Marta", "Pasto", "Montería", "Armenia"]
pesos_ciudad = [30, 18, 12, 8, 5, 4, 3, 3, 4, 3, 2, 2, 2, 2, 2]
df["Ciudad"] = np.random.choice(ciudades, size=len(df), p=[p/sum(pesos_ciudad) for p in pesos_ciudad])

# Tipo de contrato (correlacionado con ingresos y antigüedad)
def asignar_contrato(row):
    ing = row["Ingresos_Mensuales_MCOP"]
    ant = row["Antiguedad_Laboral_Anos"]
    if ing >= 6 and ant >= 3:
        return np.random.choice(["Indefinido", "Término fijo"], p=[0.80, 0.20])
    elif ing >= 3:
        return np.random.choice(["Indefinido", "Término fijo", "Independiente"], p=[0.50, 0.30, 0.20])
    elif ing >= 1.5:
        return np.random.choice(["Término fijo", "Independiente", "Informal"], p=[0.30, 0.45, 0.25])
    else:
        return np.random.choice(["Independiente", "Informal"], p=[0.40, 0.60])

df["Tipo_Contrato"] = df.apply(asignar_contrato, axis=1)

# Tipo de vivienda solicitada y valor
def asignar_vivienda(row):
    ing = row["Ingresos_Mensuales_MCOP"]
    monto = row["Monto_Solicitado_MCOP"]
    if ing <= 4.5 and monto <= 150:
        tipo = "VIS"
        valor = round(monto + np.random.uniform(10, 40), 1)
        valor = min(valor, 190)
    else:
        tipo = "No VIS"
        valor = round(monto + np.random.uniform(30, 150), 1)
    return pd.Series({"Tipo_Vivienda": tipo, "Valor_Vivienda_MCOP": valor})

df[["Tipo_Vivienda", "Valor_Vivienda_MCOP"]] = df.apply(asignar_vivienda, axis=1)

# Plazo del crédito
df["Plazo_Anos"] = np.random.choice([10, 15, 20, 25, 30], size=len(df), p=[0.05, 0.15, 0.35, 0.30, 0.15])

# Ahorro disponible (cuota inicial)
df["Ahorro_Disponible_MCOP"] = (df["Valor_Vivienda_MCOP"] - df["Monto_Solicitado_MCOP"]).clip(0).round(1)

# Nivel de endeudamiento (%)
df["Nivel_Endeudamiento_Pct"] = df["Relacion_Cuota_Ingreso_Pct"].apply(
    lambda x: round(min(x * np.random.uniform(1.5, 2.5), 85), 1))

# Subsidio Mi Casa Ya
df["Aplica_Subsidio"] = df.apply(
    lambda r: "Sí" if (r["Ingresos_Mensuales_MCOP"] <= 5.2 and
                       r["Valor_Vivienda_MCOP"] <= 190 and
                       r["Tipo_Vivienda"] == "VIS") else "No", axis=1)

# Nivel de riesgo
def nivel_riesgo(row):
    score = 0
    if row["Historial_Crediticio"] == "Bueno": score += 3
    elif row["Historial_Crediticio"] == "Regular": score += 1
    elif row["Historial_Crediticio"] == "Malo": score -= 2
    if row["Nivel_Endeudamiento_Pct"] < 30: score += 2
    elif row["Nivel_Endeudamiento_Pct"] < 55: score += 1
    else: score -= 2
    if row["Tipo_Contrato"] == "Indefinido": score += 2
    elif row["Tipo_Contrato"] == "Término fijo": score += 1
    elif row["Tipo_Contrato"] == "Informal": score -= 2
    if row["Mora_Previa"] == "Sí": score -= 3
    if row["Ingresos_Mensuales_MCOP"] >= 5: score += 1
    if score >= 5: return "Bajo"
    elif score >= 1: return "Medio"
    else: return "Alto"

df["Nivel_Riesgo"] = df.apply(nivel_riesgo, axis=1)

# Segmento cliente
def segmento(row):
    if row["Ingresos_Mensuales_MCOP"] >= 10 and row["Historial_Crediticio"] == "Bueno":
        return "Alta capacidad de pago"
    elif row["Aplica_Subsidio"] == "Sí":
        return "Posible subsidio Mi Casa Ya"
    elif row["Historial_Crediticio"] == "Inexistente":
        return "Historial crediticio limitado"
    elif row["Tipo_Contrato"] in ["Independiente", "Informal"]:
        return "Independiente o ingreso variable"
    elif row["Ingresos_Mensuales_MCOP"] >= 3 and row["Nivel_Endeudamiento_Pct"] < 40:
        return "Ingreso medio - bajo endeudamiento"
    else:
        return "Alto riesgo financiero"

df["Segmento_Cliente"] = df.apply(segmento, axis=1)

# ─── 5. SELECCIONAR Y ORDENAR COLUMNAS FINALES ────────────────────────────────
columnas_finales = [
    "Edad", "Ciudad", "Tipo_Contrato", "Antiguedad_Laboral_Anos",
    "Ingresos_Mensuales_MCOP", "Vivienda_Actual", "Historial_Crediticio",
    "Anos_Historial_Crediticio", "Mora_Previa", "Nivel_Endeudamiento_Pct",
    "Relacion_Cuota_Ingreso_Pct", "Ahorro_Disponible_MCOP", "Tipo_Vivienda",
    "Valor_Vivienda_MCOP", "Monto_Solicitado_MCOP", "Plazo_Anos",
    "Tasa_Interes_EA_Pct", "Aplica_Subsidio", "Nivel_Riesgo",
    "Estado_Solicitud", "Segmento_Cliente"
]

df_final = df[columnas_finales].reset_index(drop=True)
df_final.insert(0, "ID_Solicitud", [f"SOL-{i+1:05d}" for i in range(len(df_final))])

# ─── 6. EXPORTAR ──────────────────────────────────────────────────────────────
output = "dataset_credito_vivienda_colombia_final.csv"
df_final.to_csv(output, index=False, encoding="utf-8")

print("=" * 55)
print("✅ DATASET GENERADO EXITOSAMENTE")
print("=" * 55)
print(f"   Archivo:  {output}")
print(f"   Filas:    {len(df_final):,}")
print(f"   Columnas: {len(df_final.columns)}")
print(f"\n📊 Estado de solicitudes:")
for est, cnt in df_final["Estado_Solicitud"].value_counts().items():
    print(f"   {est}: {cnt:,} ({cnt/len(df_final)*100:.1f}%)")
print(f"\n🎯 Nivel de riesgo:")
for niv, cnt in df_final["Nivel_Riesgo"].value_counts().items():
    print(f"   {niv}: {cnt:,} ({cnt/len(df_final)*100:.1f}%)")
print(f"\n🏠 Subsidio Mi Casa Ya:")
for s, cnt in df_final["Aplica_Subsidio"].value_counts().items():
    print(f"   {s}: {cnt:,} ({cnt/len(df_final)*100:.1f}%)")
print(f"\n👥 Segmentos:")
for seg, cnt in df_final["Segmento_Cliente"].value_counts().items():
    print(f"   {seg}: {cnt:,}")
print(f"\n📁 Listo para importar en Power BI / Looker Studio / Excel")