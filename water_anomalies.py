# %% [markdown]
# # Аномалии потребления воды — тестовое задание, этап 2
#
# **Вход:** `pokazaniya.csv` (накопительные показания, м³), `pribory.csv` (тип/модель/месяц выпуска), `sobytiya.csv` (магнит / сброс / Холл).
#
# **Логика решения**
# 1. Чтение и проверка качества данных (период, дубли, пропуски дней).
# 2. Суточное потребление: Δ = показание(d) − показание(предыдущей даты прибора). Если между показаниями пропущены дни, Δ делится на число дней (`rate`, м³/сут).
#    Показания округляются до 0,001 м³ (1 л — разрешение счётчика): в данных есть усреднённые значения с 13–16 знаками после запятой, и без округления появляются «отрицательные» разности порядка −1e-13.
# 3. Отделение сбоев данных от реального расхода. Настоящий расход необратим: если после скачка показание возвращается назад, это сбой передачи, а не вода.
#    Сбойные интервалы (откат, выше физического предела счётчика, «вернувшийся» скачок, скачок кратный 327,68 м³)
#    исключаются из статистик потребления (пункты 2, 4–7) и разбираются отдельно (пункты 1 и 3).
# 4. Пункты 1–8 задания. Все пороги вынесены в словарь `P`.
# 5. Итог: приоритетный список приборов на проверку и Excel со всеми списками.
#
# Запуск: `python water_anomalies.py [папка_с_csv]`. Если CSV не найдены (например, в Colab), архив скачивается с Google Drive автоматически.

# %%
import glob
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

IN_NB = "ipykernel" in sys.modules


def show(*objs):
    """Вывод и в ноутбуке (display), и в консоли (print)."""
    for o in objs:
        if isinstance(o, dict):
            o = pd.Series({str(k): str(val) for k, val in o.items()}, dtype=object).to_frame("значение")
        if IN_NB:
            from IPython.display import display
            display(o)
        else:
            print(o, "\n")

GDRIVE_FILE_ID = "1D364uUaiNrGZPtkYsX1EreuiBe3WFWcs"  # архив из формы задания
OUT_DIR = os.environ.get("OUT_DIR", "results")


def find_data_dir(base):
    hits = glob.glob(os.path.join(base, "**", "pokazaniya.csv"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


cli_dirs = [a for a in sys.argv[1:] if os.path.isdir(a)]  # в Jupyter argv содержит служебные флаги
DATA_DIR = find_data_dir(cli_dirs[0] if cli_dirs else os.environ.get("DATA_DIR", "."))
if DATA_DIR is None:  # Colab / пустая папка: скачиваем архив задания
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    try:
        gdown.download(id=GDRIVE_FILE_ID, output="test_etap2.zip", quiet=False)
        with zipfile.ZipFile("test_etap2.zip") as z:
            z.extractall("data")
    except Exception as e:
        raise SystemExit(f"Не удалось скачать архив ({e}). Положите pokazaniya.csv, pribory.csv, sobytiya.csv "
                         "в рабочую папку (в Colab — панель «Файлы» слева) и перезапустите.")
    DATA_DIR = find_data_dir("data")
print("Данные:", DATA_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Параметры (все пороги в одном месте)

# %%
P = dict(
    ROUND=3,              # округление показаний до 0,001 м³ = 1 л
    ZERO=0.002,           # |Δ| ≤ 2 л/сут считаем «нулём» (дрожание младшего разряда / усреднение пакетов)
    APT_MAX=10.0,         # > 10 м³/сут — нереально для квартиры (≈ 7 л/мин круглые сутки)
    QMAX_DAY={"D15": 75.0, "D20": 120.0},  # физ. предел: Q4 × 24 ч (DN15: 3,125 м³/ч; DN20: 5 м³/ч)
    BIG_JUMP=10.0,        # скачки ≥ 10 м³ проверяем «тестом возврата»
    RETURN_DAYS=60,       # окно, в котором ищем возврат показания назад
    BIT=327.68,           # 2^15 × 0,01 м³ — квант битовых сбоев
    SPIKE_ABS=2.0,        # всплеск, абсолютный порог, м³/сут
    SPIKE_REL=10.0,       # всплеск, относительный порог: в N раз выше обычного расхода прибора
    SPIKE_REL_MIN=0.5,    # ... и при этом не меньше 0,5 м³/сут (чтобы 0,01 → 0,1 не считать всплеском)
    BASE_MIN_DAYS=20,     # «обычный расход» = медиана ненулевых суток, если их ≥ 20
    LEAK_DAYS=60,         # утечка: ≥ 60 суток подряд без нулевых суток
    LEAK_MAX_GAP=3,       # ... пропуск связи до 2 дней не разрывает серию, если расход за пропуск > 0
    LEAK_FLOOR=0.1,       # ... и минимум суточного расхода в этом окне ≥ 0,1 м³ (постоянный фон ≥ 100 л/сут)
    ZERO_DAYS=30,         # длительный ноль: ≥ 30 суток подряд Δ ≈ 0
    ZERO_COVER=0.5,       # ... и прибор выходил на связь хотя бы в половину этих дней
    STEP_RATIO=3.0,       # ступенька: уровень изменился в ≥ 3 раза
    STEP_MIN_LEVEL=0.1,   # ... больший из уровней ≥ 0,1 м³/сут
    STEP_MIN_DAYS=30,     # ... каждый участок ≥ 30 суток
    STEP_CAP=2.0,         # для оценки уровня сутки обрезаются сверху 2 м³ (единичный всплеск не делает «ступеньку»)
    STEP_SMOOTH=0.01,     # добавка к уровням при расчёте отношения (устойчивость к нулю)
    STEP_P=1e-3,          # значимость различия участков (тест Манна — Уитни)
)

# %% [markdown]
# ## 1. Загрузка и качество данных

# %%
def read(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), sep=";", encoding="utf-8-sig", dtype={"anon_id": str})


pok = read("pokazaniya.csv").rename(columns={"data": "date", "pokazanie": "reading", "paketov_za_sutki": "packets"})
dev = read("pribory.csv")
ev = read("sobytiya.csv").rename(columns={"data": "date", "sobytie": "event"})
pok["date"] = pd.to_datetime(pok["date"])
ev["date"] = pd.to_datetime(ev["date"])
dev["dn"] = dev["model"].str.extract(r"(D\d+)")[0].fillna("D15")
dev["release"] = pd.to_datetime(dev["mesyac_vypuska"] + "-01")

pok = pok.sort_values(["anon_id", "date"]).reset_index(drop=True)
span = pok.groupby("anon_id")["date"].agg(["min", "max", "size"])
span["days"] = (span["max"] - span["min"]).dt.days + 1
quality = {
    "строк показаний": len(pok),
    "приборов": pok["anon_id"].nunique(),
    "период": f"{pok['date'].min():%Y-%m-%d} — {pok['date'].max():%Y-%m-%d}",
    "дубли (прибор, дата)": int(pok.duplicated(["anon_id", "date"]).sum()),
    "приборов с пропусками дней": int((span["days"] > span["size"]).sum()),
    "показаний с > 4 знаками после запятой (усреднённые)": int(
        (pok["reading"].astype(str).str.split(".").str[1].str.len() > 4).sum()),
    "приборов без паспорта": int((~pok["anon_id"].drop_duplicates().isin(dev["anon_id"])).sum()),
    "событий": ev["event"].value_counts().to_dict(),
    "последний день": f"{pok['date'].max():%Y-%m-%d}: показаний {int((pok['date'] == pok['date'].max()).sum())} (неполные сутки)",
}
show(quality)

# %% [markdown]
# ## 2. Суточное потребление
# Одна строка = интервал между двумя соседними показаниями прибора. Расход относим к дате второго показания.
# `gap` — число дней между показаниями, `delta` — разность показаний, `rate = delta / gap` — средний суточный расход.

# %%
pok["delta_raw"] = pok.groupby("anon_id")["reading"].diff()  # без округления — только для демонстрации шума
pok["reading"] = pok["reading"].round(P["ROUND"])
g = pok.groupby("anon_id")
pok["prev_date"] = g["date"].shift()
pok["prev_reading"] = g["reading"].shift()
pok["gap"] = (pok["date"] - pok["prev_date"]).dt.days
pok["delta"] = (pok["reading"] - pok["prev_reading"]).round(P["ROUND"])
pok["rate"] = pok["delta"] / pok["gap"]

iv = pok[pok["gap"].notna()].merge(dev[["anon_id", "tip", "bs", "dn"]], on="anon_id", how="left").reset_index(drop=True)
iv["qmax_day"] = iv["dn"].map(P["QMAX_DAY"]).fillna(75.0)
print(f"Интервалов: {len(iv):,}; из них соседние сутки (gap = 1): {(iv['gap'] == 1).mean():.1%}")
print("Шум округления: Δ < 0 до округления —", int((iv["delta_raw"] < 0).sum()),
      "интервалов, после округления —", int((iv["delta"] < 0).sum()))
show(iv[["delta", "rate"]].describe(percentiles=[.5, .9, .99, .999]).T)

# %% [markdown]
# ### 2.1. Тест возврата: сбой данных или реальный расход
# Для каждого скачка ≥ 10 м³ (или > 10 м³/сут) смотрим, вернулось ли показание назад в течение 60 дней
# (ниже, чем `показание − 0,5·скачок`). Вернулось — сбой (ложное показание). Для провалов вниз — симметрично.
# Дополнительно проверяем кратность скачка 327,68 м³ = 2^15 × 0,01 м³ (переключение бита в регистре счётчика).

# %%
def returns_back(df_dev, idx_rows, up=True):
    """Для строк idx_rows прибора: вернулось ли показание в течение RETURN_DAYS."""
    out = {}
    dates, readings = df_dev["date"].values, df_dev["reading"].values
    pos = {ix: k for k, ix in enumerate(df_dev.index)}
    for ix in idx_rows:
        k = pos[ix]
        jump = df_dev.at[ix, "delta"]
        lim = dates[k] + np.timedelta64(P["RETURN_DAYS"], "D")
        fut = readings[k + 1:][dates[k + 1:] <= lim]
        if up:
            # сбой, если показание потом вернулось вниз ИЛИ уже было на этом уровне раньше (восстановление после провала)
            past = readings[:k - 1][dates[:k - 1] >= dates[k] - np.timedelta64(P["RETURN_DAYS"], "D")] if k > 1 else readings[:0]
            out[ix] = bool((fut < readings[k] - 0.5 * jump).any() or (past >= readings[k] - 0.5 * jump).any())
        else:
            out[ix] = bool((fut > readings[k] - 0.5 * jump).any())  # jump < 0
    return out


pk = pok.reset_index(drop=True)
big_up = pk.index[(pk["delta"] >= P["BIG_JUMP"]) | (pk["rate"] > P["APT_MAX"])]
big_dn = pk.index[pk["delta"] <= -P["BIG_JUMP"]]
ret_up, ret_dn = {}, {}
for aid, sub in pk.loc[pk["anon_id"].isin(pk.loc[big_up.union(big_dn), "anon_id"])].groupby("anon_id"):
    ret_up.update(returns_back(sub, sub.index.intersection(big_up), up=True))
    ret_dn.update(returns_back(sub, sub.index.intersection(big_dn), up=False))
pk["returned"] = False
_ret = {**ret_up, **ret_dn}
pk.loc[list(_ret), "returned"] = list(_ret.values())
pk["k_bit"] = pk["delta"] / P["BIT"]
pk["bit_multiple"] = (pk["delta"].abs() >= 100) & ((pk["k_bit"] - pk["k_bit"].round()).abs() < 0.02)

iv = iv.merge(pk[["anon_id", "date", "returned", "k_bit", "bit_multiple"]], on=["anon_id", "date"], how="left")
iv["returned"] = iv["returned"].astype(bool)

# Валидный интервал для статистик потребления: не откат, не выше физ. предела счётчика, не «вернувшийся» скачок,
# не кратный 327,68 м³ битовый сбой. Экстремальный, но физически возможный расход (10–75 м³/сут без возврата) остаётся.
iv["valid"] = (iv["delta"] >= -P["ZERO"]) & (iv["rate"] <= iv["qmax_day"]) & ~iv["returned"] & ~iv["bit_multiple"]
iv["cons"] = np.where(iv["valid"], iv["delta"].clip(lower=0), np.nan)  # м³ за интервал
iv["cons_rate"] = iv["cons"] / iv["gap"]                                 # м³/сут
print("Валидных интервалов:", f"{iv['valid'].mean():.2%}", "| исключено как сбой:", int((~iv["valid"]).sum()))

# Суточное потребление (итоговая таблица): только валидные интервалы; cons_rate — м³/сут
daily = iv.loc[iv["valid"], ["anon_id", "date", "gap", "reading", "cons", "cons_rate", "packets"]]
show(daily.head())

# %% [markdown]
# ## Пункт 1. Отрицательный расход (счётчик «пошёл назад»)
# Критерий: Δ < 0 после округления до 0,001 м³. Делим по величине и проверяем,
# был ли перед откатом ложный скачок вверх (возврат к истинному значению), кратность 327,68 м³ и событие «сброс» ±1 день.

# %%
neg = iv[iv["delta"] < 0].copy()
neg["class"] = pd.cut(-neg["delta"], [0, P["ZERO"], 1, P["BIG_JUMP"], np.inf],
                      labels=["шум ≤ 2 л", "мелкий откат 2 л–1 м³", "откат 1–10 м³", "крупный откат ≥ 10 м³"])

# был ли в предыдущие 60 дней скачок вверх, который этот откат компенсирует
ups = iv[(iv["delta"] >= P["BIG_JUMP"])][["anon_id", "date", "delta"]].rename(columns={"date": "up_date", "delta": "up"})
m = neg[neg["delta"] <= -P["BIG_JUMP"]].reset_index().merge(ups, on="anon_id")
m = m[(m["up_date"] < m["date"]) & (m["date"] - m["up_date"] <= pd.Timedelta(days=P["RETURN_DAYS"]))
      & ((m["up"] + m["delta"]).abs() <= 0.3 * m["delta"].abs() + 1)]
neg["after_false_jump"] = neg.index.isin(m["index"])
resets = ev[ev["event"] == "сброс"][["anon_id", "date"]].drop_duplicates()
rs = pd.concat([resets.assign(date=resets["date"] + pd.Timedelta(days=k)) for k in (-1, 0, 1)]).drop_duplicates()
neg["reset_event"] = neg.set_index(["anon_id", "date"]).index.isin(rs.set_index(["anon_id", "date"]).index)

neg_summary = neg.groupby("class", observed=False).agg(
    интервалов=("delta", "size"), приборов=("anon_id", "nunique"),
    после_ложного_скачка=("after_false_jump", "sum"), кратно_327_68=("bit_multiple", "sum"),
    со_сбросом=("reset_event", "sum"))
print("Всего Δ < 0:", len(neg), "интервалов,", neg["anon_id"].nunique(), "приборов")
print("Δ < −2 л (без шума):", int((neg["delta"] < -P["ZERO"]).sum()), "интервалов,",
      neg.loc[neg["delta"] < -P["ZERO"], "anon_id"].nunique(), "приборов")
show(neg_summary)

# %%
neg_dev = (neg[neg["delta"] < -P["ZERO"]].groupby("anon_id")
           .agg(откатов=("delta", "size"), max_откат_м3=("delta", "min"),
                после_ложного_скачка=("after_false_jump", "sum"), кратно_327_68=("bit_multiple", "sum"),
                сброс=("reset_event", "sum"), tip=("tip", "first"), bs=("bs", "first"))
           .sort_values("max_откат_м3"))
big_neg = neg[neg["delta"] <= -P["BIG_JUMP"]]
print("Крупные откаты: сбой подтверждён (компенсирует скачок или кратен 327,68):",
      int((big_neg["after_false_jump"] | big_neg["bit_multiple"]).sum()), "из", len(big_neg),
      f"| на приборах AQUA2: {(big_neg['tip'] == 'AQUA2').mean():.0%}")
show(neg_dev.head(15))

# %% [markdown]
# ## Пункт 3. Нереальные значения (физически невозможный расход)
# Порог 1 — физический предел счётчика: Q4 × 24 ч = 75 м³/сут для DN15 (120 для DN20).
# Порог 2 — нереально для квартиры: > 10 м³/сут.
# Дополнительно: возврат показания назад (сбой), кратность 327,68 м³ и показания ≥ 10 000 м³.

# %%
unreal = iv[(iv["rate"] > P["APT_MAX"])].copy()
unreal["level"] = np.where(unreal["rate"] > unreal["qmax_day"], "> Q4×24 (физически невозможно)", "10 м³/сут – Q4×24")
unreal_summary = unreal.groupby("level").agg(интервалов=("rate", "size"), приборов=("anon_id", "nunique"),
                                             с_возвратом=("returned", "sum"), кратно_327_68=("bit_multiple", "sum"))
unreal_dev = (unreal.groupby("anon_id")
              .agg(интервалов=("rate", "size"), max_м3_сут=("rate", "max"), с_возвратом=("returned", "sum"),
                   кратно_327_68=("bit_multiple", "sum"), tip=("tip", "first"), bs=("bs", "first"))
              .sort_values("max_м3_сут", ascending=False))
unreal_dev = unreal_dev.join(unreal.groupby("anon_id")["qmax_day"].first())
unreal_dev["тип"] = np.select(
    [unreal_dev["max_м3_сут"] > unreal_dev["qmax_day"], unreal_dev["с_возвратом"] > 0],
    ["сбой данных: выше физ. предела счётчика", "сбой данных: показание вернулось"],
    "реальный экстремальный расход (порыв / нежилой) — проверить")
huge = pok.groupby("anon_id")["reading"].max()
print("Приборов с расходом > 10 м³/сут:", unreal["anon_id"].nunique(),
      "| из них > Q4×24:", unreal.loc[unreal["rate"] > unreal["qmax_day"], "anon_id"].nunique())
show(unreal_dev["тип"].value_counts())
print("Показания ≥ 10 000 м³:", huge[huge >= 1e4].round(1).to_dict())
display_cols = ["интервалов", "max_м3_сут", "с_возвратом", "кратно_327_68", "тип", "tip", "bs"]
show(unreal_summary, unreal_dev[display_cols].head(15))

# %% [markdown]
# ## Пункт 2. Всплески (только валидные интервалы: сбои из п. 1 и 3 исключены)
# Абсолютный порог: ≥ 2 м³/сут. Относительный: ≥ 10 × «обычного» расхода прибора (медиана ненулевых суток, если их ≥ 20) и ≥ 0,5 м³/сут.

# %%
v = iv[iv["valid"]].copy()
base = (v[(v["gap"] == 1) & (v["cons"] > P["ZERO"])].groupby("anon_id")["cons"]
        .agg(base="median", n_pos="size"))
v = v.join(base, on="anon_id")
v["ratio"] = v["cons_rate"] / v["base"]
v["spike_abs"] = v["cons_rate"] >= P["SPIKE_ABS"]
v["spike_rel"] = (v["n_pos"] >= P["BASE_MIN_DAYS"]) & (v["ratio"] >= P["SPIKE_REL"]) & (v["cons_rate"] >= P["SPIKE_REL_MIN"])
pct = v.loc[(v["gap"] == 1) & (v["cons"] > P["ZERO"]), "cons"].quantile([.5, .99, .995, .999])
print("Ненулевые сутки, перцентили (м³):", pct.round(3).to_dict())
print(f"Порог 2 м³/сут ≈ {(v.loc[(v['gap'] == 1) & (v['cons'] > P['ZERO']), 'cons'] < 2).mean():.2%} перцентиль ненулевых суток")

spk = v[v["spike_abs"] | v["spike_rel"]]
spike_dev = (spk.groupby("anon_id")
             .agg(всплесков_абс=("spike_abs", "sum"), всплесков_отн=("spike_rel", "sum"),
                  max_м3_сут=("cons_rate", "max"), обычный_м3=("base", "first"), max_кратность=("ratio", "max"),
                  дата_max=("date", lambda s: s.loc[spk.loc[s.index, "cons_rate"].idxmax()].date()),
                  tip=("tip", "first"), bs=("bs", "first"))
             .sort_values("max_м3_сут", ascending=False))
spike_summary = {
    "абсолютный ≥ 2 м³/сут: приборов": int((spike_dev["всплесков_абс"] > 0).sum()),
    "абсолютный: интервалов": int(v["spike_abs"].sum()),
    "относительный ≥ 10× и ≥ 0,5 м³/сут: приборов": int((spike_dev["всплесков_отн"] > 0).sum()),
    "относительный: интервалов": int(v["spike_rel"].sum()),
    "оба критерия: приборов": int(((spike_dev["всплесков_абс"] > 0) & (spike_dev["всплесков_отн"] > 0)).sum()),
}
show(spike_summary, spike_dev.head(10), spike_dev.sort_values("max_кратность", ascending=False).head(10))

# %% [markdown]
# ### Вспомогательная функция: серии подряд идущих интервалов

# %%
def runs(frame, flag, max_gap=None):
    """Серии подряд идущих интервалов прибора, где flag = True (и gap ≤ max_gap). Возвращает таблицу серий."""
    f = frame[["anon_id", "date", "prev_date", "gap", "cons", "cons_rate", "packets"]].copy()
    f["ok"] = flag.values & ((f["gap"] <= max_gap).values if max_gap else True)
    brk = (~f["ok"]) | (f["anon_id"] != f["anon_id"].shift())
    f["run"] = brk.cumsum()
    f = f[f["ok"]]
    r = f.groupby("run").agg(anon_id=("anon_id", "first"), start=("prev_date", "min"), end=("date", "max"),
                             days=("gap", "sum"), readings=("date", "size"), packets=("packets", "sum"),
                             cons_sum=("cons", "sum"), cons_min=("cons_rate", "min"))
    r["cons_mean"] = r["cons_sum"] / r["days"]
    return r.reset_index(drop=True)


# Работаем по всем интервалам; сбойный интервал разрывает серию
ivs = iv.sort_values(["anon_id", "date"]).reset_index(drop=True)
last_date = ivs.groupby("anon_id")["date"].max()
first_date = ivs.groupby("anon_id")["prev_date"].min()

# %% [markdown]
# ## Пункт 4. Утечка — прибор «никогда не отдыхает»
# Серия подряд идущих суток, в каждые из которых расход > 2 л (пропуск связи до 2 дней не разрывает серию, если расход за пропуск > 0;
# строгий вариант — только подтверждённые соседние сутки — дан как нижняя оценка).
# Кандидат: серия ≥ 60 суток. Сильный признак: в этой серии даже минимальные сутки ≥ 0,1 м³ (постоянный фон ≥ 100 л/сут).
# Ограничение: по суточным данным утечку нельзя отличить от большой семьи; окончательно — ночной расход по часовым пакетам.

# %%
pos_flag = ivs["valid"] & (ivs["cons_rate"] > P["ZERO"])
pos_runs = runs(ivs, pos_flag, max_gap=P["LEAK_MAX_GAP"])
pos_runs_strict = runs(ivs, pos_flag, max_gap=1)
leak_c = pos_runs[pos_runs["days"] >= P["LEAK_DAYS"]].sort_values("days", ascending=False).drop_duplicates("anon_id")
best_floor = (pos_runs[pos_runs["days"] >= P["LEAK_DAYS"]].sort_values("cons_min", ascending=False)
              .drop_duplicates("anon_id"))
leak_strong = best_floor[best_floor["cons_min"] >= P["LEAK_FLOOR"]].sort_values("cons_min", ascending=False)
leak_summary = {f"серия ≥ {t} сут без нулей: приборов (строго gap=1 / с пропусками ≤ 2 дн)":
                f"{int((pos_runs_strict.groupby('anon_id')['days'].max() >= t).sum())} / "
                f"{int((pos_runs.groupby('anon_id')['days'].max() >= t).sum())}"
                for t in (30, 60, 90, 180)}
leak_summary["сильный признак (≥ 60 сут и минимум ≥ 0,1 м³/сут): приборов"] = len(leak_strong)
leak_summary["... из них минимум ≥ 0,3 м³/сут"] = int((leak_strong["cons_min"] >= 0.3).sum())
show(leak_summary, leak_strong.head(15))

# %% [markdown]
# ## Пункт 5. Нулевое потребление при идущих пакетах
# Серия интервалов с |Δ| ≤ 2 л (пропуски дней допускаются: если показание не изменилось за 5 дней, расход за все 5 дней = 0).
# Критерий: ≥ 30 суток и показания приходили хотя бы в половину дней серии (прибор на связи).
# Классы: «весь период» — ни одного литра за всё наблюдение; «остановился» — до серии был расход, серия длится до конца данных
# (застрявший счётчик / вмешательство / выезд); «временно» — расход потом возобновился (отъезд).

# %%
zero_flag = ivs["valid"] & (ivs["delta"].abs() <= P["ZERO"])
z = runs(ivs, zero_flag)
z["coverage"] = z["readings"] / z["days"]
z = z[(z["days"] >= P["ZERO_DAYS"]) & (z["coverage"] >= P["ZERO_COVER"])]
valid_by_dev = {aid: s for aid, s in ivs[ivs["valid"]].groupby("anon_id")[["date", "cons"]]}


def classify_zero(row):
    sub = valid_by_dev[row.anon_id]
    before = sub.loc[sub["date"] <= row.start, "cons"].sum()
    after = sub.loc[sub["date"] > row.end, "cons"].sum()
    if before <= 0.01 and after <= 0.01:
        return "весь период ноль"
    if row.end >= last_date[row.anon_id] and before > 0.01:
        return "остановился и не возобновился"
    return "временный простой"


z["class"] = z.apply(classify_zero, axis=1)
zero_dev = z.sort_values("days", ascending=False).drop_duplicates("anon_id")
zero_summary = {f"ноль ≥ {t} сут при связи: приборов": int((zero_dev["days"] >= t).sum()) for t in (30, 60, 90, 180)}
zero_summary.update(zero_dev["class"].value_counts().rename(lambda s: f"класс: {s}").to_dict())
stopped = zero_dev[zero_dev["class"] == "остановился и не возобновился"].sort_values("days", ascending=False)
show(zero_summary, stopped.head(10))

# %% [markdown]
# ## Пункт 6. Ступенька — резкое устойчивое изменение уровня
# Метод A (как в задании): средний расход во второй половине периода прибора vs в первой.
# Метод B (точнее): точка смены уровня по CUSUM (максимум накопленного отклонения от среднего), участки ≥ 30 суток.
# Условия: отношение уровней ≥ 3, больший уровень ≥ 0,1 м³/сут, различие значимо (Манн — Уитни, p < 0,001).
# Сутки обрезаются сверху 2 м³, чтобы единичный всплеск не создавал «ступеньку».

# %%
def step_for_device(sub):
    sub = sub.sort_values("date")
    x = sub["cons_rate"].clip(upper=P["STEP_CAP"]).values
    w = sub["gap"].values.astype(float)
    if w.sum() < 2 * P["STEP_MIN_DAYS"]:
        return None
    res = {}
    # A: половины периода
    mid = sub["prev_date"].min() + (sub["date"].max() - sub["prev_date"].min()) / 2
    h1, h2 = (sub["date"] <= mid).values, (sub["date"] > mid).values
    if w[h1].sum() >= P["STEP_MIN_DAYS"] and w[h2].sum() >= P["STEP_MIN_DAYS"]:
        a1, a2 = np.average(x[h1], weights=w[h1]), np.average(x[h2], weights=w[h2])
        res.update(half1=a1, half2=a2)
    # B: CUSUM
    mu = np.average(x, weights=w)
    cs = np.cumsum(w * (x - mu))
    cw = np.cumsum(w)
    okk = (cw >= P["STEP_MIN_DAYS"]) & (cw <= w.sum() - P["STEP_MIN_DAYS"])
    if okk.any():
        k = np.where(okk, np.abs(cs), -1).argmax()
        l1 = np.average(x[:k + 1], weights=w[:k + 1])
        l2 = np.average(x[k + 1:], weights=w[k + 1:])
        p = stats.mannwhitneyu(x[:k + 1], x[k + 1:]).pvalue if min(k + 1, len(x) - k - 1) >= 5 else 1.0
        res.update(cp_date=sub["date"].iloc[k].date(), lvl1=l1, lvl2=l2, p=p)
    return res


_res = {aid: step_for_device(s) for aid, s in v.groupby("anon_id")}
steps = pd.DataFrame({k: r for k, r in _res.items() if r}).T
sm = P["STEP_SMOOTH"]
for a, b, name in (("half1", "half2", "A"), ("lvl1", "lvl2", "B")):
    s1, s2 = steps[a].astype(float), steps[b].astype(float)
    steps[f"ratio_{name}"] = (s2 + sm) / (s1 + sm)
    cond = (np.maximum(s1, s2) >= P["STEP_MIN_LEVEL"]) & (
        (steps[f"ratio_{name}"] >= P["STEP_RATIO"]) | (steps[f"ratio_{name}"] <= 1 / P["STEP_RATIO"]))
    if name == "B":
        cond &= steps["p"].astype(float) < P["STEP_P"]
    steps[f"step_{name}"] = np.where(cond, np.where(steps[f"ratio_{name}"] > 1, "вверх", "вниз"), "")
step_summary = {
    "A (половины): вверх": int((steps["step_A"] == "вверх").sum()),
    "A (половины): вниз": int((steps["step_A"] == "вниз").sum()),
    "B (CUSUM): вверх": int((steps["step_B"] == "вверх").sum()),
    "B (CUSUM): вниз": int((steps["step_B"] == "вниз").sum()),
    "B: вниз почти до нуля (новый уровень < 0,01)": int(((steps["step_B"] == "вниз") & (steps["lvl2"].astype(float) < 0.01)).sum()),
    "A и B одновременно": int(((steps["step_A"] != "") & (steps["step_B"] != "")).sum()),
}
step_dev = steps[steps["step_B"] != ""].copy()
step_dev["кратность"] = np.where(step_dev["ratio_B"] >= 1, step_dev["ratio_B"], 1 / step_dev["ratio_B"])
l1, l2 = step_dev["lvl1"].astype(float), step_dev["lvl2"].astype(float)
step_dev["подтип"] = np.select(
    [(step_dev["step_B"] == "вверх") & (l1 < 0.01), step_dev["step_B"] == "вверх", l2 < 0.01],
    ["вверх с нуля (заселение / начало пользования)", "вверх с ненулевого уровня (утечка? больше жильцов?)",
     "вниз до нуля (выезд / остановка счётчика)"], "вниз до ненулевого (недоучёт? вмешательство?)")
step_summary.update(step_dev["подтип"].value_counts().rename(lambda s: f"B, {s}").to_dict())
step_dev = step_dev.sort_values("кратность", ascending=False)
show(step_summary, step_dev[["cp_date", "lvl1", "lvl2", "кратность", "p", "подтип", "half1", "half2", "step_A"]]
     .sort_values(["подтип", "кратность"], ascending=[True, False]).groupby("подтип").head(4))

# %% [markdown]
# ## Пункт 7. Связь с событиями «магнит»
# Проверяем на нескольких уровнях:
# (a) где и когда вообще бывают события (тип прибора, время, частота пакетов);
# (b) сутки с магнитом vs сутки без магнита **того же прибора в том же месяце** (контроль сезонности), парный тест Уилкоксона;
# (c) расход за 30 дней до и после первого «магнита»;
# (d) доля приборов с магнитом среди аномалий vs общая доля.

# %%
mag = ev[ev["event"] == "магнит"].groupby(["anon_id", "date"]).size().rename("n_mag").reset_index()
mag_dev = set(mag["anon_id"])
dev["has_magnet"] = dev["anon_id"].isin(mag_dev)
print("Приборов с «магнитом»:", len(mag_dev), f"({len(mag_dev) / len(dev):.1%}); событий:", int(mag["n_mag"].sum()))
by_tip = dev.groupby("tip")["has_magnet"].agg(["mean", "sum", "size"]).rename(columns={"mean": "доля", "sum": "с_магнитом", "size": "всего"})
by_month = ev.groupby([ev["date"].dt.to_period("M"), "event"]).size().unstack(fill_value=0)
packets_month = pok.groupby(pok["date"].dt.to_period("M"))["packets"].mean().round(1).rename("пакетов_в_сутки_ср")
show(by_tip, pd.concat([by_month, packets_month], axis=1))

# %%
d1 = v[v["gap"] == 1].merge(mag, on=["anon_id", "date"], how="left")
d1["magnet_day"] = d1["n_mag"].notna()
d1 = d1[d1["anon_id"].isin(mag_dev)]
d1["month"] = d1["date"].dt.to_period("M")
# только месяцы, где у прибора есть и дни с магнитом, и без
mm = d1.groupby(["anon_id", "month"])["magnet_day"].agg(["any", "all"])
keep = mm[mm["any"] & ~mm["all"]].index
d1m = d1.set_index(["anon_id", "month"]).loc[keep].reset_index()
def zero_share(s):
    return (s <= P["ZERO"]).mean()


pair = d1m.groupby(["anon_id", "magnet_day"])["cons"].agg(["mean", "median", "size", zero_share]).unstack()
pair.columns = [f"{a}_{'mag' if b else 'nomag'}" for a, b in pair.columns]
pair = pair[(pair["size_mag"] >= 3) & (pair["size_nomag"] >= 3)]
w_test = stats.wilcoxon(pair["mean_mag"], pair["mean_nomag"])
magnet_day_summary = {
    "приборов в парном сравнении": len(pair),
    "медиана расхода, сутки с магнитом (м³)": round(pair["median_mag"].median(), 4),
    "медиана расхода, сутки без магнита (м³)": round(pair["median_nomag"].median(), 4),
    "доля нулевых суток с магнитом": round(pair["zero_share_mag"].mean(), 3),
    "доля нулевых суток без магнита": round(pair["zero_share_nomag"].mean(), 3),
    "приборов, у которых в дни магнита расход ниже": f"{(pair['mean_mag'] < pair['mean_nomag']).mean():.1%}",
    "Wilcoxon p-value": f"{w_test.pvalue:.2e}",
}
# доза — отклик: число срабатываний в сутки
d1m["n_bin"] = pd.cut(d1m["n_mag"].fillna(0), [-1, 0, 1, 4, 9, 1000], labels=["0", "1", "2–4", "5–9", "10+"])
dose = d1m.groupby("n_bin", observed=False)["cons"].agg(сутки="size", медиана="median",
                                                        доля_нулей=lambda s: (s <= P["ZERO"]).mean())
# Проверка на парадокс Симпсона: сутки с ≥ 5 срабатываниями vs сутки без магнита ТОГО ЖЕ прибора в те же месяцы
hi_rows = []
for aid, sub in d1m.groupby("anon_id"):
    h = sub[sub["n_mag"].fillna(0) >= 5]
    o = sub[sub["n_mag"].isna() & sub["month"].isin(set(h["month"]))]
    if len(h) >= 3 and len(o) >= 3:
        hi_rows.append((aid, len(h), h["cons"].mean(), zero_share(h["cons"]), len(o), o["cons"].mean(), zero_share(o["cons"])))
hi = pd.DataFrame(hi_rows, columns=["anon_id", "дней_5+", "расход_5+", "нули_5+", "дней_без", "расход_без", "нули_без"])
magnet_day_summary.update({
    "сутки с ≥ 5 срабатываниями: приборов в парном сравнении": len(hi),
    "доля нулевых суток: ≥ 5 срабатываний / без магнита (те же приборы)":
        f"{hi['нули_5+'].mean():.2f} / {hi['нули_без'].mean():.2f}",
    "приборов, у которых в «магнитные» дни нулей больше на ≥ 30 п.п.": int((hi["нули_5+"] > hi["нули_без"] + 0.3).sum()),
})
# мелкие откаты (2 л – 10 м³) рядом с магнитом?
mm1 = pd.concat([mag.assign(date=mag["date"] + pd.Timedelta(days=k)) for k in (-1, 0, 1)]).drop_duplicates(["anon_id", "date"])
small_neg = neg[(neg["delta"] < -P["ZERO"]) & (neg["delta"] > -P["BIG_JUMP"])]
near = small_neg.merge(mm1[["anon_id", "date"]], on=["anon_id", "date"], how="inner")
magnet_day_summary["мелких откатов в ±1 день от «магнита»"] = f"{len(near)} из {len(small_neg)}"
show(magnet_day_summary, dose)

# %%
first_mag = mag.groupby("anon_id")["date"].min()
rows = []
for aid, fd in first_mag.items():
    sub = v[(v["anon_id"] == aid) & (v["gap"] == 1)]
    b = sub.loc[(sub["date"] < fd) & (sub["date"] >= fd - pd.Timedelta(days=30)), "cons"]
    a = sub.loc[(sub["date"] > fd) & (sub["date"] <= fd + pd.Timedelta(days=30)), "cons"]
    if len(b) >= 10 and len(a) >= 10:
        rows.append((aid, fd.date(), b.mean(), a.mean()))
ba = pd.DataFrame(rows, columns=["anon_id", "first_magnet", "before30", "after30"])
ba["ratio"] = (ba["after30"] + 0.01) / (ba["before30"] + 0.01)
ba_summary = {
    "приборов с данными ±30 дн": len(ba),
    "медиана до (м³/сут)": round(ba["before30"].median(), 3),
    "медиана после (м³/сут)": round(ba["after30"].median(), 3),
    "падение ≥ 2 раз (при уровне до ≥ 0,05)": int(((ba["before30"] >= 0.05) & (ba["ratio"] <= 0.5)).sum()),
    "из приборов с уровнем до ≥ 0,05": int((ba["before30"] >= 0.05).sum()),
    "Wilcoxon p-value до/после": f"{stats.wilcoxon(ba['before30'], ba['after30']).pvalue:.2f}",
}

# доля «магнитных» приборов среди аномалий (база — только типы, которые вообще шлют «магнит»)
mag_types = by_tip[by_tip["с_магнитом"] > 0].index
base_ids = set(dev.loc[dev["tip"].isin(mag_types), "anon_id"])
groups = {
    "все приборы этих типов": base_ids,
    "ступенька вниз (B)": set(steps.index[steps["step_B"] == "вниз"]),
    "ступенька вверх (B)": set(steps.index[steps["step_B"] == "вверх"]),
    "ноль: остановился": set(stopped["anon_id"]),
    "ноль: весь период": set(zero_dev.loc[zero_dev["class"] == "весь период ноль", "anon_id"]),
    "утечка (сильный признак)": set(leak_strong["anon_id"]),
    "всплески": set(spike_dev.index),
    "откат < −2 л": set(neg_dev.index),
}
overlap = pd.DataFrame({k: {"приборов": len(s & base_ids),
                            "доля с магнитом": round(len(s & base_ids & mag_dev) / max(len(s & base_ids), 1), 3)}
                        for k, s in groups.items()}).T
show(ba_summary, overlap)

# %%
# Приборы с подозрением на вмешательство: магнит + падение расхода
pair["ratio_mag"] = (pair["mean_mag"] + 0.01) / (pair["mean_nomag"] + 0.01)
susp_a = pair[(pair["size_mag"] >= 5) & (pair["mean_nomag"] >= 0.1) & (pair["ratio_mag"] <= 0.3)]
cp = steps.loc[steps["step_B"] == "вниз", ["cp_date"]].copy()
cp["cp_date"] = pd.to_datetime(cp["cp_date"])
cpm = cp.reset_index(names="anon_id").merge(mag, on="anon_id")
cpm = cpm[(cpm["date"] - cpm["cp_date"]).abs() <= pd.Timedelta(days=7)]
zs = stopped[["anon_id", "start"]].merge(mag, on="anon_id")
zs = zs[(zs["date"] - zs["start"]).abs() <= pd.Timedelta(days=7)]
magnet_suspects = pd.DataFrame(index=sorted(set(susp_a.index) | set(cpm["anon_id"]) | set(zs["anon_id"])))
magnet_suspects["расход_в_дни_магнита_ниже_в_3+_раз"] = magnet_suspects.index.isin(susp_a.index)
magnet_suspects["ступенька_вниз_±7дн_от_магнита"] = magnet_suspects.index.isin(cpm["anon_id"])
magnet_suspects["остановка_±7дн_от_магнита"] = magnet_suspects.index.isin(zs["anon_id"])
magnet_suspects["дней_с_магнитом"] = mag.groupby("anon_id").size().reindex(magnet_suspects.index)
# Магнит на фоне длительного нуля: пустая квартира или постоянно приложенный магнит — различит только выезд
zm = zero_dev[["anon_id", "start", "end", "days", "class"]].merge(mag, on="anon_id")
zm = zm[(zm["date"] > zm["start"]) & (zm["date"] <= zm["end"])]
magnet_zero = (zm.groupby(["anon_id", "class", "days"]).agg(дней_с_магнитом=("date", "nunique"), событий=("n_mag", "sum"))
               .reset_index().query("дней_с_магнитом >= 5").sort_values("событий", ascending=False))
print("Длительный ноль + ≥ 5 дней с «магнитом» внутри него:", len(magnet_zero), "приборов")
show(magnet_zero.head(10))
magnet_suspects["событий_магнит"] = mag.groupby("anon_id")["n_mag"].sum().reindex(magnet_suspects.index)
print("Подозрение на вмешательство (магнит + падение):", len(magnet_suspects))
show(magnet_suspects.sort_values("событий_магнит", ascending=False).head(15))

# %% [markdown]
# ## Пункт 8. Собственные находки

# %%
# 8.1 Битовые сбои: кратность скачков 327,68 м³ = 2^15 × 0,01 м³; дробные кратности — следствие усреднения пакетов за сутки
bigj = iv[iv["delta"].abs() >= 100].copy()
bigj["k"] = bigj["delta"] / P["BIT"]
bigj["frac"] = (bigj["k"] - bigj["k"].round()).abs()
bit_finding = {
    "скачков |Δ| ≥ 100 м³": len(bigj), "приборов": bigj["anon_id"].nunique(),
    "доля кратных 327,68 (±2%)": f"{(bigj['frac'] < 0.02).mean():.1%}",
    "доля на приборах типа AQUA2": f"{(bigj['tip'] == 'AQUA2').mean():.1%}",
    "кратности (k)": bigj.loc[bigj["frac"] < 0.02, "k"].round().astype(int).value_counts().sort_index().to_dict(),
    "максимальный ложный скачок, м³": round(bigj["delta"].max(), 1),
}
glitch_ids = set(iv.loc[iv["returned"] | iv["bit_multiple"] | (iv["rate"] > iv["qmax_day"]), "anon_id"])
dev["glitch"] = dev["anon_id"].isin(glitch_ids)
glitch_by = pd.concat({"по типу": dev.groupby("tip")["glitch"].agg(["mean", "sum", "size"]),
                       "по БС": dev.groupby("bs")["glitch"].agg(["mean", "sum", "size"])})

# 8.2 Сдвиг десятичной запятой (×10): одна дата на несколько приборов
pk2 = pok.copy()
pk2["ratio_prev"] = pk2["reading"] / pk2["prev_reading"]
x10 = pk2[(pk2["prev_reading"] > 0.5) & pk2["ratio_prev"].between(9.5, 10.5) & (pk2["delta"] > 5)]
x10_finding = {"случаев": len(x10), "приборов": x10["anon_id"].nunique(),
               "даты": x10["date"].dt.date.astype(str).value_counts().to_dict(),
               "примеры": x10[["anon_id", "prev_reading", "reading"]].head(6).values.tolist()}

# 8.3 Сезонность и крупные потребители
season = v[v["gap"] == 1].groupby(v["date"].dt.to_period("M"))["cons"].agg(
    медиана="median", среднее="mean", доля_нулей=lambda s: (s <= P["ZERO"]).mean())
heavy = (v[v["gap"] == 1].groupby("anon_id")["cons"].agg(["median", "mean", "size"])
         .query("size >= 60").sort_values("median", ascending=False).head(10)).join(dev.set_index("anon_id")[["bs", "tip"]])
allzero = dev[dev["anon_id"].isin(zero_dev.loc[zero_dev["class"] == "весь период ноль", "anon_id"])]
allzero_by = pd.concat({"по БС": allzero["bs"].value_counts(),
                        "по году выпуска": allzero["mesyac_vypuska"].str[:4].value_counts().sort_index()})
bs04 = dev[dev["bs"] == "BS-04"]["mesyac_vypuska"].str[:4].value_counts().sort_index()
show(bit_finding, glitch_by, x10_finding)

# %%
show(season, heavy, allzero_by, bs04.rename("BS-04: приборов по году выпуска"))

# %% [markdown]
# ## Итог: приоритет проверки
# Баллы: подозрение на вмешательство (магнит + падение) — 5; сильная утечка — 4; остановился при связи — 3;
# реальный крупный расход без возврата (> 10 м³/сут) — 3; ступенька — 2; всплеск (оба критерия) — 2; откат 2 л–10 м³ без сбоя — 1.
# Сбойные приборы (битовые ошибки) — отдельный список в службу АСКУЭ / производителю: это не выезд контролёра, а прошивка / модем.

# %%
score = pd.Series(0, index=dev["anon_id"], dtype=float)
reason = {k: [] for k in dev["anon_id"]}


def add(ids, pts, text):
    for i in ids:
        if i in score.index:
            score[i] += pts
            reason[i].append(text)


add(magnet_suspects.index, 5, "магнит + падение расхода")
add(leak_strong["anon_id"], 4, "утечка: постоянный фон ≥ 0,1 м³/сут ≥ 60 дн")
add(stopped["anon_id"], 3, "расход остановился, пакеты идут")
add(unreal_dev.index[unreal_dev["тип"].str.startswith("реальный")], 3, "> 10 м³/сут без возврата (порыв?)")
add(step_dev.index, 2, "ступенька")
add(spike_dev.index[(spike_dev["всплесков_абс"] > 0) & (spike_dev["всплесков_отн"] > 0)], 2, "всплеск (абс. и отн.)")
add(neg.loc[(neg["delta"] < -P["ZERO"]) & (neg["delta"] > -P["BIG_JUMP"]), "anon_id"].unique(), 1, "откат показаний")
priority = (pd.DataFrame({"балл": score, "причины": {k: "; ".join(v_) for k, v_ in reason.items()}})
            .query("балл > 0").join(dev.set_index("anon_id")[["bs", "tip", "model", "mesyac_vypuska"]])
            .sort_values("балл", ascending=False))
glitch_list = (iv[iv["anon_id"].isin(glitch_ids)].groupby("anon_id")
               .agg(сбойных_интервалов=("returned", "sum"), max_скачок_м3=("delta", "max"), tip=("tip", "first"), bs=("bs", "first"))
               .sort_values("max_скачок_м3", ascending=False))
print("Приборов с баллом > 0:", len(priority), "| сбойных (в АСКУЭ):", len(glitch_list))
show(priority.head(20))

# %% [markdown]
# ## Сохранение результатов

# %%
summary_rows = []
for title, d in [("Качество данных", quality), ("1. Отрицательный расход", neg_summary.to_dict("index")),
                 ("2. Всплески", spike_summary), ("3. Нереальные", unreal_summary.to_dict("index")),
                 ("4. Утечка", leak_summary), ("5. Нулевое потребление", zero_summary), ("6. Ступенька", step_summary),
                 ("7. Магнит: сутки с/без", magnet_day_summary), ("7. Магнит: до/после", ba_summary),
                 ("8. Битовые сбои", bit_finding), ("8. Сдвиг запятой", x10_finding)]:
    for k, val in d.items():
        summary_rows.append((title, str(k), str(val)))
summary = pd.DataFrame(summary_rows, columns=["раздел", "показатель", "значение"])
params = pd.DataFrame(list(P.items()), columns=["параметр", "значение"]).astype(str)

out_xlsx = os.path.join(OUT_DIR, "water_anomalies_results.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
    summary.to_excel(xw, sheet_name="Сводка", index=False)
    params.to_excel(xw, sheet_name="Параметры", index=False)
    priority.reset_index(names="anon_id").to_excel(xw, sheet_name="Приоритет проверки", index=False)
    neg_dev.reset_index().to_excel(xw, sheet_name="1 Отрицательный", index=False)
    spike_dev.reset_index().to_excel(xw, sheet_name="2 Всплески", index=False)
    unreal_dev.reset_index().to_excel(xw, sheet_name="3 Нереальные", index=False)
    leak_strong.to_excel(xw, sheet_name="4 Утечка сильная", index=False)
    leak_c.to_excel(xw, sheet_name="4 Утечка кандидаты", index=False)
    zero_dev.to_excel(xw, sheet_name="5 Нулевое", index=False)
    step_dev.reset_index(names="anon_id").to_excel(xw, sheet_name="6 Ступенька", index=False)
    magnet_suspects.reset_index(names="anon_id").to_excel(xw, sheet_name="7 Магнит подозрение", index=False)
    pair.reset_index().to_excel(xw, sheet_name="7 Магнит сутки", index=False)
    ba.to_excel(xw, sheet_name="7 Магнит до-после", index=False)
    overlap.reset_index(names="группа").to_excel(xw, sheet_name="7 Магнит в аномалиях", index=False)
    glitch_list.reset_index().to_excel(xw, sheet_name="8 Сбойные приборы", index=False)
    glitch_by.reset_index().to_excel(xw, sheet_name="8 Сбои по типу и БС", index=False)
    hi.to_excel(xw, sheet_name="7 Магнит 5+ в сутки", index=False)
    magnet_zero.to_excel(xw, sheet_name="7 Магнит на фоне нуля", index=False)
    heavy.reset_index().to_excel(xw, sheet_name="8 Крупные потребители", index=False)
    allzero_by.reset_index().to_excel(xw, sheet_name="8 Ноль весь период", index=False)
    season.reset_index().astype({"date": str}).to_excel(xw, sheet_name="8 Сезонность", index=False)
daily.to_csv(os.path.join(OUT_DIR, "daily_consumption.csv.gz"), index=False, compression="gzip")
print("Сохранено:", out_xlsx, "и daily_consumption.csv.gz")
