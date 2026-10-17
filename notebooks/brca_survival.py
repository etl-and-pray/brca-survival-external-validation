# %% [markdown]
# # Переносятся ли модели выживаемости при раке молочной железы на новую когорту? TCGA → METABRIC
#
# Один ноутбук, четыре части:
#
# 1. Данные и когорты (загрузка, приведение к общему виду, таблица 1)
# 2. Каплан–Мейер и регрессия Кокса
# 3. Прогностические модели, внутренняя кросс-валидация на TCGA
# 4. Внешняя валидация TCGA → METABRIC (**главный результат**)
#
# Запускать сверху вниз. Настройки (конечная точка, горизонт, использование стадии) лежат в `src/config.py`.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))  # чтобы из notebooks/ был виден пакет `src`

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from src import config, viz
from src.data import download_study, load_clinical
from src.evaluate import calibration_table, evaluate_external, repeated_cv, summarize_cv
from src.harmonize import SUBTYPE_LEVELS, build_datasets, model_frame
from src.models import combos

pd.set_option("display.width", 140)
viz.apply_style()
config.ROOT.joinpath("reports").mkdir(exist_ok=True)

# %% [markdown]
# # Часть 1. Данные и когорты
#
# Две публичные когорты из cBioPortal DataHub: TCGA-BRCA (обучение) и METABRIC (внешняя проверка).
# Архивы скачиваются один раз, в `data/raw/` остаются только три нужных файла (папка в `.gitignore`).

# %%
for key in config.STUDIES:
    download_study(key)

# %% [markdown]
# ## 1.1 Проверка имён полей
# Названия колонок в cBioPortal иногда меняются между релизами. Если следующая ячейка упадёт,
# сравните этот список с именами в `src/harmonize.py` и `src/config.py`.

# %%
KEYS = ("OS_", "DFS_", "RFS_", "PFS_", "STAGE", "SUBTYPE", "AGE")
for key in config.STUDIES:
    raw = load_clinical(key)
    print(f"{key}: {raw.shape[0]} пациентов, {raw.shape[1]} колонок")
    print("  нужные колонки:", [c for c in raw.columns if any(k in c for k in KEYS)])

# %% [markdown]
# ## 1.2 Приведение к общему виду
# Общая конечная точка, административное цензурирование на горизонте `HORIZON_MONTHS`,
# экспрессия PAM50 (log2 для RNA-seq, затем z-преобразование по каждому гену внутри когорты).

# %%
print(f"Конечная точка: {config.ENDPOINT}, горизонт: {config.HORIZON_MONTHS} мес.")
data = build_datasets()
for key, d in data.items():
    print(key, "клиника", d["clin"].shape, "экспрессия", d["expr"].shape)

tcga = model_frame(data["tcga"]["clin"], data["tcga"]["expr"])
meta = model_frame(data["metabric"]["clin"], data["metabric"]["expr"])
genes = list(data["tcga"]["expr"].columns)

# %% [markdown]
# ## 1.3 Таблица 1: описание когорт
# Медиана наблюдения считается **до** обрезки на горизонте (обратный метод Каплана–Мейера:
# цензурирование рассматривается как событие). Иначе в METABRIC получилось бы ровно 120 месяцев
# просто из-за обрезки.

# %%
def median_followup_months(clin):
    """Обратный Каплан–Мейер на необрезанном времени наблюдения."""
    kmf = KaplanMeierFitter().fit(clin["time_raw"], 1 - clin["event_raw"])
    return kmf.median_survival_time_


rows = {}
for key, d in data.items():
    c = d["clin"]
    rows[key] = {
        "пациентов": len(c),
        "событий (смертей)": int(c["event"].sum()),
        "событий, %": round(100 * c["event"].mean(), 1),
        "медиана наблюдения, мес. (обратный КМ, до обрезки)": round(median_followup_months(c), 1),
        "возраст, медиана [МКР]": f"{c['age'].median():.0f} [{c['age'].quantile(.25):.0f}-{c['age'].quantile(.75):.0f}]",
        "стадия не указана, %": round(100 * c["stage"].isna().mean(), 1),
    }
table1 = pd.DataFrame(rows)
table1

# %%
subtype_mix = (pd.concat({k: d["clin"]["subtype"].value_counts(normalize=True).mul(100).round(1)
                          for k, d in data.items()}, axis=1)
               .fillna(0))
subtype_mix.index.name = "подтип, %"
subtype_mix

# %%
stage_mix = (pd.concat({k: d["clin"]["stage"].value_counts(normalize=True, dropna=False).mul(100).round(1)
                        for k, d in data.items()}, axis=1)
             .sort_index())
stage_mix.index.name = "стадия, %"
stage_mix

# %% [markdown]
# **Что записать в README после этих чисел**
#
# - Сколько смертей в TCGA на выбранной конечной точке и сколько событий приходится на один признак
#   (главный индикатор переобучения).
# - Насколько короче наблюдение в TCGA, чем в METABRIC. Это влияет на выбор метрики (см. часть 4).
# - Различаются ли доли подтипов и стадий. Одно только это сдвигает и C-индекс, и базовый риск.
# - Поле стадии в METABRIC определено иначе, чем AJCC в TCGA (см. `USE_STAGE` в `config.py`).

# %% [markdown]
# # Часть 2. Каплан–Мейер и регрессия Кокса
#
# Описательный анализ выживаемости в каждой когорте отдельно, с явной проверкой допущения
# пропорциональности рисков.
#
# ## 2.1 Кривые Каплана–Мейера по подтипу и стадии

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
for row, (key, d) in enumerate(data.items()):
    c = d["clin"].copy()
    c["stage_label"] = c["stage"].map(lambda v: f"Стадия {int(v)}" if pd.notna(v) else np.nan)
    viz.plot_km(c, "subtype", ax=axes[row, 0], order=SUBTYPE_LEVELS,
                title=f"{key.upper()}: по подтипу")
    viz.plot_km(c, "stage_label", ax=axes[row, 1], title=f"{key.upper()}: по стадии")
fig.tight_layout()
viz.save(fig, "km_by_subtype_and_stage.png")

# %% [markdown]
# ## 2.2 Многофакторная модель Кокса (в каждой когорте отдельно)
# Отношения рисков для возраста (на 10 лет), стадии (на одну ступень) и подтипа (референс: LumA).
# Строки без стадии отбрасываются; подтипы `Other` и `Unknown` исключены, чтобы референсная
# категория была чистой.

# %%
def cox_design(clin: pd.DataFrame) -> pd.DataFrame:
    d = clin[clin["subtype"].isin(SUBTYPE_LEVELS)].copy()
    out = pd.DataFrame({"time": d["time"], "event": d["event"], "age_per_10y": d["age"] / 10})
    if config.USE_STAGE:
        out["stage"] = d["stage"]
    for s in SUBTYPE_LEVELS[1:]:  # LumA - референс
        out[f"subtype_{s}"] = (d["subtype"] == s).astype(int)
    return out.dropna()


fits = {}
for key, d in data.items():
    design = cox_design(d["clin"])
    cph = CoxPHFitter().fit(design, duration_col="time", event_col="event")
    fits[key] = (cph, design)
    print(f"\n=== {key.upper()}: n={len(design)}, событий={int(design['event'].sum())} ===")
    print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))

# %%
# Форест-плот отношений рисков: обе когорты рядом
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = list(fits["tcga"][0].summary.index)
ypos = np.arange(len(names))[::-1]
for i, (key, (cph, _)) in enumerate(fits.items()):
    s = cph.summary.reindex(names)
    off = (i - 0.5) * 0.22
    ax.errorbar(s["exp(coef)"], ypos + off,
                xerr=[s["exp(coef)"] - s["exp(coef) lower 95%"], s["exp(coef) upper 95%"] - s["exp(coef)"]],
                fmt="o", ms=5, color=viz.CATEGORICAL[i], markeredgecolor=viz.SURFACE,
                markeredgewidth=1.3, elinewidth=1.1, label=key.upper())
ax.axvline(1, color=viz.MUTED, lw=0.9, ls="--")
ax.set_xscale("log")
ax.set_yticks(ypos, names)
ax.set_xlabel("Отношение рисков (логарифмическая шкала, 95% ДИ)")
ax.set_title("Модель Кокса: одинаковое ли направление эффектов в обеих когортах?")
ax.grid(axis="y", visible=False)
ax.legend()
viz.save(fig, "cox_forest.png")

# %% [markdown]
# ## 2.3 Проверка пропорциональности рисков
# Тест на остатках Шенфельда. Малое p означает, что эффект признака меняется со временем.
# Интерпретировать нужно вместе с графиками, а не только по p-значениям, особенно при малом числе событий.

# %%
for key, (cph, design) in fits.items():
    res = proportional_hazard_test(cph, design, time_transform="rank")
    print(f"\n=== {key.upper()} ===")
    print(res.summary.round(3))

# %% [markdown]
# Если признак нарушает пропорциональность, варианты (и что описать в выводах): стратификация
# по нему (`CoxPHFitter().fit(..., strata=["..."])`), взаимодействие со временем или
# ограниченное среднее время дожития вместо одного отношения рисков.

# %% [markdown]
# # Часть 3. Внутренняя валидация внутри TCGA
#
# Три семейства моделей на трёх наборах признаков, повторная стратифицированная 5-кратная
# кросс-валидация. Штраф elastic net подбирается **внутри** каждой обучающей выборки, поэтому
# тестовые фолды никак не влияют ни на гиперпараметры, ни на отбор генов.
# Это «оптимистичная» оценка, которую в части 4 проверяет независимая когорта.

# %%
n_events = int(tcga["event"].sum())
print(f"TCGA: {len(tcga)} пациентов, {n_events} событий, {len(genes)} генов в панели")
print(f"Событий на признак при использовании всех генов PAM50: {n_events / len(genes):.1f}  (ориентир: не менее 10)")

# %% [markdown]
# ## 3.1 Запуск сравнения
# Занимает несколько минут (дольше всего подбор Coxnet и случайный лес). Для быстрого
# первого прогона уменьшите `N_REPEATS`.

# %%
N_SPLITS, N_REPEATS = 5, 3

cv_parts = []
for feature_set, model in combos():
    print(f"{feature_set:>15s} / {model}")
    cv_parts.append(repeated_cv(model, feature_set, tcga, genes,
                                n_splits=N_SPLITS, n_repeats=N_REPEATS, n_jobs=-1))
cv = pd.concat(cv_parts, ignore_index=True)
cv.to_csv(config.ROOT / "reports" / "cv_results.csv", index=False)

# %%
summarize_cv(cv)

# %% [markdown]
# `std` показывает разброс по фолдам (они перекрываются между повторами), то есть
# стабильность, а не стандартную ошибку. При ~100 событиях ожидайте большой разброс.

# %% [markdown]
# ## 3.2 Распределение по фолдам

# %%
order = [f"{f} / {m}" for f, m in combos()]
cv["label"] = cv["features"] + " / " + cv["model"]

fig, ax = plt.subplots(figsize=(7, 4.2))
for i, lab in enumerate(order[::-1]):
    vals = cv.loc[cv["label"] == lab, "harrell_c"]
    ax.scatter(vals, [i] * len(vals), s=16, color=viz.CATEGORICAL[0], alpha=0.55, edgecolor="none")
    ax.scatter([vals.mean()], [i], s=60, color=viz.CATEGORICAL[0],
               edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
ax.axvline(0.5, color=viz.MUTED, lw=0.9, ls="--")
ax.set_yticks(range(len(order)), order[::-1])
ax.set_xlabel("C-индекс Харрела по фолдам (крупная точка - среднее)")
ax.set_title("Внутренняя кросс-валидация на TCGA-BRCA")
ax.grid(axis="y", visible=False)
viz.save(fig, "internal_cv_c_index.png")

# %% [markdown]
# **Вопросы для выводов**
#
# - Добавляет ли экспрессия хоть что-то к клиническим признакам? (Подтип сам получен из
#   экспрессии, поэтому большого прироста ждать не стоит.)
# - Лучше ли штрафованные и нелинейные модели обычной клинической модели Кокса за пределами
#   шума между фолдами?
# - Согласуются ли C-индексы Харрела и Уно? Расхождение говорит о том, что оценку искажает
#   цензурирование (подробнее в разделе 4.3). Уно в TCGA шумный: событий мало, наблюдение короткое.

# %% [markdown]
# # Часть 4. Внешняя валидация: TCGA → METABRIC
#
# Обучаем на TCGA-BRCA и применяем к METABRIC **без дообучения и подбора чего-либо заново**.
# Сколько качества из внутренней кросс-валидации сохраняется?
#
# ## 4.1 Состав когорт и базовый риск
# Даже идеально ранжирующая модель будет плохо откалибрована, если базовая выживаемость в когортах разная.

# %%
print(f"обучение TCGA: {len(tcga)} (событий {int(tcga['event'].sum())}) | "
      f"тест METABRIC: {len(meta)} (событий {int(meta['event'].sum())})")

pooled = pd.concat([data[k]["clin"].assign(cohort=k.upper()) for k in data])
fig, ax = plt.subplots(figsize=(6, 4))
viz.plot_km(pooled, "cohort", ax=ax, title="Общая выживаемость, одинаковый горизонт в обеих когортах")
viz.save(fig, "km_by_cohort.png")

# %% [markdown]
# ## 4.2 Обучение на TCGA, оценка на METABRIC
# Долго работают подбор Coxnet и случайный лес: каждый обучается один раз на всей TCGA.

# %%
results = {}
for feature_set, model in combos():
    print(f"{feature_set:>15s} / {model}")
    results[(feature_set, model)] = evaluate_external(model, feature_set, tcga, meta, genes, n_jobs=-1)

external = pd.DataFrame([r["summary"] for r in results.values()])
external.to_csv(config.ROOT / "reports" / "external_results.csv", index=False)
external.round(3)

# %% [markdown]
# ## 4.3 Внутренняя CV и внешняя валидация (главный рисунок)
#
# **Важно про выбор метрики.** С-индекс Харрела зависит от длины наблюдения и доли
# цензурированных пациентов. В TCGA наблюдение короткое (медиана около 2,6 года), в METABRIC
# длинное (около 16 лет), поэтому значения Харрела в разных когортах не вполне сопоставимы.
# С-индекс Уно взвешивает пары обратной вероятностью цензурирования и обрезается на 5 годах
# (`UNO_TAU = 60` в `src/evaluate.py`). Вес считается по цензурированию **той когорты, на которой
# оцениваем**, а не обучающей: иначе распределение цензурирования TCGA исказило бы веса для METABRIC.
# На рисунке две панели; выводы о «потере качества при переносе» стоит делать по Уно.

# %%
internal = cv.groupby(["features", "model"], as_index=False)[["harrell_c", "uno_c"]].mean()

fig = viz.plot_performance(internal, external)
viz.save(fig, "internal_vs_external.png")

comparison = internal.merge(external, on=["features", "model"], suffixes=("_внутр", "_внеш"))
comparison["разница Харрел"] = comparison["harrell_c_внутр"] - comparison["harrell_c_внеш"]
comparison["разница Уно"] = comparison["uno_c_внутр"] - comparison["uno_c_внеш"]
comparison[["features", "model", "harrell_c_внутр", "harrell_c_внеш", "разница Харрел",
            "uno_c_внутр", "uno_c_внеш", "разница Уно"]].round(3)

# %% [markdown]
# ## 4.4 AUC(t) и оценка Брайера
# Оценка Брайера сравнивается с базовой моделью без признаков (одна кривая Каплана–Мейера
# для всех): `scaled_brier > 0` значит, что модель лучше, чем «каждому одна средняя кривая».

# %%
auc = pd.DataFrame({f"{f} / {m}": r["auc"] for (f, m), r in results.items()}).T
auc.columns = [f"AUC({int(t)} мес.)" for t in auc.columns]
auc.round(3)

# %%
rows = []
for (f, m), r in results.items():
    b = r["brier"]
    if b.empty:
        continue
    rows.append({"модель": f"{f} / {m}", "IBS": b.attrs.get("ibs", np.nan),
                 "scaled Brier (среднее по сетке)": b["scaled_brier"].mean()})
pd.DataFrame(rows).round(3)

# %% [markdown]
# ## 4.5 Калибровка на 5 лет
# Группы риска по **предсказанному** риску и риск, **наблюдаемый** в METABRIC (Каплан–Мейер).
# Точки выше диагонали означают, что в новой когорте модель занижает риск.

# %%
T = 60.0
i60 = list(results[("clinical", "cox")]["times"]).index(T)
tables = {}
for key in [("clinical", "cox"), ("pam50", "coxnet"), ("clinical+pam50", "rsf")]:
    r = results[key]
    tables[f"{key[0]} / {key[1]}"] = calibration_table(r["surv"][:, i60], r["y_test"], T, n_groups=5)

ax = viz.plot_calibration(tables, T)
viz.save(ax.figure, "calibration_5y.png")
tables

# %% [markdown]
# ## 4.6 Проверка в обратную сторону: METABRIC → TCGA
# В METABRIC намного больше событий, поэтому модель, обученная там, лучше обеспечена данными.
# Если она хорошо переносится на TCGA, то слабый результат TCGA → METABRIC в основном
# объясняется размером обучающей выборки, а не «непереносимостью» самой модели.

# %%
reverse = []
for feature_set, model in [("clinical", "cox"), ("pam50", "coxnet"), ("clinical+pam50", "coxnet")]:
    r = evaluate_external(model, feature_set, meta, tcga, genes, n_jobs=-1)
    reverse.append({**r["summary"], "направление": "METABRIC -> TCGA"})
pd.DataFrame(reverse).round(3)

# %% [markdown]
# ## 4.7 Чувствительность: пациенты METABRIC с известной стадией
# В METABRIC у 26% пациентов стадия не указана, а модель при предсказании подставляет туда
# медиану из TCGA. Это может занижать внешний C-индекс клинических моделей. Здесь оцениваем
# только тех, у кого стадия известна (обучение прежнее, на всей TCGA).

# %%
meta_known = meta[meta["stage"].notna()]
print(f"METABRIC с известной стадией: {len(meta_known)} из {len(meta)}, событий {int(meta_known['event'].sum())}")

rows = []
for feature_set, model in [("clinical", "cox"), ("clinical+pam50", "coxnet"), ("clinical+pam50", "rsf")]:
    r = evaluate_external(model, feature_set, tcga, meta_known, genes, n_jobs=-1)
    rows.append({**r["summary"], "выборка": "стадия известна"})
sens = pd.DataFrame(rows)
base = external[external.set_index(["features", "model"]).index.isin(
    [(r["features"], r["model"]) for r in rows])].assign(выборка="все пациенты")
pd.concat([base, sens]).sort_values(["features", "model", "выборка"]).round(3)

# %% [markdown]
# ## 4.8 Выводы
#
# Основная метрика - C-индекс Уно (горизонт 5 лет, веса по цензурированию оцениваемой когорты).
# Числа ниже взяты из прогона с `SEED = 42`; при изменении настроек обновите их.
#
# 1. **Качество переносится с небольшой потерей.** Клиническая модель Кокса: Уно 0,696 при
#    внутренней CV на TCGA и 0,657 на всех пациентах METABRIC. Но у 26% METABRIC стадия не указана
#    и подставляется медианой; на пациентах с известной стадией (4.7) Уно 0,675, то есть разница
#    с внутренней оценкой около 0,02. Разброс по фолдам внутри TCGA (SD около 0,06) заметно
#    больше, поэтому значимой потери качества при переносе мы не видим.
# 2. **Харрел вводит в заблуждение.** По Харрелу разница 0,095 (0,744 против 0,649). Это в основном
#    свойство когорт: короткое наблюдение в TCGA (медиана 32 мес. против 196 мес. в METABRIC)
#    и более широкий разброс стадий.
# 3. **Обратное направление подтверждает это.** Модель, обученная на METABRIC, на TCGA даёт
#    Уно 0,696, то же значение, что и внутренняя CV на TCGA. То есть «потолок» качества
#    определяется когортой, а не направлением переноса.
# 4. **Экспрессия PAM50 не добавляет к клинике.** Клиника + PAM50 против одной клиники: Уно 0,697
#    против 0,696 внутри и 0,659 против 0,657 снаружи. Одна экспрессия сохраняет слабый сигнал
#    (0,604 → 0,558 для Coxnet, 0,624 → 0,607 для леса), ослабленный сменой платформы.
# 5. **Сложные модели не лучше простых.** Случайный лес не превосходит регуляризованную Кокса
#    ни на клинических признаках, ни на комбинации.
# 6. **Калибровка и точность скромные.** Клиническая Cox занижает пятилетний риск во всех группах
#    (в среднем предсказано около 17%, наблюдается около 22%). AUC(60 мес.) около 0,67,
#    масштабированный Brier 0,04-0,06: модель лишь немного лучше «всем одну среднюю кривую».
#
# **Ограничения.** Мало событий в TCGA (139), у части признаков нарушена пропорциональность рисков,
# общая выживаемость включает смерти не от рака, стадия определена по-разному в когортах и
# пропущена у 26% METABRIC (подстановка медианы занижает внешний Уно примерно на 0,02, см. 4.7),
# лечение не учитывается. Доверительные интервалы для
# внешнего C-индекса отражают только разброс тестовой выборки.
