"""CALCE veri seti yükleyicisi.

CS2 cycling (.xlsx), OCV-SOC (.zip içinde .xls/.csv), DST/FUDS dinamik profil (.csv/.xls),
yüksek akım stresi verileri.

Dosyalar büyük olduğu için pickle cache kullanılır — ilk yüklemeden sonra hızlı erişim.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.config import dataset_root, load_yaml, project_root

# ----------------------------------------------------------------------------
# Yol yardımcıları
# ----------------------------------------------------------------------------

def _paths() -> dict:
    return load_yaml("paths")


def _ds_root() -> Path:
    return dataset_root()


def _cache_dir() -> Path:
    p = project_root() / _paths()["cache_dir"]
    p.mkdir(exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# CS2 cycling (CS2_3 klasörü, çoklu .xlsx)
# ----------------------------------------------------------------------------

REQUIRED_CS2_COLS = [
    "Step_Index",
    "Cycle_Index",
    "Current(A)",
    "Voltage(V)",
    "Discharge_Capacity(Ah)",
]

# Sıralı talep seviyesi etiketleri (mevcut rapor kodu uyumlu)
DEMAND_LABEL_ORDER = ["low", "low", "medium", "high", "very_high", "peak"]
DEMAND_LEVEL_MAP = {"low": 0, "medium": 1, "high": 2, "very_high": 3, "peak": 4}


def list_cs2_files() -> list[Path]:
    root = _ds_root() / _paths()["cs2_cycling_dir"]
    return sorted(root.glob("*.xlsx"))


def _read_channel_sheet(xlsx_path: Path) -> pd.DataFrame | None:
    """Arbin 'Channel_x' sheet'ini bulup numerik olarak okur."""
    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception:
        return None
    channels = [s for s in xls.sheet_names if s.startswith("Channel")]
    if not channels:
        return None
    df = pd.read_excel(xls, sheet_name=channels[0])
    df.columns = (
        df.columns.astype(str).str.strip().str.replace("\n", "", regex=False).str.replace("\r", "", regex=False)
    )
    if any(c not in df.columns for c in REQUIRED_CS2_COLS):
        return None
    df = df[REQUIRED_CS2_COLS].copy()
    for col in REQUIRED_CS2_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df.dropna().reset_index(drop=True)


def _label_discharge_steps(df: pd.DataFrame) -> pd.DataFrame | None:
    """Bir dosyadaki deşarj step'lerini ortalama mutlak akıma göre etiketle.

    Rapor uyumlu mantık: deşarj step'leri (negatif akım + kapasite artışı + min satır)
    seçilir, mutlak akıma göre sıralanır, son 6'sı low..peak etiketlenir.
    """
    summary = (
        df.groupby("Step_Index")
        .agg(
            mean_current=("Current(A)", "mean"),
            mean_abs_current=("Current(A)", lambda x: abs(x.mean())),
            row_count=("Current(A)", "count"),
            cap_start=("Discharge_Capacity(Ah)", "min"),
            cap_end=("Discharge_Capacity(Ah)", "max"),
        )
        .reset_index()
    )
    summary["cap_inc"] = summary["cap_end"] - summary["cap_start"]
    dis = summary[
        (summary["mean_current"] < -0.01) & (summary["cap_inc"] > 0.001) & (summary["row_count"] > 20)
    ].sort_values("mean_abs_current")
    if len(dis) < 6:
        return None
    dis = dis.tail(6).reset_index(drop=True)
    dis["Demand_Level"] = DEMAND_LABEL_ORDER
    step_to_label = dict(zip(dis["Step_Index"], dis["Demand_Level"]))

    clean = df[df["Step_Index"].isin(step_to_label.keys())].copy()
    clean["Demand_Level"] = clean["Step_Index"].map(step_to_label)
    clean["P_Load"] = clean["Voltage(V)"] * clean["Current(A)"].abs()
    clean["Step_Start_Discharge"] = clean.groupby(["Cycle_Index", "Step_Index"])[
        "Discharge_Capacity(Ah)"
    ].transform("min")
    Q_nom = 1.1
    clean["Battery_SOC"] = (1 - (clean["Discharge_Capacity(Ah)"] - clean["Step_Start_Discharge"]) / Q_nom).clip(0, 1)
    return clean


def load_cs2_cycling(use_cache: bool = True, balance: bool = True, random_state: int = 42) -> pd.DataFrame:
    """CS2_3 tüm xlsx dosyalarını birleştirip etiketle.

    Çıkış kolonları: Step_Index, Cycle_Index, Current(A), Voltage(V), Discharge_Capacity(Ah),
    Demand_Level, P_Load, Battery_SOC, Demand_Level_Code.
    """
    cache = _cache_dir() / "cs2_cycling.pkl"
    if use_cache and cache.exists():
        return pickle.loads(cache.read_bytes())

    frames: list[pd.DataFrame] = []
    for f in list_cs2_files():
        raw = _read_channel_sheet(f)
        if raw is None:
            continue
        labeled = _label_discharge_steps(raw)
        if labeled is None:
            continue
        labeled["source_file"] = f.name
        frames.append(labeled)

    if not frames:
        raise RuntimeError("CS2 deşarj verisi üretilemedi. Klasör yolunu kontrol et.")

    df = pd.concat(frames, ignore_index=True)
    df["Demand_Level_Code"] = df["Demand_Level"].map(DEMAND_LEVEL_MAP)

    if balance:
        min_n = int(df["Demand_Level"].value_counts().min())
        df = (
            df.groupby("Demand_Level", group_keys=False)
            .apply(lambda g: g.sample(min_n, random_state=random_state))
            .reset_index(drop=True)
        )
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    cache.write_bytes(pickle.dumps(df))
    return df


# ----------------------------------------------------------------------------
# OCV-SOC eğrisi
# ----------------------------------------------------------------------------

@dataclass
class OcvSocCurve:
    """OCV-SOC lookup table. lin interp ile sorgulanır."""

    soc: np.ndarray
    ocv: np.ndarray

    def __call__(self, soc: float | np.ndarray) -> np.ndarray:
        return np.interp(np.clip(soc, self.soc.min(), self.soc.max()), self.soc, self.ocv)


def _try_read_calce_ocv_file(buf: bytes, ext: str) -> pd.DataFrame | None:
    """CALCE OCV dosyaları .xls/.xlsx/.csv olabilir. Tüm sheet'leri tara."""
    bio = io.BytesIO(buf)
    try:
        if ext in {".xls", ".xlsx"}:
            xls = pd.ExcelFile(bio)
            for s in xls.sheet_names:
                if s.lower().startswith("channel"):
                    df = pd.read_excel(xls, sheet_name=s)
                    df.columns = df.columns.astype(str).str.strip()
                    if all(c in df.columns for c in ["Voltage(V)", "Discharge_Capacity(Ah)", "Current(A)"]):
                        return df
            return None
        elif ext == ".csv":
            return pd.read_csv(bio)
    except Exception:
        return None
    return None


def _literature_ocv_soc(n: int = 51) -> OcvSocCurve:
    """CS2 LiCoO2 hücre için literatür-tabanlı analitik OCV-SOC eğrisi.

    Plett, Chen-Mora model türevi. CS2 sınırları: OCV(SOC=0) ≈ 3.0 V, OCV(SOC=1) ≈ 4.2 V.
    """
    soc = np.linspace(0.0, 1.0, n)
    ocv = (
        3.0
        + 0.85 * soc
        + 0.35 * soc ** 2
        - 0.20 * np.exp(-20 * soc)
        + 0.15 * np.exp(-2 * (1 - soc))
    )
    ocv = np.clip(ocv, 3.0, 4.2)
    # Uçları zorla
    ocv[0] = 3.0
    ocv[-1] = 4.2
    return OcvSocCurve(soc=soc, ocv=ocv)


def load_ocv_soc_curve(use_cache: bool = True, q_nom: float = 1.1) -> OcvSocCurve:
    """`OCV-SOC/` ZIP'lerinden düşük akımlı (C/20) deşarj eğrisinden OCV-SOC çıkar.

    Strateji:
      1) ZIP içinden en uygun (Step_Index, Cycle_Index) deşarj segmentini seç:
         - Negatif akım (deşarj)
         - Akım büyüklüğü küçük (C/20 - C/10 aralığı)
         - Segment 1000+ satır ve kapasite artışı 0.8 Q_nom civarı
      2) SOC = 1 - (Q - Q_min)/Q_nom_used (segment-relative)
      3) Monotonik temizle, lookup tablosu döndür.
    """
    cache = _cache_dir() / "ocv_soc.pkl"
    if use_cache and cache.exists():
        return pickle.loads(cache.read_bytes())

    ocv_dir = _ds_root() / _paths()["ocv_soc_dir"]
    zips = list(ocv_dir.glob("*.zip"))
    candidate: pd.DataFrame | None = None
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not (name.lower().endswith(".xls") or name.lower().endswith(".xlsx") or name.lower().endswith(".csv")):
                    continue
                ext = "." + name.rsplit(".", 1)[-1].lower()
                df_try = _try_read_calce_ocv_file(zf.read(name), ext)
                if df_try is None or len(df_try) < 100:
                    continue
                # Kolonlar ok mu
                cols_needed = {"Step_Index", "Cycle_Index", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)"}
                if not cols_needed.issubset(set(df_try.columns)):
                    continue
                candidate = df_try.copy()
                break
        if candidate is not None:
            break

    if candidate is None:
        curve = _literature_ocv_soc()
        cache.write_bytes(pickle.dumps(curve))
        return curve

    df = candidate
    for c in ["Step_Index", "Cycle_Index", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna()

    # Step bazlı özet → en uygun yavaş deşarj segmentini bul
    summ = (
        df.groupby(["Cycle_Index", "Step_Index"])
        .agg(
            mean_I=("Current(A)", "mean"),
            n=("Current(A)", "count"),
            q_start=("Discharge_Capacity(Ah)", "min"),
            q_end=("Discharge_Capacity(Ah)", "max"),
            v_max=("Voltage(V)", "max"),
            v_min=("Voltage(V)", "min"),
        )
        .reset_index()
    )
    summ["cap_inc"] = summ["q_end"] - summ["q_start"]
    summ["abs_I"] = summ["mean_I"].abs()
    # Deşarj segmentleri: I<0, kapasite artışı büyük (>0.5*Q_nom = derin deşarj), yeterli satır
    cand = summ[(summ["mean_I"] < -0.001) & (summ["cap_inc"] > 0.5 * q_nom) & (summ["n"] > 100)]
    if len(cand) == 0:
        curve = _literature_ocv_soc()
        cache.write_bytes(pickle.dumps(curve))
        return curve
    # En küçük akım = en yavaş = OCV'ye en yakın
    cand = cand.sort_values("abs_I")
    cyc = int(cand.iloc[0]["Cycle_Index"])
    stp = int(cand.iloc[0]["Step_Index"])
    q_used = float(cand.iloc[0]["cap_inc"])

    seg = df[(df["Cycle_Index"] == cyc) & (df["Step_Index"] == stp)].copy()
    seg = seg.sort_values("Discharge_Capacity(Ah)")
    q_min = seg["Discharge_Capacity(Ah)"].min()
    soc = (1.0 - (seg["Discharge_Capacity(Ah)"].values - q_min) / q_used).clip(0, 1)
    ocv = seg["Voltage(V)"].values

    order = np.argsort(soc)
    soc = soc[order]
    ocv = ocv[order]
    unique_soc, inv = np.unique(soc, return_inverse=True)
    avg_ocv = np.array([ocv[inv == i].mean() for i in range(len(unique_soc))])

    # Sağlık kontrolü: OCV(SOC=1) > OCV(SOC=0) olmalı
    if avg_ocv[-1] < avg_ocv[0]:
        curve = _literature_ocv_soc()
        cache.write_bytes(pickle.dumps(curve))
        return curve

    curve = OcvSocCurve(soc=unique_soc, ocv=avg_ocv)
    cache.write_bytes(pickle.dumps(curve))
    return curve


# ----------------------------------------------------------------------------
# DST / FUDS dinamik sürüş profilleri
# ----------------------------------------------------------------------------

def _safe_read_table(p: Path) -> pd.DataFrame | None:
    """CSV/XLS okumayı birden fazla format kombinasyonu ile dene.

    CALCE dosyalarında karşılaşılan kombinasyonlar:
      - sep=',' decimal='.'  (US)
      - sep=';' decimal=','  (Avrupa/TR)
      - sep=';' decimal=' '  (bazı CALCE)
    """
    if p.suffix.lower() in (".xls", ".xlsx"):
        try:
            xls = pd.ExcelFile(p)
            # Arbin Channel_x sheet'ini tercih et
            channels = [s for s in xls.sheet_names if s.lower().startswith("channel")]
            sheet = channels[0] if channels else xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception:
            return None
    # CSV — birden çok denemeyi dene, en çok kolon üretene git
    best = None
    best_ncols = 0
    for sep, dec in [(",", "."), (";", ","), (";", "."), (",", ","), ("\t", ".")]:
        try:
            df = pd.read_csv(p, sep=sep, decimal=dec, low_memory=False, nrows=5000)
        except Exception:
            continue
        if df.shape[1] > best_ncols:
            best = (sep, dec)
            best_ncols = df.shape[1]
    if best is None:
        return None
    sep, dec = best
    df = pd.read_csv(p, sep=sep, decimal=dec, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


def list_dynamic_profile_files() -> list[Path]:
    """OCV + Initial Capacity klasöründeki DST/FUDS dosyalarını listele.

    Veri seti yapısı: DST_80SOC, FUDS_80SOC gibi isimler.
    """
    root = _ds_root() / _paths()["dst_fuds_dir"]
    files = []
    for ext in ("*.csv", "*.xls", "*.xlsx"):
        files.extend(root.glob(ext))
    # DST veya FUDS adı geçen dosyalar
    return sorted([f for f in files if ("DST" in f.name.upper() or "FUDS" in f.name.upper())])


def load_dynamic_profile(path: Path | str, use_cache: bool = True) -> pd.DataFrame:
    """Bir dinamik profili (DST/FUDS) yükle, standartlaştır.

    Çıkış: time_s, current_A, voltage_V (varsa), p_load_W (V*|I|)
    """
    p = Path(path)
    cache = _cache_dir() / f"profile_{p.stem}.pkl"
    if use_cache and cache.exists():
        return pickle.loads(cache.read_bytes())

    df = _safe_read_table(p)
    if df is None:
        raise RuntimeError(f"Profil okunamadı: {p}")

    # Kolon eşleştirme — CALCE şemaları farklı olabilir
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if "test_time" in cl or "time(s)" in cl or cl == "time":
            col_map[c] = "time_s"
        elif "current" in cl:
            col_map[c] = "current_A"
        elif "voltage" in cl:
            col_map[c] = "voltage_V"
        elif "discharge_capacity" in cl:
            col_map[c] = "discharge_capacity_Ah"
        elif cl == "soc":
            col_map[c] = "soc"
    df = df.rename(columns=col_map)
    # Duplicate kolonları kaldır (XLS report sheet'lerinde olabiliyor)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    needed = {"time_s", "current_A"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"Profilde gerekli kolonlar yok ({p}): {df.columns.tolist()}")

    df = df[[c for c in ["time_s", "current_A", "voltage_V", "discharge_capacity_Ah", "soc"] if c in df.columns]].copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_s", "current_A"]).reset_index(drop=True)
    df["abs_current_A"] = df["current_A"].abs()
    if "voltage_V" in df.columns:
        df["p_load_W"] = df["voltage_V"] * df["abs_current_A"]
    else:
        df["p_load_W"] = 3.7 * df["abs_current_A"]  # nominal V varsayım

    cache.write_bytes(pickle.dumps(df))
    return df


# ----------------------------------------------------------------------------
# Manifest — UI'da hangi dosyalar var göstermek için
# ----------------------------------------------------------------------------

def dataset_manifest() -> dict[str, list[str]]:
    out = {
        "cs2_cycling": [p.name for p in list_cs2_files()],
        "dynamic_profiles": [p.name for p in list_dynamic_profile_files()],
        "ocv_soc_zips": [p.name for p in (_ds_root() / _paths()["ocv_soc_dir"]).glob("*.zip")],
        "ocv_init_cap_zips": [p.name for p in (_ds_root() / _paths()["ocv_init_capacity_dir"]).glob("*.zip")],
        "high_current_zips": [p.name for p in (_ds_root() / _paths()["high_current_dir"]).glob("*.zip")],
    }
    return out
