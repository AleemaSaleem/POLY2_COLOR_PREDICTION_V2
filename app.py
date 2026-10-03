# ============================================================
# POLY-II BASELINE C — SINGLE DATE/TIME RAW-DCS COLOUR PREDICTOR
# ============================================================
#
# USER WORKFLOW
# --------------
# 1. Select a date.
# 2. Select a DCS time.
# 3. Click PREDICT COLOUR.
#
# The application then:
#   RAW DCS
#      -> Nova inventory conversion
#      -> residence time
#      -> RT-valid production check
#      -> backward causal trajectory
#      -> causal process features
#      -> localized paste chemistry
#      -> EST-2 H3PO4 features
#      -> process-control features
#      -> chemistry/TPD interactions
#      -> Baseline-C throughput regime features
#      -> deployment-time dynamic features
#      -> exact saved feature lists
#      -> saved CatBoost models
#      -> LAB_L / LAB_A / LAB_B
#
# IMPORTANT
# ----------
# * No previous LAB colour value is used.
# * BA202060-F0 is excluded.
# * TPD <= 0 is treated as shutdown.
# * A prediction is generated ONLY for the selected timestamp.
# * The app refuses to predict if the saved feature definition
#   cannot be reproduced.
# * Features that cannot be calculated reliably are created as
#   NaN so the feature schema remains consistent.
# * NaN values are converted to 0.0 immediately before the
#   CatBoost model receives the input.
#
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ============================================================
# APPLICATION FILE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NOVA_FILE = BASE_DIR / "Nova_Part-A.xlsx"

MODEL_FILES = {
    "L": BASE_DIR / "catboost_LAB_L_BaselineC.cbm",
    "A": BASE_DIR / "catboost_LAB_A_BaselineC.cbm",
    "B": BASE_DIR / "catboost_LAB_B_BaselineC.cbm",
}

FEATURE_FILE = BASE_DIR / "baseline_c_selected_features.json"

TARGETS = ["L", "A", "B"]

DYNAMIC_LOOKBACK_HOURS = 4.0


# ============================================================
# DCS TAGS
# ============================================================

DCS_SENSOR_COLS = [
    "YK-11001",
    "TPD",
    "FIC-12001",
    "LIC-12019",
    "TIC-12012",
    "PIC-12013",
    "LIC-13017",
    "TIC-13012",
    "LIC-14003",
    "TIC-14004",
    "PIC-14005",
    "LIC-15006",
    "TIC-15005",
    "TIC-15008",
    "PIC-15024",
    "SIK-17015",
    "SIK-17028",
    "IC-17013",
    "PIC-17024",
    "PV-17024",
    "LI-17017",
    "LI-17023",
    "LIC-17016",
    "TI-17019",
    "TI-17020",
    "TI-17022",
    "VIC-18020",
    "TI-17113",
    "TI-17114",
    "PIC-17183",
    "LIC-17175",
    "TI-50002",
    "TI-50003",
    "FIC-21009",
    "YK-21010",
    "FIC-41007",
    "YK-41008",
    "FIC-43007",
    "YK-43008",
    "FIC-42007",
    "YK-42008",
]


PASTE_CHEM_TAGS = [
    "FIC-21009",
    "YK-21010",
    "FIC-41007",
    "YK-41008",
    "FIC-42007",
    "YK-42008",
]


EST2_CHEM_TAGS = [
    "FIC-43007",
    "YK-43008",
]


PROCESS_TAGS = [
    "TPD",
    "TIC-12012",
    "PIC-12013",
    "TIC-13012",
    "TIC-14004",
    "PIC-14005",
    "TIC-15005",
    "TIC-15008",
    "PIC-15024",
    "TI-17019",
    "TI-17020",
    "TI-17022",
    "PIC-17024",
]


AREA_TAGS = {

    "PASTE": [
        "YK-11001"
    ],

    "THROUGHPUT": [
        "TPD"
    ],

    "EST1": [
        "FIC-12001",
        "LIC-12019",
        "TIC-12012",
        "PIC-12013",
    ],

    "EST2": [
        "LIC-13017",
        "TIC-13012",
    ],

    "PP1": [
        "LIC-14003",
        "TIC-14004",
        "PIC-14005",
    ],

    "PP2": [
        "LIC-15006",
        "TIC-15005",
        "TIC-15008",
        "PIC-15024",
    ],

    "DRR": [
        "SIK-17015",
        "SIK-17028",
        "IC-17013",
        "PIC-17024",
        "PV-17024",
        "LI-17017",
        "LI-17023",
        "LIC-17016",
        "TI-17019",
        "TI-17020",
        "TI-17022",
        "VIC-18020",
    ],

    "JET": [
        "TI-17113",
        "TI-17114",
        "PIC-17183",
        "LIC-17175",
    ],

    "HTM": [
        "TI-50002",
        "TI-50003",
    ],
}


SECTION_TAGS = {
    "EST1": AREA_TAGS["EST1"],
    "EST2": AREA_TAGS["EST2"],
    "PP1": AREA_TAGS["PP1"],
    "PP2": AREA_TAGS["PP2"],
    "DRR": AREA_TAGS["DRR"],
}


EQUIPMENT_MAP = {

    "EST1": {
        "DCS_TAG": "LIC-12019",
        "EQUIPMENT": "12-R01-EST-I",
    },

    "EST2": {
        "DCS_TAG": "LIC-13017",
        "EQUIPMENT": "13-R01-EST-II",
    },

    "PP1": {
        "DCS_TAG": "LIC-14003",
        "EQUIPMENT": "14-R01-PP-I",
    },

    "PP2": {
        "DCS_TAG": "LIC-15006",
        "EQUIPMENT": "15-R01-PP-II",
    },

    "DRR": {
        "DCS_TAG": "LIC-17016",
        "EQUIPMENT": "17-R01- (DRR)",
    },
}


SECTION_ORDER = [
    ("DRR", "DRR_RT_HR"),
    ("PP2", "PP2_RT_HR"),
    ("PP1", "PP1_RT_HR"),
    ("EST2", "EST2_RT_HR"),
    ("EST1", "EST1_RT_HR"),
]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature JSON not found:\n{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        saved_features = json.load(f)

    models = {}

    for target in TARGETS:

        path = MODEL_FILES[target]

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        model = CatBoostRegressor()

        model.load_model(
            str(path)
        )

        models[target] = model

    return models, saved_features


# ============================================================
# RAW DCS
# ============================================================

@st.cache_data
def load_raw_dcs(
    uploaded_file_bytes,
    uploaded_file_name,
):

    if not uploaded_file_bytes:

        raise ValueError(
            "Please upload the raw DCS Excel file."
        )

    from io import BytesIO

    # --------------------------------------------------------
    # Read raw Excel
    # --------------------------------------------------------

    raw = pd.read_excel(
        BytesIO(uploaded_file_bytes),
        header=None,
    )

    if len(raw) < 6:

        raise ValueError(
            "Unexpected DCS Excel structure. "
            "Expected timestamp/tag rows followed by DCS data."
        )

    # --------------------------------------------------------
    # Read tag names
    # --------------------------------------------------------

    tag_row = raw.iloc[2]

    columns = []

    for i, value in enumerate(tag_row):

        if i == 0:

            columns.append(
                "TIMESTAMP"
            )

        elif pd.isna(value):

            columns.append(
                f"UNNAMED_{i}"
            )

        else:

            columns.append(
                str(value).strip()
            )

    # --------------------------------------------------------
    # Extract actual DCS data
    # --------------------------------------------------------

    dcs = raw.iloc[5:].copy()

    dcs.columns = columns

    dcs = (
        dcs
        .dropna(how="all")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    dcs["TIMESTAMP"] = pd.to_datetime(
        dcs["TIMESTAMP"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Clean timestamp rows
    # --------------------------------------------------------

    dcs = (
        dcs
        .dropna(subset=["TIMESTAMP"])
        .sort_values("TIMESTAMP")
        .drop_duplicates("TIMESTAMP")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check required DCS tags
    # --------------------------------------------------------

    missing = [
        c
        for c in DCS_SENSOR_COLS
        if c not in dcs.columns
    ]

    if missing:

        raise ValueError(
            "Required DCS tags are missing:\n\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in DCS_SENSOR_COLS:

        dcs[col] = pd.to_numeric(
            dcs[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove FLOAT32 invalid sentinel
    # --------------------------------------------------------

    sentinel = (
        dcs["YK-43008"].abs() > 1e30
    )

    dcs.loc[
        sentinel,
        "YK-43008"
    ] = np.nan

    # --------------------------------------------------------
    # DCS interval
    # --------------------------------------------------------

    dcs["INTERVAL_MIN"] = (
        dcs["TIMESTAMP"].diff()
        .dt.total_seconds()
        / 60.0
    )

    return dcs


# ============================================================
# NOVA
# ============================================================

@st.cache_data
def load_nova_curves():

    if not NOVA_FILE.exists():

        raise FileNotFoundError(
            f"Nova file not found:\n{NOVA_FILE}"
        )

    nova = pd.read_excel(
        NOVA_FILE,
        sheet_name="Poly-2",
        header=1,
    )

    curves = {}

    for section, info in EQUIPMENT_MAP.items():

        mask = (
            nova["silo_no"]
            .astype(str)
            .str.strip()
            .eq(info["EQUIPMENT"])
        )

        curve = nova.loc[
            mask,
            [
                "Level Percent",
                "Level Weight Kgs.",
            ],
        ].copy()

        curve["Level Percent"] = pd.to_numeric(
            curve["Level Percent"],
            errors="coerce",
        )

        curve["Level Weight Kgs."] = pd.to_numeric(
            curve["Level Weight Kgs."],
            errors="coerce",
        )

        curve = (
            curve
            .dropna()
            .drop_duplicates("Level Percent")
            .sort_values("Level Percent")
            .reset_index(drop=True)
        )

        if len(curve) < 2:

            raise ValueError(
                f"Insufficient Nova curve for {section}."
            )

        curves[section] = curve

    return curves


def add_inventory(
    dcs,
    curves,
):

    dcs = dcs.copy()

    for section, info in EQUIPMENT_MAP.items():

        level = pd.to_numeric(
            dcs[info["DCS_TAG"]],
            errors="coerce",
        )

        curve = curves[section]

        x = curve[
            "Level Percent"
        ].to_numpy(float)

        y = curve[
            "Level Weight Kgs."
        ].to_numpy(float)

        inventory = np.full(
            len(dcs),
            np.nan,
        )

        valid = (
            level.notna()
            & level.ge(x.min())
            & level.le(x.max())
        )

        inventory[valid] = np.interp(
            level.loc[valid].to_numpy(float),
            x,
            y,
        )

        dcs[
            f"{section}_LEVEL_PCT"
        ] = level

        dcs[
            f"{section}_INVENTORY_KG"
        ] = inventory

    return dcs


# ============================================================
# RESIDENCE TIME
# ============================================================

def calculate_rt(dcs):

    dcs = dcs.copy()

    dcs["TPD"] = pd.to_numeric(
        dcs["TPD"],
        errors="coerce",
    )

    dcs["TPH"] = (
        dcs["TPD"] / 24.0
    )

    for section in EQUIPMENT_MAP:

        dcs[
            f"{section}_RT_HR"
        ] = (
            dcs[
                f"{section}_INVENTORY_KG"
            ]
            /
            (
                dcs["TPH"]
                * 1000.0
            )
        )

        dcs.loc[
            dcs["TPH"] <= 0,
            f"{section}_RT_HR",
        ] = np.nan

    rt_cols = [
        "EST1_RT_HR",
        "EST2_RT_HR",
        "PP1_RT_HR",
        "PP2_RT_HR",
        "DRR_RT_HR",
    ]

    dcs["TOTAL_RT_HR"] = (
        dcs[rt_cols]
        .sum(
            axis=1,
            min_count=5,
        )
    )

    dcs["RT_VALID"] = (
        dcs["TPD"].ge(375)
        &
        dcs[rt_cols]
        .notna()
        .all(axis=1)
    )

    dcs["RT_STATUS"] = np.where(
        dcs["RT_VALID"],
        "VALID",
        np.where(
            dcs["TPD"].le(0),
            "SHUTDOWN_TPD_LE_0",
            "NOT_RT_VALID",
        ),
    )

    return dcs


def build_runs(dcs):

    gap = (
        dcs["TIMESTAMP"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )

    continuous = gap.eq(0.25)

    run_break = (
        dcs["RT_VALID"]
        .ne(dcs["RT_VALID"].shift())
        | ~continuous
    )

    dcs = dcs.copy()

    dcs["VALID_RUN_ID"] = (
        run_break.cumsum()
    )

    runs = (
        dcs.loc[dcs["RT_VALID"]]
        .groupby("VALID_RUN_ID")
        .agg(
            START_TS=("TIMESTAMP", "min"),
            END_TS=("TIMESTAMP", "max"),
            N_ROWS=("TIMESTAMP", "size"),
        )
        .reset_index()
    )

    runs["DURATION_HR"] = (
        runs["END_TS"]
        - runs["START_TS"]
    ).dt.total_seconds() / 3600.0

    return dcs, runs


# ============================================================
# CAUSAL TRAJECTORY
# ============================================================

def find_run(
    runs,
    timestamp,
):

    match = runs[
        (runs["START_TS"] <= timestamp)
        &
        (runs["END_TS"] >= timestamp)
    ]

    if match.empty:

        return None

    return match.iloc[0]


def get_rt_at_time(
    dcs,
    timestamp,
    rt_column,
):

    timestamp = pd.Timestamp(
        timestamp
    )

    if (
        timestamp < dcs["TIMESTAMP"].min()
        or
        timestamp > dcs["TIMESTAMP"].max()
    ):

        return np.nan, "OUTSIDE_RANGE"

    before = dcs[
        (dcs["TIMESTAMP"] <= timestamp)
        &
        dcs["RT_VALID"]
    ].tail(1)

    after = dcs[
        (dcs["TIMESTAMP"] >= timestamp)
        &
        dcs["RT_VALID"]
    ].head(1)

    if before.empty or after.empty:

        return np.nan, "NO_BRACKET"

    t0 = before.iloc[0]["TIMESTAMP"]
    t1 = after.iloc[0]["TIMESTAMP"]

    gap_hr = (
        t1 - t0
    ).total_seconds() / 3600.0

    if gap_hr > 1.0:

        return np.nan, "RT_HISTORY_GAP"

    r0 = before.iloc[0][rt_column]
    r1 = after.iloc[0][rt_column]

    if pd.isna(r0) or pd.isna(r1):

        return np.nan, "RT_MISSING"

    if t0 == t1:

        return float(r0), "DIRECT"

    fraction = (
        timestamp - t0
    ).total_seconds() / (
        t1 - t0
    ).total_seconds()

    return (
        float(
            r0 + fraction * (r1 - r0)
        ),
        "INTERPOLATED",
    )


def calculate_trajectory(
    dcs,
    runs,
    timestamp,
):

    timestamp = pd.Timestamp(
        timestamp
    )

    run = find_run(
        runs,
        timestamp,
    )

    if run is None:

        return None

    current = timestamp

    result = {
        "TIMESTAMP": timestamp,
        "TRAJECTORY_VALID": True,
        "VALID_RUN_ID": int(
            run["VALID_RUN_ID"]
        ),
        "RUN_START": run["START_TS"],
        "RUN_END": run["END_TS"],
    }

    for section, rt_col in SECTION_ORDER:

        rt, status = get_rt_at_time(
            dcs,
            current,
            rt_col,
        )

        if pd.isna(rt):

            return None

        result[
            f"{section}_RT_HR"
        ] = rt

        result[
            f"{section}_RT_STATUS"
        ] = status

        boundary = (
            current
            - pd.Timedelta(
                hours=float(rt)
            )
        )

        result[
            f"{section}_BOUNDARY_TS"
        ] = boundary

        current = boundary

    result[
        "EST1_UPSTREAM_TS"
    ] = current

    boundaries = [
        result["DRR_BOUNDARY_TS"],
        result["PP2_BOUNDARY_TS"],
        result["PP1_BOUNDARY_TS"],
        result["EST2_BOUNDARY_TS"],
        result["EST1_BOUNDARY_TS"],
    ]

    if not all(
        run["START_TS"] <= x <= run["END_TS"]
        for x in boundaries
    ):

        return None

    return result


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

def extract_window_features(
    dcs_df,
    start_ts,
    end_ts,
    tags,
    prefix,
):

    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)

    if (
        pd.isna(start_ts)
        or pd.isna(end_ts)
        or end_ts <= start_ts
    ):

        return {}

    window = dcs_df[
        (dcs_df["TIMESTAMP"] >= start_ts)
        &
        (dcs_df["TIMESTAMP"] <= end_ts)
    ].sort_values(
        "TIMESTAMP"
    ).copy()

    if window.empty:

        return {}

    duration_hr = (
        end_ts - start_ts
    ).total_seconds() / 3600.0

    result = {}

    for tag in tags:

        if tag not in window.columns:

            continue

        values = pd.to_numeric(
            window[tag],
            errors="coerce",
        )

        valid = values.notna()

        if valid.sum() == 0:

            continue

        v = values.loc[
            valid
        ].astype(float)

        first = float(
            v.iloc[0]
        )

        last = float(
            v.iloc[-1]
        )

        result[
            f"{prefix}{tag}_MEAN"
        ] = float(v.mean())

        result[
            f"{prefix}{tag}_STD"
        ] = float(
            v.std(ddof=0)
        )

        result[
            f"{prefix}{tag}_MIN"
        ] = float(v.min())

        result[
            f"{prefix}{tag}_MAX"
        ] = float(v.max())

        result[
            f"{prefix}{tag}_RANGE"
        ] = float(
            v.max() - v.min()
        )

        result[
            f"{prefix}{tag}_FIRST"
        ] = first

        result[
            f"{prefix}{tag}_LAST"
        ] = last

        result[
            f"{prefix}{tag}_DELTA"
        ] = last - first

        result[
            f"{prefix}{tag}_N"
        ] = int(valid.sum())

        for q in [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]:

            result[
                f"{prefix}{tag}_P{int(q * 100)}"
            ] = float(
                v.quantile(q)
            )

        if (
            len(v) >= 2
            and duration_hr > 0
        ):

            times = (
                window.loc[
                    valid,
                    "TIMESTAMP"
                ]
                -
                window.loc[
                    valid,
                    "TIMESTAMP"
                ].iloc[0]
            ).dt.total_seconds().to_numpy() / 3600.0

            y = v.to_numpy(float)

            if len(
                np.unique(times)
            ) >= 2:

                slope = np.polyfit(
                    times,
                    y,
                    1,
                )[0]

            else:

                slope = 0.0

        else:

            slope = 0.0

        result[
            f"{prefix}{tag}_SLOPE"
        ] = float(slope)

    return result


# ============================================================
# CAUSAL STATIC FEATURES
# ============================================================

def build_static_features(
    trajectory,
    dcs,
):

    ts = pd.Timestamp(
        trajectory["TIMESTAMP"]
    )

    est1 = pd.Timestamp(
        trajectory["EST1_BOUNDARY_TS"]
    )

    est2 = pd.Timestamp(
        trajectory["EST2_BOUNDARY_TS"]
    )

    pp1 = pd.Timestamp(
        trajectory["PP1_BOUNDARY_TS"]
    )

    pp2 = pd.Timestamp(
        trajectory["PP2_BOUNDARY_TS"]
    )

    drr = pd.Timestamp(
        trajectory["DRR_BOUNDARY_TS"]
    )

    row = {

        "LAB_TIMESTAMP": ts,

        "EST1_RT_HR": trajectory[
            "EST1_RT_HR"
        ],

        "EST2_RT_HR": trajectory[
            "EST2_RT_HR"
        ],

        "PP1_RT_HR": trajectory[
            "PP1_RT_HR"
        ],

        "PP2_RT_HR": trajectory[
            "PP2_RT_HR"
        ],

        "DRR_RT_HR": trajectory[
            "DRR_RT_HR"
        ],

        "TOTAL_RT_HR": sum(
            trajectory[x]
            for x in [
                "EST1_RT_HR",
                "EST2_RT_HR",
                "PP1_RT_HR",
                "PP2_RT_HR",
                "DRR_RT_HR",
            ]
        ),

        "EST1_BOUNDARY_TS": est1,
        "EST2_BOUNDARY_TS": est2,
        "PP1_BOUNDARY_TS": pp1,
        "PP2_BOUNDARY_TS": pp2,
        "DRR_BOUNDARY_TS": drr,
    }


    # ========================================================
    # FIVE CAUSAL SECTION WINDOWS
    # ========================================================

    section_windows = {

        "EST1": (
            est1,
            est2,
        ),

        "EST2": (
            est2,
            pp1,
        ),

        "PP1": (
            pp1,
            pp2,
        ),

        "PP2": (
            pp2,
            drr,
        ),

        "DRR": (
            drr,
            ts,
        ),
    }


    for section, (
        start,
        end,
    ) in section_windows.items():

        row.update(
            extract_window_features(
                dcs,
                start,
                end,
                SECTION_TAGS[section],
                f"{section}_",
            )
        )

        row[
            f"{section}_WINDOW_START"
        ] = start

        row[
            f"{section}_WINDOW_END"
        ] = end

        row[
            f"{section}_WINDOW_HR"
        ] = (
            end - start
        ).total_seconds() / 3600.0

        row[
            f"{section}_DCS_N"
        ] = len(
            dcs[
                (dcs["TIMESTAMP"] >= start)
                &
                (dcs["TIMESTAMP"] <= end)
            ]
        )


    # ========================================================
    # FULL CAUSAL WINDOW
    # ========================================================

    global_start = est1
    global_end = ts


    row.update(
        extract_window_features(
            dcs,
            global_start,
            global_end,
            AREA_TAGS["PASTE"],
            "PASTE_",
        )
    )


    row.update(
        extract_window_features(
            dcs,
            global_start,
            global_end,
            AREA_TAGS["THROUGHPUT"],
            "THROUGHPUT_",
        )
    )


    row.update(
        extract_window_features(
            dcs,
            global_start,
            global_end,
            AREA_TAGS["JET"],
            "JET_",
        )
    )


    row.update(
        extract_window_features(
            dcs,
            global_start,
            global_end,
            AREA_TAGS["HTM"],
            "HTM_",
        )
    )


    # ========================================================
    # CHEMISTRY TRAJECTORY
    # ========================================================

    chemistry_groups = {

        "ANTIMONY": [
            "FIC-21009",
            "YK-21010",
        ],

        "RED_TONER": [
            "FIC-41007",
            "YK-41008",
        ],

        "PHOSPHORIC_ACID": [
            "FIC-43007",
            "YK-43008",
        ],

        "BLUE_TONER": [
            "FIC-42007",
            "YK-42008",
        ],
    }


    for group, tags in chemistry_groups.items():

        row.update(
            extract_window_features(
                dcs,
                global_start,
                global_end,
                tags,
                f"CHEM_{group}_",
            )
        )


    # ========================================================
    # HISTORICAL COBALT ALIASES
    # ========================================================

    cobalt = extract_window_features(
        dcs,
        global_start,
        global_end,
        [
            "FIC-41007",
            "YK-41008",
        ],
        "CHEM_COBALT_",
    )

    row.update(cobalt)


    # ========================================================
    # LOCALIZED PASTE CHEMISTRY
    # ========================================================

    for minutes in [
        30,
        60,
        120,
        240,
    ]:

        row.update(
            extract_window_features(
                dcs,
                est1
                - pd.Timedelta(
                    minutes=minutes
                ),
                est1,
                PASTE_CHEM_TAGS,
                f"PASTE_{minutes}M_",
            )
        )


    row.update(
        extract_window_features(
            dcs,
            est1
            - pd.Timedelta(
                hours=4
            ),
            est1,
            PASTE_CHEM_TAGS,
            "PASTE_4H_",
        )
    )


    # ========================================================
    # H3PO4 LOCALIZED AT EST-2
    # ========================================================

    row.update(
        extract_window_features(
            dcs,
            est2,
            pp1,
            EST2_CHEM_TAGS,
            "EST2_H3PO4_FULL_",
        )
    )


    for minutes in [
        30,
        60,
        120,
    ]:

        end = min(
            est2
            + pd.Timedelta(
                minutes=minutes
            ),
            pp1,
        )

        if end > est2:

            row.update(
                extract_window_features(
                    dcs,
                    est2,
                    end,
                    EST2_CHEM_TAGS,
                    f"EST2_H3PO4_{minutes}M_",
                )
            )


    # ========================================================
    # PROCESS CONTROL LAYER
    # ========================================================

    window = dcs[
        (dcs["TIMESTAMP"] >= est1)
        &
        (dcs["TIMESTAMP"] <= ts)
    ].copy()


    if not window.empty:

        # ----------------------------------------------------
        # TPD SUMMARY
        # ----------------------------------------------------

        if "TPD" in window.columns:

            tpd = pd.to_numeric(
                window["TPD"],
                errors="coerce",
            ).dropna()

            if len(tpd):

                row[
                    "CTRL_TPD_MEAN"
                ] = float(tpd.mean())

                row[
                    "CTRL_TPD_STD"
                ] = float(
                    tpd.std(ddof=0)
                )

                row[
                    "CTRL_TPD_MIN"
                ] = float(tpd.min())

                row[
                    "CTRL_TPD_MAX"
                ] = float(tpd.max())

                row[
                    "CTRL_TPD_RANGE"
                ] = float(
                    tpd.max() - tpd.min()
                )

                row[
                    "CTRL_TPD_FIRST"
                ] = float(
                    tpd.iloc[0]
                )

                row[
                    "CTRL_TPD_LAST"
                ] = float(
                    tpd.iloc[-1]
                )

                row[
                    "CTRL_TPD_DELTA"
                ] = float(
                    tpd.iloc[-1]
                    - tpd.iloc[0]
                )

                for q in [
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]:

                    row[
                        f"CTRL_TPD_P{int(q * 100)}"
                    ] = float(
                        tpd.quantile(q)
                    )


        # ----------------------------------------------------
        # PROCESS SUMMARY
        # ----------------------------------------------------

        process_summary = {}


        for tag in PROCESS_TAGS:

            if (
                tag == "TPD"
                or tag not in window.columns
            ):

                continue

            values = pd.to_numeric(
                window[tag],
                errors="coerce",
            ).dropna()

            if len(values) == 0:

                continue

            clean = tag.replace(
                "-",
                "_",
            )

            process_summary[tag] = {

                "MEAN": float(
                    values.mean()
                ),

                "STD": float(
                    values.std(ddof=0)
                ),

                "MIN": float(
                    values.min()
                ),

                "MAX": float(
                    values.max()
                ),

                "RANGE": float(
                    values.max()
                    - values.min()
                ),

                "FIRST": float(
                    values.iloc[0]
                ),

                "LAST": float(
                    values.iloc[-1]
                ),

                "DELTA": float(
                    values.iloc[-1]
                    - values.iloc[0]
                ),
            }


            for stat, value in process_summary[
                tag
            ].items():

                row[
                    f"CTRL_{clean}_{stat}"
                ] = value


        # ====================================================
        # SAFE TPD
        # ====================================================

        tpd_mean = row.get(
            "CTRL_TPD_MEAN",
            np.nan,
        )

        if (
            pd.notna(tpd_mean)
            and abs(tpd_mean) >= 50
        ):

            safe_tpd = tpd_mean

        else:

            safe_tpd = np.nan


        # ====================================================
        # IMPORTANT FIX 1
        #
        # ALWAYS CREATE CTRL_RATIO FEATURES
        #
        # If safe_tpd is unavailable, the feature is NaN.
        # It is later converted to 0.0 before prediction.
        # ====================================================

        for tag, stats in process_summary.items():

            clean = tag.replace(
                "-",
                "_",
            )

            ratio_feature = (
                f"CTRL_RATIO_{clean}"
                "_MEAN_PER_TPD"
            )

            if pd.notna(safe_tpd):

                row[
                    ratio_feature
                ] = (
                    stats["MEAN"]
                    / safe_tpd
                )

            else:

                row[
                    ratio_feature
                ] = np.nan


        # ====================================================
        # TPD DELTA
        # ====================================================

        tpd_delta = row.get(
            "CTRL_TPD_DELTA",
            np.nan,
        )


        # ====================================================
        # IMPORTANT FIX 2
        #
        # SAFE DELTA MUST BE DEFINED BEFORE USE
        # ====================================================

        if (
            pd.notna(tpd_delta)
            and abs(tpd_delta) >= 1
        ):

            safe_delta = tpd_delta

        else:

            safe_delta = np.nan


        # ====================================================
        # IMPORTANT FIX 3
        #
        # ALWAYS CREATE CTRL_RESPONSE FEATURES
        #
        # If TPD change is too small, use NaN.
        # Later NaN -> 0.0 before CatBoost.
        # ====================================================

        for tag, stats in process_summary.items():

            clean = tag.replace(
                "-",
                "_",
            )

            response_feature = (
                f"CTRL_RESPONSE_{clean}"
                "_PER_TPD_CHANGE"
            )

            if pd.notna(safe_delta):

                row[
                    response_feature
                ] = (
                    stats["DELTA"]
                    / safe_delta
                )

            else:

                row[
                    response_feature
                ] = np.nan


    # ========================================================
    # CHEMISTRY / THROUGHPUT INTERACTIONS
    # ========================================================

    safe_tpd = row.get(
        "CTRL_TPD_MEAN",
        np.nan,
    )

    if (
        pd.notna(safe_tpd)
        and abs(safe_tpd) >= 50
    ):

        safe_tpd = safe_tpd

    else:

        safe_tpd = np.nan


    chemistry_tokens = {

        "ANTIMONY": "FIC-21009",

        "RED_TONER": "FIC-41007",

        "BLUE_TONER": "FIC-42007",

        "H3PO4": "FIC-43007",
    }


    for chemistry, tag in chemistry_tokens.items():

        matching = [

            c
            for c in list(row.keys())

            if tag in str(c)

            and (
                "_MEAN" in str(c)
                or "_LAST" in str(c)
                or "_FIRST" in str(c)
                or "_DELTA" in str(c)
            )
        ]


        for col in matching:

            value = pd.to_numeric(
                pd.Series(
                    [row[col]]
                ),
                errors="coerce",
            ).iloc[0]


            feature_name = (
                f"CHEM_CTRL_{chemistry}_"
                f"{col}_PER_TPD"
            )


            if pd.notna(safe_tpd):

                row[
                    feature_name
                ] = (
                    value
                    / safe_tpd
                )

            else:

                row[
                    feature_name
                ] = np.nan


    # ========================================================
    # BASELINE C THROUGHPUT REGIME
    # ========================================================

    if (
        not window.empty
        and "TPD" in window.columns
    ):

        tpd = pd.to_numeric(
            window["TPD"],
            errors="coerce",
        )

        valid = tpd.notna()

        if valid.any():

            times = window.loc[
                valid,
                "TIMESTAMP",
            ]

            values = tpd.loc[
                valid
            ].astype(float)

            current = float(
                values.iloc[-1]
            )

            full_mean = float(
                values.mean()
            )

            first = float(
                values.iloc[0]
            )


            # ------------------------------------------------
            # RECENT TPD WINDOWS
            # ------------------------------------------------

            for minutes in [
                30,
                60,
                120,
                240,
            ]:

                cutoff = (
                    ts
                    - pd.Timedelta(
                        minutes=minutes
                    )
                )

                recent = values.loc[
                    times >= cutoff
                ]

                if recent.empty:

                    recent = values

                suffix = (
                    f"{minutes}M"
                )


                row[
                    f"C_TPD_{suffix}_MEAN"
                ] = float(
                    recent.mean()
                )


                row[
                    f"C_TPD_{suffix}_STD"
                ] = float(
                    recent.std(
                        ddof=0
                    )
                )


                row[
                    f"C_TPD_{suffix}_MIN"
                ] = float(
                    recent.min()
                )


                row[
                    f"C_TPD_{suffix}_MAX"
                ] = float(
                    recent.max()
                )


                row[
                    f"C_TPD_{suffix}_DELTA"
                ] = float(
                    recent.iloc[-1]
                    - recent.iloc[0]
                )


                row[
                    f"C_TPD_{suffix}_RATE"
                ] = float(
                    (
                        recent.iloc[-1]
                        - recent.iloc[0]
                    )
                    / (
                        minutes / 60.0
                    )
                )


            # ------------------------------------------------
            # ABSOLUTE THROUGHPUT FEATURES
            # ------------------------------------------------

            row[
                "C_TPD_CURRENT"
            ] = current

            row[
                "C_TPD_MINUS_400"
            ] = current - 400.0

            row[
                "C_TPD_RATIO_TO_400"
            ] = current / 400.0

            row[
                "C_TPD_MEAN_MINUS_400"
            ] = full_mean - 400.0

            row[
                "C_TPD_MEAN_RATIO_TO_400"
            ] = full_mean / 400.0


            # ------------------------------------------------
            # THROUGHPUT REGIME FRACTIONS
            # ------------------------------------------------

            row[
                "C_TPD_FRAC_BELOW_500"
            ] = float(
                (values < 500).mean()
            )


            row[
                "C_TPD_FRAC_500_600"
            ] = float(
                (
                    (values >= 500)
                    &
                    (values < 600)
                ).mean()
            )


            row[
                "C_TPD_FRAC_600_800"
            ] = float(
                (
                    (values >= 600)
                    &
                    (values <= 800)
                ).mean()
            )


            row[
                "C_TPD_FRAC_ABOVE_800"
            ] = float(
                (values > 800).mean()
            )


            row[
                "C_TPD_FRAC_GE_600"
            ] = float(
                (values >= 600).mean()
            )


            # ------------------------------------------------
            # CURRENT REGIME
            # ------------------------------------------------

            if current < 500:

                regime = 0

            elif current < 600:

                regime = 1

            elif current <= 800:

                regime = 2

            else:

                regime = 3


            row[
                "C_TPD_CURRENT_REGIME"
            ] = regime


            # ------------------------------------------------
            # REGIME CROSSING FLAGS
            # ------------------------------------------------

            row[
                "C_TPD_CROSSED_500"
            ] = float(
                first < 500
                and current >= 500
            )


            row[
                "C_TPD_CROSSED_600"
            ] = float(
                first < 600
                and current >= 600
            )


            row[
                "C_TPD_CROSSED_800"
            ] = float(
                first < 800
                and current >= 800
            )


            row[
                "C_TPD_STARTED_BELOW_600_ENDED_GE_600"
            ] = float(
                first < 600
                and current >= 600
            )


            row[
                "C_TPD_STARTED_BELOW_800_ENDED_GE_800"
            ] = float(
                first < 800
                and current >= 800
            )


            # ------------------------------------------------
            # THRESHOLD DISTANCES
            # ------------------------------------------------

            row[
                "C_TPD_MINUS_600"
            ] = current - 600.0

            row[
                "C_TPD_MINUS_800"
            ] = current - 800.0

            row[
                "C_TPD_MEAN_MINUS_600"
            ] = full_mean - 600.0

            row[
                "C_TPD_MEAN_MINUS_800"
            ] = full_mean - 800.0


            # ------------------------------------------------
            # RECENT VS FULL TPD
            # ------------------------------------------------

            for minutes in [
                30,
                60,
                120,
                240,
            ]:

                cutoff = (
                    ts
                    - pd.Timedelta(
                        minutes=minutes
                    )
                )

                recent = values.loc[
                    times >= cutoff
                ]

                if recent.empty:

                    recent = values


                row[
                    f"C_TPD_{minutes}M_VS_FULL_DELTA"
                ] = float(
                    recent.mean()
                    - full_mean
                )


                row[
                    f"C_TPD_{minutes}M_VS_FULL_RATIO"
                ] = (

                    float(
                        recent.mean()
                        / full_mean
                    )

                    if full_mean != 0

                    else np.nan
                )


    return row


# ============================================================
# DYNAMIC FEATURES
# ============================================================

DYNAMIC_EXCLUDE = {

    "LAB_TIMESTAMP",

    "EST1_BOUNDARY_TS",

    "EST2_BOUNDARY_TS",

    "PP1_BOUNDARY_TS",

    "PP2_BOUNDARY_TS",

    "DRR_BOUNDARY_TS",
}


def add_pairwise_dynamic(
    current,
    previous,
):

    """
    Raw-DCS deployment equivalent of the training CHANGE /
    RELCHANGE / RATE feature family.

    No previous LAB colour is used.
    """

    result = current.copy()

    current_ts = pd.Timestamp(
        current["LAB_TIMESTAMP"]
    )

    previous_ts = pd.Timestamp(
        previous["LAB_TIMESTAMP"]
    )

    interval_hr = (
        current_ts - previous_ts
    ).total_seconds() / 3600.0


    result[
        "LAB_INTERVAL_HR"
    ] = interval_hr


    result[
        "DYNAMIC_CHANGE_VALID"
    ] = (
        0 < interval_hr <= 12
    )


    # ========================================================
    # IMPORTANT:
    # Include columns even when their current value is NaN.
    #
    # This prevents a dynamic feature from disappearing from
    # the feature schema simply because the current process
    # value could not be calculated.
    # ========================================================

    numeric_cols = []


    all_cols = set(
        current.keys()
    ).union(
        previous.keys()
    )


    for col in all_cols:

        if col in DYNAMIC_EXCLUDE:

            continue

        if col in {
            "LAB_INTERVAL_HR",
            "DYNAMIC_CHANGE_VALID",
        }:

            continue


        cur_value = current.get(
            col,
            np.nan,
        )

        prev_value = previous.get(
            col,
            np.nan,
        )


        cur = pd.to_numeric(
            pd.Series(
                [cur_value]
            ),
            errors="coerce",
        ).iloc[0]


        prev = pd.to_numeric(
            pd.Series(
                [prev_value]
            ),
            errors="coerce",
        ).iloc[0]


        # Include the feature family if either current or
        # previous value is numeric / potentially numeric.
        if (
            pd.notna(cur)
            or pd.notna(prev)
        ):

            numeric_cols.append(col)


    for col in numeric_cols:

        cur = pd.to_numeric(
            pd.Series(
                [current.get(
                    col,
                    np.nan
                )]
            ),
            errors="coerce",
        ).iloc[0]


        prev = pd.to_numeric(
            pd.Series(
                [previous.get(
                    col,
                    np.nan
                )]
            ),
            errors="coerce",
        ).iloc[0]


        if (
            pd.notna(cur)
            and pd.notna(prev)
        ):

            change = (
                cur - prev
            )

        else:

            change = np.nan


        result[
            f"CHANGE__{col}"
        ] = change


        denominator = (

            abs(prev)

            if pd.notna(prev)

            else np.nan
        )


        if pd.notna(
            denominator
        ):

            denominator = max(
                denominator,
                1e-3,
            )


        if (
            pd.notna(change)
            and pd.notna(denominator)
        ):

            relchange = (
                change
                / denominator
            )

        else:

            relchange = np.nan


        result[
            f"RELCHANGE__{col}"
        ] = relchange


        if (
            pd.notna(change)
            and interval_hr > 0
        ):

            rate = (
                change
                / interval_hr
            )

        else:

            rate = np.nan


        result[
            f"RATE__{col}"
        ] = rate


    return result


# ============================================================
# FIND PREVIOUS VALID DCS PREDICTION STATE
# ============================================================

def find_previous_timestamp(
    dcs,
    selected_ts,
):

    lower = (
        selected_ts
        - pd.Timedelta(
            hours=DYNAMIC_LOOKBACK_HOURS
        )
    )

    candidates = dcs[
        (dcs["TIMESTAMP"] < selected_ts)
        &
        (dcs["TIMESTAMP"] >= lower)
        &
        dcs["RT_VALID"]
    ]

    if candidates.empty:

        return None

    return candidates.iloc[
        -1
    ]["TIMESTAMP"]


# ============================================================
# BUILD MODEL INPUT FOR ONE SELECTED TIMESTAMP
# ============================================================

def build_prediction_feature_row(
    dcs,
    runs,
    selected_ts,
):

    trajectory = calculate_trajectory(
        dcs,
        runs,
        selected_ts,
    )


    if trajectory is None:

        raise ValueError(
            "The selected timestamp does not have a valid "
            "continuous causal RT trajectory."
        )


    current_static = build_static_features(
        trajectory,
        dcs,
    )


    previous_ts = find_previous_timestamp(
        dcs,
        selected_ts,
    )


    dynamic_row = (
        current_static.copy()
    )


    if previous_ts is not None:

        previous_trajectory = (
            calculate_trajectory(
                dcs,
                runs,
                previous_ts,
            )
        )


        if previous_trajectory is not None:

            previous_static = (
                build_static_features(
                    previous_trajectory,
                    dcs,
                )
            )


            dynamic_row = (
                add_pairwise_dynamic(
                    current_static,
                    previous_static,
                )
            )


    return (
        pd.DataFrame(
            [dynamic_row]
        ),
        trajectory,
        previous_ts,
    )


# ============================================================
# FEATURE AUDIT + PREDICTION
# ============================================================

def prepare_model_matrix(
    feature_row,
    saved_features,
    target,
):

    selected = saved_features[
        target
    ]


    # --------------------------------------------------------
    # Check missing columns
    # --------------------------------------------------------

    missing = [
        c
        for c in selected
        if c not in feature_row.columns
    ]


    if missing:

        return None, missing


    # --------------------------------------------------------
    # Prepare exact model feature order
    #
    # NaN is converted to 0.0 HERE.
    #
    # This means feature engineering can preserve NaN to
    # indicate "not reliably calculable", while CatBoost
    # receives the same numeric input expected by deployment.
    # --------------------------------------------------------

    X = (
        feature_row[selected]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )


    return X, []


def run_prediction(
    feature_row,
    models,
    saved_features,
):

    matrices = {}

    failures = {}


    for target in TARGETS:

        key = (
            f"LAB_{target}"
        )


        selected = saved_features.get(
            key
        )


        if selected is None:

            failures[target] = [
                f"Missing JSON key: {key}"
            ]

            continue


        X, missing = (
            prepare_model_matrix(
                feature_row,
                saved_features,
                key,
            )
        )


        if missing:

            failures[target] = (
                missing
            )

        else:

            matrices[target] = X


    if failures:

        return None, failures


    result = {}


    for target in TARGETS:

        result[
            f"PRED_{target}"
        ] = float(
            models[target].predict(
                matrices[target]
            )[0]
        )


    return result, {}


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="POLY-II Colour Prediction",
    page_icon="🎨",
    layout="wide",
)


# ============================================================
# APPLICATION RESET CONTROL
# ============================================================

if (
    "app_reset_id"
    not in st.session_state
):

    st.session_state.app_reset_id = 0


# ============================================================
# TOP HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [8, 1.5]
)


with header_col1:

    st.title(
        "POLY-II Colour Prediction"
    )

    st.caption(
        "Single timestamp prediction "
        "from raw 15-minute DCS"
    )


with header_col2:

    st.markdown(
        "<div style='height: 15px'></div>",
        unsafe_allow_html=True,
    )


    reset_clicked = st.button(
        "↻  New Prediction",
        use_container_width=True,
        help=(
            "Clear the uploaded DCS file, selected "
            "date/time and all prediction results."
        ),
    )


# ============================================================
# RESET EVERYTHING
# ============================================================

if reset_clicked:

    current_reset_id = (
        st.session_state.app_reset_id
    )


    for key in list(
        st.session_state.keys()
    ):

        if key != "app_reset_id":

            del st.session_state[
                key
            ]


    st.session_state.app_reset_id = (
        current_reset_id + 1
    )


    st.cache_data.clear()

    st.rerun()


st.divider()


# ============================================================
# 1. UPLOAD DCS DATA
# ============================================================

st.subheader(
    "1. Upload DCS Data"
)


st.write(
    "Upload the raw 15-minute DCS Excel file to begin prediction."
)


uploaded_dcs = st.file_uploader(
    "Raw DCS Excel File",
    type=[
        "xlsx",
        "xls",
    ],
    help=(
        "Upload the raw POLY-II DCS Excel file."
    ),
)


if uploaded_dcs is None:

    st.info(
        "Please upload the raw DCS Excel file."
    )


    st.caption(
        "The application uses the bundled Nova residence-time "
        "data and Baseline C colour prediction models automatically."
    )

    st.stop()


# ============================================================
# 2. VALIDATE APPLICATION FILES
# ============================================================

required_paths = {

    "Nova_Part-A.xlsx":
        NOVA_FILE,

    "catboost_LAB_L_BaselineC.cbm":
        MODEL_FILES["L"],

    "catboost_LAB_A_BaselineC.cbm":
        MODEL_FILES["A"],

    "catboost_LAB_B_BaselineC.cbm":
        MODEL_FILES["B"],

    "baseline_c_selected_features.json":
        FEATURE_FILE,
}


missing_files = [
    name
    for name, path
    in required_paths.items()
    if not path.exists()
]


if missing_files:

    st.error(
        "Required application files are missing."
    )


    st.code(
        "\n".join(
            missing_files
        ),
        language="text",
    )


    st.stop()


# ============================================================
# 3. LOAD DATA
# ============================================================

try:

    with st.spinner(
        "Loading DCS data and residence-time configuration..."
    ):

        models, saved_features = (
            load_models()
        )


        dcs = load_raw_dcs(
            uploaded_dcs.getvalue(),
            uploaded_dcs.name,
        )


        curves = load_nova_curves()


        dcs = add_inventory(
            dcs,
            curves,
        )


        dcs = calculate_rt(
            dcs
        )


        dcs, runs = build_runs(
            dcs
        )


except Exception as exc:

    st.error(
        "Application could not load the supplied data."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================

min_timestamp = (
    dcs["TIMESTAMP"].min()
)

max_timestamp = (
    dcs["TIMESTAMP"].max()
)


valid_count = int(
    dcs["RT_VALID"].sum()
)


total_count = len(dcs)


s1, s2, s3 = st.columns(3)


s1.metric(
    "DCS Records",
    f"{total_count:,}",
)


s2.metric(
    "RT-Valid Records",
    f"{valid_count:,}",
)


s3.metric(
    "DCS Period",
    (
        f"{min_timestamp:%d %b %Y}"
        f" – "
        f"{max_timestamp:%d %b %Y}"
    ),
)


st.divider()


# ============================================================
# 4. SELECT DATE AND TIME
# ============================================================

st.subheader(
    "2. Select Prediction Date & Time"
)


st.write(
    "Select the production date and enter the DCS time "
    "for which you want to predict colour."
)


valid_dcs = dcs.loc[
    dcs["RT_VALID"],
    [
        "TIMESTAMP",
        "TPD",
    ],
].copy()


if valid_dcs.empty:

    st.error(
        "No RT-valid production observations were found."
    )

    st.stop()


# ============================================================
# DATE INPUT
# ============================================================

min_date = (
    valid_dcs["TIMESTAMP"]
    .dt.date
    .min()
)


max_date = (
    valid_dcs["TIMESTAMP"]
    .dt.date
    .max()
)


default_date = max_date


c1, c2, c3 = st.columns(
    [
        1.5,
        1,
        0.8,
    ]
)


with c1:

    selected_date = st.date_input(
        "Prediction Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )


# ============================================================
# TIME INPUT
# ============================================================

with c2:

    selected_time_text = st.text_input(
        "Time",
        value="08:02:49",
        placeholder="HH:MM:SS",
        help="Enter time as HH:MM:SS.",
    )


# ============================================================
# AM / PM
# ============================================================

with c3:

    selected_ampm = st.selectbox(
        "AM / PM",
        [
            "AM",
            "PM",
        ],
        index=0,
    )


# ============================================================
# PARSE USER TIME
# ============================================================

def parse_user_time(
    time_text,
    ampm,
):

    time_text = str(
        time_text
    ).strip()


    formats = [
        "%I:%M:%S",
        "%I:%M",
        "%H:%M:%S",
        "%H:%M",
    ]


    parsed = None


    for fmt in formats:

        try:

            parsed = pd.to_datetime(
                time_text,
                format=fmt,
            )

            break

        except Exception:

            continue


    if parsed is None:

        raise ValueError(
            "Invalid time format. "
            "Please enter HH:MM or HH:MM:SS."
        )


    hour = parsed.hour

    minute = parsed.minute

    second = parsed.second


    if ampm == "AM":

        if hour == 12:

            hour = 0

    else:

        if hour < 12:

            hour += 12


    return pd.Timestamp(
        year=selected_date.year,
        month=selected_date.month,
        day=selected_date.day,
        hour=hour,
        minute=minute,
        second=second,
    )


# ============================================================
# BUILD SELECTED TIMESTAMP
# ============================================================

try:

    selected_ts = parse_user_time(
        selected_time_text,
        selected_ampm,
    )

except ValueError as exc:

    st.warning(
        str(exc)
    )

    st.stop()


# ============================================================
# FIND NEAREST / EXACT DCS TIMESTAMP
# ============================================================

selected_rows = dcs[
    dcs["TIMESTAMP"]
    == selected_ts
]


if selected_rows.empty:

    nearest_idx = (
        (
            dcs["TIMESTAMP"]
            - selected_ts
        )
        .abs()
        .idxmin()
    )


    nearest_ts = dcs.loc[
        nearest_idx,
        "TIMESTAMP",
    ]


    difference_seconds = abs(
        (
            nearest_ts
            - selected_ts
        ).total_seconds()
    )


    if difference_seconds <= 60:

        st.info(
            f"No exact DCS timestamp exists at "
            f"{selected_ts.strftime('%I:%M:%S %p')}. "
            f"Using nearest DCS timestamp: "
            f"{nearest_ts.strftime('%I:%M:%S %p')}."
        )


        selected_ts = nearest_ts


        selected_rows = dcs[
            dcs["TIMESTAMP"]
            == selected_ts
        ]


    else:

        st.error(
            "The entered time does not exist in the "
            "uploaded DCS data."
        )


        st.caption(
            f"Entered: "
            f"{selected_ts.strftime('%d/%m/%Y %I:%M:%S %p')}"
        )


        st.caption(
            "Please enter a time available in the "
            "15-minute DCS data."
        )


        st.stop()


selected_row = (
    selected_rows.iloc[0]
)


# ============================================================
# SELECTED PROCESS CONDITION
# ============================================================

st.markdown(
    "### Selected Process Condition"
)


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Selected Timestamp",
    selected_ts.strftime(
        "%d/%m/%Y %I:%M:%S %p"
    ),
)


m2.metric(
    "TPD",
    (
        f"{selected_row['TPD']:.2f}"
        if pd.notna(
            selected_row["TPD"]
        )
        else "N/A"
    ),
)


m3.metric(
    "RT Status",
    str(
        selected_row[
            "RT_STATUS"
        ]
    ),
)


m4.metric(
    "Total Residence Time",
    (
        f"{selected_row['TOTAL_RT_HR']:.2f} h"
        if pd.notna(
            selected_row[
                "TOTAL_RT_HR"
            ]
        )
        else "N/A"
    ),
)


# ============================================================
# SHUTDOWN / INVALID STATUS
# ============================================================

if not bool(
    selected_row["RT_VALID"]
):

    st.error(
        "Prediction is unavailable for this timestamp."
    )


    if (
        pd.notna(
            selected_row["TPD"]
        )
        and
        selected_row["TPD"] <= 0
    ):

        st.warning(
            "TPD ≤ 0 indicates shutdown/non-production."
        )

    else:

        st.warning(
            "The selected timestamp does not satisfy "
            "the RT-valid production criteria."
        )


    st.stop()


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()


predict_clicked = st.button(
    "🎨  PREDICT COLOUR",
    type="primary",
    use_container_width=True,
)


if predict_clicked:

    try:

        with st.spinner(
            "Calculating residence time and extracting "
            "the causal DCS window..."
        ):

            (
                feature_row,
                trajectory,
                previous_ts,
            ) = (
                build_prediction_feature_row(
                    dcs,
                    runs,
                    selected_ts,
                )
            )


            prediction, failures = (
                run_prediction(
                    feature_row,
                    models,
                    saved_features,
                )
            )


        # ====================================================
        # FEATURE FAILURE
        # ====================================================

        if failures:

            st.error(
                "Prediction stopped because the saved "
                "Baseline C feature definition could not "
                "be reproduced from the supplied DCS data."
            )


            for target, cols in failures.items():

                st.write(
                    f"**LAB_{target}: "
                    f"{len(cols)} missing feature(s)**"
                )


                st.code(
                    "\n".join(cols),
                    language="text",
                )


            st.info(
                "The model feature definition must match "
                "the feature engineering used during training."
            )


            st.stop()


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "Prediction completed successfully."
        )


        # ====================================================
        # PREDICTED COLOUR
        # ====================================================

        st.subheader(
            "3. Predicted Colour"
        )


        p1, p2, p3 = st.columns(3)


        p1.metric(
            "L",
            f"{prediction['PRED_L']:.3f}",
        )


        p2.metric(
            "a",
            f"{prediction['PRED_A']:.3f}",
        )


        p3.metric(
            "b",
            f"{prediction['PRED_B']:.3f}",
        )


        # ====================================================
        # CAUSAL TRAJECTORY
        # ====================================================

        st.subheader(
            "4. Causal Residence-Time Window"
        )


        trajectory_table = pd.DataFrame(
            {

                "Point": [

                    "Prediction Time",

                    "DRR Boundary",

                    "PP2 Boundary",

                    "PP1 Boundary",

                    "EST2 Boundary",

                    "EST1 Upstream Boundary",
                ],


                "Timestamp": [

                    selected_ts,

                    trajectory[
                        "DRR_BOUNDARY_TS"
                    ],

                    trajectory[
                        "PP2_BOUNDARY_TS"
                    ],

                    trajectory[
                        "PP1_BOUNDARY_TS"
                    ],

                    trajectory[
                        "EST2_BOUNDARY_TS"
                    ],

                    trajectory[
                        "EST1_BOUNDARY_TS"
                    ],
                ],
            }
        )


        trajectory_table[
            "Timestamp"
        ] = (
            trajectory_table[
                "Timestamp"
            ]
            .dt.strftime(
                "%d/%m/%Y %I:%M:%S %p"
            )
        )


        st.dataframe(
            trajectory_table,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # SECTION RESIDENCE TIMES
        # ====================================================

        r1, r2, r3, r4, r5 = (
            st.columns(5)
        )


        r1.metric(
            "EST1 RT",
            f"{trajectory['EST1_RT_HR']:.2f} h",
        )


        r2.metric(
            "EST2 RT",
            f"{trajectory['EST2_RT_HR']:.2f} h",
        )


        r3.metric(
            "PP1 RT",
            f"{trajectory['PP1_RT_HR']:.2f} h",
        )


        r4.metric(
            "PP2 RT",
            f"{trajectory['PP2_RT_HR']:.2f} h",
        )


        r5.metric(
            "DRR RT",
            f"{trajectory['DRR_RT_HR']:.2f} h",
        )


        # ====================================================
        # DEPLOYMENT DYNAMIC REFERENCE
        # ====================================================

        st.subheader(
            "5. Process Dynamic Reference"
        )


        if previous_ts is not None:

            d1, d2 = st.columns(2)


            d1.metric(
                "Current Prediction Time",
                selected_ts.strftime(
                    "%d/%m/%Y %I:%M:%S %p"
                ),
            )


            d2.metric(
                "Previous Valid DCS State",
                pd.Timestamp(
                    previous_ts
                ).strftime(
                    "%d/%m/%Y %I:%M:%S %p"
                ),
            )


        else:

            st.warning(
                "No previous valid DCS state was found "
                f"within {DYNAMIC_LOOKBACK_HOURS:.1f} hours."
            )


        # ====================================================
        # DOWNLOAD RESULT
        # ====================================================

        export = pd.DataFrame(
            [

                {

                    "TIMESTAMP":
                        selected_ts,

                    "TPD":
                        selected_row["TPD"],

                    "TOTAL_RT_HR":
                        selected_row[
                            "TOTAL_RT_HR"
                        ],

                    "PRED_L":
                        prediction[
                            "PRED_L"
                        ],

                    "PRED_A":
                        prediction[
                            "PRED_A"
                        ],

                    "PRED_B":
                        prediction[
                            "PRED_B"
                        ],
                }
            ]
        )


        st.download_button(
            "Download Prediction CSV",

            export.to_csv(
                index=False
            ),

            file_name=(
                "POLY2_prediction_"
                f"{selected_ts.strftime('%Y%m%d_%H%M')}"
                ".csv"
            ),

            mime="text/csv",
        )


    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "POLY-II Baseline C | Raw 15-minute DCS | "
    "Causal residence-time inference | "
    "202-F0 excluded | No previous Lab colour values used"
)


# # # ============================================================
# # # POLY-II BASELINE C — SINGLE DATE/TIME RAW-DCS COLOUR PREDICTOR
# # # ============================================================
# # #
# # # USER WORKFLOW
# # # --------------
# # # 1. Select a date.
# # # 2. Select a DCS time.
# # # 3. Click PREDICT COLOUR.
# # #
# # # The application then:
# # #   RAW DCS
# # #      -> Nova inventory conversion
# # #      -> residence time
# # #      -> RT-valid production check
# # #      -> backward causal trajectory
# # #      -> causal process features
# # #      -> localized paste chemistry
# # #      -> EST-2 H3PO4 features
# # #      -> process-control features
# # #      -> chemistry/TPD interactions
# # #      -> Baseline-C throughput regime features
# # #      -> deployment-time dynamic features
# # #      -> exact saved feature lists
# # #      -> saved CatBoost models
# # #      -> LAB_L / LAB_A / LAB_B
# # #
# # # IMPORTANT
# # # ----------
# # # * No previous LAB colour value is used.
# # # * BA202060-F0 is excluded.
# # # * TPD <= 0 is treated as shutdown.
# # # * A prediction is generated ONLY for the selected timestamp.
# # # * The app refuses to predict if the saved feature definition
# # #   cannot be reproduced.
# # #
# # # ============================================================

# # from pathlib import Path
# # import json
# # import warnings

# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # from catboost import CatBoostRegressor

# # warnings.filterwarnings("ignore")

# # # ============================================================
# # # PATH CONFIGURATION
# # # ============================================================
# # #
# # # No machine-specific Windows path is hardcoded.
# # #
# # # USER SETUP
# # # ----------
# # # Enter a Base Folder in the Streamlit sidebar. Put these inside it:
# # #
# # #   Nova_Part-A.xlsx
# # #   poly2_baseline_c_models_TP_Regime_Change_v4/
# # #
# # # The raw DCS Excel file is uploaded through the Streamlit file
# # # uploader, so its filename/path is NOT hardcoded.
# # #
# # # If your model folder has another name, change it in the sidebar.
# # # ============================================================

# # DEFAULT_MODEL_DIR_NAME = "poly2_baseline_c_models_TP_Regime_Change_v4"

# # BASE_DIR = None
# # NOVA_FILE = None
# # MODEL_DIR = None
# # MODEL_FILES = {}
# # FEATURE_FILE = None
# # RAW_DCS_FILE = None

# # TARGETS = ["L", "A", "B"]

# # # Deployment dynamic comparison.
# # #
# # # Training dynamics were calculated between successive Lab
# # # observations. Raw DCS has no future Lab observation available
# # # at prediction time, so deployment compares the selected causal
# # # feature state against the closest earlier valid DCS prediction
# # # state within this lookback.
# # #
# # # This is intentionally explicit rather than silently inventing
# # # previous Lab colour values.
# # DYNAMIC_LOOKBACK_HOURS = 4.0

# # # ============================================================
# # # DCS TAGS
# # # ============================================================

# # DCS_SENSOR_COLS = [
# #     "YK-11001",
# #     "TPD",
# #     "FIC-12001",
# #     "LIC-12019",
# #     "TIC-12012",
# #     "PIC-12013",
# #     "LIC-13017",
# #     "TIC-13012",
# #     "LIC-14003",
# #     "TIC-14004",
# #     "PIC-14005",
# #     "LIC-15006",
# #     "TIC-15005",
# #     "TIC-15008",
# #     "PIC-15024",
# #     "SIK-17015",
# #     "SIK-17028",
# #     "IC-17013",
# #     "PIC-17024",
# #     "PV-17024",
# #     "LI-17017",
# #     "LI-17023",
# #     "LIC-17016",
# #     "TI-17019",
# #     "TI-17020",
# #     "TI-17022",
# #     "VIC-18020",
# #     "TI-17113",
# #     "TI-17114",
# #     "PIC-17183",
# #     "LIC-17175",
# #     "TI-50002",
# #     "TI-50003",
# #     "FIC-21009",
# #     "YK-21010",
# #     "FIC-41007",
# #     "YK-41008",
# #     "FIC-43007",
# #     "YK-43008",
# #     "FIC-42007",
# #     "YK-42008",
# # ]

# # PASTE_CHEM_TAGS = [
# #     "FIC-21009",
# #     "YK-21010",
# #     "FIC-41007",
# #     "YK-41008",
# #     "FIC-42007",
# #     "YK-42008",
# # ]

# # EST2_CHEM_TAGS = [
# #     "FIC-43007",
# #     "YK-43008",
# # ]

# # PROCESS_TAGS = [
# #     "TPD",
# #     "TIC-12012",
# #     "PIC-12013",
# #     "TIC-13012",
# #     "TIC-14004",
# #     "PIC-14005",
# #     "TIC-15005",
# #     "TIC-15008",
# #     "PIC-15024",
# #     "TI-17019",
# #     "TI-17020",
# #     "TI-17022",
# #     "PIC-17024",
# # ]

# # AREA_TAGS = {
# #     "PASTE": ["YK-11001"],
# #     "THROUGHPUT": ["TPD"],

# #     "EST1": [
# #         "FIC-12001",
# #         "LIC-12019",
# #         "TIC-12012",
# #         "PIC-12013",
# #     ],

# #     "EST2": [
# #         "LIC-13017",
# #         "TIC-13012",
# #     ],

# #     "PP1": [
# #         "LIC-14003",
# #         "TIC-14004",
# #         "PIC-14005",
# #     ],

# #     "PP2": [
# #         "LIC-15006",
# #         "TIC-15005",
# #         "TIC-15008",
# #         "PIC-15024",
# #     ],

# #     "DRR": [
# #         "SIK-17015",
# #         "SIK-17028",
# #         "IC-17013",
# #         "PIC-17024",
# #         "PV-17024",
# #         "LI-17017",
# #         "LI-17023",
# #         "LIC-17016",
# #         "TI-17019",
# #         "TI-17020",
# #         "TI-17022",
# #         "VIC-18020",
# #     ],

# #     "JET": [
# #         "TI-17113",
# #         "TI-17114",
# #         "PIC-17183",
# #         "LIC-17175",
# #     ],

# #     "HTM": [
# #         "TI-50002",
# #         "TI-50003",
# #     ],
# # }

# # SECTION_TAGS = {
# #     "EST1": AREA_TAGS["EST1"],
# #     "EST2": AREA_TAGS["EST2"],
# #     "PP1": AREA_TAGS["PP1"],
# #     "PP2": AREA_TAGS["PP2"],
# #     "DRR": AREA_TAGS["DRR"],
# # }

# # EQUIPMENT_MAP = {
# #     "EST1": {
# #         "DCS_TAG": "LIC-12019",
# #         "EQUIPMENT": "12-R01-EST-I",
# #     },
# #     "EST2": {
# #         "DCS_TAG": "LIC-13017",
# #         "EQUIPMENT": "13-R01-EST-II",
# #     },
# #     "PP1": {
# #         "DCS_TAG": "LIC-14003",
# #         "EQUIPMENT": "14-R01-PP-I",
# #     },
# #     "PP2": {
# #         "DCS_TAG": "LIC-15006",
# #         "EQUIPMENT": "15-R01-PP-II",
# #     },
# #     "DRR": {
# #         "DCS_TAG": "LIC-17016",
# #         "EQUIPMENT": "17-R01- (DRR)",
# #     },
# # }

# # SECTION_ORDER = [
# #     ("DRR", "DRR_RT_HR"),
# #     ("PP2", "PP2_RT_HR"),
# #     ("PP1", "PP1_RT_HR"),
# #     ("EST2", "EST2_RT_HR"),
# #     ("EST1", "EST1_RT_HR"),
# # ]


# # # ============================================================
# # # MODEL LOADING
# # # ============================================================

# # @st.cache_resource
# # def load_models():
# #     if not FEATURE_FILE.exists():
# #         raise FileNotFoundError(
# #             f"Feature JSON not found:\n{FEATURE_FILE}"
# #         )

# #     with open(FEATURE_FILE, "r", encoding="utf-8") as f:
# #         saved_features = json.load(f)

# #     models = {}

# #     for target in TARGETS:
# #         path = MODEL_FILES[target]

# #         if not path.exists():
# #             raise FileNotFoundError(
# #                 f"Model not found:\n{path}"
# #             )

# #         model = CatBoostRegressor()
# #         model.load_model(str(path))
# #         models[target] = model

# #     return models, saved_features


# # # ============================================================
# # # RAW DCS
# # # ============================================================

# # @st.cache_data
# # def load_raw_dcs(uploaded_file_bytes, uploaded_file_name):
# #     if not uploaded_file_bytes:
# #         raise ValueError("Please upload the raw DCS Excel file.")

# #     from io import BytesIO

# #     raw = pd.read_excel(
# #         BytesIO(uploaded_file_bytes),
# #         header=None,
# #     )

# #     if len(raw) < 6:
# #         raise ValueError(
# #             "Unexpected DCS Excel structure."
# #         )

# #     tag_row = raw.iloc[2]

# #     columns = []

# #     for i, value in enumerate(tag_row):
# #         if i == 0:
# #             columns.append("TIMESTAMP")
# #         elif pd.isna(value):
# #             columns.append(f"UNNAMED_{i}")
# #         else:
# #             columns.append(str(value).strip())

# #     dcs = raw.iloc[5:].copy()
# #     dcs.columns = columns
# #     dcs = dcs.dropna(how="all").reset_index(drop=True)

# #     dcs["TIMESTAMP"] = pd.to_datetime(
# #         dcs["TIMESTAMP"],
# #         errors="coerce",
# #     )

# #     dcs = (
# #         dcs.dropna(subset=["TIMESTAMP"])
# #         .sort_values("TIMESTAMP")
# #         .drop_duplicates("TIMESTAMP")
# #         .reset_index(drop=True)
# #     )

# #     missing = [
# #         c for c in DCS_SENSOR_COLS
# #         if c not in dcs.columns
# #     ]

# #     if missing:
# #         raise ValueError(
# #             "Required DCS tags missing:\n"
# #             + "\n".join(missing)
# #         )

# #     for col in DCS_SENSOR_COLS:
# #         dcs[col] = pd.to_numeric(
# #             dcs[col],
# #             errors="coerce",
# #         )

# #     # FLOAT32 invalid sentinel.
# #     sentinel = dcs["YK-43008"].abs() > 1e30
# #     dcs.loc[sentinel, "YK-43008"] = np.nan

# #     dcs["INTERVAL_MIN"] = (
# #         dcs["TIMESTAMP"].diff()
# #         .dt.total_seconds()
# #         / 60.0
# #     )

# #     return dcs


# # # ============================================================
# # # NOVA
# # # ============================================================

# # @st.cache_data
# # def load_nova_curves():
# #     if not NOVA_FILE.exists():
# #         raise FileNotFoundError(
# #             f"Nova file not found:\n{NOVA_FILE}"
# #         )

# #     nova = pd.read_excel(
# #         NOVA_FILE,
# #         sheet_name="Poly-2",
# #         header=1,
# #     )

# #     curves = {}

# #     for section, info in EQUIPMENT_MAP.items():

# #         mask = (
# #             nova["silo_no"]
# #             .astype(str)
# #             .str.strip()
# #             .eq(info["EQUIPMENT"])
# #         )

# #         curve = nova.loc[
# #             mask,
# #             [
# #                 "Level Percent",
# #                 "Level Weight Kgs.",
# #             ],
# #         ].copy()

# #         curve["Level Percent"] = pd.to_numeric(
# #             curve["Level Percent"],
# #             errors="coerce",
# #         )

# #         curve["Level Weight Kgs."] = pd.to_numeric(
# #             curve["Level Weight Kgs."],
# #             errors="coerce",
# #         )

# #         curve = (
# #             curve.dropna()
# #             .drop_duplicates("Level Percent")
# #             .sort_values("Level Percent")
# #             .reset_index(drop=True)
# #         )

# #         if len(curve) < 2:
# #             raise ValueError(
# #                 f"Insufficient Nova curve for {section}."
# #             )

# #         curves[section] = curve

# #     return curves


# # def add_inventory(dcs, curves):
# #     dcs = dcs.copy()

# #     for section, info in EQUIPMENT_MAP.items():

# #         level = pd.to_numeric(
# #             dcs[info["DCS_TAG"]],
# #             errors="coerce",
# #         )

# #         curve = curves[section]

# #         x = curve["Level Percent"].to_numpy(float)
# #         y = curve["Level Weight Kgs."].to_numpy(float)

# #         inventory = np.full(len(dcs), np.nan)

# #         valid = (
# #             level.notna()
# #             & level.ge(x.min())
# #             & level.le(x.max())
# #         )

# #         inventory[valid] = np.interp(
# #             level.loc[valid].to_numpy(float),
# #             x,
# #             y,
# #         )

# #         dcs[f"{section}_LEVEL_PCT"] = level
# #         dcs[f"{section}_INVENTORY_KG"] = inventory

# #     return dcs


# # # ============================================================
# # # RESIDENCE TIME
# # # ============================================================

# # def calculate_rt(dcs):
# #     dcs = dcs.copy()

# #     dcs["TPD"] = pd.to_numeric(
# #         dcs["TPD"],
# #         errors="coerce",
# #     )

# #     dcs["TPH"] = dcs["TPD"] / 24.0

# #     for section in EQUIPMENT_MAP:

# #         dcs[f"{section}_RT_HR"] = (
# #             dcs[f"{section}_INVENTORY_KG"]
# #             /
# #             (dcs["TPH"] * 1000.0)
# #         )

# #         dcs.loc[
# #             dcs["TPH"] <= 0,
# #             f"{section}_RT_HR",
# #         ] = np.nan

# #     rt_cols = [
# #         "EST1_RT_HR",
# #         "EST2_RT_HR",
# #         "PP1_RT_HR",
# #         "PP2_RT_HR",
# #         "DRR_RT_HR",
# #     ]

# #     dcs["TOTAL_RT_HR"] = dcs[rt_cols].sum(
# #         axis=1,
# #         min_count=5,
# #     )

# #     # Validated operational construction.
# #     dcs["RT_VALID"] = (
# #         dcs["TPD"].ge(375)
# #         &
# #         dcs[rt_cols].notna().all(axis=1)
# #     )

# #     dcs["RT_STATUS"] = np.where(
# #         dcs["RT_VALID"],
# #         "VALID",
# #         np.where(
# #             dcs["TPD"].le(0),
# #             "SHUTDOWN_TPD_LE_0",
# #             "NOT_RT_VALID",
# #         ),
# #     )

# #     return dcs


# # def build_runs(dcs):
# #     gap = (
# #         dcs["TIMESTAMP"].diff()
# #         .dt.total_seconds()
# #         / 3600.0
# #     )

# #     continuous = gap.eq(0.25)

# #     run_break = (
# #         dcs["RT_VALID"].ne(dcs["RT_VALID"].shift())
# #         | ~continuous
# #     )

# #     dcs = dcs.copy()
# #     dcs["VALID_RUN_ID"] = run_break.cumsum()

# #     runs = (
# #         dcs.loc[dcs["RT_VALID"]]
# #         .groupby("VALID_RUN_ID")
# #         .agg(
# #             START_TS=("TIMESTAMP", "min"),
# #             END_TS=("TIMESTAMP", "max"),
# #             N_ROWS=("TIMESTAMP", "size"),
# #         )
# #         .reset_index()
# #     )

# #     runs["DURATION_HR"] = (
# #         runs["END_TS"] - runs["START_TS"]
# #     ).dt.total_seconds() / 3600.0

# #     return dcs, runs


# # # ============================================================
# # # CAUSAL TRAJECTORY
# # # ============================================================

# # def find_run(runs, timestamp):
# #     match = runs[
# #         (runs["START_TS"] <= timestamp)
# #         &
# #         (runs["END_TS"] >= timestamp)
# #     ]

# #     if match.empty:
# #         return None

# #     return match.iloc[0]


# # def get_rt_at_time(dcs, timestamp, rt_column):
# #     timestamp = pd.Timestamp(timestamp)

# #     if (
# #         timestamp < dcs["TIMESTAMP"].min()
# #         or
# #         timestamp > dcs["TIMESTAMP"].max()
# #     ):
# #         return np.nan, "OUTSIDE_RANGE"

# #     before = dcs[
# #         (dcs["TIMESTAMP"] <= timestamp)
# #         & dcs["RT_VALID"]
# #     ].tail(1)

# #     after = dcs[
# #         (dcs["TIMESTAMP"] >= timestamp)
# #         & dcs["RT_VALID"]
# #     ].head(1)

# #     if before.empty or after.empty:
# #         return np.nan, "NO_BRACKET"

# #     t0 = before.iloc[0]["TIMESTAMP"]
# #     t1 = after.iloc[0]["TIMESTAMP"]

# #     gap_hr = (
# #         t1 - t0
# #     ).total_seconds() / 3600.0

# #     if gap_hr > 1.0:
# #         return np.nan, "RT_HISTORY_GAP"

# #     r0 = before.iloc[0][rt_column]
# #     r1 = after.iloc[0][rt_column]

# #     if pd.isna(r0) or pd.isna(r1):
# #         return np.nan, "RT_MISSING"

# #     if t0 == t1:
# #         return float(r0), "DIRECT"

# #     fraction = (
# #         timestamp - t0
# #     ).total_seconds() / (
# #         t1 - t0
# #     ).total_seconds()

# #     return float(r0 + fraction * (r1 - r0)), "INTERPOLATED"


# # def calculate_trajectory(dcs, runs, timestamp):
# #     timestamp = pd.Timestamp(timestamp)

# #     run = find_run(runs, timestamp)

# #     if run is None:
# #         return None

# #     current = timestamp

# #     result = {
# #         "TIMESTAMP": timestamp,
# #         "TRAJECTORY_VALID": True,
# #         "VALID_RUN_ID": int(run["VALID_RUN_ID"]),
# #         "RUN_START": run["START_TS"],
# #         "RUN_END": run["END_TS"],
# #     }

# #     for section, rt_col in SECTION_ORDER:

# #         rt, status = get_rt_at_time(
# #             dcs,
# #             current,
# #             rt_col,
# #         )

# #         if pd.isna(rt):
# #             return None

# #         result[f"{section}_RT_HR"] = rt
# #         result[f"{section}_RT_STATUS"] = status

# #         boundary = (
# #             current
# #             - pd.Timedelta(hours=float(rt))
# #         )

# #         result[f"{section}_BOUNDARY_TS"] = boundary

# #         current = boundary

# #     result["EST1_UPSTREAM_TS"] = current

# #     boundaries = [
# #         result["DRR_BOUNDARY_TS"],
# #         result["PP2_BOUNDARY_TS"],
# #         result["PP1_BOUNDARY_TS"],
# #         result["EST2_BOUNDARY_TS"],
# #         result["EST1_BOUNDARY_TS"],
# #     ]

# #     if not all(
# #         run["START_TS"] <= x <= run["END_TS"]
# #         for x in boundaries
# #     ):
# #         return None

# #     return result


# # # ============================================================
# # # WINDOW FEATURE EXTRACTION
# # # EXACT BASELINE-C CELL 4 LOGIC
# # # ============================================================

# # def extract_window_features(
# #     dcs_df,
# #     start_ts,
# #     end_ts,
# #     tags,
# #     prefix,
# # ):
# #     start_ts = pd.Timestamp(start_ts)
# #     end_ts = pd.Timestamp(end_ts)

# #     if (
# #         pd.isna(start_ts)
# #         or pd.isna(end_ts)
# #         or end_ts <= start_ts
# #     ):
# #         return {}

# #     window = dcs_df[
# #         (dcs_df["TIMESTAMP"] >= start_ts)
# #         &
# #         (dcs_df["TIMESTAMP"] <= end_ts)
# #     ].sort_values("TIMESTAMP").copy()

# #     if window.empty:
# #         return {}

# #     duration_hr = (
# #         end_ts - start_ts
# #     ).total_seconds() / 3600.0

# #     result = {}

# #     for tag in tags:

# #         if tag not in window.columns:
# #             continue

# #         values = pd.to_numeric(
# #             window[tag],
# #             errors="coerce",
# #         )

# #         valid = values.notna()

# #         if valid.sum() == 0:
# #             continue

# #         v = values.loc[valid].astype(float)

# #         first = float(v.iloc[0])
# #         last = float(v.iloc[-1])

# #         result[f"{prefix}{tag}_MEAN"] = float(v.mean())
# #         result[f"{prefix}{tag}_STD"] = float(v.std(ddof=0))
# #         result[f"{prefix}{tag}_MIN"] = float(v.min())
# #         result[f"{prefix}{tag}_MAX"] = float(v.max())
# #         result[f"{prefix}{tag}_RANGE"] = float(
# #             v.max() - v.min()
# #         )
# #         result[f"{prefix}{tag}_FIRST"] = first
# #         result[f"{prefix}{tag}_LAST"] = last
# #         result[f"{prefix}{tag}_DELTA"] = last - first
# #         result[f"{prefix}{tag}_N"] = int(valid.sum())

# #         for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
# #             result[
# #                 f"{prefix}{tag}_P{int(q * 100)}"
# #             ] = float(v.quantile(q))

# #         if len(v) >= 2 and duration_hr > 0:

# #             times = (
# #                 window.loc[valid, "TIMESTAMP"]
# #                 - window.loc[valid, "TIMESTAMP"].iloc[0]
# #             ).dt.total_seconds().to_numpy() / 3600.0

# #             y = v.to_numpy(float)

# #             if len(np.unique(times)) >= 2:
# #                 slope = np.polyfit(times, y, 1)[0]
# #             else:
# #                 slope = 0.0
# #         else:
# #             slope = 0.0

# #         result[
# #             f"{prefix}{tag}_SLOPE"
# #         ] = float(slope)

# #     return result


# # # ============================================================
# # # CAUSAL STATIC FEATURES
# # # ============================================================

# # def build_static_features(trajectory, dcs):
# #     ts = pd.Timestamp(trajectory["TIMESTAMP"])

# #     est1 = pd.Timestamp(trajectory["EST1_BOUNDARY_TS"])
# #     est2 = pd.Timestamp(trajectory["EST2_BOUNDARY_TS"])
# #     pp1 = pd.Timestamp(trajectory["PP1_BOUNDARY_TS"])
# #     pp2 = pd.Timestamp(trajectory["PP2_BOUNDARY_TS"])
# #     drr = pd.Timestamp(trajectory["DRR_BOUNDARY_TS"])

# #     row = {
# #         "LAB_TIMESTAMP": ts,

# #         # Explicit RT predictors used by the model.
# #         "EST1_RT_HR": trajectory["EST1_RT_HR"],
# #         "EST2_RT_HR": trajectory["EST2_RT_HR"],
# #         "PP1_RT_HR": trajectory["PP1_RT_HR"],
# #         "PP2_RT_HR": trajectory["PP2_RT_HR"],
# #         "DRR_RT_HR": trajectory["DRR_RT_HR"],

# #         "TOTAL_RT_HR": sum(
# #             trajectory[x]
# #             for x in [
# #                 "EST1_RT_HR",
# #                 "EST2_RT_HR",
# #                 "PP1_RT_HR",
# #                 "PP2_RT_HR",
# #                 "DRR_RT_HR",
# #             ]
# #         ),

# #         "EST1_BOUNDARY_TS": est1,
# #         "EST2_BOUNDARY_TS": est2,
# #         "PP1_BOUNDARY_TS": pp1,
# #         "PP2_BOUNDARY_TS": pp2,
# #         "DRR_BOUNDARY_TS": drr,
# #     }

# #     # --------------------------------------------------------
# #     # Five causal section windows
# #     # --------------------------------------------------------

# #     section_windows = {
# #         "EST1": (est1, est2),
# #         "EST2": (est2, pp1),
# #         "PP1": (pp1, pp2),
# #         "PP2": (pp2, drr),
# #         "DRR": (drr, ts),
# #     }

# #     for section, (start, end) in section_windows.items():

# #         row.update(
# #             extract_window_features(
# #                 dcs,
# #                 start,
# #                 end,
# #                 SECTION_TAGS[section],
# #                 f"{section}_",
# #             )
# #         )

# #         row[f"{section}_WINDOW_START"] = start
# #         row[f"{section}_WINDOW_END"] = end
# #         row[f"{section}_WINDOW_HR"] = (
# #             end - start
# #         ).total_seconds() / 3600.0
# #         row[f"{section}_DCS_N"] = len(
# #             dcs[
# #                 (dcs["TIMESTAMP"] >= start)
# #                 &
# #                 (dcs["TIMESTAMP"] <= end)
# #             ]
# #         )

# #     # --------------------------------------------------------
# #     # Full causal window
# #     # --------------------------------------------------------

# #     global_start = est1
# #     global_end = ts

# #     row.update(
# #         extract_window_features(
# #             dcs,
# #             global_start,
# #             global_end,
# #             AREA_TAGS["PASTE"],
# #             "PASTE_",
# #         )
# #     )

# #     row.update(
# #         extract_window_features(
# #             dcs,
# #             global_start,
# #             global_end,
# #             AREA_TAGS["THROUGHPUT"],
# #             "THROUGHPUT_",
# #         )
# #     )

# #     row.update(
# #         extract_window_features(
# #             dcs,
# #             global_start,
# #             global_end,
# #             AREA_TAGS["JET"],
# #             "JET_",
# #         )
# #     )

# #     row.update(
# #         extract_window_features(
# #             dcs,
# #             global_start,
# #             global_end,
# #             AREA_TAGS["HTM"],
# #             "HTM_",
# #         )
# #     )

# #     # --------------------------------------------------------
# #     # Chemistry trajectory
# #     #
# #     # IMPORTANT:
# #     # Historical model feature names such as CHEM_COBALT_*
# #     # are preserved exactly where applicable.
# #     # --------------------------------------------------------

# #     chemistry_groups = {
# #         "ANTIMONY": ["FIC-21009", "YK-21010"],
# #         "RED_TONER": ["FIC-41007", "YK-41008"],
# #         "PHOSPHORIC_ACID": ["FIC-43007", "YK-43008"],
# #         "BLUE_TONER": ["FIC-42007", "YK-42008"],
# #     }

# #     for group, tags in chemistry_groups.items():

# #         row.update(
# #             extract_window_features(
# #                 dcs,
# #                 global_start,
# #                 global_end,
# #                 tags,
# #                 f"CHEM_{group}_",
# #             )
# #         )

# #     # Historical saved models used CHEM_COBALT_* for the
# #     # FIC-41007/YK-41008 channels. Preserve those aliases.
# #     cobalt = extract_window_features(
# #         dcs,
# #         global_start,
# #         global_end,
# #         ["FIC-41007", "YK-41008"],
# #         "CHEM_COBALT_",
# #     )
# #     row.update(cobalt)

# #     # --------------------------------------------------------
# #     # Localized paste chemistry
# #     # --------------------------------------------------------

# #     for minutes in [30, 60, 120, 240]:

# #         row.update(
# #             extract_window_features(
# #                 dcs,
# #                 est1 - pd.Timedelta(minutes=minutes),
# #                 est1,
# #                 PASTE_CHEM_TAGS,
# #                 f"PASTE_{minutes}M_",
# #             )
# #         )

# #     row.update(
# #         extract_window_features(
# #             dcs,
# #             est1 - pd.Timedelta(hours=4),
# #             est1,
# #             PASTE_CHEM_TAGS,
# #             "PASTE_4H_",
# #         )
# #     )

# #     # --------------------------------------------------------
# #     # H3PO4 localized at EST-2
# #     # --------------------------------------------------------

# #     row.update(
# #         extract_window_features(
# #             dcs,
# #             est2,
# #             pp1,
# #             EST2_CHEM_TAGS,
# #             "EST2_H3PO4_FULL_",
# #         )
# #     )

# #     for minutes in [30, 60, 120]:

# #         end = min(
# #             est2 + pd.Timedelta(minutes=minutes),
# #             pp1,
# #         )

# #         if end > est2:
# #             row.update(
# #                 extract_window_features(
# #                     dcs,
# #                     est2,
# #                     end,
# #                     EST2_CHEM_TAGS,
# #                     f"EST2_H3PO4_{minutes}M_",
# #                 )
# #             )

# #     # --------------------------------------------------------
# #     # Process-control layer
# #     # EXACT Baseline C Cell 6 logic
# #     # --------------------------------------------------------

# #     window = dcs[
# #         (dcs["TIMESTAMP"] >= est1)
# #         &
# #         (dcs["TIMESTAMP"] <= ts)
# #     ].copy()

# #     if not window.empty:

# #         if "TPD" in window.columns:

# #             tpd = pd.to_numeric(
# #                 window["TPD"],
# #                 errors="coerce",
# #             ).dropna()

# #             if len(tpd):

# #                 row["CTRL_TPD_MEAN"] = float(tpd.mean())
# #                 row["CTRL_TPD_STD"] = float(tpd.std(ddof=0))
# #                 row["CTRL_TPD_MIN"] = float(tpd.min())
# #                 row["CTRL_TPD_MAX"] = float(tpd.max())
# #                 row["CTRL_TPD_RANGE"] = float(
# #                     tpd.max() - tpd.min()
# #                 )
# #                 row["CTRL_TPD_FIRST"] = float(tpd.iloc[0])
# #                 row["CTRL_TPD_LAST"] = float(tpd.iloc[-1])
# #                 row["CTRL_TPD_DELTA"] = float(
# #                     tpd.iloc[-1] - tpd.iloc[0]
# #                 )

# #                 for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
# #                     row[
# #                         f"CTRL_TPD_P{int(q * 100)}"
# #                     ] = float(tpd.quantile(q))

# #         process_summary = {}

# #         for tag in PROCESS_TAGS:

# #             if tag == "TPD" or tag not in window.columns:
# #                 continue

# #             values = pd.to_numeric(
# #                 window[tag],
# #                 errors="coerce",
# #             ).dropna()

# #             if len(values) == 0:
# #                 continue

# #             clean = tag.replace("-", "_")

# #             process_summary[tag] = {
# #                 "MEAN": float(values.mean()),
# #                 "STD": float(values.std(ddof=0)),
# #                 "MIN": float(values.min()),
# #                 "MAX": float(values.max()),
# #                 "RANGE": float(
# #                     values.max() - values.min()
# #                 ),
# #                 "FIRST": float(values.iloc[0]),
# #                 "LAST": float(values.iloc[-1]),
# #                 "DELTA": float(
# #                     values.iloc[-1] - values.iloc[0]
# #                 ),
# #             }

# #             for stat, value in process_summary[tag].items():
# #                 row[
# #                     f"CTRL_{clean}_{stat}"
# #                 ] = value

# #         tpd_mean = row.get("CTRL_TPD_MEAN", np.nan)

# #         safe_tpd = (
# #             tpd_mean
# #             if pd.notna(tpd_mean) and abs(tpd_mean) >= 50
# #             else np.nan
# #         )

# #         if pd.notna(safe_tpd):

# #             for tag, stats in process_summary.items():

# #                 clean = tag.replace("-", "_")

# #                 row[
# #                     f"CTRL_RATIO_{clean}_MEAN_PER_TPD"
# #                 ] = stats["MEAN"] / safe_tpd

# #         tpd_delta = row.get("CTRL_TPD_DELTA", np.nan)

# #         safe_delta = (
# #             tpd_delta
# #             if pd.notna(tpd_delta) and abs(tpd_delta) >= 1
# #             else np.nan
# #         )

# #         if pd.notna(safe_delta):

# #             for tag, stats in process_summary.items():

# #                 clean = tag.replace("-", "_")

# #                 row[
# #                     f"CTRL_RESPONSE_{clean}_PER_TPD_CHANGE"
# #                 ] = stats["DELTA"] / safe_delta

# #     # --------------------------------------------------------
# #     # Chemistry / throughput interactions
# #     # EXACT Baseline C Cell 7 logic
# #     # --------------------------------------------------------

# #     safe_tpd = row.get("CTRL_TPD_MEAN", np.nan)

# #     safe_tpd = (
# #         safe_tpd
# #         if pd.notna(safe_tpd) and abs(safe_tpd) >= 50
# #         else np.nan
# #     )

# #     chemistry_tokens = {
# #         "ANTIMONY": "FIC-21009",
# #         "RED_TONER": "FIC-41007",
# #         "BLUE_TONER": "FIC-42007",
# #         "H3PO4": "FIC-43007",
# #     }

# #     for chemistry, tag in chemistry_tokens.items():

# #         matching = [
# #             c for c in list(row.keys())
# #             if tag in str(c)
# #             and (
# #                 "_MEAN" in str(c)
# #                 or "_LAST" in str(c)
# #                 or "_FIRST" in str(c)
# #                 or "_DELTA" in str(c)
# #             )
# #         ]

# #         for col in matching:

# #             value = pd.to_numeric(
# #                 pd.Series([row[col]]),
# #                 errors="coerce",
# #             ).iloc[0]

# #             row[
# #                 f"CHEM_CTRL_{chemistry}_{col}_PER_TPD"
# #             ] = (
# #                 value / safe_tpd
# #                 if pd.notna(safe_tpd)
# #                 else np.nan
# #             )

# #     # --------------------------------------------------------
# #     # Baseline C throughput regime
# #     # EXACT Cell 7C logic
# #     # --------------------------------------------------------

# #     if not window.empty and "TPD" in window.columns:

# #         tpd = pd.to_numeric(
# #             window["TPD"],
# #             errors="coerce",
# #         )

# #         valid = tpd.notna()

# #         if valid.any():

# #             times = window.loc[valid, "TIMESTAMP"]
# #             values = tpd.loc[valid].astype(float)

# #             current = float(values.iloc[-1])
# #             full_mean = float(values.mean())
# #             first = float(values.iloc[0])

# #             for minutes in [30, 60, 120, 240]:

# #                 cutoff = (
# #                     ts - pd.Timedelta(minutes=minutes)
# #                 )

# #                 recent = values.loc[times >= cutoff]

# #                 if recent.empty:
# #                     recent = values

# #                 suffix = f"{minutes}M"

# #                 row[f"C_TPD_{suffix}_MEAN"] = float(
# #                     recent.mean()
# #                 )
# #                 row[f"C_TPD_{suffix}_STD"] = float(
# #                     recent.std(ddof=0)
# #                 )
# #                 row[f"C_TPD_{suffix}_MIN"] = float(
# #                     recent.min()
# #                 )
# #                 row[f"C_TPD_{suffix}_MAX"] = float(
# #                     recent.max()
# #                 )
# #                 row[f"C_TPD_{suffix}_DELTA"] = float(
# #                     recent.iloc[-1] - recent.iloc[0]
# #                 )
# #                 row[f"C_TPD_{suffix}_RATE"] = float(
# #                     (
# #                         recent.iloc[-1]
# #                         - recent.iloc[0]
# #                     )
# #                     / (minutes / 60.0)
# #                 )

# #             row["C_TPD_CURRENT"] = current
# #             row["C_TPD_MINUS_400"] = current - 400.0
# #             row["C_TPD_RATIO_TO_400"] = current / 400.0
# #             row["C_TPD_MEAN_MINUS_400"] = full_mean - 400.0
# #             row["C_TPD_MEAN_RATIO_TO_400"] = full_mean / 400.0

# #             row["C_TPD_FRAC_BELOW_500"] = float(
# #                 (values < 500).mean()
# #             )
# #             row["C_TPD_FRAC_500_600"] = float(
# #                 (
# #                     (values >= 500)
# #                     & (values < 600)
# #                 ).mean()
# #             )
# #             row["C_TPD_FRAC_600_800"] = float(
# #                 (
# #                     (values >= 600)
# #                     & (values <= 800)
# #                 ).mean()
# #             )
# #             row["C_TPD_FRAC_ABOVE_800"] = float(
# #                 (values > 800).mean()
# #             )
# #             row["C_TPD_FRAC_GE_600"] = float(
# #                 (values >= 600).mean()
# #             )

# #             if current < 500:
# #                 regime = 0
# #             elif current < 600:
# #                 regime = 1
# #             elif current <= 800:
# #                 regime = 2
# #             else:
# #                 regime = 3

# #             row["C_TPD_CURRENT_REGIME"] = regime

# #             row["C_TPD_CROSSED_500"] = float(
# #                 first < 500 and current >= 500
# #             )
# #             row["C_TPD_CROSSED_600"] = float(
# #                 first < 600 and current >= 600
# #             )
# #             row["C_TPD_CROSSED_800"] = float(
# #                 first < 800 and current >= 800
# #             )
# #             row[
# #                 "C_TPD_STARTED_BELOW_600_ENDED_GE_600"
# #             ] = float(
# #                 first < 600 and current >= 600
# #             )
# #             row[
# #                 "C_TPD_STARTED_BELOW_800_ENDED_GE_800"
# #             ] = float(
# #                 first < 800 and current >= 800
# #             )

# #             row["C_TPD_MINUS_600"] = current - 600.0
# #             row["C_TPD_MINUS_800"] = current - 800.0
# #             row["C_TPD_MEAN_MINUS_600"] = full_mean - 600.0
# #             row["C_TPD_MEAN_MINUS_800"] = full_mean - 800.0

# #             for minutes in [30, 60, 120, 240]:

# #                 cutoff = (
# #                     ts - pd.Timedelta(minutes=minutes)
# #                 )

# #                 recent = values.loc[times >= cutoff]

# #                 if recent.empty:
# #                     recent = values

# #                 row[
# #                     f"C_TPD_{minutes}M_VS_FULL_DELTA"
# #                 ] = float(
# #                     recent.mean() - full_mean
# #                 )

# #                 row[
# #                     f"C_TPD_{minutes}M_VS_FULL_RATIO"
# #                 ] = (
# #                     float(recent.mean() / full_mean)
# #                     if full_mean != 0
# #                     else np.nan
# #                 )

# #     return row


# # # ============================================================
# # # DYNAMIC FEATURES
# # # ============================================================

# # DYNAMIC_EXCLUDE = {
# #     "LAB_TIMESTAMP",
# #     "EST1_BOUNDARY_TS",
# #     "EST2_BOUNDARY_TS",
# #     "PP1_BOUNDARY_TS",
# #     "PP2_BOUNDARY_TS",
# #     "DRR_BOUNDARY_TS",
# # }


# # def add_pairwise_dynamic(current, previous):
# #     """
# #     Raw-DCS deployment equivalent of the training CHANGE /
# #     RELCHANGE / RATE feature family.

# #     No previous LAB colour is used.
# #     """

# #     result = current.copy()

# #     current_ts = pd.Timestamp(current["LAB_TIMESTAMP"])
# #     previous_ts = pd.Timestamp(previous["LAB_TIMESTAMP"])

# #     interval_hr = (
# #         current_ts - previous_ts
# #     ).total_seconds() / 3600.0

# #     result["LAB_INTERVAL_HR"] = interval_hr
# #     result["DYNAMIC_CHANGE_VALID"] = (
# #         0 < interval_hr <= 12
# #     )

# #     numeric_cols = []

# #     for col, value in current.items():

# #         if col in DYNAMIC_EXCLUDE:
# #             continue

# #         if col in {
# #             "LAB_INTERVAL_HR",
# #             "DYNAMIC_CHANGE_VALID",
# #         }:
# #             continue

# #         try:
# #             a = pd.to_numeric(
# #                 pd.Series([value]),
# #                 errors="coerce",
# #             ).iloc[0]
# #         except Exception:
# #             continue

# #         if pd.notna(a):
# #             numeric_cols.append(col)

# #     for col in numeric_cols:

# #         cur = pd.to_numeric(
# #             pd.Series([current.get(col)]),
# #             errors="coerce",
# #         ).iloc[0]

# #         prev = pd.to_numeric(
# #             pd.Series([previous.get(col)]),
# #             errors="coerce",
# #         ).iloc[0]

# #         change = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan

# #         result[f"CHANGE__{col}"] = change

# #         denominator = (
# #             abs(prev)
# #             if pd.notna(prev)
# #             else np.nan
# #         )

# #         if pd.notna(denominator):
# #             denominator = max(denominator, 1e-3)

# #         result[f"RELCHANGE__{col}"] = (
# #             change / denominator
# #             if pd.notna(change) and pd.notna(denominator)
# #             else np.nan
# #         )

# #         result[f"RATE__{col}"] = (
# #             change / interval_hr
# #             if pd.notna(change) and interval_hr > 0
# #             else np.nan
# #         )

# #     return result


# # # ============================================================
# # # FIND PREVIOUS VALID DCS PREDICTION STATE
# # # ============================================================

# # def find_previous_timestamp(dcs, selected_ts):
# #     lower = selected_ts - pd.Timedelta(
# #         hours=DYNAMIC_LOOKBACK_HOURS
# #     )

# #     candidates = dcs[
# #         (dcs["TIMESTAMP"] < selected_ts)
# #         &
# #         (dcs["TIMESTAMP"] >= lower)
# #         &
# #         dcs["RT_VALID"]
# #     ]

# #     if candidates.empty:
# #         return None

# #     return candidates.iloc[-1]["TIMESTAMP"]


# # # ============================================================
# # # BUILD MODEL INPUT FOR ONE SELECTED TIMESTAMP
# # # ============================================================

# # def build_prediction_feature_row(
# #     dcs,
# #     runs,
# #     selected_ts,
# # ):
# #     trajectory = calculate_trajectory(
# #         dcs,
# #         runs,
# #         selected_ts,
# #     )

# #     if trajectory is None:
# #         raise ValueError(
# #             "The selected timestamp does not have a valid "
# #             "continuous causal RT trajectory."
# #         )

# #     current_static = build_static_features(
# #         trajectory,
# #         dcs,
# #     )

# #     previous_ts = find_previous_timestamp(
# #         dcs,
# #         selected_ts,
# #     )

# #     dynamic_row = current_static.copy()

# #     if previous_ts is not None:

# #         previous_trajectory = calculate_trajectory(
# #             dcs,
# #             runs,
# #             previous_ts,
# #         )

# #         if previous_trajectory is not None:

# #             previous_static = build_static_features(
# #                 previous_trajectory,
# #                 dcs,
# #             )

# #             dynamic_row = add_pairwise_dynamic(
# #                 current_static,
# #                 previous_static,
# #             )

# #     return (
# #         pd.DataFrame([dynamic_row]),
# #         trajectory,
# #         previous_ts,
# #     )


# # # ============================================================
# # # FEATURE AUDIT + PREDICTION
# # # ============================================================

# # def prepare_model_matrix(feature_row, saved_features, target):
# #     selected = saved_features[target]

# #     missing = [
# #         c for c in selected
# #         if c not in feature_row.columns
# #     ]

# #     if missing:
# #         return None, missing

# #     X = (
# #         feature_row[selected]
# #         .replace([np.inf, -np.inf], np.nan)
# #         .apply(pd.to_numeric, errors="coerce")
# #         .fillna(0.0)
# #     )

# #     return X, []


# # def run_prediction(
# #     feature_row,
# #     models,
# #     saved_features,
# # ):
# #     matrices = {}
# #     failures = {}

# #     for target in TARGETS:

# #         key = f"LAB_{target}"

# #         # JSON supports LAB_L / LAB_A / LAB_B.
# #         selected = saved_features.get(key)

# #         if selected is None:
# #             failures[target] = [
# #                 f"Missing JSON key: {key}"
# #             ]
# #             continue

# #         X, missing = prepare_model_matrix(
# #             feature_row,
# #             saved_features,
# #             key,
# #         )

# #         if missing:
# #             failures[target] = missing
# #         else:
# #             matrices[target] = X

# #     if failures:
# #         return None, failures

# #     result = {}

# #     for target in TARGETS:

# #         result[f"PRED_{target}"] = float(
# #             models[target].predict(
# #                 matrices[target]
# #             )[0]
# #         )

# #     return result, {}


# # # ============================================================
# # # STREAMLIT UI
# # # ============================================================

# # st.set_page_config(
# #     page_title="POLY-II Colour Prediction",
# #     page_icon="🎨",
# #     layout="wide",
# # )

# # st.title("POLY-II Colour Prediction")
# # st.caption(
# #     "Baseline C — single date/time prediction from raw 15-minute DCS"
# # )

# # # ------------------------------------------------------------
# # # User-provided Base Folder + raw DCS upload
# # # ------------------------------------------------------------

# # with st.sidebar:
# #     st.header("Data & Model Configuration")

# #     base_dir_input = st.text_input(
# #         "Base folder",
# #         value=".",
# #         help=(
# #             "Enter the folder containing Nova_Part-A.xlsx and "
# #             "the Baseline-C model directory. Use . for the "
# #             "folder from which Streamlit was launched."
# #         ),
# #     )

# #     model_dir_name = st.text_input(
# #         "Model folder name",
# #         value=DEFAULT_MODEL_DIR_NAME,
# #         help=(
# #             "The model directory must be inside the Base folder "
# #             "and must contain the three Baseline-C .cbm files "
# #             "and baseline_c_selected_features.json."
# #         ),
# #     )

# #     st.markdown("---")

# #     uploaded_dcs = st.file_uploader(
# #         "Upload raw DCS Excel file",
# #         type=["xlsx", "xls"],
# #         help=(
# #             "Select the raw 15-minute DCS Excel file. "
# #             "The file is used directly from the upload for "
# #             "the current Streamlit session."
# #         ),
# #     )

# # # Resolve the user-provided folder. No machine-specific path is used.
# # # BASE_DIR = Path(base_dir_input).expanduser()
# # BASE_DIR = Path(__file__).resolve().parent

# # if not BASE_DIR.is_absolute():
# #     BASE_DIR = (Path.cwd() / BASE_DIR).resolve()
# # else:
# #     BASE_DIR = BASE_DIR.resolve()

# # NOVA_FILE = BASE_DIR / "Nova_Part-A.xlsx"
# # MODEL_DIR = BASE_DIR / "poly2_baseline_c_models_TP_Regime_Change_v4"

# # MODEL_FILES = {
# #     "L": MODEL_DIR / "catboost_LAB_L_BaselineC.cbm",
# #     "A": MODEL_DIR / "catboost_LAB_A_BaselineC.cbm",
# #     "B": MODEL_DIR / "catboost_LAB_B_BaselineC.cbm",
# # }

# # FEATURE_FILE = MODEL_DIR / "baseline_c_selected_features.json"

# # if uploaded_dcs is not None:
# #     RAW_DCS_FILE = Path(uploaded_dcs.name)
# # else:
# #     RAW_DCS_FILE = None

# # # ------------------------------------------------------------
# # # Validate Base Folder / Nova / Model Folder
# # # ------------------------------------------------------------

# # if not BASE_DIR.exists():
# #     st.error(f"Base folder does not exist:\n{BASE_DIR}")
# #     st.stop()

# # if not NOVA_FILE.exists():
# #     st.error(
# #         "Nova_Part-A.xlsx was not found in the provided Base folder:\n"
# #         f"{NOVA_FILE}"
# #     )
# #     st.stop()

# # if not MODEL_DIR.exists():
# #     st.error(
# #         "Model directory was not found in the provided Base folder:\n"
# #         f"{MODEL_DIR}"
# #     )
# #     st.stop()

# # required_paths = [
# #     MODEL_FILES["L"],
# #     MODEL_FILES["A"],
# #     MODEL_FILES["B"],
# #     FEATURE_FILE,
# # ]

# # missing_paths = [str(p) for p in required_paths if not p.exists()]

# # if missing_paths:
# #     st.error(
# #         "Required Baseline-C model files are missing:\n"
# #         + "\n".join(missing_paths)
# #     )
# #     st.stop()

# # # # ------------------------------------------------------------
# # # # Sidebar status
# # # # ------------------------------------------------------------

# # # with st.sidebar:
# # #     st.markdown("---")
# # #     st.write(f"**Base folder:**\n`{BASE_DIR}`")
# # #     st.write(f"**Nova file:**\n`{NOVA_FILE}`")
# # #     st.write(f"**Model folder:**\n`{MODEL_DIR}`")

# # #     if uploaded_dcs is not None:
# # #         st.write(f"**Raw DCS:**\n`{uploaded_dcs.name}`")
# # #     else:
# # #         st.warning("Upload the raw DCS Excel file to continue.")

# # #     st.info(
# # #         "202-F0 is excluded. "
# # #         "TPD <= 0 is treated as shutdown."
# # #     )

# # # ------------------------------------------------------------
# # # Sidebar
# # # ------------------------------------------------------------

# # with st.sidebar:

# #     st.header("DCS Data")

# #     uploaded_dcs = st.file_uploader(
# #         "Upload Raw DCS Excel File",
# #         type=["xlsx", "xls"],
# #         help="Upload the raw 15-minute DCS Excel file."
# #     )

# #     st.markdown("---")

# #     st.info(
# #         "202-F0 is excluded.\n\n"
# #         "TPD <= 0 is treated as shutdown."
# #     )

# # # ------------------------------------------------------------
# # # Require DCS upload
# # # ------------------------------------------------------------

# # if uploaded_dcs is None:
# #     st.info(
# #         "Enter the Base folder, make sure Nova_Part-A.xlsx and "
# #         "the model folder are inside it, then upload the raw "
# #         "DCS Excel file."
# #     )
# #     st.stop()

# # # ------------------------------------------------------------
# # # Load
# # # ------------------------------------------------------------

# # try:
# #     models, saved_features = load_models()

# #     uploaded_dcs_bytes = uploaded_dcs.getvalue()

# #     dcs = load_raw_dcs(
# #         uploaded_dcs_bytes,
# #         uploaded_dcs.name,
# #     )

# #     curves = load_nova_curves()

# #     dcs = add_inventory(dcs, curves)
# #     dcs = calculate_rt(dcs)
# #     dcs, runs = build_runs(dcs)

# # except Exception as exc:
# #     st.error(str(exc))
# #     st.exception(exc)
# #     st.stop()


# # # ------------------------------------------------------------
# # # Date/time selection
# # # ------------------------------------------------------------

# # valid_dcs = dcs.loc[
# #     dcs["RT_VALID"],
# #     ["TIMESTAMP", "TPD"],
# # ].copy()

# # if valid_dcs.empty:
# #     st.error("No RT-valid production observations found.")
# #     st.stop()

# # valid_dates = sorted(
# #     valid_dcs["TIMESTAMP"].dt.date.unique()
# # )

# # st.subheader("1. Select prediction timestamp")

# # c1, c2 = st.columns(2)

# # with c1:
# #     selected_date = st.selectbox(
# #         "Date",
# #         valid_dates,
# #         index=len(valid_dates) - 1,
# #     )

# # date_rows = valid_dcs[
# #     valid_dcs["TIMESTAMP"].dt.date == selected_date
# # ]

# # available_times = sorted(
# #     date_rows["TIMESTAMP"].dt.time.unique()
# # )

# # with c2:
# #     selected_time = st.selectbox(
# #         "DCS Time",
# #         available_times,
# #     )

# # selected_ts = pd.Timestamp(
# #     f"{selected_date} {selected_time}"
# # )


# # # ------------------------------------------------------------
# # # Selected DCS status
# # # ------------------------------------------------------------

# # selected_rows = dcs[
# #     dcs["TIMESTAMP"] == selected_ts
# # ]

# # if selected_rows.empty:
# #     st.error(
# #         "Selected timestamp does not exist in the raw DCS data."
# #     )
# #     st.stop()

# # selected_row = selected_rows.iloc[0]

# # m1, m2, m3, m4 = st.columns(4)

# # m1.metric(
# #     "Selected Time",
# #     selected_ts.strftime("%Y-%m-%d %H:%M"),
# # )

# # m2.metric(
# #     "TPD",
# #     f"{selected_row['TPD']:.2f}"
# #     if pd.notna(selected_row["TPD"])
# #     else "N/A",
# # )

# # m3.metric(
# #     "RT Status",
# #     str(selected_row["RT_STATUS"]),
# # )

# # m4.metric(
# #     "Total RT",
# #     f"{selected_row['TOTAL_RT_HR']:.2f} h"
# #     if pd.notna(selected_row["TOTAL_RT_HR"])
# #     else "N/A",
# # )


# # # ------------------------------------------------------------
# # # Predict
# # # ------------------------------------------------------------

# # if st.button(
# #     "🎨 PREDICT COLOUR",
# #     type="primary",
# #     use_container_width=True,
# # ):

# #     if not bool(selected_row["RT_VALID"]):

# #         st.error(
# #             "Prediction unavailable. "
# #             "The selected timestamp is not RT-valid production."
# #         )

# #         if selected_row["TPD"] <= 0:
# #             st.warning(
# #                 "TPD <= 0 indicates shutdown/non-production."
# #             )

# #         st.stop()

# #     try:

# #         with st.spinner(
# #             "Calculating residence time and extracting causal DCS window..."
# #         ):

# #             feature_row, trajectory, previous_ts = (
# #                 build_prediction_feature_row(
# #                     dcs,
# #                     runs,
# #                     selected_ts,
# #                 )
# #             )

# #             prediction, failures = run_prediction(
# #                 feature_row,
# #                 models,
# #                 saved_features,
# #             )

# #         if failures:

# #             st.error(
# #                 "Prediction stopped because the saved Baseline C "
# #                 "feature definition cannot be reproduced."
# #             )

# #             for target, cols in failures.items():

# #                 st.write(
# #                     f"**LAB_{target}: "
# #                     f"{len(cols)} missing feature(s)**"
# #                 )

# #                 st.code(
# #                     "\n".join(cols),
# #                     language="text",
# #                 )

# #             st.info(
# #                 "Do not substitute these columns manually. "
# #                 "The raw-DCS feature builder must reproduce the "
# #                 "saved training feature definitions."
# #             )

# #             st.stop()

# #         # ----------------------------------------------------
# #         # Prediction results
# #         # ----------------------------------------------------

# #         st.success(
# #             "Prediction completed successfully."
# #         )

# #         st.subheader("2. Predicted Colour")

# #         p1, p2, p3 = st.columns(3)

# #         p1.metric(
# #             "L",
# #             f"{prediction['PRED_L']:.3f}",
# #         )

# #         p2.metric(
# #             "a",
# #             f"{prediction['PRED_A']:.3f}",
# #         )

# #         p3.metric(
# #             "b",
# #             f"{prediction['PRED_B']:.3f}",
# #         )

# #         # ----------------------------------------------------
# #         # Causal trajectory
# #         # ----------------------------------------------------

# #         st.subheader("3. Causal Residence-Time Window")

# #         trajectory_table = pd.DataFrame(
# #             {
# #                 "Point": [
# #                     "Prediction / Lab boundary",
# #                     "DRR boundary",
# #                     "PP2 boundary",
# #                     "PP1 boundary",
# #                     "EST2 boundary",
# #                     "EST1 upstream boundary",
# #                 ],
# #                 "Timestamp": [
# #                     selected_ts,
# #                     trajectory["DRR_BOUNDARY_TS"],
# #                     trajectory["PP2_BOUNDARY_TS"],
# #                     trajectory["PP1_BOUNDARY_TS"],
# #                     trajectory["EST2_BOUNDARY_TS"],
# #                     trajectory["EST1_BOUNDARY_TS"],
# #                 ],
# #             }
# #         )

# #         st.dataframe(
# #             trajectory_table,
# #             use_container_width=True,
# #             hide_index=True,
# #         )

# #         r1, r2, r3, r4, r5 = st.columns(5)

# #         r1.metric(
# #             "EST1 RT",
# #             f"{trajectory['EST1_RT_HR']:.2f} h",
# #         )
# #         r2.metric(
# #             "EST2 RT",
# #             f"{trajectory['EST2_RT_HR']:.2f} h",
# #         )
# #         r3.metric(
# #             "PP1 RT",
# #             f"{trajectory['PP1_RT_HR']:.2f} h",
# #         )
# #         r4.metric(
# #             "PP2 RT",
# #             f"{trajectory['PP2_RT_HR']:.2f} h",
# #         )
# #         r5.metric(
# #             "DRR RT",
# #             f"{trajectory['DRR_RT_HR']:.2f} h",
# #         )

# #         # ----------------------------------------------------
# #         # Dynamic information
# #         # ----------------------------------------------------

# #         st.subheader("4. Deployment Dynamic Reference")

# #         if previous_ts is not None:
# #             st.write(
# #                 "Previous valid causal state used for deployment "
# #                 "dynamic features:"
# #             )
# #             st.code(
# #                 str(previous_ts),
# #                 language="text",
# #             )
# #         else:
# #             st.warning(
# #                 "No previous valid state was available within "
# #                 f"{DYNAMIC_LOOKBACK_HOURS:.1f} hours. "
# #                 "Dynamic features therefore remain unavailable "
# #                 "where appropriate and are filled by the model "
# #                 "input preparation."
# #             )

# #         # ----------------------------------------------------
# #         # Important process state
# #         # ----------------------------------------------------

# #         st.subheader("5. Process State Used")

# #         state_cols = [
# #             "CTRL_TPD_MEAN",
# #             "CTRL_TPD_MIN",
# #             "CTRL_TPD_MAX",
# #             "CTRL_TPD_LAST",
# #             "C_TPD_CURRENT",
# #             "C_TPD_30M_MEAN",
# #             "C_TPD_60M_MEAN",
# #             "C_TPD_120M_MEAN",
# #             "C_TPD_240M_MEAN",
# #         ]

# #         state = {}

# #         for col in state_cols:
# #             if col in feature_row.columns:
# #                 state[col] = feature_row.iloc[0][col]

# #         if state:
# #             state_df = pd.DataFrame(
# #                 [
# #                     {
# #                         "Feature": k,
# #                         "Value": v,
# #                     }
# #                     for k, v in state.items()
# #                 ]
# #             )

# #             st.dataframe(
# #                 state_df,
# #                 use_container_width=True,
# #                 hide_index=True,
# #             )

# #         # ----------------------------------------------------
# #         # Optional feature audit
# #         # ----------------------------------------------------

# #         with st.expander(
# #             "Show model feature audit"
# #         ):

# #             for target in TARGETS:

# #                 key = f"LAB_{target}"
# #                 selected = saved_features[key]

# #                 present = [
# #                     x for x in selected
# #                     if x in feature_row.columns
# #                 ]

# #                 st.write(
# #                     f"**{key}: "
# #                     f"{len(present)}/{len(selected)} features reproduced**"
# #                 )

# #         # ----------------------------------------------------
# #         # Save one-row CSV
# #         # ----------------------------------------------------

# #         export = pd.DataFrame(
# #             [
# #                 {
# #                     "TIMESTAMP": selected_ts,
# #                     "TPD": selected_row["TPD"],
# #                     "TOTAL_RT_HR": selected_row["TOTAL_RT_HR"],
# #                     "PRED_L": prediction["PRED_L"],
# #                     "PRED_A": prediction["PRED_A"],
# #                     "PRED_B": prediction["PRED_B"],
# #                 }
# #             ]
# #         )

# #         st.download_button(
# #             "Download Prediction CSV",
# #             export.to_csv(index=False),
# #             file_name=(
# #                 f"POLY2_prediction_"
# #                 f"{selected_ts.strftime('%Y%m%d_%H%M')}.csv"
# #             ),
# #             mime="text/csv",
# #         )



# #     except Exception as exc:
# #         st.error("Prediction failed.")
# #         st.exception(exc)


# # # ------------------------------------------------------------
# # # Footer
# # # ------------------------------------------------------------

# # st.divider()

# # st.caption(
# #     "POLY-II Baseline C | Raw 15-minute DCS | "
# #     "Causal residence-time inference | "
# #     "No previous Lab colour values used"
# # )


# # ============================================================
# # POLY-II BASELINE C — SINGLE DATE/TIME RAW-DCS COLOUR PREDICTOR
# # ============================================================
# #
# # USER WORKFLOW
# # --------------
# # 1. Select a date.
# # 2. Select a DCS time.
# # 3. Click PREDICT COLOUR.
# #
# # The application then:
# #   RAW DCS
# #      -> Nova inventory conversion
# #      -> residence time
# #      -> RT-valid production check
# #      -> backward causal trajectory
# #      -> causal process features
# #      -> localized paste chemistry
# #      -> EST-2 H3PO4 features
# #      -> process-control features
# #      -> chemistry/TPD interactions
# #      -> Baseline-C throughput regime features
# #      -> deployment-time dynamic features
# #      -> exact saved feature lists
# #      -> saved CatBoost models
# #      -> LAB_L / LAB_A / LAB_B
# #
# # IMPORTANT
# # ----------
# # * No previous LAB colour value is used.
# # * BA202060-F0 is excluded.
# # * TPD <= 0 is treated as shutdown.
# # * A prediction is generated ONLY for the selected timestamp.
# # * The app refuses to predict if the saved feature definition
# #   cannot be reproduced.
# #
# # ============================================================

# from pathlib import Path
# import json
# import warnings

# import numpy as np
# import pandas as pd
# import streamlit as st
# from catboost import CatBoostRegressor

# warnings.filterwarnings("ignore")

# # ============================================================
# # APPLICATION FILE CONFIGURATION
# # ============================================================
# #
# # The app is self-contained.
# #
# # Keep these files in the SAME GitHub/repository folder as this
# # Streamlit Python file:
# #
# #   Nova_Part-A.xlsx
# #   baseline_c_selected_features.json
# #   catboost_LAB_L_BaselineC.cbm
# #   catboost_LAB_A_BaselineC.cbm
# #   catboost_LAB_B_BaselineC.cbm
# #
# # The raw DCS Excel file is the ONLY file the user uploads.
# #
# # No Windows path, Base Folder input, or model-folder input
# # is required.
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent

# NOVA_FILE = BASE_DIR / "Nova_Part-A.xlsx"

# MODEL_FILES = {
#     "L": BASE_DIR / "catboost_LAB_L_BaselineC.cbm",
#     "A": BASE_DIR / "catboost_LAB_A_BaselineC.cbm",
#     "B": BASE_DIR / "catboost_LAB_B_BaselineC.cbm",
# }

# FEATURE_FILE = BASE_DIR / "baseline_c_selected_features.json"

# TARGETS = ["L", "A", "B"]
# DYNAMIC_LOOKBACK_HOURS = 4.0
# # ============================================================
# # DCS TAGS
# # ============================================================

# DCS_SENSOR_COLS = [
#     "YK-11001",
#     "TPD",
#     "FIC-12001",
#     "LIC-12019",
#     "TIC-12012",
#     "PIC-12013",
#     "LIC-13017",
#     "TIC-13012",
#     "LIC-14003",
#     "TIC-14004",
#     "PIC-14005",
#     "LIC-15006",
#     "TIC-15005",
#     "TIC-15008",
#     "PIC-15024",
#     "SIK-17015",
#     "SIK-17028",
#     "IC-17013",
#     "PIC-17024",
#     "PV-17024",
#     "LI-17017",
#     "LI-17023",
#     "LIC-17016",
#     "TI-17019",
#     "TI-17020",
#     "TI-17022",
#     "VIC-18020",
#     "TI-17113",
#     "TI-17114",
#     "PIC-17183",
#     "LIC-17175",
#     "TI-50002",
#     "TI-50003",
#     "FIC-21009",
#     "YK-21010",
#     "FIC-41007",
#     "YK-41008",
#     "FIC-43007",
#     "YK-43008",
#     "FIC-42007",
#     "YK-42008",
# ]

# PASTE_CHEM_TAGS = [
#     "FIC-21009",
#     "YK-21010",
#     "FIC-41007",
#     "YK-41008",
#     "FIC-42007",
#     "YK-42008",
# ]

# EST2_CHEM_TAGS = [
#     "FIC-43007",
#     "YK-43008",
# ]

# PROCESS_TAGS = [
#     "TPD",
#     "TIC-12012",
#     "PIC-12013",
#     "TIC-13012",
#     "TIC-14004",
#     "PIC-14005",
#     "TIC-15005",
#     "TIC-15008",
#     "PIC-15024",
#     "TI-17019",
#     "TI-17020",
#     "TI-17022",
#     "PIC-17024",
# ]

# AREA_TAGS = {
#     "PASTE": ["YK-11001"],
#     "THROUGHPUT": ["TPD"],

#     "EST1": [
#         "FIC-12001",
#         "LIC-12019",
#         "TIC-12012",
#         "PIC-12013",
#     ],

#     "EST2": [
#         "LIC-13017",
#         "TIC-13012",
#     ],

#     "PP1": [
#         "LIC-14003",
#         "TIC-14004",
#         "PIC-14005",
#     ],

#     "PP2": [
#         "LIC-15006",
#         "TIC-15005",
#         "TIC-15008",
#         "PIC-15024",
#     ],

#     "DRR": [
#         "SIK-17015",
#         "SIK-17028",
#         "IC-17013",
#         "PIC-17024",
#         "PV-17024",
#         "LI-17017",
#         "LI-17023",
#         "LIC-17016",
#         "TI-17019",
#         "TI-17020",
#         "TI-17022",
#         "VIC-18020",
#     ],

#     "JET": [
#         "TI-17113",
#         "TI-17114",
#         "PIC-17183",
#         "LIC-17175",
#     ],

#     "HTM": [
#         "TI-50002",
#         "TI-50003",
#     ],
# }

# SECTION_TAGS = {
#     "EST1": AREA_TAGS["EST1"],
#     "EST2": AREA_TAGS["EST2"],
#     "PP1": AREA_TAGS["PP1"],
#     "PP2": AREA_TAGS["PP2"],
#     "DRR": AREA_TAGS["DRR"],
# }

# EQUIPMENT_MAP = {
#     "EST1": {
#         "DCS_TAG": "LIC-12019",
#         "EQUIPMENT": "12-R01-EST-I",
#     },
#     "EST2": {
#         "DCS_TAG": "LIC-13017",
#         "EQUIPMENT": "13-R01-EST-II",
#     },
#     "PP1": {
#         "DCS_TAG": "LIC-14003",
#         "EQUIPMENT": "14-R01-PP-I",
#     },
#     "PP2": {
#         "DCS_TAG": "LIC-15006",
#         "EQUIPMENT": "15-R01-PP-II",
#     },
#     "DRR": {
#         "DCS_TAG": "LIC-17016",
#         "EQUIPMENT": "17-R01- (DRR)",
#     },
# }

# SECTION_ORDER = [
#     ("DRR", "DRR_RT_HR"),
#     ("PP2", "PP2_RT_HR"),
#     ("PP1", "PP1_RT_HR"),
#     ("EST2", "EST2_RT_HR"),
#     ("EST1", "EST1_RT_HR"),
# ]


# # ============================================================
# # MODEL LOADING
# # ============================================================

# @st.cache_resource
# def load_models():
#     if not FEATURE_FILE.exists():
#         raise FileNotFoundError(
#             f"Feature JSON not found:\n{FEATURE_FILE}"
#         )

#     with open(FEATURE_FILE, "r", encoding="utf-8") as f:
#         saved_features = json.load(f)

#     models = {}

#     for target in TARGETS:
#         path = MODEL_FILES[target]

#         if not path.exists():
#             raise FileNotFoundError(
#                 f"Model not found:\n{path}"
#             )

#         model = CatBoostRegressor()
#         model.load_model(str(path))
#         models[target] = model

#     return models, saved_features


# # ============================================================
# # RAW DCS
# # ============================================================

# @st.cache_data
# def load_raw_dcs(uploaded_file_bytes, uploaded_file_name):

#     if not uploaded_file_bytes:
#         raise ValueError(
#             "Please upload the raw DCS Excel file."
#         )

#     from io import BytesIO

#     # --------------------------------------------------------
#     # Read raw Excel
#     # --------------------------------------------------------

#     raw = pd.read_excel(
#         BytesIO(uploaded_file_bytes),
#         header=None,
#     )

#     if len(raw) < 6:
#         raise ValueError(
#             "Unexpected DCS Excel structure. "
#             "Expected timestamp/tag rows followed by DCS data."
#         )

#     # --------------------------------------------------------
#     # Read tag names
#     # --------------------------------------------------------

#     tag_row = raw.iloc[2]

#     columns = []

#     for i, value in enumerate(tag_row):

#         if i == 0:
#             columns.append("TIMESTAMP")

#         elif pd.isna(value):
#             columns.append(
#                 f"UNNAMED_{i}"
#             )

#         else:
#             columns.append(
#                 str(value).strip()
#             )

#     # --------------------------------------------------------
#     # Extract actual DCS data
#     # --------------------------------------------------------

#     dcs = raw.iloc[5:].copy()

#     dcs.columns = columns

#     dcs = (
#         dcs
#         .dropna(how="all")
#         .reset_index(drop=True)
#     )

#     # --------------------------------------------------------
#     # Convert timestamp
#     #
#     # IMPORTANT:
#     # Keep this as a real datetime.
#     # Do NOT convert to AM/PM text here.
#     # --------------------------------------------------------

#     dcs["TIMESTAMP"] = pd.to_datetime(
#         dcs["TIMESTAMP"],
#         errors="coerce",
#     )

#     # --------------------------------------------------------
#     # Clean timestamp rows
#     # --------------------------------------------------------

#     dcs = (
#         dcs
#         .dropna(subset=["TIMESTAMP"])
#         .sort_values("TIMESTAMP")
#         .drop_duplicates("TIMESTAMP")
#         .reset_index(drop=True)
#     )

#     # --------------------------------------------------------
#     # Check required DCS tags
#     # --------------------------------------------------------

#     missing = [
#         c
#         for c in DCS_SENSOR_COLS
#         if c not in dcs.columns
#     ]

#     if missing:

#         raise ValueError(
#             "Required DCS tags are missing:\n\n"
#             + "\n".join(missing)
#         )

#     # --------------------------------------------------------
#     # Numeric conversion
#     # --------------------------------------------------------

#     for col in DCS_SENSOR_COLS:

#         dcs[col] = pd.to_numeric(
#             dcs[col],
#             errors="coerce",
#         )

#     # --------------------------------------------------------
#     # Remove FLOAT32 invalid sentinel
#     # --------------------------------------------------------

#     sentinel = (
#         dcs["YK-43008"].abs() > 1e30
#     )

#     dcs.loc[
#         sentinel,
#         "YK-43008"
#     ] = np.nan

#     # --------------------------------------------------------
#     # DCS interval
#     # --------------------------------------------------------

#     dcs["INTERVAL_MIN"] = (
#         dcs["TIMESTAMP"].diff()
#         .dt.total_seconds()
#         / 60.0
#     )

#     return dcs

# # ============================================================
# # NOVA
# # ============================================================

# @st.cache_data
# def load_nova_curves():
#     if not NOVA_FILE.exists():
#         raise FileNotFoundError(
#             f"Nova file not found:\n{NOVA_FILE}"
#         )

#     nova = pd.read_excel(
#         NOVA_FILE,
#         sheet_name="Poly-2",
#         header=1,
#     )

#     curves = {}

#     for section, info in EQUIPMENT_MAP.items():

#         mask = (
#             nova["silo_no"]
#             .astype(str)
#             .str.strip()
#             .eq(info["EQUIPMENT"])
#         )

#         curve = nova.loc[
#             mask,
#             [
#                 "Level Percent",
#                 "Level Weight Kgs.",
#             ],
#         ].copy()

#         curve["Level Percent"] = pd.to_numeric(
#             curve["Level Percent"],
#             errors="coerce",
#         )

#         curve["Level Weight Kgs."] = pd.to_numeric(
#             curve["Level Weight Kgs."],
#             errors="coerce",
#         )

#         curve = (
#             curve.dropna()
#             .drop_duplicates("Level Percent")
#             .sort_values("Level Percent")
#             .reset_index(drop=True)
#         )

#         if len(curve) < 2:
#             raise ValueError(
#                 f"Insufficient Nova curve for {section}."
#             )

#         curves[section] = curve

#     return curves


# def add_inventory(dcs, curves):
#     dcs = dcs.copy()

#     for section, info in EQUIPMENT_MAP.items():

#         level = pd.to_numeric(
#             dcs[info["DCS_TAG"]],
#             errors="coerce",
#         )

#         curve = curves[section]

#         x = curve["Level Percent"].to_numpy(float)
#         y = curve["Level Weight Kgs."].to_numpy(float)

#         inventory = np.full(len(dcs), np.nan)

#         valid = (
#             level.notna()
#             & level.ge(x.min())
#             & level.le(x.max())
#         )

#         inventory[valid] = np.interp(
#             level.loc[valid].to_numpy(float),
#             x,
#             y,
#         )

#         dcs[f"{section}_LEVEL_PCT"] = level
#         dcs[f"{section}_INVENTORY_KG"] = inventory

#     return dcs


# # ============================================================
# # RESIDENCE TIME
# # ============================================================

# def calculate_rt(dcs):
#     dcs = dcs.copy()

#     dcs["TPD"] = pd.to_numeric(
#         dcs["TPD"],
#         errors="coerce",
#     )

#     dcs["TPH"] = dcs["TPD"] / 24.0

#     for section in EQUIPMENT_MAP:

#         dcs[f"{section}_RT_HR"] = (
#             dcs[f"{section}_INVENTORY_KG"]
#             /
#             (dcs["TPH"] * 1000.0)
#         )

#         dcs.loc[
#             dcs["TPH"] <= 0,
#             f"{section}_RT_HR",
#         ] = np.nan

#     rt_cols = [
#         "EST1_RT_HR",
#         "EST2_RT_HR",
#         "PP1_RT_HR",
#         "PP2_RT_HR",
#         "DRR_RT_HR",
#     ]

#     dcs["TOTAL_RT_HR"] = dcs[rt_cols].sum(
#         axis=1,
#         min_count=5,
#     )

#     # Validated operational construction.
#     dcs["RT_VALID"] = (
#         dcs["TPD"].ge(375)
#         &
#         dcs[rt_cols].notna().all(axis=1)
#     )

#     dcs["RT_STATUS"] = np.where(
#         dcs["RT_VALID"],
#         "VALID",
#         np.where(
#             dcs["TPD"].le(0),
#             "SHUTDOWN_TPD_LE_0",
#             "NOT_RT_VALID",
#         ),
#     )

#     return dcs


# def build_runs(dcs):
#     gap = (
#         dcs["TIMESTAMP"].diff()
#         .dt.total_seconds()
#         / 3600.0
#     )

#     continuous = gap.eq(0.25)

#     run_break = (
#         dcs["RT_VALID"].ne(dcs["RT_VALID"].shift())
#         | ~continuous
#     )

#     dcs = dcs.copy()
#     dcs["VALID_RUN_ID"] = run_break.cumsum()

#     runs = (
#         dcs.loc[dcs["RT_VALID"]]
#         .groupby("VALID_RUN_ID")
#         .agg(
#             START_TS=("TIMESTAMP", "min"),
#             END_TS=("TIMESTAMP", "max"),
#             N_ROWS=("TIMESTAMP", "size"),
#         )
#         .reset_index()
#     )

#     runs["DURATION_HR"] = (
#         runs["END_TS"] - runs["START_TS"]
#     ).dt.total_seconds() / 3600.0

#     return dcs, runs


# # ============================================================
# # CAUSAL TRAJECTORY
# # ============================================================

# def find_run(runs, timestamp):
#     match = runs[
#         (runs["START_TS"] <= timestamp)
#         &
#         (runs["END_TS"] >= timestamp)
#     ]

#     if match.empty:
#         return None

#     return match.iloc[0]


# def get_rt_at_time(dcs, timestamp, rt_column):
#     timestamp = pd.Timestamp(timestamp)

#     if (
#         timestamp < dcs["TIMESTAMP"].min()
#         or
#         timestamp > dcs["TIMESTAMP"].max()
#     ):
#         return np.nan, "OUTSIDE_RANGE"

#     before = dcs[
#         (dcs["TIMESTAMP"] <= timestamp)
#         & dcs["RT_VALID"]
#     ].tail(1)

#     after = dcs[
#         (dcs["TIMESTAMP"] >= timestamp)
#         & dcs["RT_VALID"]
#     ].head(1)

#     if before.empty or after.empty:
#         return np.nan, "NO_BRACKET"

#     t0 = before.iloc[0]["TIMESTAMP"]
#     t1 = after.iloc[0]["TIMESTAMP"]

#     gap_hr = (
#         t1 - t0
#     ).total_seconds() / 3600.0

#     if gap_hr > 1.0:
#         return np.nan, "RT_HISTORY_GAP"

#     r0 = before.iloc[0][rt_column]
#     r1 = after.iloc[0][rt_column]

#     if pd.isna(r0) or pd.isna(r1):
#         return np.nan, "RT_MISSING"

#     if t0 == t1:
#         return float(r0), "DIRECT"

#     fraction = (
#         timestamp - t0
#     ).total_seconds() / (
#         t1 - t0
#     ).total_seconds()

#     return float(r0 + fraction * (r1 - r0)), "INTERPOLATED"


# def calculate_trajectory(dcs, runs, timestamp):
#     timestamp = pd.Timestamp(timestamp)

#     run = find_run(runs, timestamp)

#     if run is None:
#         return None

#     current = timestamp

#     result = {
#         "TIMESTAMP": timestamp,
#         "TRAJECTORY_VALID": True,
#         "VALID_RUN_ID": int(run["VALID_RUN_ID"]),
#         "RUN_START": run["START_TS"],
#         "RUN_END": run["END_TS"],
#     }

#     for section, rt_col in SECTION_ORDER:

#         rt, status = get_rt_at_time(
#             dcs,
#             current,
#             rt_col,
#         )

#         if pd.isna(rt):
#             return None

#         result[f"{section}_RT_HR"] = rt
#         result[f"{section}_RT_STATUS"] = status

#         boundary = (
#             current
#             - pd.Timedelta(hours=float(rt))
#         )

#         result[f"{section}_BOUNDARY_TS"] = boundary

#         current = boundary

#     result["EST1_UPSTREAM_TS"] = current

#     boundaries = [
#         result["DRR_BOUNDARY_TS"],
#         result["PP2_BOUNDARY_TS"],
#         result["PP1_BOUNDARY_TS"],
#         result["EST2_BOUNDARY_TS"],
#         result["EST1_BOUNDARY_TS"],
#     ]

#     if not all(
#         run["START_TS"] <= x <= run["END_TS"]
#         for x in boundaries
#     ):
#         return None

#     return result


# # ============================================================
# # WINDOW FEATURE EXTRACTION
# # EXACT BASELINE-C CELL 4 LOGIC
# # ============================================================

# def extract_window_features(
#     dcs_df,
#     start_ts,
#     end_ts,
#     tags,
#     prefix,
# ):
#     start_ts = pd.Timestamp(start_ts)
#     end_ts = pd.Timestamp(end_ts)

#     if (
#         pd.isna(start_ts)
#         or pd.isna(end_ts)
#         or end_ts <= start_ts
#     ):
#         return {}

#     window = dcs_df[
#         (dcs_df["TIMESTAMP"] >= start_ts)
#         &
#         (dcs_df["TIMESTAMP"] <= end_ts)
#     ].sort_values("TIMESTAMP").copy()

#     if window.empty:
#         return {}

#     duration_hr = (
#         end_ts - start_ts
#     ).total_seconds() / 3600.0

#     result = {}

#     for tag in tags:

#         if tag not in window.columns:
#             continue

#         values = pd.to_numeric(
#             window[tag],
#             errors="coerce",
#         )

#         valid = values.notna()

#         if valid.sum() == 0:
#             continue

#         v = values.loc[valid].astype(float)

#         first = float(v.iloc[0])
#         last = float(v.iloc[-1])

#         result[f"{prefix}{tag}_MEAN"] = float(v.mean())
#         result[f"{prefix}{tag}_STD"] = float(v.std(ddof=0))
#         result[f"{prefix}{tag}_MIN"] = float(v.min())
#         result[f"{prefix}{tag}_MAX"] = float(v.max())
#         result[f"{prefix}{tag}_RANGE"] = float(
#             v.max() - v.min()
#         )
#         result[f"{prefix}{tag}_FIRST"] = first
#         result[f"{prefix}{tag}_LAST"] = last
#         result[f"{prefix}{tag}_DELTA"] = last - first
#         result[f"{prefix}{tag}_N"] = int(valid.sum())

#         for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
#             result[
#                 f"{prefix}{tag}_P{int(q * 100)}"
#             ] = float(v.quantile(q))

#         if len(v) >= 2 and duration_hr > 0:

#             times = (
#                 window.loc[valid, "TIMESTAMP"]
#                 - window.loc[valid, "TIMESTAMP"].iloc[0]
#             ).dt.total_seconds().to_numpy() / 3600.0

#             y = v.to_numpy(float)

#             if len(np.unique(times)) >= 2:
#                 slope = np.polyfit(times, y, 1)[0]
#             else:
#                 slope = 0.0
#         else:
#             slope = 0.0

#         result[
#             f"{prefix}{tag}_SLOPE"
#         ] = float(slope)

#     return result


# # ============================================================
# # CAUSAL STATIC FEATURES
# # ============================================================

# def build_static_features(trajectory, dcs):
#     ts = pd.Timestamp(trajectory["TIMESTAMP"])

#     est1 = pd.Timestamp(trajectory["EST1_BOUNDARY_TS"])
#     est2 = pd.Timestamp(trajectory["EST2_BOUNDARY_TS"])
#     pp1 = pd.Timestamp(trajectory["PP1_BOUNDARY_TS"])
#     pp2 = pd.Timestamp(trajectory["PP2_BOUNDARY_TS"])
#     drr = pd.Timestamp(trajectory["DRR_BOUNDARY_TS"])

#     row = {
#         "LAB_TIMESTAMP": ts,

#         # Explicit RT predictors used by the model.
#         "EST1_RT_HR": trajectory["EST1_RT_HR"],
#         "EST2_RT_HR": trajectory["EST2_RT_HR"],
#         "PP1_RT_HR": trajectory["PP1_RT_HR"],
#         "PP2_RT_HR": trajectory["PP2_RT_HR"],
#         "DRR_RT_HR": trajectory["DRR_RT_HR"],

#         "TOTAL_RT_HR": sum(
#             trajectory[x]
#             for x in [
#                 "EST1_RT_HR",
#                 "EST2_RT_HR",
#                 "PP1_RT_HR",
#                 "PP2_RT_HR",
#                 "DRR_RT_HR",
#             ]
#         ),

#         "EST1_BOUNDARY_TS": est1,
#         "EST2_BOUNDARY_TS": est2,
#         "PP1_BOUNDARY_TS": pp1,
#         "PP2_BOUNDARY_TS": pp2,
#         "DRR_BOUNDARY_TS": drr,
#     }

#     # --------------------------------------------------------
#     # Five causal section windows
#     # --------------------------------------------------------

#     section_windows = {
#         "EST1": (est1, est2),
#         "EST2": (est2, pp1),
#         "PP1": (pp1, pp2),
#         "PP2": (pp2, drr),
#         "DRR": (drr, ts),
#     }

#     for section, (start, end) in section_windows.items():

#         row.update(
#             extract_window_features(
#                 dcs,
#                 start,
#                 end,
#                 SECTION_TAGS[section],
#                 f"{section}_",
#             )
#         )

#         row[f"{section}_WINDOW_START"] = start
#         row[f"{section}_WINDOW_END"] = end
#         row[f"{section}_WINDOW_HR"] = (
#             end - start
#         ).total_seconds() / 3600.0
#         row[f"{section}_DCS_N"] = len(
#             dcs[
#                 (dcs["TIMESTAMP"] >= start)
#                 &
#                 (dcs["TIMESTAMP"] <= end)
#             ]
#         )

#     # --------------------------------------------------------
#     # Full causal window
#     # --------------------------------------------------------

#     global_start = est1
#     global_end = ts

#     row.update(
#         extract_window_features(
#             dcs,
#             global_start,
#             global_end,
#             AREA_TAGS["PASTE"],
#             "PASTE_",
#         )
#     )

#     row.update(
#         extract_window_features(
#             dcs,
#             global_start,
#             global_end,
#             AREA_TAGS["THROUGHPUT"],
#             "THROUGHPUT_",
#         )
#     )

#     row.update(
#         extract_window_features(
#             dcs,
#             global_start,
#             global_end,
#             AREA_TAGS["JET"],
#             "JET_",
#         )
#     )

#     row.update(
#         extract_window_features(
#             dcs,
#             global_start,
#             global_end,
#             AREA_TAGS["HTM"],
#             "HTM_",
#         )
#     )

#     # --------------------------------------------------------
#     # Chemistry trajectory
#     #
#     # IMPORTANT:
#     # Historical model feature names such as CHEM_COBALT_*
#     # are preserved exactly where applicable.
#     # --------------------------------------------------------

#     chemistry_groups = {
#         "ANTIMONY": ["FIC-21009", "YK-21010"],
#         "RED_TONER": ["FIC-41007", "YK-41008"],
#         "PHOSPHORIC_ACID": ["FIC-43007", "YK-43008"],
#         "BLUE_TONER": ["FIC-42007", "YK-42008"],
#     }

#     for group, tags in chemistry_groups.items():

#         row.update(
#             extract_window_features(
#                 dcs,
#                 global_start,
#                 global_end,
#                 tags,
#                 f"CHEM_{group}_",
#             )
#         )

#     # Historical saved models used CHEM_COBALT_* for the
#     # FIC-41007/YK-41008 channels. Preserve those aliases.
#     cobalt = extract_window_features(
#         dcs,
#         global_start,
#         global_end,
#         ["FIC-41007", "YK-41008"],
#         "CHEM_COBALT_",
#     )
#     row.update(cobalt)

#     # --------------------------------------------------------
#     # Localized paste chemistry
#     # --------------------------------------------------------

#     for minutes in [30, 60, 120, 240]:

#         row.update(
#             extract_window_features(
#                 dcs,
#                 est1 - pd.Timedelta(minutes=minutes),
#                 est1,
#                 PASTE_CHEM_TAGS,
#                 f"PASTE_{minutes}M_",
#             )
#         )

#     row.update(
#         extract_window_features(
#             dcs,
#             est1 - pd.Timedelta(hours=4),
#             est1,
#             PASTE_CHEM_TAGS,
#             "PASTE_4H_",
#         )
#     )

#     # --------------------------------------------------------
#     # H3PO4 localized at EST-2
#     # --------------------------------------------------------

#     row.update(
#         extract_window_features(
#             dcs,
#             est2,
#             pp1,
#             EST2_CHEM_TAGS,
#             "EST2_H3PO4_FULL_",
#         )
#     )

#     for minutes in [30, 60, 120]:

#         end = min(
#             est2 + pd.Timedelta(minutes=minutes),
#             pp1,
#         )

#         if end > est2:
#             row.update(
#                 extract_window_features(
#                     dcs,
#                     est2,
#                     end,
#                     EST2_CHEM_TAGS,
#                     f"EST2_H3PO4_{minutes}M_",
#                 )
#             )

#     # --------------------------------------------------------
#     # Process-control layer
#     # EXACT Baseline C Cell 6 logic
#     # --------------------------------------------------------

#     window = dcs[
#         (dcs["TIMESTAMP"] >= est1)
#         &
#         (dcs["TIMESTAMP"] <= ts)
#     ].copy()

#     if not window.empty:

#         if "TPD" in window.columns:

#             tpd = pd.to_numeric(
#                 window["TPD"],
#                 errors="coerce",
#             ).dropna()

#             if len(tpd):

#                 row["CTRL_TPD_MEAN"] = float(tpd.mean())
#                 row["CTRL_TPD_STD"] = float(tpd.std(ddof=0))
#                 row["CTRL_TPD_MIN"] = float(tpd.min())
#                 row["CTRL_TPD_MAX"] = float(tpd.max())
#                 row["CTRL_TPD_RANGE"] = float(
#                     tpd.max() - tpd.min()
#                 )
#                 row["CTRL_TPD_FIRST"] = float(tpd.iloc[0])
#                 row["CTRL_TPD_LAST"] = float(tpd.iloc[-1])
#                 row["CTRL_TPD_DELTA"] = float(
#                     tpd.iloc[-1] - tpd.iloc[0]
#                 )

#                 for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
#                     row[
#                         f"CTRL_TPD_P{int(q * 100)}"
#                     ] = float(tpd.quantile(q))

#         process_summary = {}

#         for tag in PROCESS_TAGS:

#             if tag == "TPD" or tag not in window.columns:
#                 continue

#             values = pd.to_numeric(
#                 window[tag],
#                 errors="coerce",
#             ).dropna()

#             if len(values) == 0:
#                 continue

#             clean = tag.replace("-", "_")

#             process_summary[tag] = {
#                 "MEAN": float(values.mean()),
#                 "STD": float(values.std(ddof=0)),
#                 "MIN": float(values.min()),
#                 "MAX": float(values.max()),
#                 "RANGE": float(
#                     values.max() - values.min()
#                 ),
#                 "FIRST": float(values.iloc[0]),
#                 "LAST": float(values.iloc[-1]),
#                 "DELTA": float(
#                     values.iloc[-1] - values.iloc[0]
#                 ),
#             }

#             for stat, value in process_summary[tag].items():
#                 row[
#                     f"CTRL_{clean}_{stat}"
#                 ] = value

#         tpd_mean = row.get("CTRL_TPD_MEAN", np.nan)

#         safe_tpd = (
#             tpd_mean
#             if pd.notna(tpd_mean) and abs(tpd_mean) >= 50
#             else np.nan
#         )

#         if pd.notna(safe_tpd):
#             for tag, stats in process_summary.items():

#                 clean = tag.replace("-", "_")

#                 row[
#                     f"CTRL_RATIO_{clean}_MEAN_PER_TPD"
#                 ] = stats["MEAN"] / safe_tpd

#         tpd_delta = row.get("CTRL_TPD_DELTA", np.nan)

#         safe_delta = (
#             tpd_delta
#             if pd.notna(tpd_delta) and abs(tpd_delta) >= 1
#             else np.nan
#         )

#         if pd.notna(safe_delta):

#             for tag, stats in process_summary.items():

#                 clean = tag.replace("-", "_")

#                 row[
#                     f"CTRL_RESPONSE_{clean}_PER_TPD_CHANGE"
#                 ] = stats["DELTA"] / safe_delta

#     # --------------------------------------------------------
#     # Chemistry / throughput interactions
#     # EXACT Baseline C Cell 7 logic
#     # --------------------------------------------------------

#     safe_tpd = row.get("CTRL_TPD_MEAN", np.nan)

#     safe_tpd = (
#         safe_tpd
#         if pd.notna(safe_tpd) and abs(safe_tpd) >= 50
#         else np.nan
#     )

#     chemistry_tokens = {
#         "ANTIMONY": "FIC-21009",
#         "RED_TONER": "FIC-41007",
#         "BLUE_TONER": "FIC-42007",
#         "H3PO4": "FIC-43007",
#     }

#     for chemistry, tag in chemistry_tokens.items():

#         matching = [
#             c for c in list(row.keys())
#             if tag in str(c)
#             and (
#                 "_MEAN" in str(c)
#                 or "_LAST" in str(c)
#                 or "_FIRST" in str(c)
#                 or "_DELTA" in str(c)
#             )
#         ]

#         for col in matching:

#             value = pd.to_numeric(
#                 pd.Series([row[col]]),
#                 errors="coerce",
#             ).iloc[0]

#             row[
#                 f"CHEM_CTRL_{chemistry}_{col}_PER_TPD"
#             ] = (
#                 value / safe_tpd
#                 if pd.notna(safe_tpd)
#                 else np.nan
#             )

#     # --------------------------------------------------------
#     # Baseline C throughput regime
#     # EXACT Cell 7C logic
#     # --------------------------------------------------------

#     if not window.empty and "TPD" in window.columns:

#         tpd = pd.to_numeric(
#             window["TPD"],
#             errors="coerce",
#         )

#         valid = tpd.notna()

#         if valid.any():

#             times = window.loc[valid, "TIMESTAMP"]
#             values = tpd.loc[valid].astype(float)

#             current = float(values.iloc[-1])
#             full_mean = float(values.mean())
#             first = float(values.iloc[0])

#             for minutes in [30, 60, 120, 240]:

#                 cutoff = (
#                     ts - pd.Timedelta(minutes=minutes)
#                 )

#                 recent = values.loc[times >= cutoff]

#                 if recent.empty:
#                     recent = values

#                 suffix = f"{minutes}M"

#                 row[f"C_TPD_{suffix}_MEAN"] = float(
#                     recent.mean()
#                 )
#                 row[f"C_TPD_{suffix}_STD"] = float(
#                     recent.std(ddof=0)
#                 )
#                 row[f"C_TPD_{suffix}_MIN"] = float(
#                     recent.min()
#                 )
#                 row[f"C_TPD_{suffix}_MAX"] = float(
#                     recent.max()
#                 )
#                 row[f"C_TPD_{suffix}_DELTA"] = float(
#                     recent.iloc[-1] - recent.iloc[0]
#                 )
#                 row[f"C_TPD_{suffix}_RATE"] = float(
#                     (
#                         recent.iloc[-1]
#                         - recent.iloc[0]
#                     )
#                     / (minutes / 60.0)
#                 )

#             row["C_TPD_CURRENT"] = current
#             row["C_TPD_MINUS_400"] = current - 400.0
#             row["C_TPD_RATIO_TO_400"] = current / 400.0
#             row["C_TPD_MEAN_MINUS_400"] = full_mean - 400.0
#             row["C_TPD_MEAN_RATIO_TO_400"] = full_mean / 400.0

#             row["C_TPD_FRAC_BELOW_500"] = float(
#                 (values < 500).mean()
#             )
#             row["C_TPD_FRAC_500_600"] = float(
#                 (
#                     (values >= 500)
#                     & (values < 600)
#                 ).mean()
#             )
#             row["C_TPD_FRAC_600_800"] = float(
#                 (
#                     (values >= 600)
#                     & (values <= 800)
#                 ).mean()
#             )
#             row["C_TPD_FRAC_ABOVE_800"] = float(
#                 (values > 800).mean()
#             )
#             row["C_TPD_FRAC_GE_600"] = float(
#                 (values >= 600).mean()
#             )

#             if current < 500:
#                 regime = 0
#             elif current < 600:
#                 regime = 1
#             elif current <= 800:
#                 regime = 2
#             else:
#                 regime = 3

#             row["C_TPD_CURRENT_REGIME"] = regime

#             row["C_TPD_CROSSED_500"] = float(
#                 first < 500 and current >= 500
#             )
#             row["C_TPD_CROSSED_600"] = float(
#                 first < 600 and current >= 600
#             )
#             row["C_TPD_CROSSED_800"] = float(
#                 first < 800 and current >= 800
#             )
#             row[
#                 "C_TPD_STARTED_BELOW_600_ENDED_GE_600"
#             ] = float(
#                 first < 600 and current >= 600
#             )
#             row[
#                 "C_TPD_STARTED_BELOW_800_ENDED_GE_800"
#             ] = float(
#                 first < 800 and current >= 800
#             )

#             row["C_TPD_MINUS_600"] = current - 600.0
#             row["C_TPD_MINUS_800"] = current - 800.0
#             row["C_TPD_MEAN_MINUS_600"] = full_mean - 600.0
#             row["C_TPD_MEAN_MINUS_800"] = full_mean - 800.0

#             for minutes in [30, 60, 120, 240]:

#                 cutoff = (
#                     ts - pd.Timedelta(minutes=minutes)
#                 )

#                 recent = values.loc[times >= cutoff]

#                 if recent.empty:
#                     recent = values

#                 row[
#                     f"C_TPD_{minutes}M_VS_FULL_DELTA"
#                 ] = float(
#                     recent.mean() - full_mean
#                 )

#                 row[
#                     f"C_TPD_{minutes}M_VS_FULL_RATIO"
#                 ] = (
#                     float(recent.mean() / full_mean)
#                     if full_mean != 0
#                     else np.nan
#                 )

#     return row


# # ============================================================
# # DYNAMIC FEATURES
# # ============================================================

# DYNAMIC_EXCLUDE = {
#     "LAB_TIMESTAMP",
#     "EST1_BOUNDARY_TS",
#     "EST2_BOUNDARY_TS",
#     "PP1_BOUNDARY_TS",
#     "PP2_BOUNDARY_TS",
#     "DRR_BOUNDARY_TS",
# }


# def add_pairwise_dynamic(current, previous):
#     """
#     Raw-DCS deployment equivalent of the training CHANGE /
#     RELCHANGE / RATE feature family.

#     No previous LAB colour is used.
#     """

#     result = current.copy()

#     current_ts = pd.Timestamp(current["LAB_TIMESTAMP"])
#     previous_ts = pd.Timestamp(previous["LAB_TIMESTAMP"])

#     interval_hr = (
#         current_ts - previous_ts
#     ).total_seconds() / 3600.0

#     result["LAB_INTERVAL_HR"] = interval_hr
#     result["DYNAMIC_CHANGE_VALID"] = (
#         0 < interval_hr <= 12
#     )

#     numeric_cols = []

#     for col, value in current.items():

#         if col in DYNAMIC_EXCLUDE:
#             continue

#         if col in {
#             "LAB_INTERVAL_HR",
#             "DYNAMIC_CHANGE_VALID",
#         }:
#             continue

#         try:
#             a = pd.to_numeric(
#                 pd.Series([value]),
#                 errors="coerce",
#             ).iloc[0]
#         except Exception:
#             continue

#         if pd.notna(a):
#             numeric_cols.append(col)

#     for col in numeric_cols:

#         cur = pd.to_numeric(
#             pd.Series([current.get(col)]),
#             errors="coerce",
#         ).iloc[0]

#         prev = pd.to_numeric(
#             pd.Series([previous.get(col)]),
#             errors="coerce",
#         ).iloc[0]

#         change = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan

#         result[f"CHANGE__{col}"] = change

#         denominator = (
#             abs(prev)
#             if pd.notna(prev)
#             else np.nan
#         )

#         if pd.notna(denominator):
#             denominator = max(denominator, 1e-3)

#         result[f"RELCHANGE__{col}"] = (
#             change / denominator
#             if pd.notna(change) and pd.notna(denominator)
#             else np.nan
#         )

#         result[f"RATE__{col}"] = (
#             change / interval_hr
#             if pd.notna(change) and interval_hr > 0
#             else np.nan
#         )

#     return result


# # ============================================================
# # FIND PREVIOUS VALID DCS PREDICTION STATE
# # ============================================================

# def find_previous_timestamp(dcs, selected_ts):
#     lower = selected_ts - pd.Timedelta(
#         hours=DYNAMIC_LOOKBACK_HOURS
#     )

#     candidates = dcs[
#         (dcs["TIMESTAMP"] < selected_ts)
#         &
#         (dcs["TIMESTAMP"] >= lower)
#         &
#         dcs["RT_VALID"]
#     ]

#     if candidates.empty:
#         return None

#     return candidates.iloc[-1]["TIMESTAMP"]


# # ============================================================
# # BUILD MODEL INPUT FOR ONE SELECTED TIMESTAMP
# # ============================================================

# def build_prediction_feature_row(
#     dcs,
#     runs,
#     selected_ts,
# ):
#     trajectory = calculate_trajectory(
#         dcs,
#         runs,
#         selected_ts,
#     )

#     if trajectory is None:
#         raise ValueError(
#             "The selected timestamp does not have a valid "
#             "continuous causal RT trajectory."
#         )

#     current_static = build_static_features(
#         trajectory,
#         dcs,
#     )

#     previous_ts = find_previous_timestamp(
#         dcs,
#         selected_ts,
#     )

#     dynamic_row = current_static.copy()

#     if previous_ts is not None:

#         previous_trajectory = calculate_trajectory(
#             dcs,
#             runs,
#             previous_ts,
#         )

#         if previous_trajectory is not None:

#             previous_static = build_static_features(
#                 previous_trajectory,
#                 dcs,
#             )

#             dynamic_row = add_pairwise_dynamic(
#                 current_static,
#                 previous_static,
#             )

#     return (
#         pd.DataFrame([dynamic_row]),
#         trajectory,
#         previous_ts,
#     )


# # ============================================================
# # FEATURE AUDIT + PREDICTION
# # ============================================================

# def prepare_model_matrix(feature_row, saved_features, target):
#     selected = saved_features[target]

#     missing = [
#         c for c in selected
#         if c not in feature_row.columns
#     ]

#     if missing:
#         return None, missing

#     X = (
#         feature_row[selected]
#         .replace([np.inf, -np.inf], np.nan)
#         .apply(pd.to_numeric, errors="coerce")
#         .fillna(0.0)
#     )

#     return X, []


# def run_prediction(
#     feature_row,
#     models,
#     saved_features,
# ):
#     matrices = {}
#     failures = {}

#     for target in TARGETS:

#         key = f"LAB_{target}"

#         # JSON supports LAB_L / LAB_A / LAB_B.
#         selected = saved_features.get(key)

#         if selected is None:
#             failures[target] = [
#                 f"Missing JSON key: {key}"
#             ]
#             continue

#         X, missing = prepare_model_matrix(
#             feature_row,
#             saved_features,
#             key,
#         )

#         if missing:
#             failures[target] = missing
#         else:
#             matrices[target] = X

#     if failures:
#         return None, failures

#     result = {}

#     for target in TARGETS:

#         result[f"PRED_{target}"] = float(
#             models[target].predict(
#                 matrices[target]
#             )[0]
#         )

#     return result, {}


# # ============================================================
# # STREAMLIT UI
# # ============================================================

# st.set_page_config(
#     page_title="POLY-II Colour Prediction",
#     page_icon="🎨",
#     layout="wide",
# )

# # ============================================================
# # APPLICATION RESET CONTROL
# # ============================================================

# if "app_reset_id" not in st.session_state:
#     st.session_state.app_reset_id = 0


# # ------------------------------------------------------------
# # Top header
# # ------------------------------------------------------------

# header_col1, header_col2 = st.columns(
#     [8, 1.5]
# )

# with header_col1:

#     st.title(
#         "POLY-II Colour Prediction"
#     )

#     st.caption(
#         "Single timestamp prediction "
#         "from raw 15-minute DCS"
#     )


# with header_col2:

#     st.markdown(
#         "<div style='height: 15px'></div>",
#         unsafe_allow_html=True,
#     )

#     reset_clicked = st.button(
#         "↻  New Prediction",
#         use_container_width=True,
#         help=(
#             "Clear the uploaded DCS file, selected "
#             "date/time and all prediction results."
#         ),
#     )


# # ============================================================
# # RESET EVERYTHING
# # ============================================================

# if reset_clicked:

#     # --------------------------------------------------------
#     # Clear all Streamlit session/widget state
#     # --------------------------------------------------------

#     current_reset_id = (
#         st.session_state.app_reset_id
#     )

#     for key in list(
#         st.session_state.keys()
#     ):

#         if key != "app_reset_id":
#             del st.session_state[key]

#     # --------------------------------------------------------
#     # Change uploader widget key.
#     #
#     # This is IMPORTANT because simply calling rerun()
#     # does not remove the uploaded file from st.file_uploader.
#     # --------------------------------------------------------

#     st.session_state.app_reset_id = (
#         current_reset_id + 1
#     )

#     # --------------------------------------------------------
#     # Clear cached uploaded-data processing.
#     #
#     # Models/Nova are intentionally NOT cleared.
#     # --------------------------------------------------------

#     st.cache_data.clear()

#     # --------------------------------------------------------
#     # Restart application from the beginning.
#     # --------------------------------------------------------

#     st.rerun()


# st.divider()


# # ============================================================
# # 1. UPLOAD DCS DATA
# # ============================================================

# st.subheader("1. Upload DCS Data")

# st.write(
#     "Upload the raw 15-minute DCS Excel file to begin prediction."
# )

# uploaded_dcs = st.file_uploader(
#     "Raw DCS Excel File",
#     type=["xlsx", "xls"],
#     help="Upload the raw POLY-II DCS Excel file.",
# )

# if uploaded_dcs is None:

#     st.info(
#         "Please upload the raw DCS Excel file."
#     )

#     st.caption(
#         "The application uses the bundled Nova residence-time "
#         "data and Baseline C colour prediction models automatically."
#     )

#     st.stop()


# # ============================================================
# # 2. VALIDATE APPLICATION FILES
# # ============================================================

# required_paths = {
#     "Nova_Part-A.xlsx": NOVA_FILE,
#     "catboost_LAB_L_BaselineC.cbm": MODEL_FILES["L"],
#     "catboost_LAB_A_BaselineC.cbm": MODEL_FILES["A"],
#     "catboost_LAB_B_BaselineC.cbm": MODEL_FILES["B"],
#     "baseline_c_selected_features.json": FEATURE_FILE,
# }

# missing_files = [
#     name
#     for name, path in required_paths.items()
#     if not path.exists()
# ]

# if missing_files:

#     st.error(
#         "Required application files are missing."
#     )

#     st.code(
#         "\n".join(missing_files),
#         language="text",
#     )

#     st.stop()


# # ============================================================
# # 3. LOAD DATA
# # ============================================================

# try:

#     with st.spinner(
#         "Loading DCS data and residence-time configuration..."
#     ):

#         models, saved_features = load_models()

#         dcs = load_raw_dcs(
#             uploaded_dcs.getvalue(),
#             uploaded_dcs.name,
#         )

#         curves = load_nova_curves()

#         dcs = add_inventory(
#             dcs,
#             curves,
#         )

#         dcs = calculate_rt(
#             dcs
#         )

#         dcs, runs = build_runs(
#             dcs
#         )

# except Exception as exc:

#     st.error(
#         "Application could not load the supplied data."
#     )

#     st.exception(exc)

#     st.stop()


# # ============================================================
# # DATA SUMMARY
# # ============================================================

# min_timestamp = dcs["TIMESTAMP"].min()
# max_timestamp = dcs["TIMESTAMP"].max()

# valid_count = int(
#     dcs["RT_VALID"].sum()
# )

# total_count = len(dcs)

# s1, s2, s3 = st.columns(3)

# s1.metric(
#     "DCS Records",
#     f"{total_count:,}",
# )

# s2.metric(
#     "RT-Valid Records",
#     f"{valid_count:,}",
# )

# s3.metric(
#     "DCS Period",
#     (
#         f"{min_timestamp:%d %b %Y}"
#         f" – "
#         f"{max_timestamp:%d %b %Y}"
#     ),
# )

# st.divider()


# # ============================================================
# # 4. SELECT DATE AND TIME
# # ============================================================

# st.subheader("2. Select Prediction Date & Time")

# st.write(
#     "Select the production date and enter the DCS time "
#     "for which you want to predict colour."
# )

# valid_dcs = dcs.loc[
#     dcs["RT_VALID"],
#     ["TIMESTAMP", "TPD"],
# ].copy()

# if valid_dcs.empty:

#     st.error(
#         "No RT-valid production observations were found."
#     )

#     st.stop()


# # ============================================================
# # DATE INPUT
# # ============================================================

# min_date = valid_dcs["TIMESTAMP"].dt.date.min()
# max_date = valid_dcs["TIMESTAMP"].dt.date.max()

# default_date = max_date

# c1, c2, c3 = st.columns([1.5, 1, 0.8])


# with c1:

#     selected_date = st.date_input(
#         "Prediction Date",
#         value=default_date,
#         min_value=min_date,
#         max_value=max_date,
#         format="DD/MM/YYYY",
#     )


# # ============================================================
# # TIME INPUT
# # ============================================================

# with c2:

#     selected_time_text = st.text_input(
#         "Time",
#         value="08:02:49",
#         placeholder="HH:MM:SS",
#         help="Enter time as HH:MM:SS.",
#     )


# # ============================================================
# # AM / PM
# # ============================================================

# with c3:

#     selected_ampm = st.selectbox(
#         "AM / PM",
#         ["AM", "PM"],
#         index=0,
#     )


# # ============================================================
# # PARSE USER TIME
# # ============================================================

# def parse_user_time(
#     time_text,
#     ampm,
# ):

#     time_text = str(
#         time_text
#     ).strip()

#     # Allow:
#     # 08:02
#     # 08:02:49
#     # 8:02
#     # 8:02:49

#     formats = [
#         "%I:%M:%S",
#         "%I:%M",
#         "%H:%M:%S",
#         "%H:%M",
#     ]

#     parsed = None

#     for fmt in formats:

#         try:

#             parsed = pd.to_datetime(
#                 time_text,
#                 format=fmt,
#             )

#             break

#         except Exception:
#             continue

#     if parsed is None:

#         raise ValueError(
#             "Invalid time format. "
#             "Please enter HH:MM or HH:MM:SS."
#         )

#     hour = parsed.hour

#     minute = parsed.minute

#     second = parsed.second

#     # --------------------------------------------------------
#     # Convert according to selected AM / PM
#     # --------------------------------------------------------

#     if ampm == "AM":

#         if hour == 12:
#             hour = 0

#     else:

#         if hour < 12:
#             hour += 12

#     return pd.Timestamp(
#         year=selected_date.year,
#         month=selected_date.month,
#         day=selected_date.day,
#         hour=hour,
#         minute=minute,
#         second=second,
#     )


# # ============================================================
# # BUILD SELECTED TIMESTAMP
# # ============================================================

# try:

#     selected_ts = parse_user_time(
#         selected_time_text,
#         selected_ampm,
#     )

# except ValueError as exc:

#     st.warning(
#         str(exc)
#     )

#     st.stop()


# # ============================================================
# # FIND NEAREST / EXACT DCS TIMESTAMP
# # ============================================================

# selected_rows = dcs[
#     dcs["TIMESTAMP"] == selected_ts
# ]

# # ------------------------------------------------------------
# # If user enters a time that doesn't exactly exist,
# # show nearest DCS timestamp.
# # ------------------------------------------------------------

# if selected_rows.empty:

#     nearest_idx = (
#         (dcs["TIMESTAMP"] - selected_ts)
#         .abs()
#         .idxmin()
#     )

#     nearest_ts = dcs.loc[
#         nearest_idx,
#         "TIMESTAMP",
#     ]

#     difference_seconds = abs(
#         (
#             nearest_ts
#             - selected_ts
#         ).total_seconds()
#     )

#     if difference_seconds <= 60:

#         st.info(
#             f"No exact DCS timestamp exists at "
#             f"{selected_ts.strftime('%I:%M:%S %p')}. "
#             f"Using nearest DCS timestamp: "
#             f"{nearest_ts.strftime('%I:%M:%S %p')}."
#         )

#         selected_ts = nearest_ts

#         selected_rows = dcs[
#             dcs["TIMESTAMP"] == selected_ts
#         ]

#     else:

#         st.error(
#             "The entered time does not exist in the "
#             "uploaded DCS data."
#         )

#         st.caption(
#             f"Entered: "
#             f"{selected_ts.strftime('%d/%m/%Y %I:%M:%S %p')}"
#         )

#         st.caption(
#             "Please enter a time available in the "
#             "15-minute DCS data."
#         )

#         st.stop()


# selected_row = selected_rows.iloc[0]


# # ============================================================
# # SELECTED PROCESS CONDITION
# # ============================================================

# st.markdown("### Selected Process Condition")

# m1, m2, m3, m4 = st.columns(4)

# m1.metric(
#     "Selected Timestamp",
#     selected_ts.strftime(
#         "%d/%m/%Y %I:%M:%S %p"
#     ),
# )

# m2.metric(
#     "TPD",
#     (
#         f"{selected_row['TPD']:.2f}"
#         if pd.notna(selected_row["TPD"])
#         else "N/A"
#     ),
# )

# m3.metric(
#     "RT Status",
#     str(
#         selected_row["RT_STATUS"]
#     ),
# )

# m4.metric(
#     "Total Residence Time",
#     (
#         f"{selected_row['TOTAL_RT_HR']:.2f} h"
#         if pd.notna(
#             selected_row["TOTAL_RT_HR"]
#         )
#         else "N/A"
#     ),
# )


# # ============================================================
# # SHUTDOWN / INVALID STATUS
# # ============================================================

# if not bool(
#     selected_row["RT_VALID"]
# ):

#     st.error(
#         "Prediction is unavailable for this timestamp."
#     )

#     if (
#         pd.notna(selected_row["TPD"])
#         and selected_row["TPD"] <= 0
#     ):

#         st.warning(
#             "TPD ≤ 0 indicates shutdown/non-production."
#         )

#     else:

#         st.warning(
#             "The selected timestamp does not satisfy "
#             "the RT-valid production criteria."
#         )

#     st.stop()


# # ============================================================
# # PREDICT BUTTON
# # ============================================================

# st.divider()

# predict_clicked = st.button(
#     "🎨  PREDICT COLOUR",
#     type="primary",
#     use_container_width=True,
# )


# if predict_clicked:

#     try:

#         with st.spinner(
#             "Calculating residence time and extracting "
#             "the causal DCS window..."
#         ):

#             feature_row, trajectory, previous_ts = (
#                 build_prediction_feature_row(
#                     dcs,
#                     runs,
#                     selected_ts,
#                 )
#             )

#             prediction, failures = run_prediction(
#                 feature_row,
#                 models,
#                 saved_features,
#             )


#         # ====================================================
#         # FEATURE FAILURE
#         # ====================================================

#         if failures:

#             st.error(
#                 "Prediction stopped because the saved "
#                 "Baseline C feature definition could not "
#                 "be reproduced from the supplied DCS data."
#             )

#             for target, cols in failures.items():

#                 st.write(
#                     f"**LAB_{target}: "
#                     f"{len(cols)} missing feature(s)**"
#                 )

#                 st.code(
#                     "\n".join(cols),
#                     language="text",
#                 )

#             st.info(
#                 "The model feature definition must match "
#                 "the feature engineering used during training."
#             )

#             st.stop()


#         # ====================================================
#         # SUCCESS
#         # ====================================================

#         st.success(
#             "Prediction completed successfully."
#         )


#         # ====================================================
#         # PREDICTED COLOUR
#         # ====================================================

#         st.subheader(
#             "3. Predicted Colour"
#         )

#         p1, p2, p3 = st.columns(3)

#         p1.metric(
#             "L",
#             f"{prediction['PRED_L']:.3f}",
#         )

#         p2.metric(
#             "a",
#             f"{prediction['PRED_A']:.3f}",
#         )

#         p3.metric(
#             "b",
#             f"{prediction['PRED_B']:.3f}",
#         )


#         # ====================================================
#         # CAUSAL TRAJECTORY
#         # ====================================================

#         st.subheader(
#             "4. Causal Residence-Time Window"
#         )

#         trajectory_table = pd.DataFrame(
#             {
#                 "Point": [
#                     "Prediction Time",
#                     "DRR Boundary",
#                     "PP2 Boundary",
#                     "PP1 Boundary",
#                     "EST2 Boundary",
#                     "EST1 Upstream Boundary",
#                 ],

#                 "Timestamp": [
#                     selected_ts,
#                     trajectory[
#                         "DRR_BOUNDARY_TS"
#                     ],
#                     trajectory[
#                         "PP2_BOUNDARY_TS"
#                     ],
#                     trajectory[
#                         "PP1_BOUNDARY_TS"
#                     ],
#                     trajectory[
#                         "EST2_BOUNDARY_TS"
#                     ],
#                     trajectory[
#                         "EST1_BOUNDARY_TS"
#                     ],
#                 ],
#             }
#         )

#         trajectory_table[
#             "Timestamp"
#         ] = trajectory_table[
#             "Timestamp"
#         ].dt.strftime(
#             "%d/%m/%Y %I:%M:%S %p"
#         )

#         st.dataframe(
#             trajectory_table,
#             use_container_width=True,
#             hide_index=True,
#         )


#         # ====================================================
#         # SECTION RESIDENCE TIMES
#         # ====================================================

#         r1, r2, r3, r4, r5 = st.columns(5)

#         r1.metric(
#             "EST1 RT",
#             f"{trajectory['EST1_RT_HR']:.2f} h",
#         )

#         r2.metric(
#             "EST2 RT",
#             f"{trajectory['EST2_RT_HR']:.2f} h",
#         )

#         r3.metric(
#             "PP1 RT",
#             f"{trajectory['PP1_RT_HR']:.2f} h",
#         )

#         r4.metric(
#             "PP2 RT",
#             f"{trajectory['PP2_RT_HR']:.2f} h",
#         )

#         r5.metric(
#             "DRR RT",
#             f"{trajectory['DRR_RT_HR']:.2f} h",
#         )


#         # ====================================================
#         # DEPLOYMENT DYNAMIC REFERENCE
#         # ====================================================

#         st.subheader(
#             "5. Process Dynamic Reference"
#         )

#         if previous_ts is not None:

#             d1, d2 = st.columns(2)

#             d1.metric(
#                 "Current Prediction Time",
#                 selected_ts.strftime(
#                     "%d/%m/%Y %I:%M:%S %p"
#                 ),
#             )

#             d2.metric(
#                 "Previous Valid DCS State",
#                 pd.Timestamp(
#                     previous_ts
#                 ).strftime(
#                     "%d/%m/%Y %I:%M:%S %p"
#                 ),
#             )

#         else:

#             st.warning(
#                 "No previous valid DCS state was found "
#                 f"within {DYNAMIC_LOOKBACK_HOURS:.1f} hours."
#             )


#         # # ====================================================
#         # # PROCESS STATE
#         # # ====================================================

#         # st.subheader(
#         #     "6. Process State Used"
#         # )

#         # state_cols = [
#         #     "CTRL_TPD_MEAN",
#         #     "CTRL_TPD_MIN",
#         #     "CTRL_TPD_MAX",
#         #     "CTRL_TPD_LAST",
#         #     "C_TPD_CURRENT",
#         #     "C_TPD_30M_MEAN",
#         #     "C_TPD_60M_MEAN",
#         #     "C_TPD_120M_MEAN",
#         #     "C_TPD_240M_MEAN",
#         # ]

#         # state = {}

#         # for col in state_cols:

#         #     if col in feature_row.columns:

#         #         value = feature_row.iloc[0][col]

#         #         state[col] = value


#         # if state:

#         #     state_df = pd.DataFrame(
#         #         [
#         #             {
#         #                 "Feature": key,
#         #                 "Value": value,
#         #             }

#         #             for key, value
#         #             in state.items()
#         #         ]
#         #     )

#         #     st.dataframe(
#         #         state_df,
#         #         use_container_width=True,
#         #         hide_index=True,
#         #     )


#         # ====================================================
#         # FEATURE AUDIT
#         # ====================================================

#         # with st.expander(
#         #     "Show model feature audit"
#         # ):

#         #     for target in TARGETS:

#         #         key = f"LAB_{target}"

#         #         selected = (
#         #             saved_features[key]
#         #         )

#         #         present = [
#         #             x
#         #             for x in selected
#         #             if x in feature_row.columns
#         #         ]

#         #         st.write(
#         #             f"**{key}: "
#         #             f"{len(present)}/{len(selected)} "
#         #             f"features reproduced**"
#         #         )


#         # ====================================================
#         # DOWNLOAD RESULT
#         # ====================================================

#         export = pd.DataFrame(
#             [
#                 {
#                     "TIMESTAMP": selected_ts,
#                     "TPD": selected_row["TPD"],
#                     "TOTAL_RT_HR": selected_row[
#                         "TOTAL_RT_HR"
#                     ],
#                     "PRED_L": prediction[
#                         "PRED_L"
#                     ],
#                     "PRED_A": prediction[
#                         "PRED_A"
#                     ],
#                     "PRED_B": prediction[
#                         "PRED_B"
#                     ],
#                 }
#             ]
#         )

#         st.download_button(
#             "Download Prediction CSV",
#             export.to_csv(index=False),
#             file_name=(
#                 "POLY2_prediction_"
#                 f"{selected_ts.strftime('%Y%m%d_%H%M')}"
#                 ".csv"
#             ),
#             mime="text/csv",
#         )


#     except Exception as exc:

#         st.error(
#             "Prediction failed."
#         )

#         st.exception(exc)


# # ============================================================
# # FOOTER
# # ============================================================

# st.divider()

# st.caption(
#     "POLY-II Baseline C | Raw 15-minute DCS | "
#     "Causal residence-time inference | "
#     "202-F0 excluded | No previous Lab colour values used"
# )
