"""
generate_samples.py
--------------------
Generate realistic industrial sample datasets for testing.
Run:  python generate_samples.py

All text files are written with explicit encoding="utf-8" so this
works correctly on Windows (which defaults to cp1252 and would
otherwise crash on non-ASCII characters).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "sample_data"
OUT.mkdir(exist_ok=True)

N = 500  # samples per file

# -- 1. Pump Vibration CSV ---------------------------------------------------

t         = np.linspace(0, 50, N)
vibration = 2.1 + 0.4 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.15, N)
vibration[350:] += rng.exponential(1.2, N - 350)

pump_df = pd.DataFrame({
    "timestamp":        pd.date_range("2024-01-01", periods=N, freq="6min"),
    "vibration_x_mm_s": vibration,
    "vibration_y_mm_s": 1.8 + 0.3 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.12, N),
    "temperature_C":    60 + 5 * np.sin(t / 10) + rng.normal(0, 1.5, N),
    "pressure_bar":     4.2 + rng.normal(0, 0.08, N),
    "flow_rate_L_min":  120 - 0.05 * t + rng.normal(0, 2, N),
    "rpm":              1450 + rng.normal(0, 10, N),
    "current_A":        14.5 + 0.3 * rng.random(N),
    "machine_id":       "PUMP-A01",
})
pump_df.to_csv(OUT / "pump_vibration_fault.csv", index=False, encoding="utf-8")
print("Saved: pump_vibration_fault.csv")

# -- 2. Compressor Thermal Excel ---------------------------------------------

comp_df = pd.DataFrame({
    "sample":             range(1, N + 1),
    "inlet_temp_C":       35 + rng.normal(0, 2, N),
    "outlet_temp_C":      np.concatenate([
        160 + rng.normal(0, 3, 300),
        195 + np.linspace(0, 30, 100) + rng.normal(0, 4, 100),
        210 + rng.normal(0, 5, 100),
    ]),
    "oil_pressure_bar":   np.concatenate([
        4.5 + rng.normal(0, 0.1, 300),
        4.5 - np.linspace(0, 2, 200) + rng.normal(0, 0.1, 200),
    ]),
    "oil_temp_C":         85 + 0.5 * rng.random(N),
    "power_kW":           45 + rng.normal(0, 1.5, N),
    "discharge_pressure": 8.5 + rng.normal(0, 0.2, N),
    "cooling_water_flow": 12 - 0.01 * np.arange(N) + rng.normal(0, 0.2, N),
    "machine_id":         "COMP-B02",
    "shift":              ["Morning" if i % 3 == 0 else "Afternoon" if i % 3 == 1 else "Night"
                           for i in range(N)],
})
comp_df.to_excel(OUT / "compressor_thermal_fault.xlsx", index=False, engine="openpyxl")
print("Saved: compressor_thermal_fault.xlsx")

# -- 3. CNC Machine Error Log TXT --------------------------------------------
# NOTE: plain ASCII only (no unicode arrows / symbols) so this is
# guaranteed to write correctly on any OS / locale.

log_lines = [
    "=== CNC MACHINING CENTER MC-500 ERROR LOG ===",
    "Asset ID: CNC-MILL-003",
    "Log Period: 2024-01-15 06:00 to 2024-01-15 18:00",
    "",
    "[06:12:34] INFO  Spindle start OK. Speed: 8000 RPM",
    "[06:45:01] INFO  Tool change: Position 4 -> Position 7",
    "[07:23:55] WARN  Spindle load exceeded threshold: 92% (limit 85%)",
    "[07:24:10] WARN  Spindle load: 94%",
    "[07:25:33] ERROR Spindle motor overcurrent fault. Current: 28.4A (limit 25A). E-Code: E0031",
    "[07:25:33] INFO  Auto-stop triggered. Spindle halted.",
    "[07:30:12] INFO  Fault cleared by operator. Resume production.",
    "[08:14:20] WARN  X-axis servo following error: 0.045mm (limit 0.030mm)",
    "[08:15:45] WARN  X-axis servo following error: 0.051mm - degrading",
    "[09:00:00] INFO  Scheduled lubrication cycle started",
    "[09:00:45] WARN  Lube pressure low: 1.8bar (nominal 2.5bar)",
    "[09:01:00] ERROR Lubrication system failure. Pump no-flow detected. E-Code: L0012",
    "[09:01:00] ALARM Production halted - safety interlock active",
    "[10:30:00] INFO  Maintenance team notified. Lube pump inspection started.",
    "[11:15:22] INFO  Lube pump motor replaced. System primed.",
    "[11:20:00] INFO  Production resumed.",
    "[12:00:00] INFO  Mid-shift inspection: All systems nominal",
    "[13:45:10] WARN  Coolant temperature high: 42C (limit 38C)",
    "[13:46:00] WARN  Coolant temperature: 45C - rising",
    "[13:50:00] ERROR Coolant system overheat. Chiller fault. E-Code: C0021",
    "[13:50:00] INFO  Reduced feed rate to 60% to manage heat",
    "[14:30:00] INFO  Chiller refrigerant topped up. Temperature normal: 36C",
    "[15:00:00] INFO  Spindle vibration check: 1.8 mm/s (limit 4.5 mm/s) - OK",
    "[16:55:00] WARN  Tool wear index: 87% (replace at 100%)",
    "[17:50:00] INFO  End of shift. Parts produced: 143. Rejects: 4 (2.8%)",
    "",
    "=== FAULT SUMMARY ===",
    "Total errors  : 4",
    "Total warnings: 8",
    "Downtime      : 3h 45min",
    "OEE           : 68.7%",
]
(OUT / "cnc_error_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
print("Saved: cnc_error_log.txt")

# -- 4. Motor Bearing JSON ----------------------------------------------------

motor_records = []
for i in range(200):
    amp = 0.8 + (0.015 * i if i > 100 else 0)
    rec = {
        "sample_id":             i + 1,
        "timestamp":             f"2024-02-10T{6 + i//120:02d}:{(i*30)%60:02d}:00",
        "bearing_temperature_C": 72 + amp * 3 + float(rng.normal(0, 0.8)),
        "rms_velocity_mm_s":     1.2 + amp + float(rng.normal(0, 0.1)),
        "peak_velocity_mm_s":    3.5 + amp * 2.5 + float(rng.normal(0, 0.2)),
        "crest_factor":          3.1 + amp * 0.8 + float(rng.normal(0, 0.15)),
        "kurtosis":              3.0 + amp * 1.5 + float(rng.normal(0, 0.3)),
        "bpfi_amplitude":        float(rng.exponential(amp + 0.1)),
        "bpfo_amplitude":        float(rng.exponential(0.08)),
        "motor_current_A":       18.5 + amp * 0.4 + float(rng.normal(0, 0.2)),
        "motor_speed_rpm":       2960 - amp * 5 + float(rng.normal(0, 8)),
        "asset_id":              "MOTOR-D04",
        "condition":             "Normal" if i < 100 else "Degrading" if i < 160 else "Critical",
    }
    motor_records.append(rec)

with open(OUT / "motor_bearing_data.json", "w", encoding="utf-8") as f:
    json.dump(motor_records, f, indent=2)
print("Saved: motor_bearing_data.json")

# -- 5. Multi-machine CSV -----------------------------------------------------

machines = ["LINE-1-PRESS", "LINE-2-ROBOT", "LINE-3-CONVEYOR", "LINE-4-WELD"]
records  = []
for i in range(N):
    for mach in machines:
        base_load = rng.uniform(40, 80)
        records.append({
            "timestamp":      pd.Timestamp("2024-03-01") + pd.Timedelta(minutes=5 * i),
            "machine_id":     mach,
            "load_pct":       base_load + rng.normal(0, 3),
            "temperature_C":  55 + rng.normal(0, 5),
            "vibration_mm_s": 1.5 + rng.exponential(0.3) if rng.random() > 0.95 else 1.5 + rng.normal(0, 0.2),
            "cycle_time_s":   12.5 + rng.normal(0, 0.5),
            "error_count":    int(rng.poisson(0.05)),
            "energy_kWh":     base_load * 0.05 + rng.normal(0, 0.02),
        })

multi_df = pd.DataFrame(records)
multi_df.to_csv(OUT / "multi_machine_production.csv", index=False, encoding="utf-8")
print("Saved: multi_machine_production.csv")

print()
print(f"All sample files saved to: {OUT}")
