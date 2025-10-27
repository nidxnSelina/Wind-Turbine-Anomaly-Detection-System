import os
import sys
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

def init_logger(logger):
    """
    Initialize logging
    """
    logger.remove()

    logger.add(
        sink=sys.stderr,
        format="[<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g>] <le>[{file.path}]</le>\n[<lvl>{level}</lvl>:{function}:{line}] <b><y>{message}</y></b>",
        level='INFO',
    )

column_mapping = {"日期": "time", "瞬时风速[m/s]": "wind_speed", "发电机转速[rpm]": "generator_speed", "有功功率[kW]": "power", "风向[°]": "wind_direction", "风向平均值[°]": "wind_direction_mean", "偏航位置[°]": "yaw_position", "偏航速度[°/s]": "yaw_speed", "叶片1角度[°]": "pitch1_angle", "叶片2角度[°]": "pitch2_angle", "叶片3角度[°]": "pitch3_angle", "叶片1速度[°/s]": "pitch1_speed", "叶片2速度[°/s]": "pitch2_speed", "叶片3速度[°/s]": "pitch3_speed", "电机U项绕组温度[℃]": "pitch1_moto_tmp", "电机V项绕组温度[℃]": "pitch2_moto_tmp", "电机W项绕组温度[℃]": "pitch3_moto_tmp", "加速度X[m/s²]": "acc_x", "加速度Y[m/s²]": "acc_y", "环境温度[℃]": "environment_tmp", "机舱温度[℃]": "int_tmp", "叶片1NG5温度[℃]": "pitch1_ng5_tmp", "叶片2NG5温度[℃]": "pitch2_ng5_tmp", "叶片3NG5温度[℃]": "pitch3_ng5_tmp", "叶片1NG5直流电压[V]": "pitch1_ng5_DC", "叶片2NG5直流电压[V]": "pitch2_ng5_DC", "叶片3NG5直流电压[V]": "pitch3_ng5_DC", "分组": "group", "序号": "Serial Number", "PLC状态": "PLC Status", "PLC子状态": "PLC Sub-Status", "变频器状态": "Converter Status", "安全链状态": "Safety Chain Status", "叶片1状态": "Blade 1 Status", "叶片2状态": "Blade 2 Status", "叶片3状态": "Blade 3 Status", "电池状态": "Battery Status", "轮毂状态": "Hub Status", "偏航状态": "Yaw Status", "登陆状态": "Login Status", "轮毂指令状态": "Hub Command Status", "平均风速[m/s]": "Average Wind Speed [m/s]", "风湍流度": "Wind Turbulence Intensity", "齿轮箱油温[℃]": "Gearbox Oil Temperature [℃]", "齿轮箱加热": "Gearbox Heating", "叶片1故障": "Blade 1 Fault", "叶片2故障": "Blade 2 Fault", "叶片3故障": "Blade 3 Fault", "无功功率[kVar]": "Reactive Power [kVar]", "故障1": "Fault 1", "故障2": "Fault 2", "故障3": "Fault 3", "故障4": "Fault 4", "故障5": "Fault 5", "故障6": "Fault 6", "故障7": "Fault 7", "故障8": "Fault 8", "机舱位置[°]": "Nacelle Position [°]", "叶轮转速[rpm]": "Rotor Speed [rpm]", "直流母线电压[V]": "DC Bus Voltage [V]", "电网电压1[V]": "Grid Voltage 1 [V]", "电网电压2[V]": "Grid Voltage 2 [V]", "电网电压3[V]": "Grid Voltage 3 [V]", "功率比例": "Power Ratio", "400V消耗功率[kW]": "400 V Power Consumption [kW]", "偏航故障": "Yaw Fault", "变频器故障": "Converter Fault", "变频器频率[Hz]": "Converter Frequency [Hz]", "日发电量[kWh]": "Daily Generation [kWh]", "日风速平均值[m/s]": "Daily Average Wind Speed [m/s]", "齿轮箱轴承温度[℃]": "Gearbox Bearing Temperature [℃]", "总发电量[MWh]": "Total Generation [MWh]", "总发电时间[h]": "Total Operating Time [h]", "故障电流[A]": "Fault Current [A]", "主轴承温度[℃]": "Main Bearing Temperature [℃]", "机侧变频器IGBT温度[℃]": "Machine-Side Converter IGBT Temperature [℃]", "网侧变频器IGBT温度[℃]": "Grid-Side Converter IGBT Temperature [℃]", "发电机入水口温度[℃]": "Generator Inlet Water Temperature [℃]", "发电机出水口温度[℃]": "Generator Outlet Water Temperature [℃]", "叶片1扭矩[Nm]": "Blade 1 Torque [Nm]", "叶片2扭矩[Nm]": "Blade 2 Torque [Nm]", "叶片3扭矩[Nm]": "Blade 3 Torque [Nm]", "限功率标志": "Power Limitation Flag", "电池状态2": "Battery Status 2", "电池状态3": "Battery Status 3", "UPS电池状态1": "UPS Battery Status 1", "UPS电池状态2": "UPS Battery Status 2", "UPS电池状态3": "UPS Battery Status 3", "齿轮箱入口油压[Bar]": "Gearbox Oil Inlet Pressure [Bar]", "齿轮箱入口油温[℃]": "Gearbox Oil Inlet Temperature [℃]", "电网电流[A]": "Grid Current [A]", "齿轮箱轴承温度1[℃]": "Gearbox Bearing Temperature 1 [℃]", "齿轮箱轴承温度2[℃]": "Gearbox Bearing Temperature 2 [℃]", "变频器入水温度[℃]": "Converter Inlet Water Temperature [℃]", "变频器出水温度[℃]": "Converter Outlet Water Temperature [℃]", "变桨电机1温度[℃]": "Pitch Motor 1 Temperature [℃]", "变桨电机2温度[℃]": "Pitch Motor 2 Temperature [℃]", "变桨电机3温度[℃]": "Pitch Motor 3 Temperature [℃]", "塔基柜内温度[℃]": "Tower Base Cabinet Inside Temperature [℃]", "塔基柜外温度[℃]": "Tower Base Cabinet Outside Temperature [℃]", "偏航刹车压力[Bar]": "Yaw Brake Pressure [Bar]", "NCC300柜温度[℃]": "NCC300 Cabinet Temperature [℃]", "NCC310柜温度[℃]": "NCC310 Cabinet Temperature [℃]", "NCC320-1柜温度[℃]": "NCC320-1 Cabinet Temperature [℃]", "NCC320-2柜温度[℃]": "NCC320-2 Cabinet Temperature [℃]", "NCC320-3柜温度[℃]": "NCC320-3 Cabinet Temperature [℃]", "齿轮箱水油交换入水温度[℃]": "Gearbox Water-Oil Exchanger Inlet Water Temperature [℃]", "齿轮箱水油交换出水温度[℃]": "Gearbox Water-Oil Exchanger Outlet Water Temperature [℃]", "机侧IGBT2温度[℃]": "Machine-Side IGBT 2 Temperature [℃]", "机侧IGBT3温度[℃]": "Machine-Side IGBT 3 Temperature [℃]", "网侧IGBT2温度[℃]": "Grid-Side IGBT 2 Temperature [℃]", "网侧IGBT3温度[℃]": "Grid-Side IGBT 3 Temperature [℃]", "滤波板温度1[℃]": "Filter Board Temperature 1 [℃]", "滤波板温度2[℃]": "Filter Board Temperature 2 [℃]", "滤波板温度3[℃]": "Filter Board Temperature 3 [℃]", "变频器故障2": "Converter Fault 2", "变频器故障3": "Converter Fault 3"}

def load_xls(val_data_path: str, required_english_cols: list) -> pd.DataFrame:
    frames = []
    for file in os.listdir(val_data_path):
        if not (file.endswith(".xls") or file.endswith(".xlsx")):
            continue
        file_path = os.path.join(val_data_path, file)

        # Read full sheet (simplest, avoids usecols mismatch)
        df = pd.read_excel(file_path, engine="xlrd")  # for .xls; use openpyxl for .xlsx

        # Normalize odd punctuation/spaces just in case
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace("（", "(", regex=False).str.replace("）", ")", regex=False)
            .str.replace("【", "[", regex=False).str.replace("】", "]", regex=False)
        )

        # Chinese → English
        df.rename(columns=column_mapping, inplace=True)

        frames.append(df)

    if not frames:
        raise RuntimeError(f"No .xls/.xlsx files found under: {val_data_path}")

    data = pd.concat(frames, ignore_index=True)

    # Ensure all required columns exist; fill missing with NA
    for c in required_english_cols:
        if c not in data.columns:
            data[c] = pd.NA

    # Reorder to required list
    data = data[required_english_cols].copy()

    # Optional: sanitize column names (e.g., replace '/' with 'p')
    data.columns = [(col.replace('/', 'p') if isinstance(col, str) else col) for col in data.columns]

    return data
