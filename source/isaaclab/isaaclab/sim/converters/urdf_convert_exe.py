from isaaclab.app import AppLauncher

# ✅ 必须在导入任何 omni/carb 之前启动 Isaac Sim 运行时
simulation_app = AppLauncher(headless=True).app

# 你的转换配置和调用
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.sim.converters.urdf_converter import UrdfConverter

cfg = UrdfConverterCfg(
    asset_path="/home/yan/Documents/g1_urdf/g1_29dof.urdf",
    usd_dir="/home/yan/Documents/usd_files",
    usd_file_name="g1.usd",
    merge_fixed_joints=True,
    fix_base=False,
    convert_mimic_joints_to_normal_joints=True,
    replace_cylinders_with_capsules=True,  # 简化几何
    root_link_name="base_link",
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=1.0,
        ),
        target_type="position",
    ),
)

converter = UrdfConverter(cfg)
print("✅ USD generated at:", converter.usd_path)
