from setuptools import setup

package_name = "rtt_path_plan"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ian Rist",
    maintainer_email="ian@bigair.net",
    description="RTT Path Planner forked from Maxim Yastremsky(@MaxMagazin)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rtt_plan = rtt_path_plan.MaRRTPathPlanNode:main",
        ],
    },
)
