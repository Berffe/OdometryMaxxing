"""ament_python packaging for the nested layout.

THE ONE THING THAT MUST CHANGE
------------------------------
A flat ``packages=['bee_control']`` will silently install ONLY the top level:
``bee_node.py`` ships, and every subpackage is missing at runtime. The failure
shows up as ``ModuleNotFoundError: bee_control.mission`` the first time the node
is launched from an installed workspace -- and NOT when running from source,
which is what makes it easy to miss.

``find_packages()`` fixes it, provided every folder has an ``__init__.py``
(they all do).
"""
from setuptools import find_packages, setup

package_name = "bee_control"

setup(
    name=package_name,
    version="2.0.0",
    # WAS: packages=[package_name]
    # NOW: picks up bee_control.core, .vision, .mission, .mission.phases,
    #      .control, .interfaces, .diagnostics
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bee",
    maintainer_email="bee@example.com",
    description="Bio-inspired visual landing controller for a moving platform.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Unchanged: bee_node.py deliberately stayed at the top level.
            "bee_node = bee_control.bee_node:main",
        ],
    },
)
