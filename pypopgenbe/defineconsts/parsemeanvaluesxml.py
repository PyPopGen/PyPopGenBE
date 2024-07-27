import xml.etree.ElementTree as ET
from typing import Dict, Any, cast
from pathlib import Path


def parse_mean_values_xml(file_path: Path) -> Dict[str, Any]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    mean_individual = {"Sex": []}
    for sex in root.findall('Sex'):
        sex_type = sex.get('type')
        sex_dict = {"type": sex_type, "tissue": []}

        for tissue in sex.findall('tissue'):
            tissue_type = tissue.get('type')
            tissue_dict = {"type": tissue_type, "param": []}

            for param in tissue.findall('param'):
                param_type = param.get('type')
                param_unit = param.get('unit')
                param_dict = {"type": param_type,
                              "unit": param_unit, "distribution": []}

                for distribution in param.findall('distribution'):
                    dist_type = distribution.get('type')
                    dist_unit = distribution.get('unit')
                    dist_content = float(cast(str, distribution.text))
                    dist_dict = {"type": dist_type,
                                 "unit": dist_unit, "content": dist_content}
                    param_dict["distribution"].append(dist_dict)

                tissue_dict["param"].append(param_dict)

            sex_dict["tissue"].append(tissue_dict)

        mean_individual["Sex"].append(sex_dict)

    return mean_individual
