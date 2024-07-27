import numpy as np
import numpy.typing as npt


def aggregate_organs(
    tissues: np.ndarray,
    index_of_aggregate_organ: int,
    indices_of_component_organs: npt.NDArray[np.int_],
    indices_of_organ_properties: npt.NDArray[np.int_]
):
    """
    Aggregates organ masses/flows to form combined compartments.
    Parameters:
    tissues: numpy array
    index_of_aggregate_organ: int
    indices_of_component_organs: list of int
    indices_of_organ_properties: list of int
    Returns:
    None
    """
    comp_orgs = tissues[:, indices_of_component_organs]
    org_props = comp_orgs[:, :, indices_of_organ_properties]
    sum = np.sum(org_props, axis=1)
    tissues[:, index_of_aggregate_organ, indices_of_organ_properties] = sum
