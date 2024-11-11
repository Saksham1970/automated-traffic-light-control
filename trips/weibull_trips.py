import xml.etree.ElementTree as ETree
import xml.dom.minidom as minidom
import numpy as np


def generate_weibull_times(
    shape: float, number_cars: int, total_timesteps: int
) -> np.array:
    """
    Generates an array of times based on a Weibull distribution.

    Args:
        shape (float): Shape parameter for the Weibull distribution.
        number_cars (int): The number of cars/trips to generate.
        total_timesteps (int): The total number of timesteps over which the trips will be spread.

    Returns:
        numpy.array: Sorted array of times at which the trips will depart.
    """
    times = np.random.weibull(a=shape, size=number_cars)
    times = times / (max(times) + 0.1) * total_timesteps
    times = times.astype(int)
    times.sort()
    return times


def generate_root() -> ETree.Element:
    """
    Generates the root XML element for the trips file with basic route definitions.

    Returns:
        xml.etree.ElementTree.Element: The root element of the XML structure.
    """
    root = ETree.Element(
        "routes",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    emergency = ETree.Element(
        "vType",
        attrib={
            "id": "emergency",
            "vClass": "container",
            "guiShape": "passenger/van",
            "color": "1,0,0",
        },
    )
    default = ETree.Element(
        "vType",
        attrib={
            "id": "default",
            "vClass": "container",
            "guiShape": "passenger/sedan",
            "color": "0,1,0",
        },
    )
    root.append(emergency)
    root.append(default)
    return root


def choose_src_dst(src_prob: float, turn_prob: float) -> tuple[int, int]:
    """
    Chooses a source and destination node based on the provided probabilities.

    Args:
        src_prob (float): A list of probabilities for selecting the source node.
        turn_prob (float): A list of probabilities for selecting the turn (which influences the destination).

    Returns:
        tuple[int, int]: Indices of the selected source and destination nodes.
    """
    src_ind = np.random.choice(range(len(src_prob)), p=src_prob)
    turn_choice = np.random.choice(range(len(turn_prob)), p=turn_prob)
    dst_ind = (src_ind + turn_choice) % len(src_prob)
    return src_ind, dst_ind


def choose_emergency(emergency_prob: float) -> str:
    """
    Decides whether a trip should be an emergency vehicle based on probability.

    Args:
        emergency_prob (float): The probability that the vehicle will be classified as 'emergency'.

    Returns:
        str: Returns 'emergency' if chosen, otherwise 'default'.
    """
    emergency_chance = np.random.rand()
    return "emergency" if emergency_chance < emergency_prob else "default"


def save_xml(root: ETree.Element, filename: str) -> None:
    """
    Saves the XML tree to a file with proper formatting.

    Args:
        root (ETree.Element): The root element of the XML structure.
        filename (str): The name of the file to save the XML content.
    """
    xml_string = ETree.tostring(root, encoding="utf-8", method="xml")
    pretty_xml_string = minidom.parseString(xml_string).toprettyxml(indent="  ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml_string)


def main(
    src_nodes,
    dst_nodes,
    src_prob,
    turn_prob,
    emergency_probs,
    shape,
    number_cars,
    total_timesteps,
):
    """
    Main function to generate trips for a traffic simulation.

    Args:
        src_nodes (list): List of source nodes. Keep the order as SWNE
        dst_nodes (list): List of destination nodes. Keep the order as SWNE
        src_prob (list): List of probabilities for selecting a source node. Keep the order as SWNE
        turn_prob (list): List of probabilities for determining a turn. Keep the order as U turn, left, straight, right
        emergency_probs (list): List of Probability of selecting an emergency vehicle.
        shape (float): Shape parameter for the Weibull distribution (trip times).
        number_cars (int): Number of vehicles/trips to generate.
        total_timesteps (int): Total simulation timesteps over which trips will depart.
    """
    timesteps = generate_weibull_times(shape, number_cars, total_timesteps)
    root = generate_root()
    for ind, time in enumerate(timesteps):
        src_ind, dst_ind = choose_src_dst(src_prob, turn_prob)
        typ = choose_emergency(emergency_probs[src_ind])
        trip = ETree.Element(
            "trip",
            attrib={
                "id": str(ind),
                "depart": f"{time}.00",
                "from": src_nodes[src_ind],
                "to": dst_nodes[dst_ind],
                "type": typ,
            },
        )
        root.append(trip)
    save_xml(root, "trips.trips.xml")


if __name__ == "__main__":
    # Import configuration variables from an external config file
    from trips_generator.config import *

    main(
        src_nodes,
        dst_nodes,
        src_prob,
        turn_prob,
        emergency_probs,
        shape,
        number_cars,
        total_timesteps,
    )
