from packaging import version

def check_version(current_version: str, required_version: str) -> bool:
    """
    Compare the current version with the required version.

    :param current_version: The version string of the installed package.
    :param required_version: The minimum required version string.
    :return: True if the current_version is greater than or equal to the required_version, else False.
    """
    return version.parse(current_version) >= version.parse(required_version)
