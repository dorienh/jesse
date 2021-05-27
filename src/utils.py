class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def custom_print(message, underline=False, header=False):
    if header:
        print(bcolors.HEADER + message + bcolors.ENDC)
    elif underline:
        print(
            bcolors.UNDERLINE + bcolors.WARNING + message + bcolors.ENDC + bcolors.ENDC
        )
    else:
        print(bcolors.BOLD + bcolors.OKBLUE + message + bcolors.ENDC + bcolors.ENDC)
