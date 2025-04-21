import argparse


def get_args():
    parser = argparse.ArgumentParser(description="TBPS_EVA_CLIP Args")
    ######################## mode ########################
    parser.add_argument("--real_data", default=False, action='store_true')

    args = parser.parse_args()

    return args