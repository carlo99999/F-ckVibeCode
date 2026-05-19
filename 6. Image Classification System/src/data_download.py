import kagglehub


if __name__ == "__main__":
    path = kagglehub.dataset_download(
        "paultimothymooney/chest-xray-pneumonia", output_dir="data"
    )
