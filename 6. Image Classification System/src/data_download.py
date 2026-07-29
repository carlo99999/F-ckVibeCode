import kagglehub


if __name__ == "__main__":
    path = kagglehub.dataset_download(
        "muratkokludataset/rice-image-dataset", output_dir="data"
    )
