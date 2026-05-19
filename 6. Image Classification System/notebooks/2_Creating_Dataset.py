import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    from root_folders import CHEST_XRAY_DIR

    from image_classification_system.images_processing import create_train_dataloader_pipeline

    train_path = CHEST_XRAY_DIR / 'train'
    test_path = CHEST_XRAY_DIR / 'test'


    dataloader, mean , std = create_train_dataloader_pipeline(train_path)

    test_dataloader = 


    """,
    name="_"
)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
