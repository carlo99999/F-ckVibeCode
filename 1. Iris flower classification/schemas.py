import pandera as pa
from pandera.typing import DataFrame
import pandera as pa
class IrisFeatures(pa.DataFrameModel):
    sepal_length: pa.Float
    sepal_width: pa.Float
    petal_length: pa.Float
    petal_width: pa.Float

class IrisTarget(pa.DataFrameModel):
    target: pa.Int

