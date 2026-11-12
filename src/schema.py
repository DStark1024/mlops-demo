import pandera as pa
from pandera.pandas import DataFrameModel, Field

class IrisSchema(DataFrameModel):
    id: int = Field(unique=True)
    sepal_length: float = Field(gt=0.0)
    sepal_width: float = Field(gt=0.0)
    petal_length: float = Field(gt=0.0)
    petal_width: float = Field(gt=0.0)
    species: str = Field(isin=["Iris-setosa", "Iris-virginica", "Iris-versicolor"])
    target: int = Field(isin=[0,1])

    class Config:
        coerce = True
        strict = True
