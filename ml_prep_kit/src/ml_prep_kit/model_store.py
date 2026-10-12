import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.artifacts

class ModelStore:
    """Exporta modelos do MLflow para uso na API"""

    def __init__(
        self,
        model_name: str,
        tracking_uri: str | None = None,
        
    ) -> None:
        """Inicializa o store.

        Exemplo:
            store = ModelStore(
                model_name="my_model",
                tracking_uri="sqlite:///mlflow.db"
            )
        """
        self.tracking_uri = tracking_uri
        self.model_name = model_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

    @classmethod
    def from_directory(cls, model_name: str, directory: str):
        return cls(
            model_name=model_name,
            tracking_uri=f"sqlite:///{directory}/mlflow.db"
        )

    def export(self, dst_path):
        registered_models = self.client.search_registered_models(
            max_results=1,
            filter_string=f"name='{self.model_name}'"
        )

        model_version = registered_models[0].latest_versions[0]

        mlflow.artifacts.download_artifacts(
            artifact_uri=model_version.source,
            dst_path=dst_path
        )

        with open(Path(dst_path).joinpath("version.txt"), "w", encoding="utf-8") as f:
            f.write(f"{model_version.version}\n")
