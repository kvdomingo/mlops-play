from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    RANDOM_SEED: int = 709

    IMAGE_TARGET_SIZE: tuple[int, int, int] = (256, 256, 3)
    BATCH_SIZE: int = 20
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 10

    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"

    @computed_field
    @property
    def TRAIN_DIR(self) -> Path:
        return self.DATA_DIR / "train"

    @computed_field
    @property
    def TEST_DIR(self) -> Path:
        return self.DATA_DIR / "test"


settings = Settings()
