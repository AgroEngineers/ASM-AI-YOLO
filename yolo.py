from pathlib import Path
from typing import Union

import torch
from ultralytics import YOLO

import numpy
from asm.api.ai import ASMAI, AIResult, AIExpansion
from asm.api.base import ModuleTask, ModuleTaskInput, ModuleTaskOutput, ModuleConfiguration, ModuleInformation, \
    ContainerParameterResults, ModuleRequirement


class YOLOai(ASMAI):
    model = None
    current_labels = None
    name = ""

    def expansions(self) -> AIExpansion:
        return AIExpansion(["pt", "onnx"])

    def available_labels(self) -> list[str]:
        return self.current_labels

    def process(self, frame: numpy.ndarray) -> tuple[
        Union[AIResult, None], Union[list[ContainerParameterResults], None]]:
        if self.model is None:
            raise ModuleNotFoundError()

        results = self.model.predict(source=frame, conf=0.25, verbose=False)

        label = self.current_labels[int(results[0].boxes.cls[0].item())]

        return AIResult(self.name, label), None

    def load(self, model_path: Path, labels_path: Union[Path, None]) -> bool:
        self.name = model_path.name
        try:
            self.model = YOLO(str(model_path))
            self.current_labels = list(self.model.names.values())

            return True
        except Exception as e:
            return False

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
            self.current_labels = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


    def module_info(self) -> ModuleInformation:
        return ModuleInformation(
            name="YOLO",
            version="1.0",
            requirements=[
                ModuleRequirement("torch"), ModuleRequirement("ultralytics")
            ]
        )

    def configuration(self, configuration: ModuleConfiguration):
        pass

    def task(self, task: ModuleTask, task_input: ModuleTaskInput) -> Union[ModuleTaskOutput, None]:
        pass