from pydantic import BaseModel

class NLPRequest(BaseModel):
    text: str

class TaskResult(BaseModel):
    label: str
    confidence: float

class NLPResponse(BaseModel):
    emotion: TaskResult
    violence: TaskResult
    hate: TaskResult

class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool