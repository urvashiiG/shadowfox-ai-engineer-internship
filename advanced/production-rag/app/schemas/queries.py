from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value.strip()
