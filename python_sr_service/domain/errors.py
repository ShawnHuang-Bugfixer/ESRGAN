from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    SCHEMA_INVALID = 'SCHEMA_INVALID'
    TYPE_NOT_SUPPORTED = 'TYPE_NOT_SUPPORTED'
    INPUT_NOT_FOUND = 'INPUT_NOT_FOUND'
    COS_DOWNLOAD_FAILED = 'COS_DOWNLOAD_FAILED'
    MODEL_NOT_FOUND = 'MODEL_NOT_FOUND'
    INFER_RUNTIME_ERROR = 'INFER_RUNTIME_ERROR'
    GPU_OOM = 'GPU_OOM'
    COS_UPLOAD_FAILED = 'COS_UPLOAD_FAILED'
    MYSQL_WRITE_FAILED = 'MYSQL_WRITE_FAILED'
    INTERNAL_ERROR = 'INTERNAL_ERROR'


class ServiceError(Exception):
    def __init__(self, code: ErrorCode, message: str, retryable: bool = False, cause: Optional[Exception] = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.cause = cause
