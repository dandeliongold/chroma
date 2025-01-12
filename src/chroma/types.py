"""Type definitions and custom exceptions."""

class ErrorType:
    """Standard error types for consistent messaging"""
    NOT_FOUND = "Not found"
    ALREADY_EXISTS = "Already exists" 
    INVALID_INPUT = "Invalid input"
    FILTER_ERROR = "Filter error"
    OPERATION_ERROR = "Operation failed"

class DocumentOperationError(Exception):
    """Custom error for document operations"""
    def __init__(self, error: str):
        self.error = error
        super().__init__(self.error)
