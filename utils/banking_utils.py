from enum import Enum
from typing import Dict, Any, Optional
import re

class BankingQueryType(Enum):
    ACCOUNT_INQUIRY = "ACCOUNT_INQUIRY"
    CARD_SERVICES = "CARD_SERVICES"
    SECURITY = "SECURITY"
    GENERAL = "GENERAL"

class BankingQueryParser:
    @staticmethod
    def validate_query(query: str) -> bool:
        """
        Validate if query is properly formatted and contains valid content
        """
        if not query or not isinstance(query, str):
            return False
            
        # Check for minimum length (at least 3 words)
        if len(query.split()) < 3:
            return False
            
        # Check for maximum length
        if len(query) > 500:
            return False
            
        return True

    @staticmethod
    def classify_query(query: str) -> BankingQueryType:
        """Classify banking queries into predefined types"""
        # Validate query first
        if not BankingQueryParser.validate_query(query):
            raise ValueError("Invalid query: Query must be a non-empty string with at least 3 words")
            
        query = query.lower()
        
        # Card related queries
        if any(word in query for word in ["card", "credit", "debit", "block", "unblock", "pin"]):
            return BankingQueryType.CARD_SERVICES
            
        # Account related queries
        if any(word in query for word in ["balance", "account", "statement", "transfer"]):
            return BankingQueryType.ACCOUNT_INQUIRY
            
        # Security related queries
        if any(word in query for word in ["fraud", "suspicious", "security", "unauthorized"]):
            return BankingQueryType.SECURITY
            
        # Default to general inquiry
        return BankingQueryType.GENERAL

class BankingErrorHandler:
    class BankingError(Exception):
        def __init__(self, message: str, error_type: str, query_type: BankingQueryType):
            self.message = message
            self.error_type = error_type
            self.query_type = query_type
            super().__init__(self.message)

    @staticmethod
    def handle_error(error: Exception, query_type: BankingQueryType) -> Dict[str, Any]:
        """Handle different types of banking errors"""
        if isinstance(error, BankingErrorHandler.BankingError):
            return {
                "status": "error",
                "error_type": error.error_type,
                "message": error.message,
                "query_type": query_type.value
            }
        
        return {
            "status": "error",
            "error_type": "general",
            "message": str(error),
            "query_type": query_type.value
        } 