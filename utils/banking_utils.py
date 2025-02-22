from enum import Enum
from typing import Dict, Any, Optional
import re

class BankingQueryType(Enum):
    CARD_OPERATIONS = "card"
    ACCOUNT_INFO = "account"
    TRANSACTION = "transaction"
    SECURITY = "security"
    GENERAL = "general"

class BankingQueryParser:
    # Keywords for query classification
    KEYWORDS = {
        BankingQueryType.CARD_OPERATIONS: [
            "card", "block", "unblock", "freeze", "activate", "deactivate", "pin"
        ],
        BankingQueryType.ACCOUNT_INFO: [
            "balance", "statement", "account", "details", "information"
        ],
        BankingQueryType.TRANSACTION: [
            "transfer", "payment", "send money", "receive", "transaction"
        ],
        BankingQueryType.SECURITY: [
            "fraud", "suspicious", "unauthorized", "security", "password"
        ]
    }

    @staticmethod
    def classify_query(query: str) -> BankingQueryType:
        """Classify banking query type based on keywords"""
        query = query.lower()
        
        for query_type, keywords in BankingQueryParser.KEYWORDS.items():
            if any(keyword in query for keyword in keywords):
                return query_type
        return BankingQueryType.GENERAL

    @staticmethod
    def validate_query(query: str) -> bool:
        """Validate if query contains sensitive information"""
        sensitive_patterns = [
            r'\b\d{16}\b',  # Card numbers
            r'\b\d{3,4}\b',  # CVV
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Card format
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        return not any(re.search(pattern, query) for pattern in sensitive_patterns)

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