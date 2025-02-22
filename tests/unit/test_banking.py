import pytest
from utils.banking_utils import BankingQueryParser

def test_banking_query_classification():
    test_cases = [
        ("How do I block my credit card?", "CARD_SERVICES"),
        ("What's my account balance?", "ACCOUNT_INQUIRY"),
        ("I noticed suspicious activity", "SECURITY"),
    ]
    
    for query, expected_type in test_cases:
        result = BankingQueryParser.classify_query(query)
        assert result.value == expected_type

def test_invalid_query():
    with pytest.raises(ValueError):
        BankingQueryParser.classify_query("") 