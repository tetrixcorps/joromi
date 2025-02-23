import pytest
from utils.banking_utils import BankingQueryParser, BankingQueryType

def test_banking_query_classification():
    test_cases = [
        ("How do I block my credit card?", BankingQueryType.CARD_SERVICES),
        ("What's my account balance now?", BankingQueryType.ACCOUNT_INQUIRY),
        ("I noticed some suspicious activity on my account", BankingQueryType.SECURITY),
        ("What are your working hours today?", BankingQueryType.GENERAL),
    ]
    
    for query, expected_type in test_cases:
        result = BankingQueryParser.classify_query(query)
        assert result == expected_type, f"Expected {expected_type} for query '{query}', got {result}"

def test_card_services_classification():
    card_queries = [
        "I need to block my card immediately",
        "How do I change my PIN number please?",
        "My credit card was stolen yesterday",
        "Can you help me activate my new debit card",
    ]
    
    for query in card_queries:
        result = BankingQueryParser.classify_query(query)
        assert result == BankingQueryType.CARD_SERVICES

def test_invalid_query():
    invalid_queries = [
        "",  # Empty string
        "hi",  # Too short
        "a b",  # Too few words
        None,  # None type
        "x" * 501,  # Too long
    ]
    
    for query in invalid_queries:
        with pytest.raises(ValueError) as exc_info:
            BankingQueryParser.classify_query(query)
        assert "Invalid query" in str(exc_info.value)

def test_query_validation():
    # Valid queries
    assert BankingQueryParser.validate_query("Please check my balance")
    assert BankingQueryParser.validate_query("I need help with my card")
    
    # Invalid queries
    assert not BankingQueryParser.validate_query("")
    assert not BankingQueryParser.validate_query("hi")
    assert not BankingQueryParser.validate_query("a b")
    assert not BankingQueryParser.validate_query("x" * 501) 