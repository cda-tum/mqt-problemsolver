import pytest

def main():
    # Run all tests in the current directory
    test_result = pytest.main()
    
    # Exit with the appropriate status code based on the test results
    if test_result == 0:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Check the output for details.")

if __name__ == "__main__":
    main()