from marketml.data.loader.price_loader import load_price_data
from marketml.data.loader.finance_loader import load_financial_data

if __name__ == "__main__":
    print("✅ PRICE DATA")
    print(load_price_data(nrows=5))

    print("\n✅ FINANCIAL DATA")
    print(load_financial_data(nrows=5))
