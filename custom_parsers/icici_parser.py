import pdfplumber
import pandas as pd

def parse(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                all_tables.append(table)

    combined_table = []
    for table in all_tables:
        combined_table.extend(table)

    df = pd.DataFrame(combined_table[1:], columns=combined_table[0])
    df = df[df['Date'] != 'Date']
    df = df.fillna('')
    df = df.astype({'Date':'str', 'Description':'str', 'Debit Amt':'str', 'Credit Amt':'str', 'Balance':'str'})
    df = df.rename(columns={'Date': 'Date', 'Description': 'Description', 'Debit Amt': 'Debit Amt', 'Credit Amt': 'Credit Amt', 'Balance': 'Balance'})
    return df