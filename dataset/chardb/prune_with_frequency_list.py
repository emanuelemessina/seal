import sqlite3
import pandas as pd

DB_NAME = "chardb.sqlite"
FREQUENCY_LIST_FILE = "CharFreq-Classical.xls"
N = 3000  # Number of characters to keep


def get_top_n_characters_from_excel(n):
    # Read the Excel file, skipping the first two rows
    df = pd.read_excel(FREQUENCY_LIST_FILE, skiprows=2, usecols=[1], header=None)
    # Get the first n characters from the second column
    top_n_characters = df.iloc[:n, 0].tolist()
    return top_n_characters


def prune_database_with_frequency_list(n):
    top_n_characters = get_top_n_characters_from_excel(n)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Remove characters not in the top n list from the font_support table
    cursor.execute(
        """
        DELETE FROM font_support
        WHERE character_id NOT IN (
            SELECT id FROM characters WHERE character IN ({})
        )
        """.format(",".join("?" for _ in top_n_characters)),
        top_n_characters
    )

    deleted_rows = cursor.rowcount

    # Remove characters not in the top n list from the characters table
    cursor.execute(
        """
        DELETE FROM characters
        WHERE character NOT IN ({})
        """.format(",".join("?" for _ in top_n_characters)),
        top_n_characters
    )

    deleted_rows = cursor.rowcount

    print(f"Removed {deleted_rows} characters from the characters table.")

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print(f"Kept only the first {n} characters from the frequency list in the database.")


if __name__ == "__main__":
    prune_database_with_frequency_list(N)
