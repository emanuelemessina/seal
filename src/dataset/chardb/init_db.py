import sqlite3

# File paths
KANGXI_FILE = "kangxizidian-v3f.txt"
DB_FILE = "chardb.sqlite"
CJK_RANGE = range(0x4E00, 0xA000)

# Create a database connection and table
def create_database():
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create a table for storing Chinese characters with their radicals
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character TEXT UNIQUE NOT NULL,
            radical TEXT NOT NULL
        )
        """
    )

    # Commit changes and close the connection
    conn.commit()
    conn.close()


def extract_characters_and_radicals():
    """
    Extract standalone characters and their radicals from the Kangxi dictionary file.

    :return: A list of tuples where each tuple is (character, radical)
    """
    print("Loading 康熙字典...")

    with open(KANGXI_FILE, "r", encoding="utf-8") as f:
        kangxi_lines = f.readlines()

    characters_and_radicals = []

    print("Processing lines...")

    for line in kangxi_lines:
        if line.strip():  # Ignore empty lines
            parts = line.split("\t")
            main_character = parts[0]

            try:
                if ord(main_character) not in CJK_RANGE:
                    continue
            except TypeError:  # not a character entry
                continue

            # Extract the radical information (this is typically after the "【" symbol and before the "】")
            if "【" in line and "】" in line:
                radical_info = line.split("【")[1].split("】")[0][0]
            else:
                radical_info = ""  # Fallback if no radical info is found (could be adjusted as needed)

            characters_and_radicals.append((main_character, radical_info))

    return characters_and_radicals


def populate_database():
    # Extract characters and their radicals from the Kangxi dictionary
    characters_and_radicals = extract_characters_and_radicals()

    # Connect to the SQLite database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # Insert characters into the database
        for character, radical in characters_and_radicals:
            try:
                # Insert the character and its radical into the database
                cursor.execute(
                    "INSERT OR IGNORE INTO characters (character, radical) VALUES (?, ?)",
                    (character, radical),
                )
            except sqlite3.IntegrityError:
                print(f"Skipping duplicate or invalid character: {character}")

        # Commit changes
        conn.commit()
    finally:
        # Close the connection
        conn.close()


if __name__ == "__main__":
    # Step 1: Create the database and table
    create_database()

    # Step 2: Populate the database from the Kangxi dictionary
    populate_database()

    print(f"Database '{DB_FILE}' has been created and populated.")
