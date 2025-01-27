import sqlite3
import random

CHARACTER_SUPPORT_DB = "../chardb/chardb.sqlite"

# load font support

conn = sqlite3.connect(CHARACTER_SUPPORT_DB)
cursor = conn.cursor()


def close_conn():
    conn.close()


# character extraction

def extract_random_font():
    # Check if the fonts table is empty
    cursor.execute("SELECT COUNT(*) FROM fonts")
    fonts_count = cursor.fetchone()[0]
    if fonts_count == 0:
        conn.close()
        raise AssertionError("Empty fonts table")
    random_offset = random.randint(0, fonts_count - 1)
    query = """
        SELECT id, filename 
        FROM fonts
        LIMIT 1 OFFSET ?
    """
    cursor.execute(query, (random_offset,))
    row = cursor.fetchone()
    return {"id": row[0], "filename": row[1]}


def extract_random_character(font=None):
    """
       Extract a random character. Optionally filter by a specific font.

       Args:
           font (dict, optional): A dictionary containing the font's 'id' and 'filename'.
                                  If provided, only characters associated with this font
                                  will be considered.

       Returns:
           tuple: (character, radical, font, query_character)
       """
    if font:
        cursor.execute(
            "SELECT COUNT(*) FROM font_support WHERE font_id = ?",
            (font["id"],)
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM font_support")

    count = cursor.fetchone()[0]
    if count == 0:
        conn.close()
        raise AssertionError("No entries found in font_support table for the given criteria")

    # Get a random offset
    random_offset = random.randint(0, count - 1)

    query = """
            SELECT 
                fs.id, 
                c.character, 
                f.filename AS font, 
                fs.query_character, 
                c.radical
            FROM font_support fs
            INNER JOIN characters c ON fs.character_id = c.id
            INNER JOIN fonts f ON fs.font_id = f.id
        """

    parameters = []

    if font:
        query += """
        WHERE fs.font_id = ?
        """
        parameters = [font["id"]]

    query += """
        LIMIT 1 OFFSET ?
    """

    parameters.append(random_offset)

    cursor.execute(query, parameters)

    row = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    row_dict = dict(zip(columns, row))
    return row_dict["character"], row_dict["radical"], row_dict["font"], row_dict["query_character"]

