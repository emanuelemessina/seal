import sqlite3

DB_NAME = "chardb.sqlite"


def remove_n_reprs(n):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT character_id
        FROM font_support
        GROUP BY character_id
        HAVING COUNT(*) = ?
        """, (n,)
    )

    n_repr_ids = [row[0] for row in cursor.fetchall()]

    if n_repr_ids:
        # Remove corresponding entries from font_support table
        cursor.execute(
            """
            DELETE FROM font_support
            WHERE character_id IN ({})
            """.format(",".join("?" for _ in n_repr_ids)),
            n_repr_ids
        )

        # Remove corresponding entries from characters table
        cursor.execute(
            """
            DELETE FROM characters
            WHERE id IN ({})
            """.format(",".join("?" for _ in n_repr_ids)),
            n_repr_ids
        )

        # Get the count of deleted characters
        deleted_rows = cursor.rowcount
    else:
        deleted_rows = 0

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print(f"Removed {deleted_rows} characters from the characters table.")

# all characters supported by less than this (inclusive) will be dropped
max_supported_fonts = 9

if __name__ == "__main__":
    for n in range(1, max_supported_fonts + 1):
        remove_n_reprs(n)
