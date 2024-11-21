import sqlite3

DB_NAME = "chardb.sqlite"

def remove_unassociated_characters():
    """Remove characters from the Characters table that have no font association."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Delete characters from the Characters table where no font is associated in the FontSupport table
    cursor.execute(
        """
        DELETE FROM characters
        WHERE id NOT IN (
            SELECT DISTINCT character_id
            FROM font_support
        )
        """
    )

    # Commit changes and close the connection
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()

    print(f"Removed {deleted_rows} characters from the characters table.")


if __name__ == "__main__":
    remove_unassociated_characters()
