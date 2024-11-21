import csv
import os
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
from hanzipy.dictionary import HanziDictionary
import sqlite3
import logging
# Suppress debug messages from fontTools
logging.getLogger('fontTools').setLevel(logging.WARNING)

FONTS_DIR = "../fonts"
DB_FILE = "chardb.sqlite"


def can_render_character(font, character):
    """Check if the font can render a specific character and the glyph is not blank."""
    for table in font['cmap'].tables:
        glyph_name = table.cmap.get(ord(character))
        if glyph_name:
            try:
                # Check if the glyph has outline data
                glyf_table = font['glyf']
                glyph = glyf_table[glyph_name]

                glyphset = font.getGlyphSet()
                bp = BoundsPen(glyphset)
                glyphset[glyph_name].draw(bp)
                bs = bp.bounds
                if bs:
                    return True
                else:
                    return False
            except KeyError:
                # Glyph is missing or inaccessible
                return False
    return False


dictionary = HanziDictionary()


def get_simplified(character):
    return dictionary.definition_lookup(character, "traditional")[0]["simplified"]


def create_font_support_table():
    """Create a table to store font-character-query relationships."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS font_support (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character TEXT NOT NULL,
            font TEXT NOT NULL,
            query_character TEXT NOT NULL,
            UNIQUE(character, font, query_character)
        )
        """
    )

    conn.commit()
    conn.close()


def associate_characters_with_fonts():
    """Associate characters with fonts and their query characters."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Get all characters from the Characters table
    cursor.execute("SELECT character FROM characters")
    characters = [row[0] for row in cursor.fetchall()]

    # Process each font
    for file_name in os.listdir(FONTS_DIR):
        if file_name.lower().endswith('.ttf') or file_name.lower().endswith('.otf'):
            font_path = os.path.join(FONTS_DIR, file_name)
            print(f"Processing font: {file_name} ...")

            try:
                font = TTFont(font_path)
            except Exception as e:
                print(f"Failed to read font {file_name}: {e}")
                continue

            for character in characters:
                query_character = character

                # Check if the font can render the character
                if not can_render_character(font, character):
                    try:
                        # If not, try the simplified version
                        simplified = get_simplified(character)
                        if simplified and can_render_character(font, simplified):
                            query_character = simplified
                        else:
                            continue
                    except KeyError:
                        # Character is not in the dictionary
                        continue

                # Insert the font-character-query association into the database
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO font_support (character, font, query_character)
                        VALUES (?, ?, ?)
                        """,
                        (character, file_name, query_character),
                    )
                except sqlite3.IntegrityError as e:
                    print(f"Failed to insert association: {e}")

            print(f"Finished processing font: {file_name}.")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_font_support_table()
    associate_characters_with_fonts()
